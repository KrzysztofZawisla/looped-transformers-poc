# looped-transformers-poc

A from-scratch proof of concept for training a language model whose depth is a **loop**:
a small stack of transformer blocks is applied repeatedly, sharing weights across
iterations. Fewer unique parameters, more compute per token, and a knob at inference
time (`n_loops`) that trades latency for quality.

The repo trains on TinyStories (English) and includes a parameter-matched vanilla
baseline so the loop can be compared against a normal stack.

## Architecture

```
tokens ──► embedding
              │
        ┌─────▼──────┐
        │  prelude   │   n_prelude unique blocks
        └─────┬──────┘
              │  anchor ────────────────────────────┐
        ┌─────▼──────┐                              │
        │    core    │   n_core unique blocks       │
        │   loop 1   │◄── + sinusoidal loop code    │
        └─────┬──────┘                              │
              │  + inject(anchor) ◄─────────────────┘
        ┌─────▼──────┐
        │   loop 2   │   same weights, next loop code
        └─────┬──────┘
              │
             ...
        ┌─────▼──────┐
        │    coda    │   n_coda unique blocks
        └─────┬──────┘
              ▼
           LM head
```

Per forward pass, the residual stream after the prelude is passed through the shared
core `n_loops` times:

```
h_0 = prelude(embedding(x))
h_{t+1} = core(h_t + loop_encoding(t)) + inject(h_0)
logits = head(coda(h_T))
```

Key details:

- **Weight sharing** across loop iterations is what makes the model "looped".
- **Input injection** (`inject`, zero-initialised) re-anchors every iteration to the
  prelude state, which keeps repeated application stable at higher loop counts.
- **Loop encoding** tells the shared block which iteration it is on. Sinusoidal by
  default, so loops can be extrapolated beyond the trained count.
- **Auxiliary LM losses** at the last `aux_loops` iterations (logit lens). They make
  intermediate states decodable and are the basis for early exit.
- **Deep-residual init**: residual projections are scaled by `1/sqrt(2 * virtual_depth)`.

With `n_loops=8`, `n_prelude=2`, `n_core=2`, `n_coda=2` the model has **6 unique
blocks but 20 virtual layers**. Parameters ~55.7M; the vanilla baseline at 6 layers has
~55.1M, and `vanilla_deep_50m.yaml` (20 layers) matches compute instead of parameters.

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install torch  # CUDA build: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

### 1. Data

```bash
python prepare_data.py --out-dir data/tinystories
```

Downloads `roneneldan/TinyStories`, trains a 16k byte-level BPE tokenizer on a sample of
stories, and writes packed `train.bin` / `val.bin` (uint16) plus `meta.json`.

### 2. Sanity check (no data needed)

```bash
python -m loopedlm.train --config configs/smoke.yaml --dry-run
```

Builds the model, runs one forward/backward, prints parameter count.

### 3. Train

```bash
python -m loopedlm.train --config configs/looped_50m.yaml
python -m loopedlm.train --config configs/vanilla_50m.yaml
```

Checkpoints and metrics land in `runs/<name>/` (`last.pt`, `best.pt`, `metrics.jsonl`,
`config.yaml`). Resume with `--resume auto`. Override anything from the CLI:

```bash
python -m loopedlm.train --config configs/smoke.yaml \
  --set model.n_loops=6 --set train.micro_batch_size=16 --max-steps 500
```

If you hit OOM, lower `train.micro_batch_size` (keeping `batch_tokens` fixed raises
gradient accumulation), or set `train.grad_checkpoint: true`.

### 4. Sample

```bash
python -m loopedlm.generate --ckpt runs/looped_50m/best.pt --loops 8 --prompt "Once upon a time"
python -m loopedlm.generate --ckpt runs/looped_50m/best.pt --loops 2 --temperature 0.7
```

The same weights can be run with fewer loops: fewer loops = less compute and usually
worse text. With sinusoidal loop encoding the model can also be pushed past the trained
loop count.

### 5. Analyze

```bash
python -m loopedlm.analyze \
  --ckpt runs/looped_50m/best.pt runs/vanilla_50m/best.pt runs/vanilla_deep_50m/best.pt \
  --loops 1 2 4 6 8 12 16 --out runs/analysis.json
```

Prints validation loss / perplexity against number of loops, virtual depth and an
estimate of FLOPs per token, so the compute-vs-quality trade-off is explicit.

### 6. Export to Hugging Face

```bash
python export_hf.py --ckpt runs/looped_50m/best.pt \
  --out hf/looped-lm-tinystories-55M --verify
```

Writes a self-contained `transformers` repo: `model.safetensors`, `config.json`, the
custom `configuration_looped_lm.py` / `modeling_looped_lm.py` (loaded with
`trust_remote_code=True`), the tokenizer, `generation_config.json` and a model card.
`--verify` reloads the export, compares logits against the original checkpoint and
samples one continuation. Use `--dtype fp32` for an exact-precision export (bf16 is the
default and differs from the checkpoint only by rounding, ~0.1 absolute logits).

Upload it with:

```bash
pip install -U huggingface_hub
hf auth login
hf upload KrzysztofZawisla/looped-lm-tinystories-55M hf/looped-lm-tinystories-55M --repo-type model
```

Usage:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "KrzysztofZawisla/looped-lm-tinystories-55M"
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16)

inputs = tokenizer("Once upon a time", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.8,
                     top_k=50, pad_token_id=tokenizer.eos_token_id, n_loops=8)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

There is also a ready-to-run script for the exported model:

```bash
python run_hf.py --prompt "Once upon a time" --tokens 150 --loops 8
# local export instead of the Hub:
python run_hf.py --model hf/looped-lm-tinystories-55M
```

This architecture is not supported by llama.cpp / Ollama, so a GGUF file would be a
container nothing could execute. The safetensors repo with custom code is the portable
format for this model.

## Fast smoke run

```bash
python prepare_data.py --out-dir data/tiny_smoke --vocab-size 4096 \
  --max-train-stories 4000 --max-val-stories 200 --tokenizer-sample 4000
python -m loopedlm.train --config configs/smoke.yaml
python -m loopedlm.generate --ckpt runs/smoke/best.pt --tokens 60
```

This is a ~1M parameter model, useful for checking that the whole pipeline works before
committing to a full run. With 200 steps on 856k tokens it reaches val loss 4.14
(ppl 63), and the loop sweep behaves as expected: 4.57 at 1 loop, 4.22 at 2, 4.15 at 4
(training depth), 4.18 at 8 — more compute helps until the model leaves the depth it
was trained at.

## Monitoring a run

`runs/<name>/metrics.jsonl` is appended after every logging/eval interval, so it is the
live view of training:

```bash
tail -f runs/looped_50m/metrics.jsonl
# PowerShell: Get-Content runs\looped_50m\metrics.jsonl -Wait
```

`best.pt` is written whenever validation loss improves, `last.pt` every `save_every`
steps. `--resume auto` picks whichever of the two is newer (compare the step number in
the logs if you need to be sure), so after an interruption:

```bash
python -m loopedlm.train --config configs/looped_50m.yaml --resume auto
```

An explicit `--resume path/to/checkpoint.pt` always wins over the auto choice.

## What to look for

- **Loop scaling**: validation loss should drop monotonically as `n_loops` grows, with
  diminishing returns. That curve is the "test-time compute" knob.
- **Parameter efficiency**: at matched parameters, the looped model has ~3.3x the
  virtual depth of the 6-layer vanilla baseline. Compare it against both baselines:
  the 6-layer one tests the parameter-efficiency claim, the 20-layer one tests whether
  looping is as good as a real stack at the same compute.
- **Depth extrapolation**: evaluate at `n_loops` above the training value. Sinusoidal
  loop encoding usually degrades gracefully; learned loop embeddings will not.
- **Logit lens**: with `aux_loops > 0`, intermediate loops are trained to be decodable.
  Loss at early loops tells you how much of the answer is already computed by then.

## Results (TinyStories, 55.7M params)

Trained for 8000 steps (1.05B tokens, ~2.3 epochs) on one RTX 4090 at ~63k tokens/s.
Final validation loss **1.271** (perplexity **3.56**).

Validation loss as a function of the number of loops used at inference (same weights,
trained at 8 loops):

| loops | virtual depth | GFLOP/token | val loss | ppl |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 6 | 0.120 | 4.234 | 68.96 |
| 2 | 8 | 0.151 | 2.565 | 13.00 |
| 3 | 10 | 0.182 | 1.778 | 5.92 |
| 4 | 12 | 0.214 | 1.461 | 4.31 |
| 6 | 16 | 0.277 | 1.291 | 3.63 |
| **8** | **20** | **0.340** | **1.271** | **3.56** |
| 10 | 24 | 0.403 | 1.278 | 3.59 |
| 12 | 28 | 0.466 | 1.291 | 3.64 |
| 16 | 36 | 0.591 | 1.323 | 3.76 |

Two things to read from this:

- **Compute scaling works.** The first few loops buy a lot: 4 loops already recover
  most of the quality of 8 at 63% of the compute. Running below the trained depth
  produces word salad (2 loops: *"Nemo, however, is that a gentle push…"*), so the
  loop budget is a real quality knob, not a formality.
- **Depth extrapolation is graceful.** Pushing past the trained 8 loops (10, 12, 16)
  costs little, because the sinusoidal loop encoding is defined for any index. The
  optimum sits exactly at the trained depth, which is the expected behaviour.

The parameter-matched and compute-matched vanilla baselines were trained with the same
recipe (8000 steps, 131k tokens per step, same data):

| model | params | layers | GFLOP/token | val loss | ppl |
| --- | ---: | ---: | ---: | ---: | ---: |
| looped, 8 loops | 55.7M | 20 virtual | 0.340 | 1.2711 | 3.56 |
| vanilla, 6 layers | 55.1M | 6 | 0.120 | 1.2974 | 3.66 |
| vanilla, 20 layers | 154.2M | 20 | 0.340 | 1.2136 | 3.37 |

How to read it:

- **At matched parameters** the loop wins: 1.271 vs 1.297, but it spends 2.8x the
  compute to get there. Looping is a way to buy depth without buying parameters.
- **At matched compute** the explicit 20-layer stack wins: 1.214 vs 1.271, because
  matched FLOPs let it carry 2.8x more unique parameters. At this scale parameters are
  the cheaper resource, so the loop is not a free lunch.
- The loop is therefore a parameter-efficiency technique, not a compute-efficiency one.
  It should pay off when VRAM/disk/weights are the binding constraint, or once the
  token budget is large enough that extra capacity is what limits the plain stack.

## Design choices and knobs

| Option | Meaning |
| --- | --- |
| `model.n_loops` | loops used during training |
| `model.loop_sampling` | `fixed`, or `uniform` to sample `min_loops..n_loops` per step |
| `model.loop_encoding` | `sinusoidal` (extrapolates), `learned`, `none` |
| `model.n_prelude` / `n_core` / `n_coda` | how unique blocks are split |
| `model.aux_loops` / `aux_loss_weight` | per-loop LM supervision |
| `train.batch_tokens` | total tokens per optimizer step |
| `train.grad_checkpoint` | trade compute for memory |

`loop_sampling: uniform` trains one model to work at several depths; it is the
prerequisite for honest early-exit experiments.

On a desktop RTX 4090 the 55.7M looped config runs at roughly 64k tokens/s and peaks
at ~15.5 GB of VRAM using `micro_batch_size: 32` and `bf16`. Gradient checkpointing
lowers memory to ~10 GB but costs ~20% throughput. `torch.compile` requires Triton,
which is not bundled with PyTorch on Windows; the trainer falls back to eager mode.

## Not included (deliberate scope)

- KV cache for generation — sampling recomputes the full context each token.
- ACT / learned exit gates — auxiliary losses are there, the gating policy is not.
- Multi-GPU training.
- Instruction tuning or chat format.

## References

- Universal Transformers — Dehghani et al., 2018
- Scaling up Test-Time Compute with Latent Reasoning (recurrent depth) — Geiping et al., 2025
- Ouro / LoopLM — ByteDance, 2025
