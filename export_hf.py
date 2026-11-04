#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import PreTrainedTokenizerFast

SPECIAL_TOKEN = "<|endoftext|>"
TEMPLATE_DIR = Path(__file__).resolve().parent / "hf_export"
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
DTYPE_NAMES = {"bf16": "bfloat16", "fp32": "float32"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a looped LM checkpoint to a HF model repo.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="hf/looped-lm-tinystories-55M")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--data-dir", default=None, help="directory with tokenizer.json")
    parser.add_argument("--analysis", default=None, help="optional analysis.json for the model card")
    parser.add_argument("--verify", action="store_true", help="load the export and compare logits")
    return parser.parse_args()


def build_config(model_cfg: dict, dtype: str) -> dict:
    hidden = model_cfg.get("mlp_hidden")
    if hidden is None:
        hidden = int(model_cfg.get("mlp_mult", 8.0 / 3.0) * model_cfg["d_model"])
    hidden = max(64, (hidden + 63) // 64 * 64)
    depth = model_cfg["n_prelude"] + model_cfg["n_core"] * model_cfg["n_loops"] + model_cfg["n_coda"]
    return {
        "architectures": ["LoopedLMForCausalLM"],
        "model_type": "looped_lm",
        "hidden_size": model_cfg["d_model"],
        "intermediate_size": hidden,
        "num_hidden_layers": depth,
        "num_attention_heads": model_cfg["n_heads"],
        "use_cache": False,
        "auto_map": {
            "AutoConfig": "configuration_looped_lm.LoopedLMConfig",
            "AutoModelForCausalLM": "modeling_looped_lm.LoopedLMForCausalLM",
        },
        "vocab_size": model_cfg["vocab_size"],
        "d_model": model_cfg["d_model"],
        "n_heads": model_cfg["n_heads"],
        "head_dim": model_cfg["head_dim"],
        "n_prelude": model_cfg["n_prelude"],
        "n_core": model_cfg["n_core"],
        "n_coda": model_cfg["n_coda"],
        "n_loops": model_cfg["n_loops"],
        "min_loops": model_cfg.get("min_loops", 1),
        "max_loops": model_cfg.get("max_loops", 32),
        "loop_encoding": model_cfg.get("loop_encoding", "sinusoidal"),
        "use_input_injection": model_cfg.get("use_input_injection", True),
        "mlp_hidden": model_cfg.get("mlp_hidden"),
        "mlp_mult": model_cfg.get("mlp_mult", 8.0 / 3.0),
        "rope_theta": model_cfg.get("rope_theta", 10000.0),
        "max_seq_len": model_cfg.get("max_seq_len", 512),
        "max_position_embeddings": model_cfg.get("max_seq_len", 512),
        "dropout": model_cfg.get("dropout", 0.0),
        "tie_word_embeddings": False,
        "aux_loops": model_cfg.get("aux_loops", 2),
        "aux_loss_weight": model_cfg.get("aux_loss_weight", 0.25),
        "dtype": DTYPE_NAMES[dtype],
        "torch_dtype": DTYPE_NAMES[dtype],
    }


def build_model_card(
    model_cfg: dict,
    num_params: int,
    best_val: float | None,
    analysis_path: Path | None,
    train_cfg: dict,
) -> str:
    rows = []
    if analysis_path is not None and analysis_path.exists():
        for row in json.loads(analysis_path.read_text(encoding="utf-8")):
            rows.append(
                f"| {row['loops']} | {row['virtual_depth']} | "
                f"{row['flops_per_token'] / 1e9:.3f} | {row['val_loss']:.4f} | {row['perplexity']:.2f} |"
            )
    table = "\n".join(rows) if rows else "| 1 | - | - | - | - |"
    val_text = f"{best_val:.4f}" if best_val is not None else "n/a"
    ppl_text = f"{math.exp(min(best_val, 20)):.2f}" if best_val is not None else "n/a"
    steps = train_cfg.get("max_steps", "?")
    tokens = train_cfg.get("batch_tokens")
    tokens_text = f"{steps} steps x {tokens:,} tokens" if tokens else f"{steps} steps"
    total_text = f"~{steps * tokens / 1e9:.2f}B tokens" if tokens and isinstance(steps, int) else ""
    return f"""---
language: en
library_name: transformers
pipeline_tag: text-generation
tags:
  - looped-transformer
  - recurrent-depth
  - tinystories
  - custom-architecture
---

# Looped LM — {num_params / 1e6:.1f}M (TinyStories)

A decoder-only language model whose depth is a **loop**: {model_cfg['n_prelude']} prelude
blocks, {model_cfg['n_core']} shared core blocks applied {model_cfg['n_loops']} times, and
{model_cfg['n_coda']} coda blocks. Only {model_cfg['n_prelude'] + model_cfg['n_core'] + model_cfg['n_coda']}
unique transformer blocks exist, yet the model computes {model_cfg['n_prelude'] + model_cfg['n_core'] * model_cfg['n_loops'] + model_cfg['n_coda']}
virtual layers at inference.

This is a research checkpoint, not a production assistant. It was trained from scratch on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (English children's
stories) with a 16k byte-level BPE tokenizer.

## Architecture

```
h_0 = prelude(embedding(input))
h_(t+1) = core(h_t + loop_encoding(t)) + inject(h_0)
logits = head(coda(h_T))
```

- shared core weights across loop iterations,
- sinusoidal loop encoding (defined for any `t`, so loops extrapolate),
- input injection re-anchors every iteration to the prelude state,
- auxiliary LM heads on the last {model_cfg.get('aux_loops', 0)} loops during training (logit lens).

## Training

- {tokens_text} ({total_text}), batch 131k tokens, AdamW, cosine schedule, bf16.
- Final validation loss **{val_text}** (perplexity {ppl_text}) on TinyStories validation.

## Loop scaling

Same weights, different number of loops at inference:

| loops | virtual depth | GFLOP/token | val loss | ppl |
| ---: | ---: | ---: | ---: | ---: |
{table}

Quality improves sharply with the first loops and peaks at the trained depth (8). Pushing
past 8 degrades gracefully thanks to the sinusoidal loop encoding.

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "KrzysztofZawisla/looped-lm-tinystories-55M"
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16)

inputs = tokenizer("Once upon a time", return_tensors="pt")
output = model.generate(
    **inputs,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    pad_token_id=tokenizer.eos_token_id,
    n_loops=8,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

`n_loops` is a per-call compute knob: 4 loops already recover most of the quality at ~63%
of the compute, 1-2 loops produce degraded text.

## Limitations

- TinyStories only: simple English, short narratives, no world knowledge.
- No KV cache: generation recomputes the full context for every token.
- The architecture is custom and therefore requires `trust_remote_code=True`; it is not
  runnable by llama.cpp / Ollama / vLLM without a dedicated implementation.
- The checkpoint stores the tied output matrix explicitly (`tie_word_embeddings: false`
  in `config.json`); its values are identical to the token embedding.
- Trained on ~1B tokens, so it is not competitive with modern small LMs.
"""


def verify(out_dir: Path, ckpt_path: str, dtype: torch.dtype) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from loopedlm.model import load_model

    hf_model = AutoModelForCausalLM.from_pretrained(
        out_dir, trust_remote_code=True, dtype=dtype
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(out_dir, trust_remote_code=True)
    ids = tokenizer("Once upon a time, there was", return_tensors="pt").input_ids

    with torch.no_grad():
        hf_logits = hf_model(ids).logits
    original, _ = load_model(ckpt_path, device="cpu")
    original = original.to(dtype).eval()
    with torch.no_grad():
        ref_logits = original(ids)["logits"]
    diff = (hf_logits.float() - ref_logits.float()).abs().max().item()
    scale = ref_logits.float().abs().max().item()
    print(f"max |logit diff| vs checkpoint: {diff:.6f} (max |logit| {scale:.2f})")
    if diff > 0.01 * scale + 0.05:
        raise SystemExit("exported model diverges from the checkpoint")

    with torch.no_grad():
        output = hf_model.generate(
            ids,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
            n_loops=8,
        )
    print("sample:", tokenizer.decode(output[0], skip_special_tokens=True))
    print("verification passed")


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_cfg = dict(ckpt["model_config"])
    train_cfg = dict(ckpt.get("train_config", {}))
    state = dict(ckpt["model"])
    tied = bool(model_cfg.get("tie_embeddings", True))
    if tied:
        state["lm_head.weight"] = state["tok_emb.weight"].clone()

    dtype = DTYPES[args.dtype]
    state = {name: tensor.to(dtype).contiguous() for name, tensor in state.items()}
    num_params = sum(tensor.numel() for tensor in state.values())
    if tied:
        num_params -= state["lm_head.weight"].numel()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(state, str(out_dir / "model.safetensors"), metadata={"format": "pt"})

    config = build_config(model_cfg, args.dtype)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for name in ("configuration_looped_lm.py", "modeling_looped_lm.py"):
        shutil.copy2(TEMPLATE_DIR / name, out_dir / name)

    data_dir = Path(args.data_dir or train_cfg.get("data_dir", "data/tinystories"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(data_dir / "tokenizer.json"),
        eos_token=SPECIAL_TOKEN,
        bos_token=SPECIAL_TOKEN,
        pad_token=SPECIAL_TOKEN,
        model_max_length=model_cfg.get("max_seq_len", 512),
    )
    tokenizer.save_pretrained(out_dir)
    eos_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKEN)
    (out_dir / "generation_config.json").write_text(
        json.dumps(
            {
                "bos_token_id": eos_id,
                "eos_token_id": eos_id,
                "pad_token_id": eos_id,
                "use_cache": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    analysis_path = Path(args.analysis) if args.analysis else Path(args.ckpt).parent / "analysis.json"
    card = build_model_card(model_cfg, num_params, ckpt.get("best_val"), analysis_path, train_cfg)
    (out_dir / "README.md").write_text(card, encoding="utf-8")

    print(
        f"exported {num_params / 1e6:.1f}M params to {out_dir} "
        f"({sum(p.stat().st_size for p in out_dir.iterdir() if p.is_file()) / 2**20:.0f} MB)"
    )
    if args.verify:
        verify(out_dir, args.ckpt, dtype)


if __name__ == "__main__":
    main()
