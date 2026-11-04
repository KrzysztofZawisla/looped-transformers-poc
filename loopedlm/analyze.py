from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import ModelConfig
from .data import TokenDataset, read_meta
from .model import load_model
from .utils import configure_console


def estimate_flops_per_token(cfg: ModelConfig, seq_len: int, loops: int) -> float:
    if cfg.model_type == "vanilla":
        blocks = cfg.n_layers
    else:
        blocks = cfg.n_prelude + cfg.n_core * loops + cfg.n_coda
    linear = 2 * (4 * cfg.d_model**2 + 3 * cfg.d_model * cfg.mlp_width) * blocks
    attention = 4 * seq_len * cfg.d_model * blocks
    head = 2 * cfg.d_model * cfg.vocab_size
    return linear + attention + head


@torch.no_grad()
def eval_loss(model, loader, max_batches, device, ctx, n_loops) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with ctx:
            out = model(x, targets=y, n_loops=n_loops, aux=False)
        total += float(out["loss"])
        count += 1
    return total / max(1, count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate loss vs. number of loops for one or more checkpoints."
    )
    parser.add_argument("--ckpt", nargs="+", required=True)
    parser.add_argument("--loops", nargs="+", type=int, default=[1, 2, 4, 6, 8, 12, 16])
    parser.add_argument("--eval-batches", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None, help="optional path for a JSON report")
    return parser.parse_args()


def main() -> None:
    configure_console()
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    rows = []
    for ckpt_path in args.ckpt:
        model, ckpt = load_model(ckpt_path, device)
        train_cfg = ckpt.get("train_config", {})
        data_dir = Path(args.data_dir or train_cfg.get("data_dir", "data/tinystories"))
        seq_len = args.seq_len or train_cfg.get("seq_len", model.cfg.max_seq_len)
        meta = read_meta(data_dir)
        if meta["vocab_size"] != model.cfg.vocab_size:
            raise ValueError(
                f"{ckpt_path}: model vocab {model.cfg.vocab_size} != tokenizer vocab {meta['vocab_size']}"
            )
        dataset = TokenDataset(data_dir / "val.bin", seq_len)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=0
        )
        label = Path(ckpt_path).parent.name
        step = ckpt.get("step", "?")
        print(
            f"\n{ckpt_path} | type={model.cfg.model_type} | step={step} | "
            f"params={model.num_parameters() / 1e6:.1f}M"
        )

        sweep = [model.cfg.n_loops] if model.cfg.model_type == "vanilla" else args.loops
        for loops in sweep:
            loss = eval_loss(model, loader, args.eval_batches, device, ctx, loops)
            depth = model.virtual_depth(loops)
            flops = estimate_flops_per_token(model.cfg, seq_len, loops)
            rows.append(
                {
                    "checkpoint": ckpt_path,
                    "label": label,
                    "step": step,
                    "model_type": model.cfg.model_type,
                    "loops": loops if model.cfg.model_type == "looped" else None,
                    "virtual_depth": depth,
                    "flops_per_token": flops,
                    "val_loss": loss,
                    "perplexity": math.exp(min(loss, 20)),
                }
            )

    header = f"{'run':<18} {'loops':>6} {'depth':>6} {'GFLOP/tok':>11} {'val loss':>10} {'ppl':>10}"
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        loops = row["loops"] if row["loops"] is not None else "-"
        print(
            f"{row['label']:<18} {str(loops):>6} {row['virtual_depth']:>6} "
            f"{row['flops_per_token'] / 1e9:>11.3f} {row['val_loss']:>10.4f} {row['perplexity']:>10.2f}"
        )

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
