from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config, load_config, save_config
from .data import TokenDataset, load_tokenizer, read_meta
from .model import LoopedLM, sample
from .utils import configure_console


def human_count(value: int) -> str:
    for unit, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= scale:
            return f"{value / scale:.1f}{unit}"
    return str(int(value))


def lr_at(step: int, cfg: Config) -> float:
    t = cfg.train
    if step < t.warmup_steps:
        return t.lr * (step + 1) / max(1, t.warmup_steps)
    if step >= t.max_steps:
        return t.min_lr
    progress = (step - t.warmup_steps) / max(1, t.max_steps - t.warmup_steps)
    return t.min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (t.lr - t.min_lr)


def pick_loops(cfg: Config, rng: random.Random) -> int | None:
    if cfg.model.model_type != "looped":
        return None
    if cfg.model.loop_sampling == "uniform":
        return rng.randint(cfg.model.min_loops, cfg.model.n_loops)
    return cfg.model.n_loops


def make_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)
    groups = [
        {"params": decay, "weight_decay": cfg.train.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.train.lr, betas=(cfg.train.beta1, cfg.train.beta2))


def infinite_batches(dataset: TokenDataset, batch_size: int, device: torch.device):
    while True:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0
        )
        for x, y in loader:
            yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(
    model: LoopedLM,
    val_loader: DataLoader,
    max_batches: int,
    device: torch.device,
    ctx,
    n_loops: int | None,
) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch_idx, (x, y) in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with ctx:
            out = model(x, targets=y, n_loops=n_loops, aux=False)
        total += float(out["loss"])
        count += 1
    model.train()
    return total / max(1, count)


def save_checkpoint(
    path: Path,
    model: LoopedLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val: float,
    tokens_seen: int,
    cfg: Config,
) -> None:
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_val": best_val,
            "tokens_seen": tokens_seen,
            "model_config": asdict(cfg.model),
            "train_config": asdict(cfg.train),
        },
        path,
    )


def resolve_checkpoint(run_dir: Path, arg: str) -> Path | None:
    if arg == "auto":
        candidates = [p for p in (run_dir / "last.pt", run_dir / "best.pt") if p.exists()]
        return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None
    return Path(arg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a looped (or vanilla) transformer LM.")
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--out-dir", default=None, help="override train.out_dir")
    parser.add_argument("--max-steps", type=int, default=None, help="override train.max_steps")
    parser.add_argument(
        "--resume", default=None, help="checkpoint path, or 'auto' for the newest of last.pt / best.pt"
    )
    parser.add_argument("--set", action="append", default=[], help="override e.g. model.n_loops=4")
    parser.add_argument("--dry-run", action="store_true", help="build the model, run one step, exit")
    return parser.parse_args()


def main() -> None:
    configure_console()
    args = parse_args()
    cfg = load_config(args.config, overrides=args.set)
    if args.out_dir:
        cfg.train.out_dir = args.out_dir
    if args.max_steps:
        cfg.train.max_steps = args.max_steps
    cfg.validate()

    torch.manual_seed(cfg.train.seed)
    torch.cuda.manual_seed_all(cfg.train.seed)
    random.seed(cfg.train.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = cfg.train.dtype == "bf16" and device.type == "cuda"
    if cfg.train.dtype == "bf16" and not use_bf16:
        print("[warn] bf16 requested but CUDA is unavailable; falling back to fp32")
    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else nullcontext()
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    model = LoopedLM(cfg.model).to(device)
    model.enable_grad_checkpointing(cfg.train.grad_checkpoint)
    model.train()
    if cfg.train.compile:
        try:
            import triton  # noqa: F401

            model = torch.compile(model)
        except ImportError:
            print("[warn] triton is not available; torch.compile disabled")

    params = model.num_parameters()
    params_non_embed = model.num_parameters(non_embedding=True)
    loop_desc = (
        f"loops {cfg.model.n_loops} (virtual depth {model.virtual_depth()})"
        if cfg.model.model_type == "looped"
        else f"{cfg.model.n_layers} layers"
    )
    print(
        f"device={device} | {cfg.model.model_type} | params {human_count(params)} "
        f"({human_count(params_non_embed)} non-embed) | {loop_desc}"
    )

    if args.dry_run:
        length = min(64, cfg.train.seq_len)
        x = torch.randint(0, cfg.model.vocab_size, (2, length), device=device)
        y = torch.randint(0, cfg.model.vocab_size, (2, length), device=device)
        with ctx:
            out = model(x, targets=y, n_loops=cfg.model.n_loops)
            loss = out["loss"] + cfg.model.aux_loss_weight * out.get(
                "aux_loss", torch.zeros((), device=device)
            )
        loss.backward()
        print(f"dry run ok | loss={loss.item():.4f} | seq_len={length}")
        return

    meta = read_meta(cfg.train.data_dir)
    if meta["vocab_size"] != cfg.model.vocab_size:
        raise ValueError(
            f"config vocab_size={cfg.model.vocab_size} does not match the tokenizer "
            f"vocab_size={meta['vocab_size']} in {cfg.train.data_dir}/meta.json"
        )

    train_ds = TokenDataset(Path(cfg.train.data_dir) / "train.bin", cfg.train.seq_len)
    val_ds = TokenDataset(Path(cfg.train.data_dir) / "val.bin", cfg.train.seq_len)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.micro_batch_size, shuffle=False, drop_last=True, num_workers=0
    )
    batches = infinite_batches(train_ds, cfg.train.micro_batch_size, device)
    tokenizer = load_tokenizer(cfg.train.data_dir)
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    run_dir = Path(cfg.train.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    metrics_path = run_dir / "metrics.jsonl"

    optimizer = make_optimizer(model, cfg)

    start_step, best_val, tokens_seen = 0, float("inf"), 0
    resume_path = resolve_checkpoint(run_dir, args.resume) if args.resume else None
    if resume_path is not None and resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        tokens_seen = ckpt.get("tokens_seen", 0)
        print(f"resumed from {resume_path} at step {start_step}")

    print(
        f"train tokens {human_count(meta['train_tokens'])} | val tokens {human_count(meta['val_tokens'])} | "
        f"batch {human_count(cfg.train.tokens_per_step)} tokens "
        f"({cfg.train.micro_batch_size} x {cfg.train.grad_accum} accum) | "
        f"steps {start_step}->{cfg.train.max_steps}"
    )

    use_wandb = cfg.train.wandb
    if use_wandb:
        import wandb

        wandb.init(project=cfg.train.wandb_project, config=cfg.to_dict(), dir=str(run_dir))

    rng = random.Random(cfg.train.seed + start_step)
    eval_loops = cfg.model.n_loops if cfg.model.model_type == "looped" else None
    prompt_ids = tokenizer.encode("Once upon a time").ids
    interval_start = time.time()
    interval_tokens = 0
    running_loss, running_aux = 0.0, 0.0
    model.train()

    step = start_step
    try:
        for step in range(start_step, cfg.train.max_steps):
            lr = lr_at(step, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr

            n_loops = pick_loops(cfg, rng)
            optimizer.zero_grad(set_to_none=True)
            step_loss, step_aux = 0.0, 0.0

            for _ in range(cfg.train.grad_accum):
                x, y = next(batches)
                with ctx:
                    out = model(x, targets=y, n_loops=n_loops)
                    loss = out["loss"]
                    if "aux_loss" in out:
                        loss = loss + cfg.model.aux_loss_weight * out["aux_loss"]
                    scaled_loss = loss / cfg.train.grad_accum
                scaled_loss.backward()
                step_loss += float(out["loss"].item()) / cfg.train.grad_accum
                if "aux_loss" in out:
                    step_aux += float(out["aux_loss"].item()) / cfg.train.grad_accum

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

            tokens_seen += cfg.train.tokens_per_step
            interval_tokens += cfg.train.tokens_per_step
            running_loss += step_loss
            running_aux += step_aux

            if cfg.train.log_every and (step + 1) % cfg.train.log_every == 0:
                elapsed = max(time.time() - interval_start, 1e-6)
                log_loss = running_loss / cfg.train.log_every
                log_aux = running_aux / cfg.train.log_every
                tok_per_s = interval_tokens / elapsed
                vram = (
                    torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
                )
                print(
                    f"step {step + 1}/{cfg.train.max_steps} | loss {log_loss:.4f} | "
                    f"aux {log_aux:.4f} | ppl {math.exp(min(log_loss, 20)):.1f} | "
                    f"lr {lr:.2e} | {tok_per_s / 1000:.1f}k tok/s | "
                    f"grad {float(grad_norm):.2f} | {vram:.1f} GB"
                )
                record = {
                    "step": step + 1,
                    "loss": log_loss,
                    "aux_loss": log_aux,
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "tokens_per_s": tok_per_s,
                    "tokens_seen": tokens_seen,
                }
                with open(metrics_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                if use_wandb:
                    wandb.log(record, step=step + 1)
                interval_start = time.time()
                interval_tokens = 0
                running_loss, running_aux = 0.0, 0.0

            if (cfg.train.eval_every and (step + 1) % cfg.train.eval_every == 0) or (
                step + 1 == cfg.train.max_steps
            ):
                val_loss = evaluate(
                    model, val_loader, cfg.train.eval_batches, device, ctx, eval_loops
                )
                print(
                    f"  eval step {step + 1} | val loss {val_loss:.4f} | "
                    f"val ppl {math.exp(min(val_loss, 20)):.1f}"
                )
                if use_wandb:
                    wandb.log({"val_loss": val_loss}, step=step + 1)
                with open(metrics_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": step + 1, "val_loss": val_loss}) + "\n")
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(
                        run_dir / "best.pt", model, optimizer, step, best_val,
                        tokens_seen, cfg,
                    )

            if cfg.train.save_every and (step + 1) % cfg.train.save_every == 0:
                save_checkpoint(
                    run_dir / "last.pt", model, optimizer, step, best_val, tokens_seen, cfg
                )

            if cfg.train.sample_every and (step + 1) % cfg.train.sample_every == 0:
                prompt = torch.tensor([prompt_ids], device=device)
                generated = sample(
                    model,
                    prompt,
                    max_new_tokens=cfg.train.sample_tokens,
                    temperature=0.8,
                    top_k=50,
                    n_loops=n_loops,
                    eos_id=eos_id,
                )
                text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
                print("  sample:", text.replace("\n", " ")[:300])
                model.train()

    except KeyboardInterrupt:
        save_checkpoint(
            run_dir / "interrupt.pt", model, optimizer, step, best_val, tokens_seen, cfg
        )
        print("interrupted; checkpoint saved to interrupt.pt")
        return

    save_checkpoint(
        run_dir / "last.pt", model, optimizer, cfg.train.max_steps - 1, best_val, tokens_seen, cfg
    )
    print(f"done | best val loss {best_val:.4f}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
