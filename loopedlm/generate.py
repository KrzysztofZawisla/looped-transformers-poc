from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import load_tokenizer
from .model import load_model, sample
from .utils import configure_console


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from a trained looped LM.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--loops", type=int, default=None, help="override the number of loops")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    configure_console()
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, ckpt = load_model(args.ckpt, device)
    data_dir = Path(
        args.data_dir
        or ckpt.get("train_config", {}).get("data_dir", "data/tinystories")
    )
    tokenizer = load_tokenizer(data_dir)
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    loops = args.loops if args.loops is not None else model.cfg.n_loops
    prompt_ids = tokenizer.encode(args.prompt).ids
    print(
        f"checkpoint={args.ckpt} | type={model.cfg.model_type} | loops={loops} | "
        f"virtual depth={model.virtual_depth(loops)}"
    )

    for index in range(args.num_samples):
        torch.manual_seed(args.seed + index)
        ids = torch.tensor([prompt_ids], device=device)
        output = sample(
            model,
            ids,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            n_loops=loops,
            eos_id=eos_id,
        )
        text = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        print(f"\n--- sample {index + 1} ---")
        print(text)


if __name__ == "__main__":
    main()
