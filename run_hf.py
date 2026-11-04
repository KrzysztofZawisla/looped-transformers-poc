#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the exported looped LM locally.")
    parser.add_argument(
        "--model",
        default="KrzysztofZawisla/looped-lm-tinystories-55M",
        help="HF repo id or a local export directory",
    )
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=150)
    parser.add_argument("--loops", type=int, default=8, help="1-16; 8 is the trained depth")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=dtype
    ).to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    for index in range(args.samples):
        torch.manual_seed(args.seed + index)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-5),
                top_k=args.top_k,
                pad_token_id=tokenizer.eos_token_id,
                n_loops=args.loops,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\n--- sample {index + 1} (loops={args.loops}) ---")
        print(text)


if __name__ == "__main__":
    main()
