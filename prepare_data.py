from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from tqdm import tqdm

SPECIAL_TOKEN = "<|endoftext|>"


def resolve_files(repo: str) -> tuple[list[str], list[str]]:
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    train = sorted(f for f in files if f.startswith("data/train-") and f.endswith(".parquet"))
    valid = sorted(f for f in files if f.startswith("data/validation-") and f.endswith(".parquet"))
    if not train or not valid:
        raise RuntimeError(
            f"{repo} does not expose data/train-*.parquet and data/validation-*.parquet"
        )
    return train, valid


def iter_texts(repo: str, filenames: list[str], limit: int = -1, batch_size: int = 2000):
    emitted = 0
    for filename in filenames:
        path = Path(hf_hub_download(repo, filename, repo_type="dataset"))
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["text"]):
            for text in batch.column("text").to_pylist():
                yield text
                emitted += 1
                if 0 < limit <= emitted:
                    return


def batched(iterable, size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_tokenizer(texts: list[str], vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[SPECIAL_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def tokenize_split(
    texts,
    tokenizer: Tokenizer,
    eos_id: int,
    out_path: Path,
    batch_size: int,
    desc: str,
    total: int | None = None,
) -> tuple[int, int]:
    stories, tokens = 0, 0
    with open(out_path, "wb") as handle, tqdm(
        desc=desc, unit="stories", total=total if total and total > 0 else None
    ) as bar:
        for batch in batched(texts, batch_size):
            encodings = tokenizer.encode_batch(batch)
            arrays = [np.asarray(e.ids + [eos_id], dtype=np.uint16) for e in encodings]
            buffer = np.concatenate(arrays)
            buffer.tofile(handle)
            stories += len(batch)
            tokens += int(buffer.size)
            bar.update(len(batch))
    return stories, tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and tokenize TinyStories.")
    parser.add_argument("--out-dir", default="data/tinystories")
    parser.add_argument("--repo", default="roneneldan/TinyStories")
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--tokenizer-sample", type=int, default=200_000)
    parser.add_argument("--max-train-stories", type=int, default=-1)
    parser.add_argument("--max-val-stories", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.vocab_size < 300:
        raise ValueError("vocab_size must be at least 300 for byte-level BPE")
    if args.vocab_size > 65535:
        raise ValueError("vocab_size must fit in uint16 token storage")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_files, val_files = resolve_files(args.repo)
    print(f"{args.repo}: {len(train_files)} train shards, {len(val_files)} validation shards")

    sample_n = args.tokenizer_sample
    print(f"training BPE tokenizer on up to {sample_n:,} stories (vocab {args.vocab_size})")
    sample = list(iter_texts(args.repo, train_files, limit=sample_n))
    tokenizer = build_tokenizer(sample, args.vocab_size)
    tokenizer.save(str(out_dir / "tokenizer.json"))
    eos_id = tokenizer.token_to_id(SPECIAL_TOKEN)
    print(f"tokenizer vocab: {tokenizer.get_vocab_size():,} | eos id: {eos_id}")

    train_stories, train_tokens = tokenize_split(
        iter_texts(args.repo, train_files, limit=args.max_train_stories),
        tokenizer,
        eos_id,
        out_dir / "train.bin",
        args.batch_size,
        "train",
        total=args.max_train_stories,
    )
    val_stories, val_tokens = tokenize_split(
        iter_texts(args.repo, val_files, limit=args.max_val_stories),
        tokenizer,
        eos_id,
        out_dir / "val.bin",
        args.batch_size,
        "val",
        total=args.max_val_stories,
    )

    meta = {
        "dataset": args.repo,
        "vocab_size": tokenizer.get_vocab_size(),
        "eos_id": eos_id,
        "train_stories": train_stories,
        "val_stories": val_stories,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "tokenizer_file": "tokenizer.json",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"done | train {train_stories:,} stories / {train_tokens:,} tokens | "
        f"val {val_stories:,} stories / {val_tokens:,} tokens | files in {out_dir}"
    )
    if meta["vocab_size"] != args.vocab_size:
        print(
            f"[warn] tokenizer produced vocab {meta['vocab_size']} instead of {args.vocab_size}; "
            f"set model.vocab_size to {meta['vocab_size']} in your config"
        )


if __name__ == "__main__":
    main()
