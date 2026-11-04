from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


def read_meta(data_dir: str | Path) -> dict:
    path = Path(data_dir) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python prepare_data.py --out-dir {data_dir}` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_tokenizer(data_dir: str | Path) -> Tokenizer:
    path = Path(data_dir) / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python prepare_data.py --out-dir {data_dir}` first."
        )
    return Tokenizer.from_file(str(path))


class TokenDataset(Dataset):
    def __init__(self, bin_path: str | Path, seq_len: int) -> None:
        self.bin_path = Path(bin_path)
        if not self.bin_path.exists():
            raise FileNotFoundError(f"token file {self.bin_path} does not exist")
        self.seq_len = seq_len
        self.tokens = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        self.n_chunks = max(0, (len(self.tokens) - 1) // seq_len)
        if self.n_chunks == 0:
            raise ValueError(
                f"{self.bin_path} holds {len(self.tokens)} tokens, "
                f"not enough for seq_len={seq_len}"
            )

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.seq_len
        chunk = np.asarray(self.tokens[start : start + self.seq_len + 1], dtype=np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])
