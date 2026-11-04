from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    model_type: str = "looped"
    vocab_size: int = 16384
    d_model: int = 768
    n_heads: int = 12
    head_dim: int = 64
    n_prelude: int = 2
    n_core: int = 2
    n_coda: int = 2
    n_layers: int = 6
    n_loops: int = 8
    min_loops: int = 1
    max_loops: int = 32
    loop_encoding: str = "sinusoidal"
    loop_sampling: str = "fixed"
    use_input_injection: bool = True
    mlp_mult: float = 8.0 / 3.0
    mlp_hidden: int | None = None
    rope_theta: float = 10000.0
    max_seq_len: int = 512
    dropout: float = 0.0
    tie_embeddings: bool = True
    aux_loops: int = 2
    aux_loss_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.model_type not in ("looped", "vanilla"):
            raise ValueError(f"model_type must be 'looped' or 'vanilla', got {self.model_type!r}")
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError(
                f"d_model ({self.d_model}) must equal n_heads * head_dim "
                f"({self.n_heads} * {self.head_dim})"
            )
        if self.loop_encoding not in ("sinusoidal", "learned", "none"):
            raise ValueError(f"unknown loop_encoding {self.loop_encoding!r}")
        if self.loop_encoding == "sinusoidal" and self.d_model % 2:
            raise ValueError("sinusoidal loop encoding requires an even d_model")
        if self.loop_sampling not in ("fixed", "uniform"):
            raise ValueError(f"unknown loop_sampling {self.loop_sampling!r}")
        if self.model_type == "looped":
            if self.n_core < 1:
                raise ValueError("n_core must be >= 1 for looped models")
            if self.max_loops < self.n_loops:
                raise ValueError("max_loops must be >= n_loops")
            if not 1 <= self.min_loops <= self.n_loops:
                raise ValueError("require 1 <= min_loops <= n_loops")
            if not 0 <= self.aux_loops <= self.n_loops:
                raise ValueError("require 0 <= aux_loops <= n_loops")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be >= 1")

    @property
    def mlp_width(self) -> int:
        hidden = self.mlp_hidden if self.mlp_hidden is not None else int(self.mlp_mult * self.d_model)
        return max(64, (hidden + 63) // 64 * 64)


@dataclass
class TrainConfig:
    data_dir: str = "data/tinystories"
    out_dir: str = "runs/looped_50m"
    seq_len: int = 512
    batch_tokens: int = 131072
    micro_batch_size: int = 32
    max_steps: int = 8000
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dtype: str = "bf16"
    grad_checkpoint: bool = False
    compile: bool = False
    eval_every: int = 250
    eval_batches: int = 40
    save_every: int = 1000
    log_every: int = 10
    sample_every: int = 1000
    sample_tokens: int = 80
    seed: int = 1337
    wandb: bool = False
    wandb_project: str = "looped-lm"

    def __post_init__(self) -> None:
        if self.dtype not in ("bf16", "fp32"):
            raise ValueError("dtype must be 'bf16' or 'fp32'")
        if self.seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        if self.micro_batch_size < 1:
            raise ValueError("micro_batch_size must be >= 1")
        tokens_per_micro = self.micro_batch_size * self.seq_len
        if self.batch_tokens % tokens_per_micro:
            raise ValueError(
                f"batch_tokens ({self.batch_tokens}) must be divisible by "
                f"micro_batch_size * seq_len ({tokens_per_micro})"
            )
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")

    @property
    def grad_accum(self) -> int:
        return self.batch_tokens // (self.micro_batch_size * self.seq_len)

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch_size * self.seq_len * self.grad_accum


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        if self.train.seq_len > self.model.max_seq_len:
            raise ValueError(
                f"train.seq_len ({self.train.seq_len}) exceeds "
                f"model.max_seq_len ({self.model.max_seq_len})"
            )
        self.model.__post_init__()
        self.train.__post_init__()

    def to_dict(self) -> dict:
        return {"model": asdict(self.model), "train": asdict(self.train)}


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = Config(
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
    )
    if overrides:
        apply_overrides(cfg, overrides)
    cfg.validate()
    return cfg


def apply_overrides(cfg: Config, overrides: list[str]) -> None:
    for item in overrides:
        key, sep, raw_value = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like section.field=value, got {item!r}")
        section, dot, name = key.partition(".")
        if not dot:
            raise ValueError(f"override key must be dotted, got {key!r}")
        target = getattr(cfg, section, None)
        if target is None or not hasattr(target, name):
            raise ValueError(f"unknown config key {key!r}")
        setattr(target, name, yaml.safe_load(raw_value))
    cfg.validate()


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).write_text(
        yaml.safe_dump(cfg.to_dict(), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
