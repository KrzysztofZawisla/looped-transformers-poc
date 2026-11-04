from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.to(dtype=x.dtype)[None, None, :, :]
    sin = sin.to(dtype=x.dtype)[None, None, :, :]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.register_buffer(
            "inv_freq",
            1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)),
            persistent=False,
        )
        self._build(max_seq_len)

    def _build(self, seq_len: int) -> None:
        pos = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(pos, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            self._build(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).reshape(b, t, c)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = cfg.mlp_width
        self.gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class LoopEncoder(nn.Module):
    def __init__(self, d_model: int, kind: str, max_loops: int) -> None:
        super().__init__()
        self.kind = kind
        self.d_model = d_model
        self.max_loops = max_loops
        if kind == "learned":
            self.embedding = nn.Embedding(max_loops, d_model)
            nn.init.normal_(self.embedding.weight, std=0.02)
        elif kind == "sinusoidal":
            half = d_model // 2
            omega = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
            self.register_buffer("omega", omega, persistent=False)

    def forward(self, step: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.kind == "none":
            return None
        if self.kind == "learned":
            idx = torch.tensor(min(step, self.max_loops - 1), device=device)
            return self.embedding(idx).to(dtype)
        phase = self.omega.to(device=device) * float(step)
        pe = torch.empty(self.d_model, device=device, dtype=torch.float32)
        pe[0::2] = torch.sin(phase)
        pe[1::2] = torch.cos(phase)
        return pe.to(dtype)


class LoopedLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._grad_checkpoint = False
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        if cfg.model_type == "looped":
            self.prelude = nn.ModuleList([Block(cfg) for _ in range(cfg.n_prelude)])
            self.core = nn.ModuleList([Block(cfg) for _ in range(cfg.n_core)])
            self.coda = nn.ModuleList([Block(cfg) for _ in range(cfg.n_coda)])
            self.inject: nn.Linear | None = (
                nn.Linear(cfg.d_model, cfg.d_model, bias=False)
                if cfg.use_input_injection
                else None
            )
            self.loop_enc = LoopEncoder(cfg.d_model, cfg.loop_encoding, cfg.max_loops)
        else:
            self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.apply(self._init_weights)
        if cfg.model_type == "looped" and self.inject is not None:
            nn.init.zeros_(self.inject.weight)
        self._scale_residual()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual(self) -> None:
        std = 0.02 / math.sqrt(2 * self.virtual_depth())
        for name, param in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.down.weight"):
                nn.init.normal_(param, mean=0.0, std=std)

    def virtual_depth(self, n_loops: int | None = None) -> int:
        if self.cfg.model_type == "vanilla":
            return self.cfg.n_layers
        loops = self.cfg.n_loops if n_loops is None else n_loops
        return self.cfg.n_prelude + self.cfg.n_core * loops + self.cfg.n_coda

    def num_parameters(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.tok_emb.weight.numel()
        return total

    def _run_blocks(
        self,
        blocks: nn.ModuleList,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        for block in blocks:
            if self._grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        return x

    def enable_grad_checkpointing(self, enabled: bool = True) -> None:
        self._grad_checkpoint = enabled

    def _cross_entropy(
        self, hidden: torch.Tensor, targets: torch.Tensor, chunk_size: int = 64
    ) -> torch.Tensor:
        total = torch.zeros((), device=hidden.device, dtype=torch.float32)
        for start in range(0, hidden.size(1), chunk_size):
            logits = self.lm_head(hidden[:, start : start + chunk_size]).float()
            chunk_targets = targets[:, start : start + chunk_size].reshape(-1)
            total = total + F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), chunk_targets, reduction="sum"
            )
        return total / targets.numel()

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        n_loops: int | None = None,
        last_token_only: bool = False,
        aux: bool = True,
    ) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        b, t = idx.shape
        if t > cfg.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len {cfg.max_seq_len}")

        cos, sin = self.rope(t)
        x = self.drop(self.tok_emb(idx))
        aux_states: list[torch.Tensor] = []

        if cfg.model_type == "looped":
            x = self._run_blocks(self.prelude, x, cos, sin)
            anchor = x
            loops = cfg.n_loops if n_loops is None else int(n_loops)
            first_aux = max(0, loops - cfg.aux_loops) if (aux and targets is not None) else loops
            for step in range(loops):
                h = x
                loop_signal = self.loop_enc(step, h.device, h.dtype)
                if loop_signal is not None:
                    h = h + loop_signal
                h = self._run_blocks(self.core, h, cos, sin)
                if self.inject is not None:
                    h = h + self.inject(anchor)
                x = h
                if step >= first_aux:
                    aux_states.append(x)
            x = self._run_blocks(self.coda, x, cos, sin)
        else:
            x = self._run_blocks(self.layers, x, cos, sin)

        x = self.norm_f(x)
        out: dict[str, torch.Tensor] = {}

        if targets is not None:
            out["loss"] = self._cross_entropy(x, targets)
            if aux_states:
                aux_losses = [self._cross_entropy(self.norm_f(h), targets) for h in aux_states]
                out["aux_loss"] = torch.stack(aux_losses).mean()

        if targets is None:
            hidden = x[:, -1:] if last_token_only else x
            out["logits"] = self.lm_head(hidden)

        return out


@torch.no_grad()
def sample(
    model: LoopedLM,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = None,
    n_loops: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    model.eval()
    for _ in range(max_new_tokens):
        context = idx[:, -model.cfg.max_seq_len :]
        logits = model(context, n_loops=n_loops, last_token_only=True)["logits"][:, -1, :].float()

        if temperature <= 1e-5:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                cutoff = torch.topk(logits, k).values[:, [-1]]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative = probs.cumsum(dim=-1)
                remove = cumulative - probs > top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id], dim=1)
        if eos_id is not None and bool((next_id == eos_id).all()):
            break
    return idx


def load_model(ckpt_path: str | Path, device: str | torch.device = "cpu") -> tuple[LoopedLM, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = LoopedLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt
