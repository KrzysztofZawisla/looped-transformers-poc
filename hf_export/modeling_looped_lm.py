from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_looped_lm import LoopedLMConfig


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
        self.theta = theta
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        position = torch.arange(seq_len, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim)
        )
        freqs = torch.outer(position, inv_freq)
        return freqs.cos(), freqs.sin()


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LoopedLMConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        if attn_mask is None:
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
            )
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
            )
        y = y.transpose(1, 2).reshape(b, t, c)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, config: LoopedLMConfig) -> None:
        super().__init__()
        hidden = config.mlp_width
        self.gate = nn.Linear(config.d_model, hidden, bias=False)
        self.up = nn.Linear(config.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, config: LoopedLMConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, attn_mask)
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

    def forward(self, step: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.kind == "none":
            return None
        if self.kind == "learned":
            index = torch.tensor(min(step, self.max_loops - 1), device=device)
            return self.embedding(index).to(dtype)
        half = self.d_model // 2
        omega = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=device) / half
        )
        phase = omega * float(step)
        pe = torch.empty(self.d_model, device=device, dtype=torch.float32)
        pe[0::2] = torch.sin(phase)
        pe[1::2] = torch.cos(phase)
        return pe.to(dtype)


class LoopedLMForCausalLM(PreTrainedModel):
    config_class = LoopedLMConfig
    base_model_prefix = "model"
    _tied_weights_keys = {"lm_head.weight": "tok_emb.weight"}

    def __init__(self, config: LoopedLMConfig) -> None:
        super().__init__(config)
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.prelude = nn.ModuleList([Block(config) for _ in range(config.n_prelude)])
        self.core = nn.ModuleList([Block(config) for _ in range(config.n_core)])
        self.coda = nn.ModuleList([Block(config) for _ in range(config.n_coda)])
        self.inject: nn.Linear | None = (
            nn.Linear(config.d_model, config.d_model, bias=False)
            if config.use_input_injection
            else None
        )
        self.loop_enc = LoopEncoder(config.d_model, config.loop_encoding, config.max_loops)
        self.norm_f = RMSNorm(config.d_model)
        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)

        self.post_init()
        if self.inject is not None:
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
        if self.config.n_core < 1:
            return
        depth = self.config.virtual_depth
        std = 0.02 / math.sqrt(2 * depth)
        for name, param in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.down.weight"):
                nn.init.normal_(param, mean=0.0, std=std)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.tok_emb

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.tok_emb = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value

    def _attn_mask(self, attention_mask: torch.Tensor | None, length: int, device) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        if bool(attention_mask.all()):
            return None
        causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
        padding = attention_mask.to(torch.bool)[:, None, None, :]
        return causal[None, None] & padding

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        n_loops: int | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if input_ids is None:
            raise ValueError("input_ids are required")

        config = self.config
        if input_ids.size(1) > config.max_seq_len:
            input_ids = input_ids[:, -config.max_seq_len :]
            attention_mask = (
                attention_mask[:, -config.max_seq_len :] if attention_mask is not None else None
            )
            labels = labels[:, -config.max_seq_len :] if labels is not None else None

        b, t = input_ids.shape
        cos, sin = self.rope(t, input_ids.device)
        attn_mask = self._attn_mask(attention_mask, t, input_ids.device)
        x = self.drop(self.tok_emb(input_ids))

        x = self._run_blocks(self.prelude, x, cos, sin, attn_mask)
        anchor = x
        loops = config.n_loops if n_loops is None else int(n_loops)
        for step in range(loops):
            h = x
            loop_signal = self.loop_enc(step, h.device, h.dtype)
            if loop_signal is not None:
                h = h + loop_signal
            h = self._run_blocks(self.core, h, cos, sin, attn_mask)
            if self.inject is not None:
                h = h + self.inject(anchor)
            x = h
        x = self._run_blocks(self.coda, x, cos, sin, attn_mask)

        logits = self.lm_head(self.norm_f(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, config.vocab_size).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def _run_blocks(self, blocks, x, cos, sin, attn_mask):
        for block in blocks:
            x = block(x, cos, sin, attn_mask)
        return x

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        return {"input_ids": input_ids, "attention_mask": attention_mask}
