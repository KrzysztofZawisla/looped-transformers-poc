from __future__ import annotations

from transformers import PretrainedConfig


class LoopedLMConfig(PretrainedConfig):
    model_type = "looped_lm"

    def __init__(
        self,
        vocab_size: int = 16384,
        d_model: int = 768,
        n_heads: int = 12,
        head_dim: int = 64,
        n_prelude: int = 2,
        n_core: int = 2,
        n_coda: int = 2,
        n_loops: int = 8,
        min_loops: int = 1,
        max_loops: int = 32,
        loop_encoding: str = "sinusoidal",
        use_input_injection: bool = True,
        mlp_hidden: int | None = None,
        mlp_mult: float = 8.0 / 3.0,
        rope_theta: float = 10000.0,
        max_seq_len: int = 512,
        max_position_embeddings: int = 512,
        dropout: float = 0.0,
        aux_loops: int = 2,
        aux_loss_weight: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_prelude = n_prelude
        self.n_core = n_core
        self.n_coda = n_coda
        self.n_loops = n_loops
        self.min_loops = min_loops
        self.max_loops = max_loops
        self.loop_encoding = loop_encoding
        self.use_input_injection = use_input_injection
        self.mlp_hidden = mlp_hidden
        self.mlp_mult = mlp_mult
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout
        self.aux_loops = aux_loops
        self.aux_loss_weight = aux_loss_weight

    @property
    def mlp_width(self) -> int:
        hidden = self.mlp_hidden if self.mlp_hidden is not None else int(self.mlp_mult * self.d_model)
        return max(64, (hidden + 63) // 64 * 64)

    @property
    def virtual_depth(self) -> int:
        return self.n_prelude + self.n_core * self.n_loops + self.n_coda
