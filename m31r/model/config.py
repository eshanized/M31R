# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Model configuration for M31R.

Per 06_MODEL_ARCHITECTURE.md — all architecture sizes must be config-driven.
No hardcoded dimensions. The same config class supports Tiny → Large by
changing values only.

This is a plain data object (not Pydantic) because it's used inside torch
modules and needs to be lightweight. Validation happens in config/schema.py.
"""


def compute_intermediate_size(dim: int, ffn_mult: int = 4, multiple_of: int = 256) -> int:
    """
    Compute the SwiGLU intermediate dimension.

    Uses the canonical (2/3) * ffn_mult * dim formula from the LLaMA family,
    rounded up to the nearest multiple of ``multiple_of`` for hardware
    alignment efficiency.

    Args:
        dim: Model hidden dimension.
        ffn_mult: FFN expansion multiplier (default 4).
        multiple_of: Round up to nearest multiple of this value.

    Returns:
        Intermediate dimension as an integer.
    """
    intermediate = int(2 * (ffn_mult * dim) / 3)
    intermediate = multiple_of * ((intermediate + multiple_of - 1) // multiple_of)
    return intermediate


class TransformerModelConfig:
    """
    Configuration for the transformer model.

    All architecture dimensions flow from this object. The same class
    supports Tiny (30M) through Large (1.5B+) by changing field values only.

    Args:
        vocab_size: Size of the token vocabulary.
        dim: Hidden dimension of the model.
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        ffn_mult: FFN expansion multiplier. SwiGLU intermediate size is
                  computed as (2/3) * ffn_mult * dim, rounded to multiple_of.
        max_seq_len: Maximum sequence length for RoPE precomputation.
        dropout: Dropout probability for attention and FFN.
        norm_eps: Epsilon for RMSNorm.
        rope_theta: Base frequency for RoPE.
        init_std: Standard deviation for weight initialization.
        seed: Random seed for deterministic initialization.
        multiple_of: Alignment factor for intermediate FFN dimension.
    """

    __slots__ = (
        "vocab_size",
        "dim",
        "n_layers",
        "n_heads",
        "head_dim",
        "ffn_mult",
        "max_seq_len",
        "dropout",
        "norm_eps",
        "rope_theta",
        "init_std",
        "seed",
        "multiple_of",
        "mlp_type",
        "attention_type",
        "norm_type",
    )

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        ffn_mult: int = 4,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        init_std: float = 0.02,
        seed: int = 42,
        multiple_of: int = 256,
        mlp_type: str = "swiglu",
        attention_type: str = "causal",
        norm_type: str = "rmsnorm",
    ) -> None:
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.ffn_mult = ffn_mult
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.init_std = init_std
        self.seed = seed
        self.multiple_of = multiple_of
        self.mlp_type = mlp_type
        self.attention_type = attention_type
        self.norm_type = norm_type

    @property
    def intermediate_dim(self) -> int:
        """Computed SwiGLU intermediate dimension."""
        return compute_intermediate_size(self.dim, self.ffn_mult, self.multiple_of)
