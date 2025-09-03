# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Full transformer model for M31R.

Per 06_MODEL_ARCHITECTURE.md §4, the topology is:
  Input Tokens → Embedding → N × Transformer Blocks → Final Norm → Linear LM Head → Logits

Weight tying (§5): shared embedding and LM head weights.
All initialization is deterministic (§21).
"""

import torch
import torch.nn as nn

from m31r.model.blocks.transformer import TransformerBlock
from m31r.model.init.weights import init_weights
from m31r.model.layers.rmsnorm import RMSNorm
from m31r.model.layers.rotary import precompute_freqs_cis


class TransformerModelConfig:
    """
    Configuration for the transformer model.

    This is a plain data object — not a Pydantic model — because it's used
    inside the torch module and needs to be lightweight. The Pydantic schema
    in config/schema.py handles validation; this just carries the values.

    Args:
        vocab_size: Size of the token vocabulary.
        dim: Hidden dimension of the model.
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        max_seq_len: Maximum sequence length for RoPE precomputation.
        dropout: Dropout probability for attention and FFN.
        norm_eps: Epsilon for RMSNorm.
        rope_theta: Base frequency for RoPE.
        init_std: Standard deviation for weight initialization.
        seed: Random seed for deterministic initialization.
    """

    __slots__ = (
        "vocab_size", "dim", "n_layers", "n_heads", "head_dim",
        "max_seq_len", "dropout", "norm_eps", "rope_theta",
        "init_std", "seed",
    )

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        init_std: float = 0.02,
        seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.init_std = init_std
        self.seed = seed


class M31RTransformer(nn.Module):
    """
    Decoder-only transformer for Rust code generation.

    Implements the full model topology specified in 06_MODEL_ARCHITECTURE.md:
      - Token embedding (weight-tied with output head)
      - N transformer blocks (pre-norm with RMSNorm, SwiGLU, causal attention)
      - Final RMSNorm
      - Linear LM head (shared weights with embedding)

    The model precomputes RoPE frequencies once and passes them to each block.
    All weights are initialized deterministically from a single seed.

    Args:
        config: TransformerModelConfig with all architecture parameters.
    """

    def __init__(self, config: TransformerModelConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=config.dim,
                n_heads=config.n_heads,
                head_dim=config.head_dim,
                dropout=config.dropout,
                norm_eps=config.norm_eps,
            )
            for _ in range(config.n_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        # Output projection (weight-tied with embedding)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Precompute RoPE frequencies — stored as a buffer (non-parameter state)
        freqs_cis = precompute_freqs_cis(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Deterministic initialization
        init_weights(self, seed=config.seed, init_std=config.init_std)

        # Weight tying: embedding and output head share the same weight matrix
        self.output.weight = self.tok_embeddings.weight

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer.

        Args:
            tokens: Input token IDs of shape (batch, seq_len).

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size).
        """
        seq_len = tokens.shape[1]

        # Token embedding
        h = self.tok_embeddings(tokens)

        # Get RoPE frequencies for this sequence length
        freqs_cis = self.freqs_cis[:seq_len]

        # Pass through all transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis)

        # Final norm and output projection
        h = self.norm(h)
        logits = self.output(h)

        return logits

    def count_parameters(self) -> int:
        """
        Count total trainable parameters.

        Returns:
            Integer count of all parameters with requires_grad=True.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
