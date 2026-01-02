# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Transformer block for M31R.

Per 06_MODEL_ARCHITECTURE.md §7, each block contains:
  1. RMSNorm
  2. Self-Attention
  3. Residual
  4. RMSNorm
  5. FeedForward (SwiGLU)
  6. Residual

This order is fixed. Pre-norm configuration is required.
"""

import torch
import torch.nn as nn

from m31r.model.attention import CausalSelfAttention
from m31r.model.mlp import SwiGLUFeedForward
from m31r.model.norm import RMSNorm


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-norm architecture.

    The structure is:
      x = x + attention(norm(x))
      x = x + feedforward(norm(x))

    Args:
        dim: Model hidden dimension.
        n_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        intermediate_dim: FFN intermediate dimension. None for automatic.
        dropout: Dropout probability for attention and FFN.
        norm_eps: Epsilon for RMSNorm.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        intermediate_dim: int | None = None,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.attention = CausalSelfAttention(
            dim=dim,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.feed_forward = SwiGLUFeedForward(
            dim=dim,
            intermediate_dim=intermediate_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).
            freqs_cis: Precomputed RoPE frequencies.

        Returns:
            Output tensor of shape (batch, seq_len, dim).
        """
        # Pre-norm attention with residual
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        # Pre-norm FFN with residual
        x = x + self.feed_forward(self.ffn_norm(x))
        return x
