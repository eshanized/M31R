# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Transformer block for M31R.

Per 06_MODEL_ARCHITECTURE.md §7, each block contains:
  1. RMSNorm
  2. Self-Attention
  3. Residual
  4. RMSNorm
  5. FeedForward (pluggable)
  6. Residual

This order is fixed. Pre-norm configuration is required.

Layer implementations are selected via the config-driven factory system.
No layer types are hardcoded here — the block receives built layers
through constructor injection from the factory functions.
"""

import torch
import torch.nn as nn

from m31r.model.config import TransformerModelConfig
from m31r.model.factory import build_attention, build_mlp, build_norm


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-norm architecture.

    The structure is:
      x = x + attention(norm(x))
      x = x + feedforward(norm(x))

    Layer implementations are determined by the config's mlp_type,
    attention_type, and norm_type fields, resolved through the registry.

    Args:
        config: TransformerModelConfig with all architecture parameters
                including layer type selectors.
    """

    def __init__(self, config: TransformerModelConfig) -> None:
        super().__init__()
        self.attention_norm = build_norm(config, config.dim)
        self.attention = build_attention(config)
        self.ffn_norm = build_norm(config, config.dim)
        self.feed_forward = build_mlp(config)

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
