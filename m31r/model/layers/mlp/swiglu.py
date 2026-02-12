# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
SwiGLU feedforward network — pluggable layer wrapper.

Per 06_MODEL_ARCHITECTURE.md §9:
  Activation: SwiGLU
  Structure: Linear → SwiGLU → Linear
  Reason: Better parameter efficiency and performance compared to ReLU/GELU.

This module wraps the existing SwiGLUFeedForward implementation to conform
to the MLPBase interface and registers it with the layer registry as "swiglu".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from m31r.model.config import compute_intermediate_size
from m31r.model.interfaces import MLPBase
from m31r.model.registry import register_mlp


class SwiGLUMLP(MLPBase):
    """
    SwiGLU-activated feedforward network.

    The gating mechanism uses SiLU (swish) on one branch and multiplies
    it with a linear gate branch before projecting back down.

    Args:
        dim: Model hidden dimension.
        intermediate_dim: Size of the hidden layer. If None, computed
                         automatically using the standard formula.
        dropout: Dropout probability applied after the gating.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if intermediate_dim is None:
            intermediate_dim = compute_intermediate_size(dim)

        self.w1 = nn.Linear(dim, intermediate_dim, bias=False)  # gate projection
        self.w2 = nn.Linear(intermediate_dim, dim, bias=False)  # down projection
        self.w3 = nn.Linear(dim, intermediate_dim, bias=False)  # up projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU feedforward.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).

        Returns:
            Output tensor of same shape.
        """
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


# Register with the layer registry
register_mlp("swiglu", SwiGLUMLP)
