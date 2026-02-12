# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
RMSNorm — pluggable normalization layer wrapper.

Per 06_MODEL_ARCHITECTURE.md §10:
  Type: RMSNorm
  Reasons: faster than LayerNorm, lower memory, simpler computation.

This module wraps the existing RMSNorm to conform to the NormBase interface
and registers it as "rmsnorm".
"""

import torch
import torch.nn as nn

from m31r.model.interfaces import NormBase
from m31r.model.registry import register_norm


class RMSNormLayer(NormBase):
    """
    Root Mean Square Layer Normalization.

    Normalizes the input by its RMS value and scales by a learned weight.
    Does not center (no bias subtraction), making it faster and simpler
    than standard LayerNorm.

    Args:
        dim: Feature dimension to normalize over.
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMS normalization without the learned scale."""
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input tensor.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Normalized tensor of same shape, scaled by learned weight.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# Register with the layer registry
register_norm("rmsnorm", RMSNormLayer)
