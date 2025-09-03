# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
SwiGLU feedforward network for M31R.

Per 06_MODEL_ARCHITECTURE.md §9:
  Activation: SwiGLU
  Structure: Linear → SwiGLU → Linear
  Reason: Better parameter efficiency and performance compared to ReLU/GELU.

SwiGLU uses a gating mechanism: out = (Linear_gate(x) * silu(Linear_up(x)))
followed by a down projection. The intermediate dimension is typically
(4/3) * 4 * dim, rounded to the nearest multiple of 256 for efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_intermediate_size(dim: int, multiple_of: int = 256) -> int:
    """
    Compute the SwiGLU intermediate dimension.

    Uses the canonical (4/3) * 4 * dim formula from the LLaMA family,
    rounded up to the nearest multiple of `multiple_of` for hardware
    alignment efficiency.

    Args:
        dim: Model hidden dimension.
        multiple_of: Round up to nearest multiple of this value.

    Returns:
        Intermediate dimension as an integer.
    """
    intermediate = int(2 * (4 * dim) / 3)
    # Round up to nearest multiple
    intermediate = multiple_of * ((intermediate + multiple_of - 1) // multiple_of)
    return intermediate


class SwiGLUFeedForward(nn.Module):
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
            intermediate_dim = _compute_intermediate_size(dim)

        self.w1 = nn.Linear(dim, intermediate_dim, bias=False)  # gate projection
        self.w2 = nn.Linear(intermediate_dim, dim, bias=False)  # down projection
        self.w3 = nn.Linear(dim, intermediate_dim, bias=False)  # up projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU feedforward.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Output tensor of same shape.
        """
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))
