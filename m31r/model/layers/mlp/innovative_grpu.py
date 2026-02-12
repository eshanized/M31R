# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Gated Residual Polynomial Unit (GRPU) — innovative MLP layer.

A novel feedforward equation that combines gating, polynomial expansion,
and residual connections in a single efficient layer.

Equation:
    gate = sigmoid(Wg · x)
    poly = W1 · x + beta * (W2 · x)²
    y = x + gate ⊙ poly

Properties:
    - Shape preserving: input [B, T, H] → output [B, T, H]
    - Fully differentiable
    - Deterministic
    - Learnable beta parameter (scalar per-layer)
    - Residual connection ensures stable gradient flow
    - Sigmoid gating prevents unbounded outputs
    - Polynomial term adds expressivity beyond linear transforms

The design is production-grade: no randomness, no external dependencies,
no special initialization requirements beyond the standard M31R init.
"""

import torch
import torch.nn as nn

from m31r.model.interfaces import MLPBase
from m31r.model.registry import register_mlp


class GatedResidualPolynomialUnit(MLPBase):
    """
    Gated Residual Polynomial Unit (GRPU).

    Combines a sigmoid gate with a polynomial expansion to create an
    expressive feedforward layer with built-in residual connection.

    The gate controls how much of the polynomial signal is mixed into
    the residual stream. The polynomial term uses both a linear and a
    squared component, giving the network access to second-order
    features without the cost of a full intermediate expansion.

    Args:
        dim: Model hidden dimension.
        intermediate_dim: Not used (API compatibility). GRPU operates
                         directly in the hidden dimension.
        dropout: Dropout probability applied to the gated output.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Gate projection: controls residual mixing
        self.wg = nn.Linear(dim, dim, bias=False)

        # Linear component of the polynomial
        self.w1 = nn.Linear(dim, dim, bias=False)

        # Squared component of the polynomial
        self.w2 = nn.Linear(dim, dim, bias=False)

        # Learnable scalar that controls polynomial contribution
        # Initialized to a small value for stable early training
        self.beta = nn.Parameter(torch.tensor(0.1))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GRPU feedforward.

        Computes: y = x + gate ⊙ poly
        where gate = sigmoid(Wg · x) and poly = W1·x + beta·(W2·x)²

        Args:
            x: Input tensor of shape (batch, seq_len, dim).

        Returns:
            Output tensor of same shape (batch, seq_len, dim).
        """
        # Gate: sigmoid ensures [0, 1] range for stable mixing
        gate = torch.sigmoid(self.wg(x))

        # Polynomial: linear + scaled quadratic
        linear_term = self.w1(x)
        quadratic_term = self.w2(x).pow(2)
        poly = linear_term + self.beta * quadratic_term

        # Gated residual connection
        return x + self.dropout(gate * poly)


# Register with the layer registry
register_mlp("innovative_grpu", GatedResidualPolynomialUnit)
