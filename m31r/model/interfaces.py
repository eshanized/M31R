# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Abstract base classes for pluggable model layers.

Per 06_MODEL_ARCHITECTURE.md — the transformer topology is fixed:
  Embedding → N × Blocks → Norm → Head

Within each block, the MLP, Attention, and Norm components are swappable
as long as they obey the contracts defined here:

- Shape contract: input [B, T, H] → output [B, T, H]
- Deterministic behavior
- PyTorch nn.Module subclass

All custom layer implementations MUST subclass the appropriate base class.
This ensures the factory and block can work with any layer without
type-checking or conditional logic.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class MLPBase(nn.Module, ABC):
    """
    Base class for all MLP / feedforward layer implementations.

    Contract:
        forward(x) -> y  where x.shape == y.shape == [B, T, H]

    All MLP implementations must preserve the hidden dimension.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the feedforward transformation.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).

        Returns:
            Output tensor of same shape (batch, seq_len, dim).
        """
        ...


class AttentionBase(nn.Module, ABC):
    """
    Base class for all self-attention layer implementations.

    Contract:
        forward(x, freqs_cis) -> y  where x.shape == y.shape == [B, T, H]

    All attention implementations receive precomputed RoPE frequencies
    and must apply causal masking per 06_MODEL_ARCHITECTURE.md §8.
    """

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).
            freqs_cis: Precomputed RoPE frequencies.

        Returns:
            Output tensor of same shape (batch, seq_len, dim).
        """
        ...


class NormBase(nn.Module, ABC):
    """
    Base class for all normalization layer implementations.

    Contract:
        forward(x) -> y  where x.shape == y.shape == [..., dim]

    All normalization implementations must preserve shape exactly.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Normalized tensor of same shape.
        """
        ...
