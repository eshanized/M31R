# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for RMSNorm.

Validates forward shape, normalization behavior, numerical stability,
and learned weight scaling.
"""

import torch

from m31r.model.norm import RMSNorm


class TestRMSNorm:
    """Tests for the RMSNorm module."""

    def test_output_shape(self) -> None:
        """Output shape must match input shape."""
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 8, 64)
        output = norm(x)
        assert output.shape == x.shape

    def test_unit_norm_scale(self) -> None:
        """
        After RMSNorm with weight=1, the RMS of output should be ~1.

        The RMS of a normalized vector scaled by weight=1 should be close to 1.
        """
        norm = RMSNorm(dim=64)
        x = torch.randn(1, 1, 64) * 10.0  # large input
        output = norm(x)
        rms = output.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_numerical_stability(self) -> None:
        """Must not produce NaN or Inf for very small inputs."""
        norm = RMSNorm(dim=64, eps=1e-6)
        x = torch.full((1, 1, 64), 1e-10)
        output = norm(x)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_zero_input(self) -> None:
        """Zero input should produce zero output without NaN."""
        norm = RMSNorm(dim=64, eps=1e-6)
        x = torch.zeros(1, 1, 64)
        output = norm(x)
        assert not torch.isnan(output).any()
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)

    def test_weight_init(self) -> None:
        """Weight should be initialized to ones."""
        norm = RMSNorm(dim=64)
        assert torch.allclose(norm.weight, torch.ones(64))
