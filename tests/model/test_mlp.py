# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for SwiGLU MLP.

Validates output shape, intermediate dim computation, and forward pass.
"""

import torch

from m31r.model.config import compute_intermediate_size
from m31r.model.mlp import SwiGLUFeedForward


class TestSwiGLUFeedForward:
    """Tests for the SwiGLU feedforward module."""

    def test_output_shape(self) -> None:
        """Output shape must match input shape in the last dimension."""
        mlp = SwiGLUFeedForward(dim=64)
        x = torch.randn(2, 8, 64)
        output = mlp(x)
        assert output.shape == (2, 8, 64)

    def test_explicit_intermediate_dim(self) -> None:
        """Explicit intermediate_dim must be used when provided."""
        mlp = SwiGLUFeedForward(dim=64, intermediate_dim=128)
        assert mlp.w1.out_features == 128
        assert mlp.w3.out_features == 128
        assert mlp.w2.in_features == 128

    def test_auto_intermediate_dim(self) -> None:
        """When intermediate_dim is None, auto-compute from dim."""
        mlp = SwiGLUFeedForward(dim=384)
        expected = compute_intermediate_size(384)
        assert mlp.w1.out_features == expected

    def test_no_nan_output(self) -> None:
        """Output must not contain NaN."""
        mlp = SwiGLUFeedForward(dim=64)
        x = torch.randn(1, 4, 64)
        output = mlp(x)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self) -> None:
        """Gradients must flow through SwiGLU."""
        mlp = SwiGLUFeedForward(dim=64)
        x = torch.randn(1, 4, 64, requires_grad=True)
        output = mlp(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
