# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for TransformerBlock.

Validates forward pass shape, residual connections, and gradient flow.
"""

import torch

from m31r.model.block import TransformerBlock
from m31r.model.rope import precompute_freqs_cis


class TestTransformerBlock:
    """Tests for the TransformerBlock module."""

    def _make_block(self) -> TransformerBlock:
        return TransformerBlock(
            dim=64,
            n_heads=4,
            head_dim=16,
            dropout=0.0,
            norm_eps=1e-6,
        )

    def test_output_shape(self) -> None:
        """Output shape must match input shape."""
        block = self._make_block()
        x = torch.randn(2, 8, 64)
        freqs = precompute_freqs_cis(dim=16, max_seq_len=8)
        output = block(x, freqs)
        assert output.shape == (2, 8, 64)

    def test_residual_connection(self) -> None:
        """With zero-initialized weights, output should equal input (residual)."""
        block = self._make_block()
        # Zero out attention and FFN output projections
        with torch.no_grad():
            block.attention.wo.weight.zero_()
            block.feed_forward.w2.weight.zero_()

        x = torch.randn(1, 4, 64)
        freqs = precompute_freqs_cis(dim=16, max_seq_len=4)
        output = block(x, freqs)
        assert torch.allclose(output, x, atol=1e-5)

    def test_gradient_flow(self) -> None:
        """Gradients must flow through the entire block."""
        block = self._make_block()
        x = torch.randn(1, 4, 64, requires_grad=True)
        freqs = precompute_freqs_cis(dim=16, max_seq_len=4)
        output = block(x, freqs)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_no_nan_output(self) -> None:
        """Output must not contain NaN."""
        block = self._make_block()
        x = torch.randn(1, 4, 64)
        freqs = precompute_freqs_cis(dim=16, max_seq_len=4)
        output = block(x, freqs)
        assert not torch.isnan(output).any()
