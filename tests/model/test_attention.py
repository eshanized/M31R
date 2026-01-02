# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for CausalSelfAttention.

Validates output shape, causal masking behavior, and attention computation.
"""

import torch

from m31r.model.attention import CausalSelfAttention
from m31r.model.rope import precompute_freqs_cis


class TestCausalSelfAttention:
    """Tests for the CausalSelfAttention module."""

    def _make_attention(self) -> CausalSelfAttention:
        return CausalSelfAttention(dim=64, n_heads=4, head_dim=16, dropout=0.0)

    def test_output_shape(self) -> None:
        """Output must be (batch, seq_len, dim)."""
        attn = self._make_attention()
        x = torch.randn(2, 8, 64)
        freqs = precompute_freqs_cis(dim=16, max_seq_len=8)
        output = attn(x, freqs)
        assert output.shape == (2, 8, 64)

    def test_single_token(self) -> None:
        """Must handle single-token sequences."""
        attn = self._make_attention()
        x = torch.randn(1, 1, 64)
        freqs = precompute_freqs_cis(dim=16, max_seq_len=1)
        output = attn(x, freqs)
        assert output.shape == (1, 1, 64)

    def test_no_nan_output(self) -> None:
        """Output must not contain NaN."""
        attn = self._make_attention()
        x = torch.randn(1, 4, 64)
        freqs = precompute_freqs_cis(dim=16, max_seq_len=4)
        output = attn(x, freqs)
        assert not torch.isnan(output).any()

    def test_causal_mask_effect(self) -> None:
        """
        Changing a future token must not affect attention output
        for earlier positions.
        """
        attn = self._make_attention()
        attn.eval()

        x1 = torch.randn(1, 4, 64)
        x2 = x1.clone()
        # Change the last token
        x2[0, 3, :] = torch.randn(64)

        freqs = precompute_freqs_cis(dim=16, max_seq_len=4)

        with torch.no_grad():
            out1 = attn(x1, freqs)
            out2 = attn(x2, freqs)

        # First 3 positions should be identical (causal mask)
        assert torch.allclose(out1[0, :3, :], out2[0, :3, :], atol=1e-5)
