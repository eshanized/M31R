# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for model primitive layers: RMSNorm, RoPE, SwiGLU.

Per 15_TESTING_STRATEGY.md — unit tests for every component.
"""

import torch

from m31r.model.config import compute_intermediate_size
from m31r.model.mlp import SwiGLUFeedForward
from m31r.model.norm import RMSNorm
from m31r.model.rope import apply_rotary_emb, precompute_freqs_cis


class TestRMSNorm:
    """Unit tests for the RMSNorm layer."""

    def test_output_shape(self) -> None:
        """RMSNorm must preserve input shape."""
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == (2, 10, 64)

    def test_normalized_output(self) -> None:
        """After normalization, RMS should be approximately 1.0."""
        norm = RMSNorm(128)
        x = torch.randn(4, 8, 128) * 10.0  # large values
        out = norm(x)
        # RMS of output (without learned weight, which starts at 1) should be ~1
        rms = out.pow(2).mean(dim=-1).sqrt()
        assert rms.mean().item() < 2.0  # reasonable after scaling

    def test_deterministic(self) -> None:
        """Same input must produce same output."""
        norm = RMSNorm(32)
        x = torch.randn(1, 5, 32)
        out1 = norm(x.clone())
        out2 = norm(x.clone())
        assert torch.allclose(out1, out2, atol=1e-6)


class TestRoPE:
    """Unit tests for Rotary Positional Embedding."""

    def test_freqs_shape(self) -> None:
        """Precomputed frequencies must have correct shape."""
        freqs = precompute_freqs_cis(dim=64, max_seq_len=128)
        assert freqs.shape == (128, 32)  # (seq_len, dim//2)

    def test_apply_rotary_shape(self) -> None:
        """apply_rotary_emb must preserve Q/K shapes."""
        seq_len, n_heads, head_dim = 32, 8, 64
        xq = torch.randn(2, seq_len, n_heads, head_dim)
        xk = torch.randn(2, seq_len, n_heads, head_dim)
        freqs_cis = precompute_freqs_cis(dim=head_dim, max_seq_len=seq_len)

        q_out, k_out = apply_rotary_emb(xq, xk, freqs_cis)
        assert q_out.shape == xq.shape
        assert k_out.shape == xk.shape

    def test_position_sensitivity(self) -> None:
        """Different positions should produce different rotated values."""
        head_dim = 64
        freqs = precompute_freqs_cis(dim=head_dim, max_seq_len=16)
        x = torch.ones(1, 16, 1, head_dim)
        q_out, _ = apply_rotary_emb(x, x, freqs)
        # Position 0 and position 15 should differ
        assert not torch.allclose(q_out[0, 0], q_out[0, 15], atol=1e-3)


class TestSwiGLU:
    """Unit tests for SwiGLU feedforward network."""

    def test_output_shape(self) -> None:
        """SwiGLU must preserve batch/seq dimensions."""
        ffn = SwiGLUFeedForward(dim=64)
        x = torch.randn(2, 10, 64)
        out = ffn(x)
        assert out.shape == (2, 10, 64)

    def test_intermediate_size_alignment(self) -> None:
        """Intermediate size must be aligned to 256."""
        size = compute_intermediate_size(512)
        assert size % 256 == 0
        assert size > 512  # must be larger than dim

    def test_deterministic(self) -> None:
        """Same input must produce same output."""
        torch.manual_seed(42)
        ffn = SwiGLUFeedForward(dim=32)
        x = torch.randn(1, 3, 32)
        out1 = ffn(x.clone())
        out2 = ffn(x.clone())
        assert torch.allclose(out1, out2, atol=1e-6)
