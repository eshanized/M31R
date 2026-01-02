# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for RoPE (Rotary Positional Embedding).

Validates frequency precomputation shape, determinism, and
correct application to query/key tensors.
"""

import torch

from m31r.model.rope import apply_rotary_emb, precompute_freqs_cis


class TestPrecomputeFreqsCis:
    """Tests for RoPE frequency precomputation."""

    def test_output_shape(self) -> None:
        """Output must be (max_seq_len, dim // 2)."""
        freqs = precompute_freqs_cis(dim=64, max_seq_len=128)
        assert freqs.shape == (128, 32)

    def test_complex_dtype(self) -> None:
        """Precomputed frequencies must be complex tensors."""
        freqs = precompute_freqs_cis(dim=64, max_seq_len=128)
        assert freqs.is_complex()

    def test_deterministic(self) -> None:
        """Same parameters must produce identical frequencies."""
        f1 = precompute_freqs_cis(dim=64, max_seq_len=128, theta=10000.0)
        f2 = precompute_freqs_cis(dim=64, max_seq_len=128, theta=10000.0)
        assert torch.allclose(f1.abs(), f2.abs(), atol=1e-6)

    def test_unit_magnitude(self) -> None:
        """All complex values should have magnitude 1 (unit circle)."""
        freqs = precompute_freqs_cis(dim=64, max_seq_len=128)
        magnitudes = freqs.abs()
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-5)


class TestApplyRotaryEmb:
    """Tests for applying RoPE to Q and K tensors."""

    def test_output_shape(self) -> None:
        """Output shapes must match input shapes."""
        batch, seq_len, n_heads, head_dim = 2, 8, 4, 64
        xq = torch.randn(batch, seq_len, n_heads, head_dim)
        xk = torch.randn(batch, seq_len, n_heads, head_dim)
        freqs = precompute_freqs_cis(dim=head_dim, max_seq_len=seq_len)
        rq, rk = apply_rotary_emb(xq, xk, freqs)
        assert rq.shape == xq.shape
        assert rk.shape == xk.shape

    def test_preserves_magnitude(self) -> None:
        """Rotation should approximately preserve vector magnitude."""
        batch, seq_len, n_heads, head_dim = 1, 4, 2, 64
        xq = torch.randn(batch, seq_len, n_heads, head_dim)
        xk = torch.randn(batch, seq_len, n_heads, head_dim)
        freqs = precompute_freqs_cis(dim=head_dim, max_seq_len=seq_len)
        rq, rk = apply_rotary_emb(xq, xk, freqs)
        # Norms should be approximately preserved
        assert torch.allclose(xq.norm(dim=-1), rq.norm(dim=-1), atol=0.1)

    def test_position_zero_identity(self) -> None:
        """At position 0, rotation should be approximately identity."""
        batch, n_heads, head_dim = 1, 2, 64
        xq = torch.randn(batch, 1, n_heads, head_dim)
        xk = torch.randn(batch, 1, n_heads, head_dim)
        freqs = precompute_freqs_cis(dim=head_dim, max_seq_len=1)
        rq, rk = apply_rotary_emb(xq, xk, freqs)
        # At position 0 the freqs are all e^{i*0} = 1, so rotation is identity
        assert torch.allclose(rq, xq, atol=1e-5)
