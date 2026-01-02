# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for TransformerModelConfig.

Validates config defaults, ffn_mult computation, intermediate_dim property,
and scaling behavior across different model sizes.
"""

from m31r.model.config import TransformerModelConfig, compute_intermediate_size


class TestComputeIntermediateSize:
    """Tests for the SwiGLU intermediate size computation."""

    def test_basic_computation(self) -> None:
        """Formula: int(2 * ffn_mult * dim / 3), rounded to multiple_of."""
        result = compute_intermediate_size(384, ffn_mult=4, multiple_of=256)
        # 2 * 4 * 384 / 3 = 1024.0, rounded to 256 → 1024
        assert result == 1024

    def test_rounding_up(self) -> None:
        """Non-aligned values must round up to nearest multiple."""
        result = compute_intermediate_size(256, ffn_mult=4, multiple_of=256)
        # 2 * 4 * 256 / 3 = 682.67, rounded up to 256 → 768
        assert result == 768
        assert result % 256 == 0

    def test_large_dim(self) -> None:
        """Large hidden dims should still produce aligned values."""
        result = compute_intermediate_size(1536, ffn_mult=4, multiple_of=256)
        # 2 * 4 * 1536 / 3 = 4096, already a multiple of 256
        assert result == 4096
        assert result % 256 == 0

    def test_custom_multiple(self) -> None:
        """Different multiple_of should change alignment."""
        result = compute_intermediate_size(384, ffn_mult=4, multiple_of=64)
        # 2 * 4 * 384 / 3 = 1024, already a multiple of 64
        assert result == 1024


class TestTransformerModelConfig:
    """Tests for TransformerModelConfig fields and properties."""

    def test_defaults(self) -> None:
        """Verify default parameter values."""
        config = TransformerModelConfig(
            vocab_size=16384, dim=384, n_layers=6, n_heads=6, head_dim=64,
        )
        assert config.ffn_mult == 4
        assert config.max_seq_len == 2048
        assert config.dropout == 0.1
        assert config.norm_eps == 1e-6
        assert config.rope_theta == 10000.0
        assert config.init_std == 0.02
        assert config.seed == 42
        assert config.multiple_of == 256

    def test_intermediate_dim_property(self) -> None:
        """Computed intermediate_dim must match compute_intermediate_size."""
        config = TransformerModelConfig(
            vocab_size=16384, dim=384, n_layers=6, n_heads=6, head_dim=64,
        )
        expected = compute_intermediate_size(384, 4, 256)
        assert config.intermediate_dim == expected

    def test_scaling_tiny_to_large(self) -> None:
        """Different dims produce different intermediate sizes."""
        tiny = TransformerModelConfig(
            vocab_size=16384, dim=384, n_layers=6, n_heads=6, head_dim=64,
        )
        large = TransformerModelConfig(
            vocab_size=16384, dim=2048, n_layers=32, n_heads=32, head_dim=64,
        )
        assert large.intermediate_dim > tiny.intermediate_dim

    def test_slots(self) -> None:
        """Config uses __slots__ for memory efficiency."""
        config = TransformerModelConfig(
            vocab_size=16384, dim=384, n_layers=6, n_heads=6, head_dim=64,
        )
        assert hasattr(config, "__slots__")

    def test_all_fields_accessible(self) -> None:
        """Every field in __slots__ must be accessible."""
        config = TransformerModelConfig(
            vocab_size=16384, dim=384, n_layers=6, n_heads=6, head_dim=64,
        )
        for slot in TransformerModelConfig.__slots__:
            assert hasattr(config, slot)
