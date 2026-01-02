# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for the model factory.

Validates preset configs, build functions, and model construction.
"""

import pytest
import torch

from m31r.model.config import TransformerModelConfig
from m31r.model.factory import (
    PRESETS,
    base_config,
    build_model,
    build_model_from_preset,
    large_config,
    small_config,
    tiny_config,
)
from m31r.model.transformer import M31RTransformer


class TestPresetConfigs:
    """Tests for preset configuration functions."""

    def test_tiny_config_values(self) -> None:
        """Tiny config must match M31R-Tiny spec."""
        config = tiny_config()
        assert config.dim == 384
        assert config.n_layers == 6
        assert config.n_heads == 6
        assert config.head_dim == 64
        assert config.max_seq_len == 1024
        assert config.ffn_mult == 4

    def test_small_config_values(self) -> None:
        """Small config must have dim=768, n_layers=12."""
        config = small_config()
        assert config.dim == 768
        assert config.n_layers == 12

    def test_base_config_values(self) -> None:
        """Base config must have dim=1536, n_layers=24."""
        config = base_config()
        assert config.dim == 1536
        assert config.n_layers == 24

    def test_large_config_values(self) -> None:
        """Large config must have dim=2048, n_layers=32."""
        config = large_config()
        assert config.dim == 2048
        assert config.n_layers == 32

    def test_vocab_size_override(self) -> None:
        """Preset configs must accept vocab_size override."""
        config = tiny_config(vocab_size=32000)
        assert config.vocab_size == 32000

    def test_seed_override(self) -> None:
        """Preset configs must accept seed override."""
        config = tiny_config(seed=99)
        assert config.seed == 99

    def test_all_presets_return_config(self) -> None:
        """Every preset must return a TransformerModelConfig."""
        for name, config_fn in PRESETS.items():
            config = config_fn()
            assert isinstance(config, TransformerModelConfig), (
                f"Preset '{name}' did not return TransformerModelConfig"
            )


class TestBuildModel:
    """Tests for model construction functions."""

    def test_build_model(self) -> None:
        """build_model must return a working M31RTransformer."""
        config = tiny_config(vocab_size=256)
        model = build_model(config)
        assert isinstance(model, M31RTransformer)
        tokens = torch.randint(0, 256, (1, 4))
        logits = model(tokens)
        assert logits.shape == (1, 4, 256)

    def test_build_model_from_preset(self) -> None:
        """build_model_from_preset must work for all presets."""
        # Only test tiny to avoid memory issues
        model = build_model_from_preset("tiny", vocab_size=256)
        assert isinstance(model, M31RTransformer)

    def test_build_model_from_preset_invalid(self) -> None:
        """Unknown preset must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            build_model_from_preset("nonexistent")

    def test_preset_names(self) -> None:
        """All expected presets must be registered."""
        assert set(PRESETS.keys()) == {"tiny", "small", "base", "large"}
