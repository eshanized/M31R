# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for the full transformer model.

Per 15_TESTING_STRATEGY.md — model forward pass, shape correctness,
parameter count, weight tying, and determinism.
"""

import torch

from m31r.model.config import TransformerModelConfig
from m31r.model.transformer import M31RTransformer


def _tiny_config() -> TransformerModelConfig:
    """A minimal config for fast testing."""
    return TransformerModelConfig(
        vocab_size=256,
        dim=64,
        n_layers=2,
        n_heads=4,
        head_dim=16,
        max_seq_len=32,
        dropout=0.0,
        seed=42,
    )


class TestTransformerModel:
    """Unit tests for the M31RTransformer model."""

    def test_forward_shape(self) -> None:
        """Output logits must have shape (batch, seq_len, vocab_size)."""
        config = _tiny_config()
        model = M31RTransformer(config)
        tokens = torch.randint(0, 256, (2, 16))
        logits = model(tokens)
        assert logits.shape == (2, 16, 256)

    def test_weight_tying(self) -> None:
        """Embedding and output head must share the same weight tensor."""
        config = _tiny_config()
        model = M31RTransformer(config)
        assert model.tok_embeddings.weight is model.output.weight

    def test_parameter_count(self) -> None:
        """Tiny model should have a reasonable number of parameters."""
        config = _tiny_config()
        model = M31RTransformer(config)
        count = model.count_parameters()
        assert count > 0
        assert count < 500_000  # tiny model should be small

    def test_deterministic_init(self) -> None:
        """Same config and seed must produce identical model weights."""
        config = _tiny_config()
        model1 = M31RTransformer(config)
        model2 = M31RTransformer(config)

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6)

    def test_deterministic_forward(self) -> None:
        """Same input must produce same output."""
        config = _tiny_config()
        model = M31RTransformer(config)
        model.eval()
        tokens = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            out1 = model(tokens)
            out2 = model(tokens)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_different_seeds_differ(self) -> None:
        """Different seeds must produce different weights."""
        config1 = TransformerModelConfig(
            vocab_size=256,
            dim=64,
            n_layers=2,
            n_heads=4,
            head_dim=16,
            max_seq_len=32,
            seed=42,
        )
        config2 = TransformerModelConfig(
            vocab_size=256,
            dim=64,
            n_layers=2,
            n_heads=4,
            head_dim=16,
            max_seq_len=32,
            seed=123,
        )
        model1 = M31RTransformer(config1)
        model2 = M31RTransformer(config2)

        params1 = list(model1.parameters())
        params2 = list(model2.parameters())
        # At least one parameter tensor should differ
        any_different = any(
            not torch.allclose(p1, p2, atol=1e-6) for p1, p2 in zip(params1, params2)
        )
        assert any_different

    def test_variable_sequence_length(self) -> None:
        """Model must handle sequences shorter than max_seq_len."""
        config = _tiny_config()
        model = M31RTransformer(config)
        model.eval()
        for seq_len in [1, 4, 16, 32]:
            tokens = torch.randint(0, 256, (1, seq_len))
            logits = model(tokens)
            assert logits.shape == (1, seq_len, 256)
