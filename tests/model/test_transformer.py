# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for the full M31RTransformer model.

Per 15_TESTING_STRATEGY.md — model forward pass, shape correctness,
parameter count, weight tying, determinism, overfit, gradient flow,
config scaling, and CPU memory.
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
        ffn_mult=4,
        max_seq_len=32,
        dropout=0.0,
        seed=42,
    )


class TestTransformerForwardPass:
    """Forward pass shape and behavior tests."""

    def test_forward_shape(self) -> None:
        """Output logits must have shape (batch, seq_len, vocab_size)."""
        config = _tiny_config()
        model = M31RTransformer(config)
        tokens = torch.randint(0, 256, (2, 16))
        logits = model(tokens)
        assert logits.shape == (2, 16, 256)

    def test_variable_sequence_length(self) -> None:
        """Model must handle sequences shorter than max_seq_len."""
        config = _tiny_config()
        model = M31RTransformer(config)
        model.eval()
        for seq_len in [1, 4, 16, 32]:
            tokens = torch.randint(0, 256, (1, seq_len))
            logits = model(tokens)
            assert logits.shape == (1, seq_len, 256)

    def test_batch_sizes(self) -> None:
        """Model must handle different batch sizes."""
        config = _tiny_config()
        model = M31RTransformer(config)
        model.eval()
        for batch_size in [1, 2, 4]:
            tokens = torch.randint(0, 256, (batch_size, 8))
            logits = model(tokens)
            assert logits.shape == (batch_size, 8, 256)


class TestWeightTying:
    """Weight tying verification."""

    def test_weight_tying(self) -> None:
        """Embedding and output head must share the same weight tensor."""
        config = _tiny_config()
        model = M31RTransformer(config)
        assert model.tok_embeddings.weight is model.output.weight


class TestParameterCount:
    """Parameter count validation."""

    def test_parameter_count_tiny_test(self) -> None:
        """Test-tiny model should have a reasonable number of parameters."""
        config = _tiny_config()
        model = M31RTransformer(config)
        count = model.count_parameters()
        assert count > 0
        assert count < 500_000  # test-tiny should be small

    def test_parameter_count_m31r_tiny(self) -> None:
        """M31R-Tiny should have ~17M parameters (dim=384, layers=6, heads=6)."""
        config = TransformerModelConfig(
            vocab_size=16384,
            dim=384,
            n_layers=6,
            n_heads=6,
            head_dim=64,
            ffn_mult=4,
            max_seq_len=1024,
            dropout=0.0,
            seed=42,
        )
        model = M31RTransformer(config)
        count = model.count_parameters()
        # Actual: ~16.9M for these dimensions
        assert 14_000_000 < count < 20_000_000, f"M31R-Tiny has {count:,} params, expected ~17M"


class TestDeterminism:
    """Deterministic initialization and forward pass."""

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
        any_different = any(
            not torch.allclose(p1, p2, atol=1e-6) for p1, p2 in zip(params1, params2)
        )
        assert any_different


class TestGradientFlow:
    """Gradient flow through the full model."""

    def test_all_parameters_receive_gradients(self) -> None:
        """After a backward pass, all parameters must have non-None gradients."""
        config = _tiny_config()
        model = M31RTransformer(config)
        tokens = torch.randint(0, 256, (1, 8))
        logits = model(tokens)
        loss = logits.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} has no gradient"


class TestOverfit:
    """Overfit test — model should memorize a tiny dataset."""

    def test_overfit_single_sequence(self) -> None:
        """Model must memorize a single sequence in ~200 steps."""
        config = _tiny_config()
        model = M31RTransformer(config)
        model.train()

        # Create a fixed input/target pair
        torch.manual_seed(42)
        tokens = torch.randint(0, 256, (1, 16))
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()

        final_loss = float("inf")
        for step in range(200):
            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        assert final_loss < 0.5, f"Model failed to overfit: loss={final_loss:.4f} after 200 steps"


class TestConfigScaling:
    """Config-driven scaling tests."""

    def test_more_layers_more_params(self) -> None:
        """More layers must produce more parameters."""
        config_2 = TransformerModelConfig(
            vocab_size=256,
            dim=64,
            n_layers=2,
            n_heads=4,
            head_dim=16,
            max_seq_len=32,
            seed=42,
        )
        config_4 = TransformerModelConfig(
            vocab_size=256,
            dim=64,
            n_layers=4,
            n_heads=4,
            head_dim=16,
            max_seq_len=32,
            seed=42,
        )
        model_2 = M31RTransformer(config_2)
        model_4 = M31RTransformer(config_4)
        assert model_4.count_parameters() > model_2.count_parameters()

    def test_larger_dim_more_params(self) -> None:
        """Larger hidden dim must produce more parameters."""
        config_small = TransformerModelConfig(
            vocab_size=256,
            dim=64,
            n_layers=2,
            n_heads=4,
            head_dim=16,
            max_seq_len=32,
            seed=42,
        )
        config_large = TransformerModelConfig(
            vocab_size=256,
            dim=128,
            n_layers=2,
            n_heads=4,
            head_dim=32,
            max_seq_len=32,
            seed=42,
        )
        model_small = M31RTransformer(config_small)
        model_large = M31RTransformer(config_large)
        assert model_large.count_parameters() > model_small.count_parameters()


class TestCPUMemory:
    """CPU memory footprint tests."""

    def test_tiny_model_fits_in_memory(self) -> None:
        """Tiny model construction and forward pass must work on CPU."""
        config = TransformerModelConfig(
            vocab_size=16384,
            dim=384,
            n_layers=6,
            n_heads=6,
            head_dim=64,
            ffn_mult=4,
            max_seq_len=1024,
            dropout=0.0,
            seed=42,
        )
        model = M31RTransformer(config)
        model.eval()
        tokens = torch.randint(0, 16384, (1, 64))
        with torch.no_grad():
            logits = model(tokens)
        assert logits.shape == (1, 64, 16384)
        # If we get here without OOM, we pass
