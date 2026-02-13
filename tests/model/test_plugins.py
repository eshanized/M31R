# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for the pluggable layer system.

Validates the full plugin lifecycle: registry, factory, layer interfaces,
forward shapes, gradient flow, determinism, layer swapping, and end-to-end
model construction with different layer types.

Per 15_TESTING_STRATEGY.md — tests must be deterministic and reproducible.
"""

import torch
import torch.nn as nn

from m31r.model.config import TransformerModelConfig
from m31r.model.factory import (
    build_attention,
    build_mlp,
    build_model,
    build_norm,
)
from m31r.model.interfaces import AttentionBase, MLPBase, NormBase
from m31r.model.registry import (
    get_attention,
    get_mlp,
    get_norm,
    list_attention_types,
    list_mlp_types,
    list_norm_types,
)

# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_config(
    mlp_type: str = "swiglu",
    attention_type: str = "causal",
    norm_type: str = "rmsnorm",
    dim: int = 64,
    n_layers: int = 2,
    n_heads: int = 2,
    head_dim: int = 32,
    vocab_size: int = 256,
    seed: int = 42,
) -> TransformerModelConfig:
    """Create a minimal test config with the given layer types."""
    return TransformerModelConfig(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        max_seq_len=128,
        dropout=0.0,
        norm_eps=1e-6,
        rope_theta=10000.0,
        init_std=0.02,
        seed=seed,
        mlp_type=mlp_type,
        attention_type=attention_type,
        norm_type=norm_type,
    )


# ── Registry Tests ──────────────────────────────────────────────────────────


class TestRegistry:
    """Tests for the layer type registry system."""

    def test_mlp_registry_contains_swiglu(self) -> None:
        """The 'swiglu' MLP type must be registered by default."""
        assert "swiglu" in list_mlp_types()

    def test_mlp_registry_contains_grpu(self) -> None:
        """The 'innovative_grpu' MLP type must be registered by default."""
        assert "innovative_grpu" in list_mlp_types()

    def test_attention_registry_contains_causal(self) -> None:
        """The 'causal' attention type must be registered by default."""
        assert "causal" in list_attention_types()

    def test_norm_registry_contains_rmsnorm(self) -> None:
        """The 'rmsnorm' norm type must be registered by default."""
        assert "rmsnorm" in list_norm_types()

    def test_get_mlp_returns_class(self) -> None:
        """get_mlp must return a class (not an instance)."""
        cls = get_mlp("swiglu")
        assert isinstance(cls, type)
        assert issubclass(cls, nn.Module)

    def test_get_attention_returns_class(self) -> None:
        """get_attention must return a class (not an instance)."""
        cls = get_attention("causal")
        assert isinstance(cls, type)
        assert issubclass(cls, nn.Module)

    def test_get_norm_returns_class(self) -> None:
        """get_norm must return a class (not an instance)."""
        cls = get_norm("rmsnorm")
        assert isinstance(cls, type)
        assert issubclass(cls, nn.Module)

    def test_unknown_mlp_raises(self) -> None:
        """Unknown MLP type must raise KeyError."""
        try:
            get_mlp("nonexistent_mlp_type")
            assert False, "Expected KeyError"
        except KeyError:
            pass

    def test_unknown_attention_raises(self) -> None:
        """Unknown attention type must raise KeyError."""
        try:
            get_attention("nonexistent_attention_type")
            assert False, "Expected KeyError"
        except KeyError:
            pass

    def test_unknown_norm_raises(self) -> None:
        """Unknown norm type must raise KeyError."""
        try:
            get_norm("nonexistent_norm_type")
            assert False, "Expected KeyError"
        except KeyError:
            pass


# ── Factory Tests ───────────────────────────────────────────────────────────


class TestFactory:
    """Tests for the layer factory builder functions."""

    def test_factory_builds_swiglu(self) -> None:
        """Factory must return an MLPBase instance for 'swiglu'."""
        config = _make_config(mlp_type="swiglu")
        mlp = build_mlp(config)
        assert isinstance(mlp, MLPBase)

    def test_factory_builds_grpu(self) -> None:
        """Factory must return an MLPBase instance for 'innovative_grpu'."""
        config = _make_config(mlp_type="innovative_grpu")
        mlp = build_mlp(config)
        assert isinstance(mlp, MLPBase)

    def test_factory_builds_causal_attention(self) -> None:
        """Factory must return an AttentionBase instance for 'causal'."""
        config = _make_config(attention_type="causal")
        attn = build_attention(config)
        assert isinstance(attn, AttentionBase)

    def test_factory_builds_rmsnorm(self) -> None:
        """Factory must return a NormBase instance for 'rmsnorm'."""
        config = _make_config(norm_type="rmsnorm")
        norm = build_norm(config, dim=64)
        assert isinstance(norm, NormBase)


# ── Forward Shape Tests ────────────────────────────────────────────────────


class TestForwardShape:
    """Tests that all layer implementations preserve the [B, T, H] shape."""

    def test_swiglu_shape(self) -> None:
        """SwiGLU must preserve [B, T, H] shape."""
        config = _make_config(mlp_type="swiglu")
        mlp = build_mlp(config)
        x = torch.randn(2, 8, 64)
        output = mlp(x)
        assert output.shape == (2, 8, 64)

    def test_grpu_shape(self) -> None:
        """GRPU must preserve [B, T, H] shape."""
        config = _make_config(mlp_type="innovative_grpu")
        mlp = build_mlp(config)
        x = torch.randn(2, 8, 64)
        output = mlp(x)
        assert output.shape == (2, 8, 64)

    def test_norm_shape(self) -> None:
        """RMSNorm must preserve shape."""
        config = _make_config()
        norm = build_norm(config, dim=64)
        x = torch.randn(2, 8, 64)
        output = norm(x)
        assert output.shape == (2, 8, 64)


# ── Gradient Flow Tests ────────────────────────────────────────────────────


class TestGradientFlow:
    """Tests that gradients flow through all pluggable layers."""

    def test_swiglu_gradient(self) -> None:
        """Gradients must flow through SwiGLU."""
        config = _make_config(mlp_type="swiglu")
        mlp = build_mlp(config)
        x = torch.randn(1, 4, 64, requires_grad=True)
        output = mlp(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_grpu_gradient(self) -> None:
        """Gradients must flow through GRPU."""
        config = _make_config(mlp_type="innovative_grpu")
        mlp = build_mlp(config)
        x = torch.randn(1, 4, 64, requires_grad=True)
        output = mlp(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_grpu_beta_has_gradient(self) -> None:
        """The learnable beta parameter in GRPU must receive gradients."""
        config = _make_config(mlp_type="innovative_grpu")
        mlp = build_mlp(config)
        x = torch.randn(1, 4, 64)
        output = mlp(x)
        loss = output.sum()
        loss.backward()
        # Find the beta parameter
        for name, param in mlp.named_parameters():
            if "beta" in name:
                assert param.grad is not None
                assert param.grad.abs().sum() > 0
                return
        assert False, "GRPU must have a 'beta' parameter"


# ── Determinism Tests ───────────────────────────────────────────────────────


class TestDeterminism:
    """Tests that layer outputs are deterministic given the same seed."""

    def test_swiglu_deterministic(self) -> None:
        """Same seed + same input → same SwiGLU output."""
        config = _make_config(mlp_type="swiglu", seed=42)
        mlp1 = build_mlp(config)
        mlp2 = build_mlp(config)

        # Copy weights to ensure identical initialization
        mlp2.load_state_dict(mlp1.state_dict())

        x = torch.randn(1, 4, 64)
        out1 = mlp1(x)
        out2 = mlp2(x)
        assert torch.allclose(out1, out2, atol=1e-7)

    def test_grpu_deterministic(self) -> None:
        """Same seed + same input → same GRPU output."""
        config = _make_config(mlp_type="innovative_grpu", seed=42)
        mlp1 = build_mlp(config)
        mlp2 = build_mlp(config)

        mlp2.load_state_dict(mlp1.state_dict())

        x = torch.randn(1, 4, 64)
        out1 = mlp1(x)
        out2 = mlp2(x)
        assert torch.allclose(out1, out2, atol=1e-7)


# ── Layer Swap Tests ────────────────────────────────────────────────────────


class TestLayerSwap:
    """Tests that swapping layer types changes model outputs."""

    def test_swap_mlp_changes_output(self) -> None:
        """Different mlp_type must produce different outputs."""
        config_swiglu = _make_config(mlp_type="swiglu", seed=42)
        config_grpu = _make_config(mlp_type="innovative_grpu", seed=42)

        model_swiglu = build_model(config_swiglu)
        model_grpu = build_model(config_grpu)

        model_swiglu.eval()
        model_grpu.eval()

        tokens = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            out_swiglu = model_swiglu(tokens)
            out_grpu = model_grpu(tokens)

        # Outputs must differ — different equations produce different results
        assert not torch.allclose(out_swiglu, out_grpu, atol=1e-3)


# ── Full Model Tests ────────────────────────────────────────────────────────


class TestFullModel:
    """Tests for end-to-end model construction with pluggable layers."""

    def test_full_model_with_swiglu(self) -> None:
        """Full model with default SwiGLU must work."""
        config = _make_config(mlp_type="swiglu")
        model = build_model(config)
        tokens = torch.randint(0, 256, (1, 8))
        logits = model(tokens)
        assert logits.shape == (1, 8, 256)

    def test_full_model_with_grpu(self) -> None:
        """Full model with GRPU must work end-to-end."""
        config = _make_config(mlp_type="innovative_grpu")
        model = build_model(config)
        tokens = torch.randint(0, 256, (1, 8))
        logits = model(tokens)
        assert logits.shape == (1, 8, 256)
        assert not torch.isnan(logits).any()

    def test_no_nan_output(self) -> None:
        """Both MLP types must produce NaN-free outputs."""
        for mlp_type in ("swiglu", "innovative_grpu"):
            config = _make_config(mlp_type=mlp_type)
            model = build_model(config)
            tokens = torch.randint(0, 256, (1, 8))
            logits = model(tokens)
            assert not torch.isnan(logits).any(), f"NaN detected with {mlp_type}"

    def test_checkpoint_compatibility(self) -> None:
        """state_dict save/load must work across layer types."""
        config = _make_config(mlp_type="innovative_grpu")
        model1 = build_model(config)

        # Save state dict
        state = model1.state_dict()

        # Build fresh model with same config and load
        model2 = build_model(config)
        model2.load_state_dict(state)

        # Outputs must match
        model1.eval()
        model2.eval()
        tokens = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            out1 = model1(tokens)
            out2 = model2(tokens)
        assert torch.allclose(out1, out2, atol=1e-7)

    def test_config_defaults_backward_compatible(self) -> None:
        """Config with no layer type fields must default to existing behavior."""
        config = TransformerModelConfig(
            vocab_size=256,
            dim=64,
            n_layers=2,
            n_heads=2,
            head_dim=32,
        )
        assert config.mlp_type == "swiglu"
        assert config.attention_type == "causal"
        assert config.norm_type == "rmsnorm"

        # Must still build successfully
        model = build_model(config)
        tokens = torch.randint(0, 256, (1, 4))
        logits = model(tokens)
        assert logits.shape == (1, 4, 256)

    def test_grpu_tiny_overfit(self) -> None:
        """A tiny GRPU model must be able to overfit a single batch."""
        config = _make_config(
            mlp_type="innovative_grpu",
            dim=32,
            n_layers=1,
            n_heads=2,
            head_dim=16,
            vocab_size=32,
            seed=42,
        )
        model = build_model(config)
        model.train()

        # Single repeated batch for overfitting
        tokens = torch.randint(0, 32, (2, 16))
        targets = torch.randint(0, 32, (2, 16))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = nn.CrossEntropyLoss()

        initial_loss = None
        final_loss = None

        for step in range(50):
            logits = model(tokens)
            loss = loss_fn(logits.view(-1, 32), targets.view(-1))

            if step == 0:
                initial_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            final_loss = loss.item()

        # Loss must decrease — model can learn
        assert (
            final_loss < initial_loss
        ), f"Loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"
