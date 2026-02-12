# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Model factory for M31R.

Provides preset configurations, layer builder functions, and a build function
for standard model sizes. All sizes share the same code — only config values
change.

Per 06_MODEL_ARCHITECTURE.md — the architecture must scale from Tiny → Large
by configuration only, with no code modifications.

Layer builder functions use the registry to resolve layer types from config
strings. No if/else chains — the registry handles dispatch.
"""

import logging

import torch.nn as nn

from m31r.model.config import TransformerModelConfig
from m31r.model.registry import get_attention, get_mlp, get_norm

logger = logging.getLogger(__name__)


# ── Layer Builder Functions ─────────────────────────────────────────────────


def build_mlp(config: TransformerModelConfig) -> nn.Module:
    """
    Build an MLP layer from config.

    Looks up config.mlp_type in the registry and constructs the layer
    with the config's dimension, intermediate dimension, and dropout.

    Args:
        config: Model config with mlp_type, dim, intermediate_dim, dropout.

    Returns:
        An initialized nn.Module implementing the MLP contract.
    """
    mlp_cls = get_mlp(config.mlp_type)
    return mlp_cls(
        dim=config.dim,
        intermediate_dim=config.intermediate_dim,
        dropout=config.dropout,
    )


def build_attention(config: TransformerModelConfig) -> nn.Module:
    """
    Build an Attention layer from config.

    Looks up config.attention_type in the registry and constructs the layer
    with the config's dimension, head count, head dim, and dropout.

    Args:
        config: Model config with attention_type, dim, n_heads, head_dim, dropout.

    Returns:
        An initialized nn.Module implementing the Attention contract.
    """
    attn_cls = get_attention(config.attention_type)
    return attn_cls(
        dim=config.dim,
        n_heads=config.n_heads,
        head_dim=config.head_dim,
        dropout=config.dropout,
    )


def build_norm(config: TransformerModelConfig, dim: int) -> nn.Module:
    """
    Build a Normalization layer from config.

    Looks up config.norm_type in the registry and constructs the layer
    with the specified dimension and config's epsilon.

    Args:
        config: Model config with norm_type and norm_eps.
        dim: Feature dimension to normalize over.

    Returns:
        An initialized nn.Module implementing the Norm contract.
    """
    norm_cls = get_norm(config.norm_type)
    return norm_cls(dim=dim, eps=config.norm_eps)


# ── Preset Configurations ──────────────────────────────────────────────────


def tiny_config(
    vocab_size: int = 16384,
    seed: int = 42,
) -> TransformerModelConfig:
    """
    M31R-Tiny configuration (~25-30M parameters).

    Purpose: CPU-trainable model for pipeline validation and fast iteration.
    Same architecture as larger models — only dimensions differ.

    Args:
        vocab_size: Vocabulary size (config-driven, default 16384).
        seed: Random seed for deterministic initialization.

    Returns:
        TransformerModelConfig for the Tiny model.
    """
    return TransformerModelConfig(
        vocab_size=vocab_size,
        dim=384,
        n_layers=6,
        n_heads=6,
        head_dim=64,
        ffn_mult=4,
        max_seq_len=1024,
        dropout=0.1,
        norm_eps=1e-6,
        rope_theta=10000.0,
        init_std=0.02,
        seed=seed,
    )


def small_config(
    vocab_size: int = 16384,
    seed: int = 42,
) -> TransformerModelConfig:
    """
    M31R-Small configuration (~150M parameters).

    Purpose: Single-GPU training baseline.

    Args:
        vocab_size: Vocabulary size (config-driven).
        seed: Random seed for deterministic initialization.

    Returns:
        TransformerModelConfig for the Small model.
    """
    return TransformerModelConfig(
        vocab_size=vocab_size,
        dim=768,
        n_layers=12,
        n_heads=12,
        head_dim=64,
        ffn_mult=4,
        max_seq_len=2048,
        dropout=0.1,
        norm_eps=1e-6,
        rope_theta=10000.0,
        init_std=0.02,
        seed=seed,
    )


def base_config(
    vocab_size: int = 16384,
    seed: int = 42,
) -> TransformerModelConfig:
    """
    M31R-Base configuration (~700M parameters).

    Purpose: Production model for Rust code generation.

    Args:
        vocab_size: Vocabulary size (config-driven).
        seed: Random seed for deterministic initialization.

    Returns:
        TransformerModelConfig for the Base model.
    """
    return TransformerModelConfig(
        vocab_size=vocab_size,
        dim=1536,
        n_layers=24,
        n_heads=24,
        head_dim=64,
        ffn_mult=4,
        max_seq_len=2048,
        dropout=0.1,
        norm_eps=1e-6,
        rope_theta=10000.0,
        init_std=0.02,
        seed=seed,
    )


def large_config(
    vocab_size: int = 16384,
    seed: int = 42,
) -> TransformerModelConfig:
    """
    M31R-Large configuration (~1.5B parameters).

    Purpose: Upper-bound model for maximum capability.

    Args:
        vocab_size: Vocabulary size (config-driven).
        seed: Random seed for deterministic initialization.

    Returns:
        TransformerModelConfig for the Large model.
    """
    return TransformerModelConfig(
        vocab_size=vocab_size,
        dim=2048,
        n_layers=32,
        n_heads=32,
        head_dim=64,
        ffn_mult=4,
        max_seq_len=2048,
        dropout=0.1,
        norm_eps=1e-6,
        rope_theta=10000.0,
        init_std=0.02,
        seed=seed,
    )


PRESETS: dict[str, type] = {
    "tiny": tiny_config,
    "small": small_config,
    "base": base_config,
    "large": large_config,
}


def build_model(config: TransformerModelConfig) -> "M31RTransformer":
    """
    Build an M31RTransformer from a config object.

    This is the canonical entry point for model construction.

    Args:
        config: Fully populated TransformerModelConfig.

    Returns:
        Initialized M31RTransformer ready for training or inference.
    """
    # Deferred import to avoid circular dependency (transformer → block → factory)
    from m31r.model.transformer import M31RTransformer

    logger.info(
        "building_model",
        extra={
            "dim": config.dim,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "head_dim": config.head_dim,
            "ffn_mult": config.ffn_mult,
            "intermediate_dim": config.intermediate_dim,
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
            "mlp_type": config.mlp_type,
            "attention_type": config.attention_type,
            "norm_type": config.norm_type,
        },
    )
    model = M31RTransformer(config)
    param_count = model.count_parameters()
    logger.info(
        "model_built",
        extra={"total_parameters": param_count},
    )
    return model


def build_model_from_preset(
    preset: str,
    vocab_size: int = 16384,
    seed: int = 42,
) -> "M31RTransformer":
    """
    Build an M31RTransformer from a named preset.

    Args:
        preset: One of "tiny", "small", "base", "large".
        vocab_size: Override vocabulary size.
        seed: Override random seed.

    Returns:
        Initialized M31RTransformer.

    Raises:
        ValueError: If preset name is not recognized.
    """
    config_fn = PRESETS.get(preset)
    if config_fn is None:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}")
    config = config_fn(vocab_size=vocab_size, seed=seed)
    return build_model(config)
