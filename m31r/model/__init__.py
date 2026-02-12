# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
M31R model architecture package.

Decoder-only transformer optimized for Rust code generation.
Architecture per 06_MODEL_ARCHITECTURE.md:
  - Pluggable normalization (default: RMSNorm)
  - RoPE positional encoding
  - Pluggable feedforward (default: SwiGLU)
  - Pluggable self-attention (default: multi-head causal)
  - Weight-tied embedding/LM head

All sizes (Tiny → Large) share the same code.
Only TransformerModelConfig values change.

Layer implementations are selected via config strings (mlp_type,
attention_type, norm_type) resolved through the layer registry.
"""

from m31r.model.config import TransformerModelConfig
from m31r.model.factory import (
    build_attention,
    build_mlp,
    build_model,
    build_model_from_preset,
    build_norm,
)
from m31r.model.registry import (
    get_attention,
    get_mlp,
    get_norm,
    list_attention_types,
    list_mlp_types,
    list_norm_types,
    register_attention,
    register_mlp,
    register_norm,
)
from m31r.model.transformer import M31RTransformer

__all__ = [
    "TransformerModelConfig",
    "M31RTransformer",
    "build_model",
    "build_model_from_preset",
    "build_mlp",
    "build_attention",
    "build_norm",
    "register_mlp",
    "register_attention",
    "register_norm",
    "get_mlp",
    "get_attention",
    "get_norm",
    "list_mlp_types",
    "list_attention_types",
    "list_norm_types",
]
