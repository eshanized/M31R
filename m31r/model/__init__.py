# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
M31R model architecture package.

Decoder-only transformer optimized for Rust code generation.
Architecture per 06_MODEL_ARCHITECTURE.md:
  - RMSNorm
  - RoPE positional encoding
  - SwiGLU feedforward
  - Multi-head causal self-attention
  - Weight-tied embedding/LM head

All sizes (Tiny → Large) share the same code.
Only TransformerModelConfig values change.
"""

from m31r.model.config import TransformerModelConfig
from m31r.model.factory import build_model, build_model_from_preset
from m31r.model.transformer import M31RTransformer

__all__ = [
    "TransformerModelConfig",
    "M31RTransformer",
    "build_model",
    "build_model_from_preset",
]
