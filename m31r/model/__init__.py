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
"""
