# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Multi-head causal self-attention — pluggable layer wrapper.

Per 06_MODEL_ARCHITECTURE.md §8:
  Type: Multi-head self-attention
  Requirements: causal masking, FlashAttention, batch-first layout.

This module wraps the existing CausalSelfAttention to conform to the
AttentionBase interface and registers it as "causal".
"""

from m31r.model.attention import CausalSelfAttention
from m31r.model.interfaces import AttentionBase
from m31r.model.registry import register_attention


class CausalAttention(CausalSelfAttention, AttentionBase):
    """
    Multi-head causal self-attention with RoPE.

    Thin wrapper around CausalSelfAttention that registers with the plugin system.
    Inherits all functionality from the base implementation.
    """

    pass


# Register with the layer registry
register_attention("causal", CausalAttention)
