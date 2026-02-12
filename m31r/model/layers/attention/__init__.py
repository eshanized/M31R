# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Attention layer implementations.

Importing this package registers all built-in Attention types with the registry.
"""

from m31r.model.layers.attention.causal import CausalAttention

__all__ = ["CausalAttention"]
