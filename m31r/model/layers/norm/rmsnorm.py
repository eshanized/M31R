# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
RMSNorm — pluggable normalization layer wrapper.

Per 06_MODEL_ARCHITECTURE.md §10:
  Type: RMSNorm
  Reasons: faster than LayerNorm, lower memory, simpler computation.

This module wraps the existing RMSNorm to conform to the NormBase interface
and registers it as "rmsnorm".
"""

from m31r.model.interfaces import NormBase
from m31r.model.norm import RMSNorm
from m31r.model.registry import register_norm


class RMSNormLayer(RMSNorm, NormBase):
    """
    Root Mean Square Layer Normalization.

    Thin wrapper around RMSNorm that registers with the plugin system.
    Inherits all functionality from the base implementation.
    """

    pass


# Register with the layer registry
register_norm("rmsnorm", RMSNormLayer)
