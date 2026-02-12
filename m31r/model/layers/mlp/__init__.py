# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
MLP layer implementations.

Importing this package registers all built-in MLP types with the registry.
"""

from m31r.model.layers.mlp.innovative_grpu import GatedResidualPolynomialUnit
from m31r.model.layers.mlp.swiglu import SwiGLUMLP

__all__ = ["SwiGLUMLP", "GatedResidualPolynomialUnit"]
