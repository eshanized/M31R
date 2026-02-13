# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Layer type registry for M31R.

Provides a deterministic, typed registry system for pluggable neural layers.
Each layer family (MLP, Attention, Norm) has its own registry to enforce
type safety and prevent cross-family registration errors.

Per 13_CODING_STANDARDS.md §13 — no global mutable state beyond these
registries. Each registry is populated exactly once at import time via
``_register_builtins()`` and remains deterministic thereafter.

Per 11_CONFIGURATION_SPEC.md §1 — the layer type is selected by config
string alone. The registry maps that string to the concrete class.
"""

import logging

import torch.nn as nn

logger = logging.getLogger(__name__)

# ── MLP Registry ────────────────────────────────────────────────────────────

_MLP_REGISTRY: dict[str, type[nn.Module]] = {}


def register_mlp(name: str, cls: type[nn.Module]) -> None:
    """
    Register an MLP layer class under a unique name.

    Args:
        name: Config-level identifier (e.g. ``"swiglu"``, ``"innovative_grpu"``).
        cls: The ``nn.Module`` subclass to register.

    Raises:
        ValueError: If ``name`` is already registered.
    """
    if name in _MLP_REGISTRY:
        raise ValueError(
            f"MLP type '{name}' is already registered to {_MLP_REGISTRY[name].__name__}"
        )
    _MLP_REGISTRY[name] = cls
    logger.debug("registered_mlp", extra={"name": name, "cls": cls.__name__})


def get_mlp(name: str) -> type[nn.Module]:
    """
    Retrieve a registered MLP class by name.

    Args:
        name: Config-level identifier.

    Returns:
        The registered ``nn.Module`` subclass.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    if name not in _MLP_REGISTRY:
        available = sorted(_MLP_REGISTRY.keys())
        raise KeyError(f"Unknown MLP type '{name}'. Available: {available}")
    return _MLP_REGISTRY[name]


def list_mlp_types() -> list[str]:
    """Return sorted list of all registered MLP type names."""
    return sorted(_MLP_REGISTRY.keys())


# ── Attention Registry ──────────────────────────────────────────────────────

_ATTENTION_REGISTRY: dict[str, type[nn.Module]] = {}


def register_attention(name: str, cls: type[nn.Module]) -> None:
    """
    Register an Attention layer class under a unique name.

    Args:
        name: Config-level identifier (e.g. ``"causal"``).
        cls: The ``nn.Module`` subclass to register.

    Raises:
        ValueError: If ``name`` is already registered.
    """
    if name in _ATTENTION_REGISTRY:
        raise ValueError(
            f"Attention type '{name}' is already registered to "
            f"{_ATTENTION_REGISTRY[name].__name__}"
        )
    _ATTENTION_REGISTRY[name] = cls
    logger.debug("registered_attention", extra={"name": name, "cls": cls.__name__})


def get_attention(name: str) -> type[nn.Module]:
    """
    Retrieve a registered Attention class by name.

    Args:
        name: Config-level identifier.

    Returns:
        The registered ``nn.Module`` subclass.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    if name not in _ATTENTION_REGISTRY:
        available = sorted(_ATTENTION_REGISTRY.keys())
        raise KeyError(f"Unknown Attention type '{name}'. Available: {available}")
    return _ATTENTION_REGISTRY[name]


def list_attention_types() -> list[str]:
    """Return sorted list of all registered Attention type names."""
    return sorted(_ATTENTION_REGISTRY.keys())


# ── Norm Registry ───────────────────────────────────────────────────────────

_NORM_REGISTRY: dict[str, type[nn.Module]] = {}


def register_norm(name: str, cls: type[nn.Module]) -> None:
    """
    Register a Normalization layer class under a unique name.

    Args:
        name: Config-level identifier (e.g. ``"rmsnorm"``).
        cls: The ``nn.Module`` subclass to register.

    Raises:
        ValueError: If ``name`` is already registered.
    """
    if name in _NORM_REGISTRY:
        raise ValueError(
            f"Norm type '{name}' is already registered to " f"{_NORM_REGISTRY[name].__name__}"
        )
    _NORM_REGISTRY[name] = cls
    logger.debug("registered_norm", extra={"name": name, "cls": cls.__name__})


def get_norm(name: str) -> type[nn.Module]:
    """
    Retrieve a registered Normalization class by name.

    Args:
        name: Config-level identifier.

    Returns:
        The registered ``nn.Module`` subclass.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    if name not in _NORM_REGISTRY:
        available = sorted(_NORM_REGISTRY.keys())
        raise KeyError(f"Unknown Norm type '{name}'. Available: {available}")
    return _NORM_REGISTRY[name]


def list_norm_types() -> list[str]:
    """Return sorted list of all registered Norm type names."""
    return sorted(_NORM_REGISTRY.keys())


# ── Builtin Registration ───────────────────────────────────────────────────

_BUILTINS_REGISTERED: bool = False


def _register_builtins() -> None:
    """
    Register all built-in layer implementations.

    Called once at module import time. Importing the layer subpackages
    triggers their ``register_*`` calls. This function is idempotent.
    """
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return

    # Importing subpackages triggers registration via their __init__.py
    import m31r.model.layers.mlp  # noqa: F401
    import m31r.model.layers.attention  # noqa: F401
    import m31r.model.layers.norm  # noqa: F401

    _BUILTINS_REGISTERED = True
    logger.debug(
        "builtins_registered",
        extra={
            "mlp_types": list_mlp_types(),
            "attention_types": list_attention_types(),
            "norm_types": list_norm_types(),
        },
    )


# Register builtins on import
_register_builtins()
