# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Deterministic weight initialization for M31R.

Per 06_MODEL_ARCHITECTURE.md §21:
  Weights must use deterministic seeds.
  Loading external pretrained weights is forbidden.

All initialization uses a single torch.Generator seeded from the config.
Every call to init_weights produces identical parameter values given the
same seed and model config.
"""

import torch
import torch.nn as nn


def init_weights(module: nn.Module, seed: int, init_std: float = 0.02) -> None:
    """
    Initialize all parameters in a module deterministically.

    Uses Xavier-like normal initialization for linear layers and constant
    initialization for norms. A dedicated Generator ensures reproducibility
    regardless of external random state.

    Args:
        module: The nn.Module to initialize.
        seed: Random seed for the Generator.
        init_std: Standard deviation for normal initialization.

    Side effects:
        Modifies all parameter tensors in the module in-place.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    for name, param in module.named_parameters():
        if param.dim() >= 2:
            # Linear weight matrices: scaled normal init
            with torch.no_grad():
                param.normal_(0.0, init_std, generator=generator)
        elif "weight" in name:
            # Norm weights: init to 1.0
            with torch.no_grad():
                param.fill_(1.0)
        else:
            # Biases (if any exist): init to 0.0
            with torch.no_grad():
                param.zero_()
