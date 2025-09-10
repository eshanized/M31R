# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
AdamW optimizer factory for M31R.

Per 07_TRAINING_ARCHITECTURE.md §10:
  Optimizer: AdamW
  Requirements: explicit weight decay, bias correction, no momentum on
  norm/bias parameters.

The factory separates parameters into decay and no-decay groups per
standard practice: linear weights get weight decay, everything else
(norms, biases, embeddings) does not.
"""

import torch
import torch.nn as nn

from m31r.config.schema import TrainConfig


def _separate_weight_decay_params(
    model: nn.Module,
    weight_decay: float,
) -> list[dict[str, object]]:
    """
    Split model parameters into decay and no-decay groups.

    Weight decay is applied only to 2D+ parameter tensors (weight matrices).
    1D parameters (norms, biases) get no weight decay to avoid regularizing
    them, which hurts training.

    Args:
        model: The model to extract parameters from.
        weight_decay: The weight decay value for the decay group.

    Returns:
        List of parameter group dicts suitable for torch.optim.
    """
    decay_params: list[torch.Tensor] = []
    no_decay_params: list[torch.Tensor] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def create_optimizer(
    model: nn.Module,
    train_config: TrainConfig,
) -> torch.optim.AdamW:
    """
    Create an AdamW optimizer with correctly separated parameter groups.

    Per spec: weight decay on matrices only, no decay on norms/biases.
    Beta1/beta2/lr all come from the validated config.

    Args:
        model: The model whose parameters to optimize.
        train_config: Validated training configuration.

    Returns:
        Configured AdamW optimizer.
    """
    param_groups = _separate_weight_decay_params(model, train_config.weight_decay)

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay,
    )

    return optimizer
