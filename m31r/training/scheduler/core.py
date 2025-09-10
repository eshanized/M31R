# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Cosine warmup learning rate scheduler for M31R.

Per 07_TRAINING_ARCHITECTURE.md §11:
  Schedule: linear warmup followed by cosine decay to min_lr.
  The warmup phase ramps from 0 to peak LR over `warmup_steps` steps.
  After warmup, cosine decay smoothly reduces LR to `min_lr`.

This is implemented as a standalone function rather than a LambdaLR scheduler
to maintain explicit control and debuggability per coding standards.
"""

import math


def get_learning_rate(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    max_steps: int,
) -> float:
    """
    Compute the learning rate for a given step.

    Three phases:
      1. Linear warmup (step < warmup_steps): 0 → max_lr
      2. Cosine decay (warmup_steps <= step < max_steps): max_lr → min_lr
      3. Post-training (step >= max_steps): min_lr

    Args:
        step: Current optimizer step (0-indexed).
        max_lr: Peak learning rate.
        min_lr: Minimum learning rate at end of decay.
        warmup_steps: Number of warmup steps.
        max_steps: Total number of training steps.

    Returns:
        Learning rate as a float.
    """
    if step < 0:
        return 0.0

    # Phase 1: Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # Phase 3: Past max_steps, hold at min_lr
    if step >= max_steps:
        return min_lr

    # Phase 2: Cosine decay
    decay_steps = max_steps - warmup_steps
    progress = (step - warmup_steps) / decay_steps
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine_decay


def set_learning_rate(
    optimizer: object,
    lr: float,
) -> None:
    """
    Set the learning rate on all parameter groups of an optimizer.

    Args:
        optimizer: A torch.optim.Optimizer (typed as object to avoid
                   importing torch at module level).
        lr: The learning rate to set.
    """
    for param_group in optimizer.param_groups:  # type: ignore[attr-defined]
        param_group["lr"] = lr
