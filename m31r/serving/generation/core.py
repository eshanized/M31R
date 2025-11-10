# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Token generation strategies.

This module handles the "what token comes next?" decision. It supports
three modes:

  1. Greedy (temperature=0.0) — always pick the highest-probability token.
     This is deterministic and reproducible.
  2. Temperature sampling — scale logits by temperature before sampling.
     Higher temperature = more creative, lower = more focused.
  3. Top-k sampling — only consider the k most likely tokens.
     Cuts off the long tail of unlikely garbage.

You can combine temperature and top-k for controlled randomness.
"""

import logging
from dataclasses import dataclass

import torch

from m31r.logging.logger import get_logger

logger: logging.Logger = get_logger(__name__)


@dataclass(frozen=True)
class GenerationConfig:
    """
    Parameters that control how the next token gets picked.

    The defaults here give you deterministic greedy decoding — the
    safest choice for reproducibility. Bump temperature above zero
    if you want the model to be more creative.
    """

    max_tokens: int = 512
    temperature: float = 0.0
    top_k: int = 0
    seed: int = 42
    eos_token_id: int = 3


def sample_next_token(
    logits: torch.Tensor,
    config: GenerationConfig,
    generator: torch.Generator | None = None,
) -> int:
    """
    Pick the next token from a logits vector.

    This is where the magic happens during generation. The model gives
    us raw scores (logits) for every token in the vocabulary, and we
    need to decide which one to actually use.

    How it works, step by step:

      1. Start with the logits — a 1D tensor of shape [vocab_size]
         where each value represents "how much the model likes this token."

      2. If temperature is 0 (or very close), just take the argmax.
         That's greedy decoding: always pick the most confident answer.
         No randomness, perfectly reproducible.

      3. If temperature > 0, divide all logits by temperature first.
         This "softens" the distribution — high temperature makes the
         model more willing to pick less-likely tokens.

      4. If top_k > 0, zero out everything except the top k tokens.
         This prevents the model from ever picking really unlikely
         tokens, even with high temperature.

      5. Convert the adjusted logits to probabilities (softmax) and
         sample from that distribution using the provided RNG.

    Args:
        logits: Raw model output for the last position, shape [vocab_size].
        config: Generation parameters (temperature, top_k).
        generator: Optional torch RNG for reproducible sampling.

    Returns:
        The token ID that was selected.
    """
    if logits.dim() > 1:
        logits = logits[-1]

    if config.temperature <= 1e-8:
        return int(logits.argmax(dim=-1).item())

    scaled = logits / config.temperature

    if config.top_k > 0:
        top_values, _ = torch.topk(scaled, min(config.top_k, scaled.size(-1)))
        threshold = top_values[-1]
        scaled = scaled.masked_fill(scaled < threshold, float("-inf"))

    probs = torch.softmax(scaled, dim=-1)
    selected = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(selected.item())
