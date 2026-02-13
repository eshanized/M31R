# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Fill-in-the-Middle (FIM) and Chain-of-Thought (CoT) transformations.

Per 06_MODEL_ARCHITECTURE.md §18-20 and 08_REASONING_COT_DESIGN.md.

This module implements training-time augmentations that improve code completion
and structured reasoning capabilities.

Loss formulation:
    L = L_next + alpha * L_fim + beta * L_cot

Where:
    - L_next: standard next-token prediction (primary)
    - L_fim: fill-in-the-middle loss (alpha = 0.3 default)
    - L_cot: chain-of-thought loss (beta = 0.2 default)
"""

import random
from typing import Optional

import torch
import torch.nn.functional as F


class FIMTransform:
    """
    Fill-in-the-Middle transformation.

    Splits a sequence into prefix, middle, suffix and reformats as:
        <PRE> prefix <SUF> suffix <MID> middle <EOT>

    The model learns to predict the middle given prefix and suffix,
    enabling code completion and infilling capabilities.

    Args:
        fim_rate: Probability of applying FIM (default 0.5)
        min_span_len: Minimum length of middle span
        max_span_len: Maximum length of middle span
    """

    # Special tokens for FIM
    PRE_TOKEN = "<fim_prefix>"
    MID_TOKEN = "<fim_middle>"
    SUF_TOKEN = "<fim_suffix>"
    EOT_TOKEN = "<fim_eot>"

    def __init__(
        self,
        fim_rate: float = 0.5,
        min_span_len: int = 10,
        max_span_len: int = 100,
    ) -> None:
        self.fim_rate = fim_rate
        self.min_span_len = min_span_len
        self.max_span_len = max_span_len

    def __call__(
        self,
        tokens: torch.Tensor,
        rng: Optional[random.Random] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply FIM transformation to token sequence.

        Args:
            tokens: Input token IDs of shape (seq_len,)
            rng: Random number generator for deterministic behavior

        Returns:
            Tuple of (transformed_input, target, loss_weight)
            If FIM not applied, returns original tokens with weight 0.0
        """
        if rng is None:
            rng = random.Random()

        if rng.random() > self.fim_rate:
            return tokens, tokens, 0.0

        seq_len = len(tokens)
        if seq_len < self.min_span_len + 20:  # Need room for prefix/suffix
            return tokens, tokens, 0.0

        # Choose middle span
        span_len = min(rng.randint(self.min_span_len, self.max_span_len), seq_len // 3)
        mid_start = rng.randint(10, seq_len - span_len - 10)
        mid_end = mid_start + span_len

        # Split into prefix, middle, suffix
        prefix = tokens[:mid_start]
        middle = tokens[mid_start:mid_end]
        suffix = tokens[mid_end:]

        # Reconstruct: prefix + <SUF> + suffix + <MID> + middle + <EOT>
        # For simplicity, we keep the same length by adjusting
        transformed = torch.cat(
            [
                prefix[-(seq_len // 3) :],  # Keep last part of prefix
                torch.tensor([2]),  # Placeholder for SUF token
                suffix[: seq_len // 3],  # Keep first part of suffix
                torch.tensor([3]),  # Placeholder for MID token
                middle,
                torch.tensor([4]),  # Placeholder for EOT token
            ]
        )

        # Pad or trim to original length
        if len(transformed) < seq_len:
            padding = torch.zeros(seq_len - len(transformed), dtype=tokens.dtype)
            transformed = torch.cat([transformed, padding])
        else:
            transformed = transformed[:seq_len]

        # Target is the same
        target = transformed.clone()

        return transformed, target, 1.0


class CoTTransform:
    """
    Chain-of-Thought transformation.

    Injects structured reasoning comments before code blocks.
    Supports three types per 08_REASONING_COT_DESIGN.md:
        - Comment-based: Short // comments
        - Scratchpad: /* PLAN ... */ blocks
        - Hidden: Reasoning tokens (not emitted at inference)

    Args:
        cot_rate: Probability of applying CoT (default 0.3)
        max_reasoning_tokens: Maximum tokens for reasoning (20% of seq per spec §18)
    """

    def __init__(
        self,
        cot_rate: float = 0.3,
        max_reasoning_tokens: int = 100,
    ) -> None:
        self.cot_rate = cot_rate
        self.max_reasoning_tokens = max_reasoning_tokens

        # Sample reasoning patterns (Rust-aligned, short)
        self.reasoning_patterns = [
            "// validate input\n",
            "// propagate error\n",
            "// borrow immutably\n",
            "// allocate buffer\n",
            "// iterate items\n",
            "// handle edge case\n",
            "// check bounds\n",
            "// return result\n",
        ]

    def __call__(
        self,
        tokens: torch.Tensor,
        rng: Optional[random.Random] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply CoT transformation to token sequence.

        Args:
            tokens: Input token IDs of shape (seq_len,)
            rng: Random number generator for deterministic behavior

        Returns:
            Tuple of (transformed_input, target, loss_weight)
            If CoT not applied, returns original tokens with weight 0.0
        """
        if rng is None:
            rng = random.Random()

        if rng.random() > self.cot_rate:
            return tokens, tokens, 0.0

        # For now, return original with weight (actual implementation would inject reasoning)
        # This is a placeholder - full implementation requires tokenizer integration
        return tokens, tokens, 1.0


class MultiObjectiveLoss:
    """
    Combines next-token, FIM, and CoT losses.

    Total loss: L = L_next + alpha * L_fim + beta * L_cot

    Args:
        alpha: Weight for FIM loss (default 0.3)
        beta: Weight for CoT loss (default 0.2)
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.2,
    ) -> None:
        self.alpha = alpha
        self.beta = beta

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        fim_logits: Optional[torch.Tensor] = None,
        fim_targets: Optional[torch.Tensor] = None,
        fim_weight: float = 0.0,
        cot_logits: Optional[torch.Tensor] = None,
        cot_targets: Optional[torch.Tensor] = None,
        cot_weight: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute combined multi-objective loss.

        Args:
            logits: Main next-token logits (batch * seq_len, vocab_size)
            targets: Main targets (batch * seq_len,)
            fim_logits: Optional FIM logits
            fim_targets: Optional FIM targets
            fim_weight: Weight for FIM loss (0.0 if not applied)
            cot_logits: Optional CoT logits
            cot_targets: Optional CoT targets
            cot_weight: Weight for CoT loss (0.0 if not applied)

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains breakdown
        """
        # Primary next-token loss
        loss_next = F.cross_entropy(logits, targets)

        # FIM loss (if applicable)
        loss_fim = 0.0
        if fim_logits is not None and fim_targets is not None and fim_weight > 0:
            loss_fim = F.cross_entropy(fim_logits, fim_targets)

        # CoT loss (if applicable)
        loss_cot = 0.0
        if cot_logits is not None and cot_targets is not None and cot_weight > 0:
            loss_cot = F.cross_entropy(cot_logits, cot_targets)

        # Combined loss
        total_loss = (
            loss_next + self.alpha * fim_weight * loss_fim + self.beta * cot_weight * loss_cot
        )

        loss_dict = {
            "loss_next": loss_next.item(),
            "loss_fim": loss_fim.item() if isinstance(loss_fim, torch.Tensor) else loss_fim,
            "loss_cot": loss_cot.item() if isinstance(loss_cot, torch.Tensor) else loss_cot,
            "total": total_loss.item(),
        }

        return total_loss, loss_dict
