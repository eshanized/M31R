# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Token embedding layer for M31R.

Per 06_MODEL_ARCHITECTURE.md §5:
  - Convert tokens to vectors.
  - Shared embedding and LM head weights (weight tying handled in transformer.py).
  - Deterministic initialization.
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Token embedding layer.

    Wraps nn.Embedding to convert integer token IDs to dense vectors.
    Weight tying with the LM head is managed externally by the transformer
    module, not inside this class.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        dim: Embedding dimension (must match model hidden dimension).
    """

    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)

    @property
    def weight(self) -> nn.Parameter:
        """Expose the embedding weight for weight tying."""
        return self.embedding.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Embed token IDs.

        Args:
            tokens: Integer tensor of shape (batch, seq_len).

        Returns:
            Embedding tensor of shape (batch, seq_len, dim).
        """
        return self.embedding(tokens)
