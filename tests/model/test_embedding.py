# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for TokenEmbedding.

Validates output shape, weight access for tying, and dtype behavior.
"""

import torch

from m31r.model.embedding import TokenEmbedding


class TestTokenEmbedding:
    """Tests for the TokenEmbedding module."""

    def test_output_shape(self) -> None:
        """Output must be (batch, seq_len, dim)."""
        emb = TokenEmbedding(vocab_size=256, dim=64)
        tokens = torch.randint(0, 256, (2, 8))
        output = emb(tokens)
        assert output.shape == (2, 8, 64)

    def test_weight_property(self) -> None:
        """weight property must return an nn.Parameter."""
        emb = TokenEmbedding(vocab_size=256, dim=64)
        assert isinstance(emb.weight, torch.nn.Parameter)
        assert emb.weight.shape == (256, 64)

    def test_weight_is_same_as_embedding_weight(self) -> None:
        """weight property must expose the actual embedding weight."""
        emb = TokenEmbedding(vocab_size=256, dim=64)
        assert emb.weight is emb.embedding.weight

    def test_single_token(self) -> None:
        """Single-token input must work."""
        emb = TokenEmbedding(vocab_size=256, dim=64)
        tokens = torch.tensor([[42]])
        output = emb(tokens)
        assert output.shape == (1, 1, 64)
