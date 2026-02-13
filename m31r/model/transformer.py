# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Full transformer model for M31R.

Per 06_MODEL_ARCHITECTURE.md §4, the topology is:
  Input Tokens → Embedding → N × Transformer Blocks → Final Norm → Linear LM Head → Logits

Weight tying (§5): shared embedding and LM head weights.
All initialization is deterministic (§21).

Layer implementations within each block are selected via config-driven
factory system. The transformer itself uses build_norm for the final
normalization layer.
"""

import torch
import torch.nn as nn

from m31r.model.block import TransformerBlock
from m31r.model.config import TransformerModelConfig
from m31r.model.embedding import TokenEmbedding
from m31r.model.factory import build_norm
from m31r.model.rope import precompute_freqs_cis
from m31r.model.utils import count_parameters, init_weights


class M31RTransformer(nn.Module):
    """
    Decoder-only transformer for Rust code generation.

    Implements the full model topology specified in 06_MODEL_ARCHITECTURE.md:
      - Token embedding (weight-tied with output head)
      - N transformer blocks (pre-norm, pluggable layers via config)
      - Final normalization (pluggable via config)
      - Linear LM head (shared weights with embedding)

    The model precomputes RoPE frequencies once and passes them to each block.
    All weights are initialized deterministically from a single seed.

    Args:
        config: TransformerModelConfig with all architecture parameters.
    """

    def __init__(self, config: TransformerModelConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.tok_embeddings = TokenEmbedding(config.vocab_size, config.dim)

        # Transformer blocks — each block uses factory to build its layers
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        # Final normalization — uses the same pluggable norm type
        self.norm = build_norm(config, config.dim)

        # Output projection (weight-tied with embedding)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Precompute RoPE frequencies — stored as a buffer (non-parameter state)
        freqs_cis = precompute_freqs_cis(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Deterministic initialization
        init_weights(self, seed=config.seed, init_std=config.init_std)

        # Weight tying: embedding and output head share the same weight matrix
        self.output.weight = self.tok_embeddings.weight

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer.

        Args:
            tokens: Input token IDs of shape (batch, seq_len).

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size).
        """
        seq_len = tokens.shape[1]

        # Token embedding
        h = self.tok_embeddings(tokens)

        # Get RoPE frequencies for this sequence length
        freqs_cis = self.freqs_cis[:seq_len]

        # Pass through all transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis)

        # Final norm and output projection
        h = self.norm(h)
        logits = self.output(h)

        return logits

    def count_parameters(self) -> int:
        """
        Count total trainable parameters.

        Returns:
            Integer count of all parameters with requires_grad=True.
        """
        return count_parameters(self)
