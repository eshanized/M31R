# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Multi-head causal self-attention for M31R.

Per 06_MODEL_ARCHITECTURE.md §8:
  Type: Multi-head self-attention
  Requirements: causal masking, batch-first layout.

The attention computation:
  1. Project input to Q, K, V
  2. Apply RoPE to Q, K
  3. Compute scaled dot-product attention with causal mask
  4. Project output back to model dimension
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from m31r.model.rope import apply_rotary_emb


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE.

    Uses separate projections for Q, K, V and applies rotary positional
    embeddings before computing attention scores. A causal mask prevents
    attending to future positions.

    Uses PyTorch's scaled_dot_product_attention which automatically
    dispatches to FlashAttention when available.

    Args:
        dim: Model hidden dimension.
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        dropout: Attention dropout probability.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dim = dim

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute causal self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).
            freqs_cis: Precomputed RoPE frequencies (seq_len, head_dim // 2).

        Returns:
            Output tensor of shape (batch, seq_len, dim).
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V and reshape for multi-head
        xq = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Apply rotary positional embeddings to Q and K
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Transpose to (batch, n_heads, seq_len, head_dim) for attention
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Use PyTorch's scaled_dot_product_attention with causal mask
        # This automatically uses FlashAttention when available
        dropout_p = self.dropout if self.training else 0.0
        output = F.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True,
        )

        # Reshape back to (batch, seq_len, dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)
