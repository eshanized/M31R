# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Rotary Positional Embedding (RoPE) for M31R.

Per 06_MODEL_ARCHITECTURE.md §6:
  Method: RoPE
  Reasons: stable long context, no learned position parameters,
           better extrapolation, lower memory usage.
  Absolute learned positions are forbidden.

RoPE encodes position by rotating pairs of dimensions in the query/key
vectors. This makes attention naturally position-aware without adding
any learned parameters.
"""

import torch


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Precompute the complex exponential frequencies for RoPE.

    Creates a (max_seq_len, dim//2) tensor of complex values that encode
    position information. These are computed once and cached for the entire
    training run.

    Args:
        dim: Head dimension (must be even — each pair of dims gets one rotation).
        max_seq_len: Maximum sequence length to precompute for.
        theta: Base frequency for the geometric series. Default 10000.0 per the
               original RoPE paper.
        device: Device to place the tensor on.

    Returns:
        Complex tensor of shape (max_seq_len, dim // 2).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    positions = torch.arange(max_seq_len, device=device).float()
    freqs_outer = torch.outer(positions, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs_outer), freqs_outer)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to query and key tensors.

    Takes the real-valued Q and K tensors, views them as complex numbers
    (pairing consecutive dimensions), multiplies by the precomputed
    frequency tensor, then converts back to real.

    Args:
        xq: Query tensor of shape (batch, seq_len, n_heads, head_dim).
        xk: Key tensor of shape (batch, seq_len, n_kv_heads, head_dim).
        freqs_cis: Precomputed frequencies of shape (seq_len, head_dim // 2).

    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as inputs.
    """
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Reshape freqs_cis to broadcast: (1, seq_len, 1, head_dim//2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)
