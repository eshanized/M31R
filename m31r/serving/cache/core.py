# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
KV cache for transformer inference.

During generation, the model produces key and value tensors at every
attention layer for every token. Without caching, we'd recompute all
of them from scratch each time we generate a new token. That's wasteful —
the keys and values for previous tokens don't change.

The KV cache stores these tensors so we only compute the new token's
contribution at each step. This turns O(n²) per-token work into O(n),
which matters a lot when generating hundreds of tokens.

Memory is the constraint here. Each cache entry is [batch, heads, seq, head_dim]
and we pre-allocate the full context window upfront so there's no
fragmentation or reallocation during generation.
"""

import logging

import torch

from m31r.logging.logger import get_logger

logger: logging.Logger = get_logger(__name__)


class KVCache:
    """
    Pre-allocated key-value cache for autoregressive generation.

    Think of it like a scratchpad that grows as the model generates tokens.
    We allocate the maximum size upfront (based on context length) and
    fill it in progressively. When it's full, generation has to stop.

    Each transformer layer gets its own pair of (key, value) tensors
    stored in this cache. The engine feeds them back into the model
    on every forward pass so it doesn't redo past work.
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._n_layers = n_layers
        self._max_seq_len = max_seq_len
        self._current_len = 0
        self._device = device

        # Pre-allocate the full cache. Shape per layer: [1, n_heads, max_seq_len, head_dim]
        # We use batch size 1 because M31R doesn't do batched inference.
        self._keys: list[torch.Tensor] = []
        self._values: list[torch.Tensor] = []

        for _ in range(n_layers):
            k = torch.zeros(1, n_heads, max_seq_len, head_dim, device=device, dtype=dtype)
            v = torch.zeros(1, n_heads, max_seq_len, head_dim, device=device, dtype=dtype)
            self._keys.append(k)
            self._values.append(v)

    @property
    def current_length(self) -> int:
        return self._current_len

    @property
    def max_length(self) -> int:
        return self._max_seq_len

    @property
    def is_full(self) -> bool:
        return self._current_len >= self._max_seq_len

    @property
    def remaining_capacity(self) -> int:
        return self._max_seq_len - self._current_len

    def update(
        self,
        layer_idx: int,
        new_key: torch.Tensor,
        new_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Slot new key/value tensors into the cache for one layer and
        return the full accumulated key/value up to this point.

        The caller passes in tensors for just the new token(s), and we
        append them to what's already cached. We return the full
        sequence so the attention computation can see everything.
        """
        seq_len = new_key.size(2)
        end_pos = self._current_len + seq_len

        if end_pos > self._max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: trying to write to position {end_pos} "
                f"but max is {self._max_seq_len}"
            )

        self._keys[layer_idx][:, :, self._current_len : end_pos, :] = new_key
        self._values[layer_idx][:, :, self._current_len : end_pos, :] = new_value

        return (
            self._keys[layer_idx][:, :, :end_pos, :],
            self._values[layer_idx][:, :, :end_pos, :],
        )

    def advance(self, steps: int = 1) -> None:
        """Move the write cursor forward after processing new tokens."""
        self._current_len = min(self._current_len + steps, self._max_seq_len)

    def reset(self) -> None:
        """Clear the cache for a new generation sequence."""
        self._current_len = 0
        for k in self._keys:
            k.zero_()
        for v in self._values:
            v.zero_()

    def memory_bytes(self) -> int:
        """How much memory this cache is using right now."""
        total = 0
        for k, v in zip(self._keys, self._values):
            total += k.nelement() * k.element_size()
            total += v.nelement() * v.element_size()
        return total

    def memory_mb(self) -> float:
        return self.memory_bytes() / (1024 * 1024)
