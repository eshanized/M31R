# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""Tests for the KV cache."""

import torch
import pytest

from m31r.serving.cache.core import KVCache


class TestKVCacheBasics:

    def test_initial_state(self) -> None:
        cache = KVCache(
            n_layers=2, n_heads=4, head_dim=8,
            max_seq_len=32, device=torch.device("cpu"),
        )
        assert cache.current_length == 0
        assert cache.max_length == 32
        assert not cache.is_full
        assert cache.remaining_capacity == 32

    def test_update_and_advance(self) -> None:
        cache = KVCache(
            n_layers=2, n_heads=4, head_dim=8,
            max_seq_len=32, device=torch.device("cpu"),
        )

        new_k = torch.randn(1, 4, 1, 8)
        new_v = torch.randn(1, 4, 1, 8)

        keys, values = cache.update(0, new_k, new_v)
        cache.advance(1)

        assert keys.shape == (1, 4, 1, 8)
        assert values.shape == (1, 4, 1, 8)
        assert cache.current_length == 1

    def test_accumulates_across_updates(self) -> None:
        cache = KVCache(
            n_layers=1, n_heads=2, head_dim=4,
            max_seq_len=16, device=torch.device("cpu"),
        )

        for step in range(5):
            k = torch.randn(1, 2, 1, 4)
            v = torch.randn(1, 2, 1, 4)
            keys, values = cache.update(0, k, v)
            cache.advance(1)

        assert cache.current_length == 5
        assert keys.shape == (1, 2, 5, 4)

    def test_reset_clears_state(self) -> None:
        cache = KVCache(
            n_layers=1, n_heads=2, head_dim=4,
            max_seq_len=8, device=torch.device("cpu"),
        )

        k = torch.randn(1, 2, 3, 4)
        v = torch.randn(1, 2, 3, 4)
        cache.update(0, k, v)
        cache.advance(3)

        cache.reset()
        assert cache.current_length == 0

    def test_overflow_raises(self) -> None:
        cache = KVCache(
            n_layers=1, n_heads=2, head_dim=4,
            max_seq_len=4, device=torch.device("cpu"),
        )

        # Fill it up
        k = torch.randn(1, 2, 4, 4)
        v = torch.randn(1, 2, 4, 4)
        cache.update(0, k, v)
        cache.advance(4)

        # This should blow up
        with pytest.raises(RuntimeError, match="overflow"):
            extra_k = torch.randn(1, 2, 1, 4)
            extra_v = torch.randn(1, 2, 1, 4)
            cache.update(0, extra_k, extra_v)


class TestKVCacheMemory:

    def test_memory_tracking(self) -> None:
        cache = KVCache(
            n_layers=2, n_heads=4, head_dim=8,
            max_seq_len=16, device=torch.device("cpu"),
        )
        mem = cache.memory_mb()
        assert mem > 0

    def test_is_full_at_capacity(self) -> None:
        cache = KVCache(
            n_layers=1, n_heads=1, head_dim=4,
            max_seq_len=2, device=torch.device("cpu"),
        )
        cache.advance(2)
        assert cache.is_full
        assert cache.remaining_capacity == 0
