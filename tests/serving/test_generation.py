# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""Tests for token generation strategies."""

import torch
import pytest

from m31r.serving.generation.core import GenerationConfig, sample_next_token


class TestGreedyDecoding:
    """When temperature is 0, we should always pick the argmax."""

    def test_picks_highest_logit(self) -> None:
        logits = torch.tensor([1.0, 5.0, 2.0, 0.5])
        config = GenerationConfig(temperature=0.0)
        result = sample_next_token(logits, config)
        assert result == 1

    def test_deterministic_across_calls(self) -> None:
        logits = torch.randn(128)
        config = GenerationConfig(temperature=0.0)

        results = [sample_next_token(logits, config) for _ in range(10)]
        assert len(set(results)) == 1

    def test_handles_multidimensional_logits(self) -> None:
        logits = torch.randn(5, 128)
        config = GenerationConfig(temperature=0.0)
        result = sample_next_token(logits, config)
        expected = int(logits[-1].argmax().item())
        assert result == expected


class TestTemperatureSampling:
    """Higher temperature should spread the distribution out more."""

    def test_low_temperature_concentrates(self) -> None:
        torch.manual_seed(42)
        logits = torch.tensor([10.0, 1.0, 1.0, 1.0])
        config = GenerationConfig(temperature=0.1, seed=42)
        gen = torch.Generator().manual_seed(42)

        results = [sample_next_token(logits, config, gen) for _ in range(20)]
        # With very low temperature, the dominant logit should win almost every time
        assert results.count(0) >= 18

    def test_seeded_sampling_is_reproducible(self) -> None:
        logits = torch.randn(128)
        config = GenerationConfig(temperature=1.0, seed=99)

        gen1 = torch.Generator().manual_seed(99)
        gen2 = torch.Generator().manual_seed(99)

        r1 = sample_next_token(logits, config, gen1)
        r2 = sample_next_token(logits, config, gen2)
        assert r1 == r2


class TestTopKSampling:
    """Top-k should restrict sampling to the k most likely tokens."""

    def test_restricts_to_top_k(self) -> None:
        torch.manual_seed(42)
        # Make the first 3 tokens much more likely than the rest
        logits = torch.tensor([-100.0] * 100)
        logits[0] = 5.0
        logits[1] = 4.0
        logits[2] = 3.0

        config = GenerationConfig(temperature=1.0, top_k=3, seed=42)
        gen = torch.Generator().manual_seed(42)

        results = set()
        for _ in range(50):
            gen.manual_seed(42 + len(results))
            results.add(sample_next_token(logits, config, gen))

        # Should only ever pick from the top 3
        assert results.issubset({0, 1, 2})

    def test_top_k_larger_than_vocab_is_fine(self) -> None:
        logits = torch.randn(10)
        config = GenerationConfig(temperature=1.0, top_k=100, seed=42)
        gen = torch.Generator().manual_seed(42)
        result = sample_next_token(logits, config, gen)
        assert 0 <= result < 10
