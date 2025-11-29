# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""Tests for the inference engine."""

import torch
import pytest

from m31r.model.transformer import M31RTransformer, TransformerModelConfig
from m31r.serving.engine.core import InferenceEngine
from m31r.serving.generation.core import GenerationConfig


class TestInferenceEngineForward:
    """Make sure the engine can actually run a forward pass."""

    def test_forward_produces_logits(
        self,
        tiny_model: M31RTransformer,
    ) -> None:
        engine = InferenceEngine(
            model=tiny_model,
            tokenizer=None,
            device=torch.device("cpu"),
            max_context_length=128,
        )

        logits = engine._forward([1, 2, 3])
        assert logits.shape == (256,)  # vocab_size from tiny_model_config

    def test_deterministic_forward(
        self,
        tiny_model_config: TransformerModelConfig,
    ) -> None:
        """Same input should always produce the same logits."""
        torch.manual_seed(42)
        model1 = M31RTransformer(tiny_model_config)
        model1.eval()

        torch.manual_seed(42)
        model2 = M31RTransformer(tiny_model_config)
        model2.eval()

        engine1 = InferenceEngine(
            model=model1, tokenizer=None,
            device=torch.device("cpu"), max_context_length=128,
        )
        engine2 = InferenceEngine(
            model=model2, tokenizer=None,
            device=torch.device("cpu"), max_context_length=128,
        )

        logits1 = engine1._forward([1, 2, 3])
        logits2 = engine2._forward([1, 2, 3])
        assert torch.allclose(logits1, logits2)


class TestEngineGeneration:
    """Test generation with a mock tokenizer."""

    def _make_engine_with_mock_tokenizer(
        self,
        tiny_model: M31RTransformer,
    ) -> InferenceEngine:
        class MockTokenizer:
            def encode(self, text: str):
                class Result:
                    ids = [1, 2, 3]
                return Result()

            def decode(self, ids: list[int]) -> str:
                return "".join(chr(65 + (i % 26)) for i in ids)

        return InferenceEngine(
            model=tiny_model,
            tokenizer=MockTokenizer(),
            device=torch.device("cpu"),
            max_context_length=128,
        )

    def test_generate_returns_response(
        self,
        tiny_model: M31RTransformer,
    ) -> None:
        engine = self._make_engine_with_mock_tokenizer(tiny_model)
        config = GenerationConfig(max_tokens=5, temperature=0.0, eos_token_id=9999)

        response = engine.generate("test prompt", config)
        assert response.tokens_generated > 0
        assert response.total_time_ms > 0
        assert len(response.text) > 0

    def test_generate_respects_max_tokens(
        self,
        tiny_model: M31RTransformer,
    ) -> None:
        engine = self._make_engine_with_mock_tokenizer(tiny_model)
        config = GenerationConfig(max_tokens=3, temperature=0.0, eos_token_id=9999)

        response = engine.generate("test", config)
        assert response.tokens_generated <= 3

    def test_generate_stream_yields_chunks(
        self,
        tiny_model: M31RTransformer,
    ) -> None:
        engine = self._make_engine_with_mock_tokenizer(tiny_model)
        config = GenerationConfig(max_tokens=3, temperature=0.0, eos_token_id=9999)

        chunks = list(engine.generate_stream("test", config))
        assert len(chunks) > 0
        assert all(hasattr(c, "token_text") for c in chunks)

    def test_metrics_tracked_after_generate(
        self,
        tiny_model: M31RTransformer,
    ) -> None:
        engine = self._make_engine_with_mock_tokenizer(tiny_model)
        config = GenerationConfig(max_tokens=3, temperature=0.0, eos_token_id=9999)

        engine.generate("test", config)
        assert engine.metrics.total_requests == 1

    def test_no_tokenizer_raises(
        self,
        tiny_model: M31RTransformer,
    ) -> None:
        engine = InferenceEngine(
            model=tiny_model,
            tokenizer=None,
            device=torch.device("cpu"),
            max_context_length=64,
        )
        config = GenerationConfig(max_tokens=5)

        with pytest.raises(RuntimeError, match="No tokenizer"):
            engine.generate("test", config)
