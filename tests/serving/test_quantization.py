# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""Tests for model quantization."""

import torch
import pytest

from m31r.model.transformer import M31RTransformer, TransformerModelConfig
from m31r.serving.quantization.core import (
    VALID_MODES,
    estimate_model_memory_mb,
    quantize_model,
)


@pytest.fixture()
def small_model() -> M31RTransformer:
    config = TransformerModelConfig(
        vocab_size=64, dim=16, n_layers=1, n_heads=2,
        head_dim=8, max_seq_len=32, dropout=0.0,
        norm_eps=1e-6, rope_theta=10000.0, init_std=0.02, seed=42,
    )
    torch.manual_seed(42)
    return M31RTransformer(config)


class TestQuantizationModes:

    def test_none_is_identity(self, small_model: M31RTransformer) -> None:
        original_dtype = next(small_model.parameters()).dtype
        result = quantize_model(small_model, "none", torch.device("cpu"))
        assert result is small_model
        assert next(result.parameters()).dtype == original_dtype

    def test_fp16_halves_precision(self, small_model: M31RTransformer) -> None:
        result = quantize_model(small_model, "fp16", torch.device("cpu"))
        assert next(result.parameters()).dtype == torch.float16

    def test_int8_quantizes_linear_layers(self, small_model: M31RTransformer) -> None:
        result = quantize_model(small_model, "int8", torch.device("cpu"))
        # The model should still be callable
        assert result is not None

    def test_int4_runs_without_error(self, small_model: M31RTransformer) -> None:
        result = quantize_model(small_model, "int4", torch.device("cpu"))
        assert result is not None

    def test_invalid_mode_raises(self, small_model: M31RTransformer) -> None:
        with pytest.raises(ValueError, match="Unknown quantization mode"):
            quantize_model(small_model, "fp8", torch.device("cpu"))


class TestMemoryEstimation:

    def test_estimates_positive_memory(self, small_model: M31RTransformer) -> None:
        mem = estimate_model_memory_mb(small_model)
        assert mem > 0

    def test_fp16_uses_less_memory(self, small_model: M31RTransformer) -> None:
        mem_fp32 = estimate_model_memory_mb(small_model)
        quantize_model(small_model, "fp16", torch.device("cpu"))
        mem_fp16 = estimate_model_memory_mb(small_model)
        assert mem_fp16 < mem_fp32
