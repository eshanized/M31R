# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Offline mode verification tests.

These tests confirm that the serving runtime works without any network
access. Nothing should try to phone home, download anything, or connect
to an external service. Everything runs from local artifacts.
"""

import socket
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from m31r.serving.loader.core import resolve_device


class TestOfflineModelLoading:
    """The loader should never make network calls."""

    def test_resolve_device_is_local(self) -> None:
        device = resolve_device("cpu")
        assert device == torch.device("cpu")

    def test_auto_device_is_local(self) -> None:
        device = resolve_device("auto")
        assert device.type in ("cpu", "cuda")

    def test_loader_does_not_use_network(self, model_export_dir: Path) -> None:
        """
        Patch socket.create_connection to blow up if anything tries to
        connect to the network. The loader should never trigger this.
        """
        from m31r.serving.loader.core import _find_model_weights, _load_export_metadata

        original_connect = socket.create_connection

        def block_network(*args, **kwargs):
            raise RuntimeError("Network access detected during offline operation")

        with patch.object(socket, "create_connection", block_network):
            weights = _find_model_weights(model_export_dir)
            assert weights.exists()

            metadata = _load_export_metadata(model_export_dir)
            assert isinstance(metadata, dict)


class TestOfflineGeneration:
    """Generation should work without any external connectivity."""

    def test_greedy_decoding_is_local(self) -> None:
        from m31r.serving.generation.core import GenerationConfig, sample_next_token

        logits = torch.randn(128)
        config = GenerationConfig(temperature=0.0)

        original_connect = socket.create_connection

        def block_network(*args, **kwargs):
            raise RuntimeError("Network access during generation")

        with patch.object(socket, "create_connection", block_network):
            token = sample_next_token(logits, config)
            assert isinstance(token, int)


class TestOfflineQuantization:
    """Quantization should work entirely on local tensors."""

    def test_fp16_is_local(self) -> None:
        from m31r.model.transformer import M31RTransformer, TransformerModelConfig
        from m31r.serving.quantization.core import quantize_model

        config = TransformerModelConfig(
            vocab_size=64, dim=16, n_layers=1, n_heads=2,
            head_dim=8, max_seq_len=16, dropout=0.0,
            norm_eps=1e-6, rope_theta=10000.0, init_std=0.02, seed=42,
        )
        torch.manual_seed(42)
        model = M31RTransformer(config)

        original_connect = socket.create_connection

        def block_network(*args, **kwargs):
            raise RuntimeError("Network access during quantization")

        with patch.object(socket, "create_connection", block_network):
            result = quantize_model(model, "fp16", torch.device("cpu"))
            assert result is not None
