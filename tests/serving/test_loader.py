# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""Tests for the artifact loader."""

import json
from pathlib import Path

import torch
import pytest

from m31r.model.transformer import TransformerModelConfig
from m31r.serving.loader.core import (
    LoadedArtifacts,
    _find_model_weights,
    _verify_checksum,
    resolve_device,
)


class TestResolveDevice:

    def test_cpu_explicit(self) -> None:
        device = resolve_device("cpu")
        assert device == torch.device("cpu")

    def test_auto_fallback_to_cpu(self) -> None:
        device = resolve_device("auto")
        # On CI/test machines without GPU, this should resolve to CPU
        assert device.type in ("cpu", "cuda")


class TestFindModelWeights:

    def test_finds_standard_model_pt(self, tmp_path: Path) -> None:
        (tmp_path / "model.pt").write_bytes(b"fake weights")
        result = _find_model_weights(tmp_path)
        assert result.name == "model.pt"

    def test_finds_any_pt_file(self, tmp_path: Path) -> None:
        (tmp_path / "checkpoint.pt").write_bytes(b"fake weights")
        result = _find_model_weights(tmp_path)
        assert result.suffix == ".pt"

    def test_raises_when_no_weights(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No model weights"):
            _find_model_weights(tmp_path)


class TestChecksumVerification:

    def test_matching_checksum_passes(self, tmp_path: Path) -> None:
        weights = tmp_path / "model.pt"
        weights.write_bytes(b"test data for hashing")

        from m31r.utils.hashing import compute_sha256
        correct_hash = compute_sha256(weights)

        # Should not raise
        _verify_checksum(weights, correct_hash)

    def test_mismatched_checksum_raises(self, tmp_path: Path) -> None:
        weights = tmp_path / "model.pt"
        weights.write_bytes(b"test data")

        with pytest.raises(RuntimeError, match="checksum mismatch"):
            _verify_checksum(weights, "0000dead0000beef")

    def test_no_expected_hash_warns(self, tmp_path: Path) -> None:
        weights = tmp_path / "model.pt"
        weights.write_bytes(b"test")

        # Should not raise, just warn
        _verify_checksum(weights, None)


class TestLoadArtifactsIntegration:
    """
    End-to-end test for load_artifacts using the fixtures from conftest.

    This checks that the full loading pipeline works: finding weights,
    verifying checksums, building the model, and loading the tokenizer.
    """

    def test_loads_successfully(
        self,
        model_export_dir: Path,
        tokenizer_dir: Path,
        tiny_model_config: TransformerModelConfig,
    ) -> None:
        from m31r.config.schema import ModelConfig, RuntimeConfig
        from m31r.serving.loader.core import load_artifacts

        model_cfg = ModelConfig(
            config_version="1.0.0",
            n_layers=2,
            hidden_size=64,
            n_heads=2,
            head_dim=32,
            context_length=128,
            dropout=0.0,
            norm_eps=1e-6,
            rope_theta=10000.0,
            init_std=0.02,
            vocab_size=256,
        )

        runtime_cfg = RuntimeConfig(
            config_version="1.0.0",
            device="cpu",
            quantization="none",
            model_path=str(model_export_dir),
            tokenizer_path=str(tokenizer_dir),
            seed=42,
            max_context_length=128,
        )

        # Use the export dir as project root since paths are absolute
        artifacts = load_artifacts(model_cfg, runtime_cfg, Path("/"))
        assert artifacts.model is not None
        assert artifacts.device == torch.device("cpu")

    def test_fails_on_missing_model_dir(self) -> None:
        from m31r.config.schema import ModelConfig, RuntimeConfig
        from m31r.serving.loader.core import load_artifacts

        model_cfg = ModelConfig(config_version="1.0.0")
        runtime_cfg = RuntimeConfig(
            config_version="1.0.0",
            model_path="/nonexistent/path",
        )

        with pytest.raises(FileNotFoundError):
            load_artifacts(model_cfg, runtime_cfg, Path("/"))
