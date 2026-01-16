# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Shared fixtures for serving tests.

These fixtures create tiny models and tokenizers in temp directories
so we can test inference logic without needing a fully trained model.
The models are small enough to run on CPU in under a second.
"""

import json
import textwrap
from pathlib import Path

import pytest
import torch

from m31r.model.transformer import M31RTransformer, TransformerModelConfig


@pytest.fixture()
def tiny_model_config() -> TransformerModelConfig:
    """A miniature transformer config that runs fast on CPU."""
    return TransformerModelConfig(
        vocab_size=256,
        dim=64,
        n_layers=2,
        n_heads=2,
        head_dim=32,
        max_seq_len=128,
        dropout=0.0,
        norm_eps=1e-6,
        rope_theta=10000.0,
        init_std=0.02,
        seed=42,
    )


@pytest.fixture()
def tiny_model(tiny_model_config: TransformerModelConfig) -> M31RTransformer:
    """A tiny transformer model for testing."""
    torch.manual_seed(42)
    model = M31RTransformer(tiny_model_config)
    model.eval()
    return model


@pytest.fixture()
def model_export_dir(tmp_path: Path, tiny_model: M31RTransformer) -> Path:
    """
    A temporary directory that looks like a model export bundle.

    This creates the same structure that `m31r export` produces:
    model.pt, export_metadata.json, and checksums.sha256.
    """
    export_dir = tmp_path / "exports"
    export_dir.mkdir()

    weights_path = export_dir / "model.pt"
    torch.save(tiny_model.state_dict(), weights_path)

    # Compute real checksum
    from m31r.utils.hashing import compute_sha256

    weights_hash = compute_sha256(weights_path)

    metadata = {
        "step": 1000,
        "loss": 2.5,
        "seed": 42,
        "tokens_seen": 1000000,
        "weights_sha256": weights_hash,
        "model_config": {},
    }
    (export_dir / "export_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    (export_dir / "checksums.sha256").write_text(
        f"{weights_hash}  model.pt\n",
        encoding="utf-8",
    )

    return export_dir


@pytest.fixture()
def tokenizer_dir(tmp_path: Path) -> Path:
    """
    A minimal tokenizer bundle for testing.

    We create a real HuggingFace tokenizer with a tiny vocabulary so
    the encoding/decoding tests actually work end-to-end.
    """
    tok_dir = tmp_path / "tokenizer"
    tok_dir.mkdir()

    try:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace

        vocab = {chr(i): i for i in range(32, 128)}
        vocab["<pad>"] = 0
        vocab["<unk>"] = 1
        vocab["<bos>"] = 2
        vocab["<eos>"] = 3

        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.save(str(tok_dir / "tokenizer.json"))
    except ImportError:
        # If tokenizers isn't available, create a placeholder
        (tok_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    return tok_dir


@pytest.fixture()
def runtime_config_yaml(tmp_path: Path, model_export_dir: Path, tokenizer_dir: Path) -> Path:
    """A config file with global, model, and runtime sections for testing."""
    config_content = textwrap.dedent(f"""\
        global:
          config_version: "1.0.0"
          project_name: "m31r-test"
          seed: 42
          log_level: "DEBUG"
        model:
          config_version: "1.0.0"
          n_layers: 2
          hidden_size: 64
          n_heads: 2
          head_dim: 32
          context_length: 128
          dropout: 0.0
          norm_eps: 1e-6
          rope_theta: 10000.0
          init_std: 0.02
          vocab_size: 256
        runtime:
          config_version: "1.0.0"
          device: "cpu"
          quantization: "none"
          max_tokens: 10
          temperature: 0.0
          top_k: 0
          max_context_length: 128
          stream: false
          host: "127.0.0.1"
          port: 8731
          model_path: "{model_export_dir}"
          tokenizer_path: "{tokenizer_dir}"
          seed: 42
    """)
    config_file = tmp_path / "test_runtime_config.yaml"
    config_file.write_text(config_content, encoding="utf-8")
    return config_file
