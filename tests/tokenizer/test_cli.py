# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
CLI smoke tests for the tokenizer subcommands.

These test that the CLI machinery works — args get parsed, handlers get
called, exit codes are correct. They don't test the full training pipeline
(that's in test_trainer.py), just the wiring.
"""

import textwrap
from pathlib import Path
from unittest.mock import patch

from m31r.cli.commands import (
    handle_tokenizer_decode,
    handle_tokenizer_encode,
    handle_tokenizer_info,
    handle_tokenizer_train,
)
from m31r.cli.exit_codes import CONFIG_ERROR, SUCCESS, VALIDATION_ERROR


def _make_tokenizer_config_file(tmp_path: Path) -> Path:
    """Create a config file with tokenizer section for CLI tests."""
    config_content = textwrap.dedent("""\
        global:
          config_version: "1.0.0"
          project_name: "m31r-test"
          seed: 42
          log_level: "DEBUG"
        tokenizer:
          config_version: "1.0.0"
          vocab_size: 256
          tokenizer_type: "bpe"
          seed: 42
          min_frequency: 1
          dataset_directory: "data/datasets"
          output_directory: "data/tokenizer"
    """)
    config_file = tmp_path / "test_tokenizer_config.yaml"
    config_file.write_text(config_content, encoding="utf-8")
    return config_file


def _make_args(**kwargs):
    """Build a fake argparse.Namespace for testing CLI handlers."""
    import argparse

    defaults = {
        "config": None,
        "log_level": "DEBUG",
        "dry_run": False,
        "seed": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_train_requires_config() -> None:
    """tokenizer train without a config should fail with CONFIG_ERROR."""
    args = _make_args()
    # No config means config is None, tokenizer config check will fail
    # But since _load_and_bootstrap returns None config, it hits the config check
    result = handle_tokenizer_train(args)
    assert result == CONFIG_ERROR


def test_train_dry_run(tmp_path: Path) -> None:
    """tokenizer train --dry-run should succeed without actually training."""
    config_file = _make_tokenizer_config_file(tmp_path)
    args = _make_args(config=str(config_file), dry_run=True)

    with patch("m31r.cli.commands._resolve_project_root", return_value=tmp_path):
        result = handle_tokenizer_train(args)

    assert result == SUCCESS


def test_encode_no_tokenizer(tmp_path: Path) -> None:
    """Encoding without a trained tokenizer should fail with VALIDATION_ERROR."""
    config_file = _make_tokenizer_config_file(tmp_path)
    args = _make_args(config=str(config_file), text="test")

    with patch("m31r.cli.commands._resolve_project_root", return_value=tmp_path):
        result = handle_tokenizer_encode(args)

    assert result == VALIDATION_ERROR


def test_decode_no_tokenizer(tmp_path: Path) -> None:
    """Decoding without a trained tokenizer should fail with VALIDATION_ERROR."""
    config_file = _make_tokenizer_config_file(tmp_path)
    args = _make_args(config=str(config_file), token_ids="1,2,3")

    with patch("m31r.cli.commands._resolve_project_root", return_value=tmp_path):
        result = handle_tokenizer_decode(args)

    assert result == VALIDATION_ERROR


def test_info_no_bundle(tmp_path: Path) -> None:
    """Info command without a bundle should fail with VALIDATION_ERROR."""
    config_file = _make_tokenizer_config_file(tmp_path)
    args = _make_args(config=str(config_file))

    with patch("m31r.cli.commands._resolve_project_root", return_value=tmp_path):
        result = handle_tokenizer_info(args)

    assert result == VALIDATION_ERROR
