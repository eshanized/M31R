# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for the tokenizer artifact bundle.

We train a tokenizer, create a bundle, then verify that all the expected
files are present, the checksums are valid, and the metadata contains
the right fields. This is the integration layer that ties training
output to the on-disk format.
"""

import json
from pathlib import Path

from m31r.config.schema import TokenizerConfig
from m31r.tokenizer.artifacts.bundle import create_bundle
from m31r.tokenizer.trainer.core import train_tokenizer
from m31r.utils.hashing import compute_sha256


def _train_and_bundle(tmp_path: Path) -> tuple:
    """Train a tiny tokenizer and create its artifact bundle."""
    config = TokenizerConfig(
        config_version="1.0.0",
        vocab_size=256,
        tokenizer_type="bpe",
        seed=42,
        min_frequency=1,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        dataset_directory="data/datasets",
        output_directory="data/tokenizer",
    )

    corpus = [
        'fn main() { println!("Hello"); }',
        "struct Point { x: f64, y: f64 }",
        "let v: Vec<i32> = vec![1, 2, 3];",
    ] * 20

    tokenizer = train_tokenizer(config, iter(corpus))
    result = create_bundle(tokenizer, config, tmp_path, dataset_hash="abc123")

    return config, tokenizer, result, tmp_path / config.output_directory


def test_bundle_creates_all_files(tmp_path: Path) -> None:
    """The bundle should contain all six expected files."""
    _, _, _, bundle_dir = _train_and_bundle(tmp_path)

    expected_files = [
        "tokenizer.json",
        "vocab.txt",
        "merges.txt",
        "config_snapshot.yaml",
        "metadata.json",
        "checksum.txt",
    ]

    for filename in expected_files:
        assert (bundle_dir / filename).is_file(), f"Missing: {filename}"


def test_tokenizer_json_is_loadable(tmp_path: Path) -> None:
    """tokenizer.json should be valid JSON that can be loaded."""
    _, _, _, bundle_dir = _train_and_bundle(tmp_path)

    tok_json = json.loads((bundle_dir / "tokenizer.json").read_text(encoding="utf-8"))
    assert "model" in tok_json


def test_metadata_has_required_fields(tmp_path: Path) -> None:
    """metadata.json should contain all the fields we need for traceability."""
    _, _, _, bundle_dir = _train_and_bundle(tmp_path)

    metadata = json.loads((bundle_dir / "metadata.json").read_text(encoding="utf-8"))

    required = [
        "version_hash",
        "dataset_hash",
        "config_hash",
        "tokenizer_type",
        "vocab_size",
        "target_vocab_size",
        "seed",
        "special_tokens",
        "created_at",
    ]
    for field in required:
        assert field in metadata, f"Missing metadata field: {field}"

    assert metadata["tokenizer_type"] == "bpe"
    assert metadata["seed"] == 42
    assert metadata["dataset_hash"] == "abc123"


def test_checksums_verify(tmp_path: Path) -> None:
    """Every file listed in checksum.txt should match its actual SHA256."""
    _, _, _, bundle_dir = _train_and_bundle(tmp_path)

    checksum_content = (bundle_dir / "checksum.txt").read_text(encoding="utf-8")
    for line in checksum_content.strip().splitlines():
        expected_hash, filename = line.split("  ", 1)
        actual_hash = compute_sha256(bundle_dir / filename)
        assert actual_hash == expected_hash, f"Checksum mismatch for {filename}"


def test_vocab_txt_is_not_empty(tmp_path: Path) -> None:
    """vocab.txt should have content after training."""
    _, _, _, bundle_dir = _train_and_bundle(tmp_path)

    vocab_content = (bundle_dir / "vocab.txt").read_text(encoding="utf-8")
    assert len(vocab_content.strip()) > 0


def test_bundle_result_has_correct_fields(tmp_path: Path) -> None:
    """The BundleResult should report accurate counts."""
    _, tokenizer, result, _ = _train_and_bundle(tmp_path)

    assert result.vocab_size == tokenizer.get_vocab_size()
    assert result.file_count == 6
    assert len(result.version_hash) == 64  # SHA256 hex is 64 chars


def test_version_hash_determinism(tmp_path: Path) -> None:
    """Creating two bundles from identically trained tokenizers must produce identical hashes."""
    config = TokenizerConfig(
        config_version="1.0.0",
        vocab_size=256,
        tokenizer_type="bpe",
        seed=42,
        min_frequency=1,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        dataset_directory="data/datasets",
        output_directory="data/tokenizer",
    )

    corpus = ["fn main() {}", "let x = 1;"] * 20

    tok_a = train_tokenizer(config, iter(corpus))
    dir_a = tmp_path / "bundle_a"
    result_a = create_bundle(tok_a, config, dir_a)

    tok_b = train_tokenizer(config, iter(corpus))
    dir_b = tmp_path / "bundle_b"
    result_b = create_bundle(tok_b, config, dir_b)

    assert result_a.version_hash == result_b.version_hash
