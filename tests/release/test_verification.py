# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for release verification.
"""

from pathlib import Path

import pytest

from m31r.release.checksums.integrity import generate_checksums, write_checksum_file
from m31r.release.manifests.manifest import create_manifest, write_manifest
from m31r.release.verification.verifier import verify_release


@pytest.fixture()
def valid_release_dir(tmp_path: Path) -> Path:
    """Create a valid release directory structure."""
    release_dir = tmp_path / "release_v1"
    release_dir.mkdir()

    # Create content files
    (release_dir / "model.safetensors").write_bytes(b"model")
    (release_dir / "tokenizer.json").write_text("{}")
    (release_dir / "config.yaml").write_text("config: true")
    (release_dir / "README.txt").write_text("readme")

    # Create manifest
    manifest = create_manifest(
        version="1.0.0",
        dataset_hash="h1",
        tokenizer_hash="h2",
        model_hash="h3",
        config_hash="h4",
        training_seed=42,
        m31r_version="0.1.0",
    )
    write_manifest(manifest, release_dir / "metadata.json")

    # Create checksums
    checksums = generate_checksums(release_dir)
    write_checksum_file(release_dir, checksums)

    return release_dir


def test_verify_valid_release(valid_release_dir: Path):
    """Test that a perfect release passes verification."""
    report = verify_release(valid_release_dir)
    assert report.is_valid
    assert not report.errors
    assert "checksums_valid" in report.checks_passed


def test_verify_missing_file(valid_release_dir: Path):
    """Test failure when required file is missing."""
    (valid_release_dir / "model.safetensors").unlink()

    report = verify_release(valid_release_dir)
    assert not report.is_valid
    assert "model_readable" in report.checks_failed


def test_verify_invalid_metadata(valid_release_dir: Path):
    """Test failure when metadata.json is corrupted."""
    (valid_release_dir / "metadata.json").write_text("{broken json")

    report = verify_release(valid_release_dir)
    assert not report.is_valid
    assert "metadata_valid" in report.checks_failed


def test_verify_checksum_mismatch(valid_release_dir: Path):
    """Test failure when content doesn't match checksum."""
    # Modify a file after checksum generation
    (valid_release_dir / "config.yaml").write_text("modified content")

    report = verify_release(valid_release_dir)
    assert not report.is_valid
    assert "checksums_valid" in report.checks_failed
    assert any("config.yaml" in e for e in report.errors)
