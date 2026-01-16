# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for release packaging.
"""

import json
from pathlib import Path

import pytest
import yaml

from m31r.release.packaging.packager import create_release


@pytest.fixture()
def mock_checkpoint_dir(tmp_path: Path) -> Path:
    """Create a fake checkpoint directory with model weights."""
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").write_bytes(b"fake model weights")
    (checkpoint_dir / "config_snapshot.yaml").write_text(
        "global:\n  seed: 42\nmodel:\n  n_layers: 2\n"
    )
    return checkpoint_dir


@pytest.fixture()
def mock_tokenizer_dir(tmp_path: Path) -> Path:
    """Create a fake tokenizer bundle."""
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text('{"vocab": []}')
    return tokenizer_dir


def test_create_release_success(
    tmp_path: Path, mock_checkpoint_dir: Path, mock_tokenizer_dir: Path
):
    """Test that a release is created with all required files."""
    output_root = tmp_path / "releases"
    result = create_release(
        version="1.0.0",
        checkpoint_dir=mock_checkpoint_dir,
        tokenizer_dir=mock_tokenizer_dir,
        output_root=output_root,
        config_path=None,  # Should find snapshot in checkpoint
    )

    release_dir = Path(result.output_dir)
    assert release_dir.exists()
    assert (release_dir / "model.safetensors").exists()
    assert (release_dir / "tokenizer.json").exists()
    assert (release_dir / "config.yaml").exists()
    assert (release_dir / "metadata.json").exists()
    assert (release_dir / "checksum.txt").exists()
    assert (release_dir / "README.txt").exists()

    # Check manifest content
    manifest = json.loads((release_dir / "metadata.json").read_text())
    assert manifest["version"] == "1.0.0"
    assert manifest["training_seed"] == 42


def test_create_release_immutability(
    tmp_path: Path, mock_checkpoint_dir: Path, mock_tokenizer_dir: Path
):
    """Test that creating a release with an existing version fails."""
    output_root = tmp_path / "releases"
    create_release(
        version="1.0.0",
        checkpoint_dir=mock_checkpoint_dir,
        tokenizer_dir=mock_tokenizer_dir,
        output_root=output_root,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        create_release(
            version="1.0.0",
            checkpoint_dir=mock_checkpoint_dir,
            tokenizer_dir=mock_tokenizer_dir,
            output_root=output_root,
        )


def test_create_release_cleanup_on_failure(tmp_path: Path, mock_checkpoint_dir: Path):
    """Test that partial releases are cleaned up on failure."""
    output_root = tmp_path / "releases"

    # Missing tokenizer will cause failure
    with pytest.raises(FileNotFoundError):
        create_release(
            version="1.0.1",
            checkpoint_dir=mock_checkpoint_dir,
            tokenizer_dir=tmp_path / "nonexistent",
            output_root=output_root,
        )

    # Release directory should not exist
    assert not (output_root / "1.0.1").exists()
