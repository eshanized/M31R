# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for artifact manifests.
"""

import json
from pathlib import Path

import pytest

from m31r.release.manifests.manifest import (
    ArtifactManifest,
    create_manifest,
    load_manifest,
    write_manifest,
)


def test_manifest_roundtrip(tmp_path: Path):
    """Test creating, writing, and loading a manifest."""
    manifest = create_manifest(
        version="1.0.0",
        dataset_hash="hash_d",
        tokenizer_hash="hash_t",
        model_hash="hash_m",
        config_hash="hash_c",
        training_seed=42,
        m31r_version="0.1.0",
        metrics_summary={"accuracy": 0.95},
    )

    path = tmp_path / "metadata.json"
    write_manifest(manifest, path)
    loaded = load_manifest(path)

    assert loaded == manifest
    assert loaded.metrics_summary["accuracy"] == 0.95


def test_manifest_missing_fields(tmp_path: Path):
    """Test that loading a manifest with missing fields raises ValueError."""
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"version": "1.0.0"}))  # Missing required hashes

    with pytest.raises(ValueError, match="missing required fields"):
        load_manifest(path)
