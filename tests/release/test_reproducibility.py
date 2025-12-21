# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for reproducibility tools.
"""

import json
from m31r.release.reproducibility import (
    freeze_environment,
    compare_environments,
    write_lockfile,
    load_lockfile,
    EnvironmentSnapshot,
)


def test_freeze_environment():
    """Test environment snapshot structure."""
    snap = freeze_environment()
    assert snap.python_version
    assert snap.platform_system
    assert isinstance(snap.packages, dict)


def test_environment_comparison_same():
    """Test that identical environments match."""
    snap = freeze_environment()
    diffs = compare_environments(snap, snap)
    assert not diffs


def test_environment_comparison_diff(tmp_path):
    """Test detection of differences."""
    snap_a = EnvironmentSnapshot(
        python_version="3.11.0",
        platform_system="Linux",
        platform_machine="x86_64",
        platform_release="1.0",
        packages={"foo": "1.0"},
    )
    snap_b = EnvironmentSnapshot(
        python_version="3.12.0",  # Changed
        platform_system="Linux",
        platform_machine="x86_64",
        platform_release="1.0",
        packages={"foo": "2.0"},  # Changed
    )

    diffs = compare_environments(snap_a, snap_b)
    assert len(diffs) == 2
    fields = {d.field for d in diffs}
    assert "python_version" in fields
    assert "package:foo" in fields


def test_lockfile_roundtrip(tmp_path):
    """Test writing and reading lockfiles."""
    snap = freeze_environment()
    path = tmp_path / "requirements.lock"
    
    write_lockfile(snap, path)
    loaded = load_lockfile(path)

    # Timestamps will differ if we don't copy it, but load_lockfile reads it back
    assert loaded == snap
