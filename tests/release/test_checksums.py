# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for checksum generation and verification.
"""

from pathlib import Path

from m31r.release.checksums.integrity import (
    generate_checksums,
    verify_checksums,
    write_checksum_file,
)
from m31r.utils.hashing import compute_sha256


def test_checksum_generation(tmp_path: Path):
    """Test SHA256 generation for a directory."""
    (tmp_path / "file1.txt").write_text("content1")
    (tmp_path / "file2.txt").write_text("content2")

    checksums = generate_checksums(tmp_path)

    assert len(checksums) == 2
    assert checksums["file1.txt"] == compute_sha256(tmp_path / "file1.txt")
    assert checksums["file2.txt"] == compute_sha256(tmp_path / "file2.txt")


def test_checksum_verification_success(tmp_path: Path):
    """Test verification passes for valid files."""
    (tmp_path / "data.bin").write_bytes(b"\x00\x01\x02")
    checksums = generate_checksums(tmp_path)
    write_checksum_file(tmp_path, checksums)

    result = verify_checksums(tmp_path)
    assert result.is_valid
    assert result.checked_count == 1
    assert not result.mismatches
    assert not result.missing_files


def test_checksum_verification_corruption(tmp_path: Path):
    """Test verification fails when a file is modified."""
    (tmp_path / "data.bin").write_bytes(b"original")
    checksums = generate_checksums(tmp_path)
    write_checksum_file(tmp_path, checksums)

    # Corrupt the file
    (tmp_path / "data.bin").write_bytes(b"corrupted")

    result = verify_checksums(tmp_path)
    assert not result.is_valid
    assert "data.bin" in result.mismatches


def test_checksum_verification_missing_file(tmp_path: Path):
    """Test verification fails when a file is missing."""
    (tmp_path / "kept.txt").write_text("kept")
    (tmp_path / "lost.txt").write_text("lost")

    checksums = generate_checksums(tmp_path)
    write_checksum_file(tmp_path, checksums)

    # Delete a file
    (tmp_path / "lost.txt").unlink()

    result = verify_checksums(tmp_path)
    assert not result.is_valid
    assert "lost.txt" in result.missing_files
