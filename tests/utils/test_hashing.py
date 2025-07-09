# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for hashing utilities.

Per 15_TESTING_STRATEGY.md section 16, determinism tests must run twice
and compare results. SHA256 should produce identical output for identical input,
every single time.
"""

from pathlib import Path

from m31r.utils.hashing import compute_sha256, compute_sha256_bytes, verify_checksum


class TestSha256Determinism:
    def test_same_bytes_produce_same_hash(self) -> None:
        data = b"deterministic input"
        hash_a = compute_sha256_bytes(data)
        hash_b = compute_sha256_bytes(data)
        assert hash_a == hash_b

    def test_different_bytes_produce_different_hash(self) -> None:
        hash_a = compute_sha256_bytes(b"input_a")
        hash_b = compute_sha256_bytes(b"input_b")
        assert hash_a != hash_b

    def test_empty_bytes_has_known_hash(self) -> None:
        # SHA256 of empty input is a well-known constant.
        expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert compute_sha256_bytes(b"") == expected


class TestFileHashing:
    def test_file_hash_matches_bytes_hash(self, tmp_path: Path) -> None:
        content = b"some file content for hashing"
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(content)

        file_hash = compute_sha256(test_file)
        bytes_hash = compute_sha256_bytes(content)
        assert file_hash == bytes_hash

    def test_file_hash_is_deterministic(self, tmp_path: Path) -> None:
        test_file = tmp_path / "repeat.txt"
        test_file.write_text("repeat me", encoding="utf-8")

        hash_a = compute_sha256(test_file)
        hash_b = compute_sha256(test_file)
        assert hash_a == hash_b


class TestVerifyChecksum:
    def test_correct_checksum_passes(self, tmp_path: Path) -> None:
        test_file = tmp_path / "verified.txt"
        test_file.write_bytes(b"verify me")
        expected = compute_sha256(test_file)

        assert verify_checksum(test_file, expected) is True

    def test_wrong_checksum_fails(self, tmp_path: Path) -> None:
        test_file = tmp_path / "tampered.txt"
        test_file.write_bytes(b"original content")

        assert verify_checksum(test_file, "0" * 64) is False

    def test_checksum_comparison_is_case_insensitive(self, tmp_path: Path) -> None:
        test_file = tmp_path / "case.txt"
        test_file.write_bytes(b"case test")
        expected = compute_sha256(test_file)

        assert verify_checksum(test_file, expected.upper()) is True
