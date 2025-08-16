# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Manifest and integrity verification tests.

Covers manifest creation, version hash computation, checksums.txt format,
and the full verify-after-build cycle.
"""

import json
from pathlib import Path

import pytest

from m31r.data.dataset.manifest import (
    build_manifest,
    compute_version_hash,
    write_checksums,
    write_dataset_manifest,
)
from m31r.data.dataset.shard import ShardInfo
from m31r.data.hashing.integrity import (
    compute_shard_checksums,
    verify_dataset_integrity,
)


def _make_shard_info(shard_id: int, sha256: str) -> ShardInfo:
    return ShardInfo(
        shard_id=shard_id,
        filename=f"shard_{shard_id:06d}.jsonl",
        entry_count=10,
        size_bytes=1024,
        sha256=sha256,
    )


class TestVersionHash:
    def test_same_shards_produce_same_hash(self) -> None:
        infos = [
            _make_shard_info(0, "aaa"),
            _make_shard_info(1, "bbb"),
        ]
        hash_a = compute_version_hash(infos)
        hash_b = compute_version_hash(infos)
        assert hash_a == hash_b

    def test_different_shards_produce_different_hash(self) -> None:
        infos_a = [_make_shard_info(0, "aaa")]
        infos_b = [_make_shard_info(0, "bbb")]
        assert compute_version_hash(infos_a) != compute_version_hash(infos_b)

    def test_hash_is_64_char_hex(self) -> None:
        infos = [_make_shard_info(0, "test")]
        h = compute_version_hash(infos)
        assert len(h) == 64
        int(h, 16)


class TestManifest:
    def test_manifest_contains_all_fields(self) -> None:
        infos = [_make_shard_info(0, "abc123")]
        sources = [{"name": "test_repo", "commit": "deadbeef"}]
        manifest = build_manifest(infos, sources, total_files=42)

        assert manifest.total_files == 42
        assert manifest.total_shards == 1
        assert manifest.total_size_bytes == 1024
        assert len(manifest.version_hash) == 64
        assert manifest.sources == sources

    def test_manifest_writes_valid_json(self, tmp_path: Path) -> None:
        infos = [_make_shard_info(0, "abc123")]
        sources = [{"name": "test_repo", "commit": "deadbeef"}]
        manifest = build_manifest(infos, sources, total_files=10)

        manifest_path = tmp_path / "manifest.json"
        write_dataset_manifest(manifest, manifest_path)

        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert data["total_files"] == 10
        assert data["version_hash"] == manifest.version_hash


class TestChecksums:
    def test_checksums_format(self, tmp_path: Path) -> None:
        infos = [
            _make_shard_info(0, "aabbcc"),
            _make_shard_info(1, "ddeeff"),
        ]
        checksums_path = tmp_path / "checksums.txt"
        write_checksums(infos, checksums_path)

        content = checksums_path.read_text(encoding="utf-8")
        lines = content.strip().splitlines()
        assert len(lines) == 2
        assert lines[0] == "aabbcc  shard_000000.jsonl"
        assert lines[1] == "ddeeff  shard_000001.jsonl"


class TestIntegrityVerification:
    def test_valid_dataset_passes(self, tmp_path: Path) -> None:
        from m31r.utils.hashing import compute_sha256_bytes

        shard_content = '{"content": "fn main() {}"}\n'
        shard_path = tmp_path / "shard_000000.jsonl"
        shard_path.write_text(shard_content, encoding="utf-8")

        actual_hash = compute_sha256_bytes(shard_content.encode("utf-8"))
        checksums_path = tmp_path / "checksums.txt"
        checksums_path.write_text(
            f"{actual_hash}  shard_000000.jsonl\n", encoding="utf-8"
        )

        assert verify_dataset_integrity(tmp_path) is True

    def test_corrupted_shard_fails(self, tmp_path: Path) -> None:
        shard_path = tmp_path / "shard_000000.jsonl"
        shard_path.write_text("original content\n", encoding="utf-8")

        checksums_path = tmp_path / "checksums.txt"
        checksums_path.write_text(
            "0000000000000000000000000000000000000000000000000000000000000000  shard_000000.jsonl\n",
            encoding="utf-8",
        )

        assert verify_dataset_integrity(tmp_path) is False

    def test_missing_checksums_file_fails(self, tmp_path: Path) -> None:
        assert verify_dataset_integrity(tmp_path) is False

    def test_missing_shard_file_fails(self, tmp_path: Path) -> None:
        checksums_path = tmp_path / "checksums.txt"
        checksums_path.write_text(
            "abcdef  shard_000000.jsonl\n", encoding="utf-8"
        )
        assert verify_dataset_integrity(tmp_path) is False
