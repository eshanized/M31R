# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Shard writer and dataset builder tests.

Validates that the shard writer respects size limits, packs in deterministic
order, handles edge cases, and that the full dataset builder produces
correct manifests and checksums.
"""

import json
from pathlib import Path

import pytest

from m31r.data.dataset.shard import ShardEntry, ShardWriter


def _make_entry(name: str, content: str) -> ShardEntry:
    """Quick helper to build a ShardEntry for testing."""
    from m31r.utils.hashing import compute_sha256_bytes

    encoded = content.encode("utf-8")
    return ShardEntry(
        source_name="test_repo",
        relative_path=f"src/{name}.rs",
        content=content,
        sha256=compute_sha256_bytes(encoded),
        size_bytes=len(encoded),
    )


class TestShardWriter:
    def test_single_shard_for_small_data(self, tmp_path: Path) -> None:
        writer = ShardWriter(tmp_path, shard_size_bytes=10_000)
        writer.add_entry(_make_entry("a", "fn a() {}"))
        writer.add_entry(_make_entry("b", "fn b() {}"))
        shards = writer.finalize()

        assert len(shards) == 1
        assert shards[0].entry_count == 2

    def test_multiple_shards_when_size_exceeded(self, tmp_path: Path) -> None:
        writer = ShardWriter(tmp_path, shard_size_bytes=50)
        writer.add_entry(_make_entry("a", "fn a() { let x = 1; }"))
        writer.add_entry(_make_entry("b", "fn b() { let y = 2; }"))
        writer.add_entry(_make_entry("c", "fn c() { let z = 3; }"))
        shards = writer.finalize()

        assert len(shards) >= 2

    def test_empty_input_produces_no_shards(self, tmp_path: Path) -> None:
        writer = ShardWriter(tmp_path, shard_size_bytes=1000)
        shards = writer.finalize()
        assert len(shards) == 0

    def test_shard_files_written_to_disk(self, tmp_path: Path) -> None:
        writer = ShardWriter(tmp_path, shard_size_bytes=10_000)
        writer.add_entry(_make_entry("test", "fn test() {}"))
        shards = writer.finalize()

        shard_file = tmp_path / shards[0].filename
        assert shard_file.exists()

        content = shard_file.read_text(encoding="utf-8")
        records = [json.loads(line) for line in content.strip().splitlines()]
        assert len(records) == 1
        assert records[0]["source"] == "test_repo"

    def test_shard_checksums_are_populated(self, tmp_path: Path) -> None:
        writer = ShardWriter(tmp_path, shard_size_bytes=10_000)
        writer.add_entry(_make_entry("test", "fn test() {}"))
        shards = writer.finalize()

        assert len(shards[0].sha256) == 64

    def test_deterministic_output(self, tmp_path: Path) -> None:
        """Same entries in same order should always produce identical shards."""
        entries = [_make_entry(f"f{i}", f"fn f{i}() {{}}") for i in range(5)]

        dir_a = tmp_path / "run_a"
        dir_a.mkdir()
        writer_a = ShardWriter(dir_a, shard_size_bytes=10_000)
        for e in entries:
            writer_a.add_entry(e)
        shards_a = writer_a.finalize()

        dir_b = tmp_path / "run_b"
        dir_b.mkdir()
        writer_b = ShardWriter(dir_b, shard_size_bytes=10_000)
        for e in entries:
            writer_b.add_entry(e)
        shards_b = writer_b.finalize()

        assert len(shards_a) == len(shards_b)
        for sa, sb in zip(shards_a, shards_b):
            assert sa.sha256 == sb.sha256
