# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for the streaming corpus reader.

We create fake JSONL shards that mimic the dataset pipeline output, then
verify that the reader yields them in the right order, handles edge cases
gracefully, and never loads everything into memory at once.
"""

import json
from pathlib import Path

from m31r.tokenizer.streaming.reader import stream_corpus


def _write_shard(shard_dir: Path, filename: str, entries: list[dict]) -> None:
    """Helper to create a JSONL shard file for testing."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(entry, sort_keys=True) for entry in entries]
    (shard_dir / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_reads_content_from_shards(tmp_path: Path) -> None:
    """The reader should yield the 'content' field from each JSONL line."""
    version_dir = tmp_path / "v1"
    _write_shard(version_dir, "shard_000000.jsonl", [
        {"source": "a", "path": "a/main.rs", "content": "fn main() {}"},
        {"source": "a", "path": "a/lib.rs", "content": "pub fn hello() {}"},
    ])

    results = list(stream_corpus(version_dir))
    assert results == ["fn main() {}", "pub fn hello() {}"]


def test_reads_multiple_shards_in_order(tmp_path: Path) -> None:
    """When there are multiple shards, they should be read in sorted filename order."""
    version_dir = tmp_path / "v1"
    _write_shard(version_dir, "shard_000001.jsonl", [
        {"source": "b", "path": "b/mod.rs", "content": "mod second;"},
    ])
    _write_shard(version_dir, "shard_000000.jsonl", [
        {"source": "a", "path": "a/mod.rs", "content": "mod first;"},
    ])

    results = list(stream_corpus(version_dir))
    assert results == ["mod first;", "mod second;"]


def test_skips_lines_without_content(tmp_path: Path) -> None:
    """Lines that don't have a 'content' field should be silently skipped."""
    version_dir = tmp_path / "v1"
    _write_shard(version_dir, "shard_000000.jsonl", [
        {"source": "a", "path": "a/main.rs"},
        {"source": "a", "path": "a/lib.rs", "content": "valid"},
    ])

    results = list(stream_corpus(version_dir))
    assert results == ["valid"]


def test_skips_malformed_json(tmp_path: Path) -> None:
    """Broken JSON lines shouldn't crash the reader — just skip them."""
    version_dir = tmp_path / "v1"
    version_dir.mkdir(parents=True)
    shard_content = '{"content": "good"}\n{broken json\n{"content": "also good"}\n'
    (version_dir / "shard_000000.jsonl").write_text(shard_content, encoding="utf-8")

    results = list(stream_corpus(version_dir))
    assert results == ["good", "also good"]


def test_empty_directory_yields_nothing(tmp_path: Path) -> None:
    """An empty dataset dir should produce zero items without crashing."""
    version_dir = tmp_path / "empty"
    version_dir.mkdir(parents=True)

    results = list(stream_corpus(version_dir))
    assert results == []


def test_missing_directory_yields_nothing(tmp_path: Path) -> None:
    """A non-existent directory should produce zero items without crashing."""
    results = list(stream_corpus(tmp_path / "does_not_exist"))
    assert results == []


def test_skips_empty_lines(tmp_path: Path) -> None:
    """Blank lines in a shard file should be skipped."""
    version_dir = tmp_path / "v1"
    version_dir.mkdir(parents=True)
    content = '{"content": "hello"}\n\n\n{"content": "world"}\n'
    (version_dir / "shard_000000.jsonl").write_text(content, encoding="utf-8")

    results = list(stream_corpus(version_dir))
    assert results == ["hello", "world"]
