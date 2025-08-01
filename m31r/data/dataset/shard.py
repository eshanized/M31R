# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Fixed-size shard writer.

Sharding splits the filtered corpus into chunks of roughly equal size. This
serves two purposes:
  1. It makes the dataset parallelizable — you can train on different shards
     on different machines without coordination.
  2. It makes individual artifacts small enough to checksum, verify, and
     transfer quickly.

Each shard is a JSON Lines file where every line is one source file with its
metadata. We fill shards up to a target size and then start a new one.
The packing order is deterministic (sorted by file path) so the same filtered
directory always produces the same shards.
"""

import json
from pathlib import Path
from typing import NamedTuple

from m31r.logging.logger import get_logger
from m31r.utils.filesystem import atomic_write
from m31r.utils.hashing import compute_sha256_bytes
from m31r.utils.paths import ensure_directory


class ShardEntry(NamedTuple):
    """One source file packed into a shard."""

    source_name: str
    relative_path: str
    content: str
    sha256: str
    size_bytes: int


class ShardInfo(NamedTuple):
    """Metadata about a completed shard."""

    shard_id: int
    filename: str
    entry_count: int
    size_bytes: int
    sha256: str


class ShardWriter:
    """
    Accumulates filtered files and packs them into fixed-size shards.

    How to use it:
      1. Create a ShardWriter with a target directory and size limit
      2. Call add_entry() for each filtered file
      3. Call finalize() when you're done to flush the last partial shard
      4. Read completed_shards for the list of everything written

    The writer decides when to start a new shard based on accumulated size.
    When the current shard would exceed the size limit, it gets flushed to
    disk and a new one starts. This means shards are approximately
    (not exactly) the target size — the last entry might push slightly over.
    """

    def __init__(self, output_dir: Path, shard_size_bytes: int) -> None:
        self._output_dir = ensure_directory(output_dir)
        self._shard_size_bytes = shard_size_bytes
        self._current_entries: list[ShardEntry] = []
        self._current_size = 0
        self._shard_counter = 0
        self._completed_shards: list[ShardInfo] = []
        self._logger = get_logger("m31r.data.dataset.shard")

    def add_entry(self, entry: ShardEntry) -> None:
        """
        Add a file entry to the current shard.

        If adding this entry would push the shard over the size limit,
        we flush the current shard first and then start a fresh one.
        """
        if self._current_size + entry.size_bytes > self._shard_size_bytes and self._current_entries:
            self._flush_current_shard()

        self._current_entries.append(entry)
        self._current_size += entry.size_bytes

    def finalize(self) -> list[ShardInfo]:
        """
        Flush any remaining entries and return the complete shard list.

        Always call this after you've added all entries, otherwise the last
        partial shard won't get written to disk.
        """
        if self._current_entries:
            self._flush_current_shard()
        return self._completed_shards

    @property
    def completed_shards(self) -> list[ShardInfo]:
        return list(self._completed_shards)

    def _flush_current_shard(self) -> None:
        """
        Write the current accumulated entries to a shard file on disk.

        Each shard is a JSON Lines file — one JSON object per line. This format
        is easy to stream-read and works well with tools like jq.

        After writing, we compute the shard's SHA256 checksum. This goes into
        the dataset manifest so we can verify integrity later.
        """
        filename = f"shard_{self._shard_counter:06d}.jsonl"
        shard_path = self._output_dir / filename

        lines: list[str] = []
        for entry in self._current_entries:
            record = {
                "source": entry.source_name,
                "path": entry.relative_path,
                "content": entry.content,
                "sha256": entry.sha256,
                "size_bytes": entry.size_bytes,
            }
            lines.append(json.dumps(record, sort_keys=True))

        shard_content = "\n".join(lines) + "\n"
        atomic_write(shard_path, shard_content)

        shard_hash = compute_sha256_bytes(shard_content.encode("utf-8"))

        info = ShardInfo(
            shard_id=self._shard_counter,
            filename=filename,
            entry_count=len(self._current_entries),
            size_bytes=len(shard_content.encode("utf-8")),
            sha256=shard_hash,
        )
        self._completed_shards.append(info)

        self._logger.info(
            "Shard written",
            extra={
                "shard_id": info.shard_id,
                "entries": info.entry_count,
                "size_bytes": info.size_bytes,
            },
        )

        self._current_entries = []
        self._current_size = 0
        self._shard_counter += 1
