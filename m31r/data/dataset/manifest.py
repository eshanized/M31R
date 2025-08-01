# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Dataset manifest generation.

The manifest is the "table of contents" for a dataset version. It records
everything needed to reproduce or verify the dataset: which repos went in,
how many files survived filtering, checksums for every shard, and a single
version hash that uniquely identifies this exact dataset.

If anything changes — a new repo, a different filter setting, a code fix —
the version hash changes. This makes it trivial to check whether two
datasets are identical without comparing gigabytes of shard data.
"""

import json
from pathlib import Path
from typing import NamedTuple

from m31r.data.dataset.shard import ShardInfo
from m31r.utils.filesystem import atomic_write
from m31r.utils.hashing import compute_sha256_bytes


class DatasetManifest(NamedTuple):
    """Everything you need to know about a dataset version."""

    version_hash: str
    total_files: int
    total_shards: int
    total_size_bytes: int
    shard_info: list[dict[str, object]]
    sources: list[dict[str, str]]


def compute_version_hash(shard_infos: list[ShardInfo]) -> str:
    """
    Produce a single hash that uniquely identifies this dataset version.

    The version hash is the SHA256 of all shard hashes concatenated together
    in shard order. Because SHA256 is collision-resistant and the shard order
    is deterministic, two datasets will have the same version hash if and
    only if they contain the exact same data in the exact same order.

    This is how we guarantee the determinism property from the spec: same
    inputs + same config = same version hash, always.
    """
    combined = "".join(info.sha256 for info in shard_infos)
    return compute_sha256_bytes(combined.encode("utf-8"))


def build_manifest(
    shard_infos: list[ShardInfo],
    sources: list[dict[str, str]],
    total_files: int,
) -> DatasetManifest:
    """Put together the manifest from shard info and source metadata."""
    version_hash = compute_version_hash(shard_infos)

    total_size = sum(info.size_bytes for info in shard_infos)
    shard_dicts: list[dict[str, object]] = [
        {
            "shard_id": info.shard_id,
            "filename": info.filename,
            "entry_count": info.entry_count,
            "size_bytes": info.size_bytes,
            "sha256": info.sha256,
        }
        for info in shard_infos
    ]

    return DatasetManifest(
        version_hash=version_hash,
        total_files=total_files,
        total_shards=len(shard_infos),
        total_size_bytes=total_size,
        shard_info=shard_dicts,
        sources=sources,
    )


def write_dataset_manifest(manifest: DatasetManifest, output_path: Path) -> None:
    """Write the manifest as a formatted JSON file with atomic write for safety."""
    data = {
        "version_hash": manifest.version_hash,
        "total_files": manifest.total_files,
        "total_shards": manifest.total_shards,
        "total_size_bytes": manifest.total_size_bytes,
        "shards": manifest.shard_info,
        "sources": manifest.sources,
    }
    atomic_write(output_path, json.dumps(data, indent=2, sort_keys=True))


def write_checksums(shard_infos: list[ShardInfo], output_path: Path) -> None:
    """
    Write a checksums.txt file in the classic sha256sum format.

    Each line looks like: <sha256_hash>  <filename>
    This is compatible with `sha256sum -c checksums.txt` for manual verification.
    """
    lines = [f"{info.sha256}  {info.filename}" for info in shard_infos]
    content = "\n".join(lines) + "\n" if lines else ""
    atomic_write(output_path, content)
