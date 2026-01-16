# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Dataset builder — the final stage of the data pipeline.

This takes filtered files from data/filtered/ and packs them into versioned,
checksummed shards ready for training. The output is a self-contained dataset
directory with:
  - Numbered shard files (shard_000000.jsonl, shard_000001.jsonl, ...)
  - A manifest.json with full metadata and version hash
  - A checksums.txt for quick integrity verification

The version hash in the manifest is derived from all shard checksums, so it
changes if and only if the data changes. This is the determinism guarantee:
same filtered input + same config = same version hash, every time.
"""

from pathlib import Path
from typing import NamedTuple

from m31r.config.schema import DatasetConfig
from m31r.data.dataset.manifest import (
    build_manifest,
    write_checksums,
    write_dataset_manifest,
)
from m31r.data.dataset.shard import ShardEntry, ShardInfo, ShardWriter
from m31r.logging.logger import get_logger
from m31r.utils.hashing import compute_sha256_bytes
from m31r.utils.paths import ensure_directory


class DatasetResult(NamedTuple):
    """Summary of the dataset build."""

    version_hash: str
    total_files: int
    total_shards: int
    total_size_bytes: int
    output_directory: str


def _collect_filtered_files(filtered_dir: Path) -> list[tuple[Path, str]]:
    """
    Gather all filtered files in deterministic sorted order.

    We sort by the relative path string to guarantee consistent ordering
    across runs and platforms. Without sorting, different filesystems could
    yield files in different orders, breaking determinism.

    Returns a list of (absolute_path, relative_path_string) tuples.
    """
    files: list[tuple[Path, str]] = []

    if not filtered_dir.is_dir():
        return files

    for file_path in sorted(filtered_dir.rglob("*")):
        if file_path.is_file() and file_path.name != "filter_stats.json":
            relative = str(file_path.relative_to(filtered_dir))
            files.append((file_path, relative))

    return files


def build_dataset(
    config: DatasetConfig,
    project_root: Path,
) -> DatasetResult:
    """
    Build a versioned dataset from filtered files.

    This is the main entry point for the dataset stage. Here's what happens:
      1. Scan filtered_dir for all surviving files
      2. Sort them deterministically by path
      3. Read each file and pack it into a shard via ShardWriter
      4. When a shard hits the size limit, flush it and start the next one
      5. After all files are packed, write the manifest and checksums

    The output directory is data/datasets/<version_hash>/, where the version
    hash is computed from all shard checksums. This means the directory name
    itself is a content-addressable identifier for the dataset.
    """
    logger = get_logger("m31r.data.dataset")

    filtered_dir = project_root / config.filtered_directory
    dataset_base_dir = ensure_directory(project_root / config.dataset_directory)

    files = _collect_filtered_files(filtered_dir)

    if not files:
        logger.warning("No filtered files found, producing empty dataset")

    temp_shard_dir = ensure_directory(dataset_base_dir / "_building")
    writer = ShardWriter(temp_shard_dir, config.shard.shard_size_bytes)

    sources_seen: dict[str, str] = {}

    for file_path, relative_path in files:
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError:
            logger.warning(
                "Could not read filtered file, skipping",
                extra={"path": str(file_path)},
            )
            continue

        content_hash = compute_sha256_bytes(content.encode("utf-8"))
        size_bytes = len(content.encode("utf-8"))

        parts = relative_path.split("/")
        source_name = parts[0] if parts else "unknown"
        if source_name not in sources_seen:
            sources_seen[source_name] = parts[1] if len(parts) > 1 else "unknown"

        entry = ShardEntry(
            source_name=source_name,
            relative_path=relative_path,
            content=content,
            sha256=content_hash,
            size_bytes=size_bytes,
        )
        writer.add_entry(entry)

    shard_infos = writer.finalize()

    source_list = [
        {"name": name, "commit": commit} for name, commit in sorted(sources_seen.items())
    ]

    manifest = build_manifest(shard_infos, source_list, len(files))

    version_dir = ensure_directory(dataset_base_dir / manifest.version_hash)

    for shard_file in sorted(temp_shard_dir.iterdir()):
        if shard_file.is_file():
            target = version_dir / shard_file.name
            shard_file.rename(target)

    write_dataset_manifest(manifest, version_dir / "manifest.json")
    write_checksums(shard_infos, version_dir / "checksums.txt")

    _cleanup_temp_dir(temp_shard_dir)

    logger.info(
        "Dataset build complete",
        extra={
            "version_hash": manifest.version_hash,
            "total_files": manifest.total_files,
            "total_shards": manifest.total_shards,
            "total_size_bytes": manifest.total_size_bytes,
        },
    )

    return DatasetResult(
        version_hash=manifest.version_hash,
        total_files=manifest.total_files,
        total_shards=manifest.total_shards,
        total_size_bytes=manifest.total_size_bytes,
        output_directory=str(version_dir),
    )


def _cleanup_temp_dir(temp_dir: Path) -> None:
    """Remove the temporary build directory if it's empty."""
    try:
        if temp_dir.is_dir() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()
    except OSError:
        pass
