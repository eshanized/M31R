# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Integrity verification for dataset artifacts.

After building a dataset, you want to be sure nothing got corrupted — whether
from a disk error, a bad transfer, or a bug in the pipeline. This module
provides the tools to verify that shards match their recorded checksums.

It reuses the SHA256 utilities from m31r.utils.hashing so there's only one
hashing implementation in the codebase. Consistency matters.
"""

import json
from pathlib import Path

from m31r.logging.logger import get_logger
from m31r.utils.hashing import compute_sha256


def compute_shard_checksums(shard_dir: Path) -> dict[str, str]:
    """
    Compute SHA256 checksums for all shard files in a directory.

    Returns a dict mapping filename to hex digest, sorted by filename.
    Only processes .jsonl files to avoid accidentally checksumming
    the manifest or checksums.txt themselves.
    """
    checksums: dict[str, str] = {}

    shard_files = sorted(
        f for f in shard_dir.iterdir()
        if f.is_file() and f.suffix == ".jsonl"
    )

    for shard_file in shard_files:
        checksums[shard_file.name] = compute_sha256(shard_file)

    return checksums


def verify_dataset_integrity(dataset_dir: Path) -> bool:
    """
    Verify that all shards in a dataset match their recorded checksums.

    This reads the checksums.txt file (which was generated during the build)
    and compares each recorded hash against a freshly computed one. If even
    a single byte has changed in any shard, this returns False.

    This is the primary integrity check the spec describes. Run it after
    transferring a dataset to a new machine, or periodically as a health check.
    """
    logger = get_logger("m31r.data.hashing")

    checksums_path = dataset_dir / "checksums.txt"
    if not checksums_path.is_file():
        logger.error("No checksums.txt found", extra={"dir": str(dataset_dir)})
        return False

    expected: dict[str, str] = {}
    content = checksums_path.read_text(encoding="utf-8")
    for line in content.strip().splitlines():
        parts = line.strip().split("  ", 1)
        if len(parts) == 2:
            expected[parts[1]] = parts[0]

    if not expected:
        logger.warning("checksums.txt is empty", extra={"dir": str(dataset_dir)})
        return True

    actual = compute_shard_checksums(dataset_dir)

    all_valid = True
    for filename, expected_hash in expected.items():
        actual_hash = actual.get(filename)
        if actual_hash is None:
            logger.error(
                "Missing shard file",
                extra={"filename": filename, "dir": str(dataset_dir)},
            )
            all_valid = False
        elif actual_hash != expected_hash:
            logger.error(
                "Checksum mismatch",
                extra={
                    "filename": filename,
                    "expected": expected_hash,
                    "actual": actual_hash,
                },
            )
            all_valid = False

    if all_valid:
        logger.info(
            "Integrity check passed",
            extra={"shard_count": len(expected), "dir": str(dataset_dir)},
        )

    return all_valid


def load_manifest(dataset_dir: Path) -> dict:
    """Load and return the dataset manifest as a dict."""
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No manifest.json found in {dataset_dir}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))
