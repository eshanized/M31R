# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Filter pipeline orchestrator.

This is where all the individual filter modules come together into a single
streaming pipeline. The key design decision here is that we process files
one at a time through an iterator — we never load the entire dataset into
memory. This means we can handle repos that are hundreds of gigabytes
without running out of RAM.

The filter order is deterministic and matters:
  1. Extension check (cheapest, eliminates most files immediately)
  2. Path exclusion (drops entire directory trees)
  3. Size limits (drops oversized files)
  4. License check (per-repo, cached)
  5. AST validation (needs to read content)
  6. Deduplication (needs content hash)
  7. Normalization (transforms content for output)

Each step is a pure function that takes a file and says "keep" or "drop".
The pipeline just chains them together and copies survivors to the output.
"""

import json
from pathlib import Path, PurePosixPath
from typing import Iterator, NamedTuple

from m31r.config.schema import DatasetConfig
from m31r.data.filter.ast_validator import validate_rust_syntax
from m31r.data.filter.deduplicator import ContentDeduplicator
from m31r.data.filter.exclusions import exceeds_size_limits, should_exclude_path
from m31r.data.filter.license_filter import detect_license, is_permissive_license
from m31r.data.filter.normalizer import normalize_content
from m31r.data.filter.rust_filter import is_allowed_extension
from m31r.logging.logger import get_logger
from m31r.utils.filesystem import atomic_write
from m31r.utils.paths import ensure_directory


class FilterStats(NamedTuple):
    """Breakdown of what happened during filtering."""

    total_files_scanned: int
    kept: int
    rejected_extension: int
    rejected_path: int
    rejected_size: int
    rejected_license: int
    rejected_syntax: int
    rejected_duplicate: int


class FilteredFile(NamedTuple):
    """A file that survived all the filters."""

    relative_path: str
    source_name: str
    commit: str
    license: str
    content: str
    original_size: int
    normalized_size: int


def _iter_source_files(
    raw_dir: Path,
    config: DatasetConfig,
) -> Iterator[tuple[Path, str, str, str]]:
    """
    Walk through all crawled repositories and yield their files one at a time.

    This is a generator, so it doesn't load everything into memory. Each
    yield gives you: (absolute_file_path, source_name, commit_hash, license).

    We iterate sources in sorted order to keep the output deterministic.
    Filesystem ordering can vary between runs on some systems, so sorting
    by name ensures we always process files in the same sequence.
    """
    for source in sorted(config.sources, key=lambda s: s.name):
        repo_dir = raw_dir / source.name / source.commit
        if not repo_dir.is_dir():
            continue

        all_files = sorted(repo_dir.rglob("*"))
        for file_path in all_files:
            if file_path.is_file():
                yield file_path, source.name, source.commit, source.license


def run_filter_pipeline(
    config: DatasetConfig,
    project_root: Path,
) -> FilterStats:
    """
    Run the complete filter pipeline over all crawled data.

    This is the main entry point for the filter stage. It reads files from
    data/raw/, applies every filter in sequence, and writes survivors to
    data/filtered/. The whole thing streams — files go in one end and come
    out the other without accumulating in memory.

    Here's the flow for each file:
      raw_dir/source/commit/path/to/file.rs
        → extension check → path check → size check
        → license check → syntax check → dedup check
        → normalize → write to filtered_dir/source/commit/path/to/file.rs

    The function returns a FilterStats with detailed counts of what got
    rejected and why, which is essential for data quality auditing.
    """
    logger = get_logger("m31r.data.filter")

    raw_dir = project_root / config.raw_directory
    filtered_dir = ensure_directory(project_root / config.filtered_directory)

    deduplicator = ContentDeduplicator()
    license_cache: dict[str, str] = {}

    total = 0
    kept = 0
    rejected_ext = 0
    rejected_path = 0
    rejected_size = 0
    rejected_lic = 0
    rejected_syntax = 0
    rejected_dup = 0

    for file_path, source_name, commit, expected_license in _iter_source_files(raw_dir, config):
        total += 1

        relative = file_path.relative_to(raw_dir / source_name / commit)
        posix_path = PurePosixPath(str(relative))

        if not is_allowed_extension(posix_path, config.filter.allowed_extensions):
            rejected_ext += 1
            continue

        if should_exclude_path(posix_path, config.filter.excluded_directories):
            rejected_path += 1
            continue

        cache_key = f"{source_name}/{commit}"
        if cache_key not in license_cache:
            repo_dir = raw_dir / source_name / commit
            detected = detect_license(repo_dir)
            if detected == "unknown":
                detected = expected_license
            license_cache[cache_key] = detected

        repo_license = license_cache[cache_key]
        if not is_permissive_license(repo_license, config.filter.allowed_licenses):
            rejected_lic += 1
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            rejected_syntax += 1
            continue

        if exceeds_size_limits(content, config.filter.max_file_size_bytes, config.filter.max_lines):
            rejected_size += 1
            continue

        if not validate_rust_syntax(content):
            rejected_syntax += 1
            continue

        if config.filter.enable_deduplication and deduplicator.is_duplicate(content):
            rejected_dup += 1
            continue

        normalized = normalize_content(content)

        output_path = filtered_dir / source_name / commit / str(relative)
        ensure_directory(output_path.parent)
        atomic_write(output_path, normalized)

        kept += 1

    stats = FilterStats(
        total_files_scanned=total,
        kept=kept,
        rejected_extension=rejected_ext,
        rejected_path=rejected_path,
        rejected_size=rejected_size,
        rejected_license=rejected_lic,
        rejected_syntax=rejected_syntax,
        rejected_duplicate=rejected_dup,
    )

    stats_path = filtered_dir / "filter_stats.json"
    atomic_write(stats_path, json.dumps(stats._asdict(), indent=2, sort_keys=True))

    logger.info(
        "Filter pipeline complete",
        extra={
            "total_scanned": stats.total_files_scanned,
            "kept": stats.kept,
            "rejected_extension": stats.rejected_extension,
            "rejected_path": stats.rejected_path,
            "rejected_size": stats.rejected_size,
            "rejected_license": stats.rejected_license,
            "rejected_syntax": stats.rejected_syntax,
            "rejected_duplicate": stats.rejected_duplicate,
        },
    )

    return stats
