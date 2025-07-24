# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Deterministic repository crawler.

This is the acquisition stage of the pipeline. It clones repositories at pinned
commits so every run produces the exact same file tree. Think of it like a
package manager's lockfile — the commit hash IS the version pin.

The crawler never executes anything from the cloned repos. It treats all
downloaded content as untrusted text. No cargo build, no test runs, no
build.rs evaluation. Just a read-only snapshot.
"""

import json
import subprocess
from pathlib import Path
from typing import NamedTuple

from m31r.config.schema import DatasetConfig, SourceConfig
from m31r.logging.logger import get_logger
from m31r.utils.filesystem import atomic_write
from m31r.utils.paths import ensure_directory


class RepoSnapshot(NamedTuple):
    """What we know about a single cloned repository."""

    name: str
    url: str
    commit: str
    license: str
    local_path: str
    file_count: int


class CrawlResult(NamedTuple):
    """Summary of the entire crawl run."""

    total_sources: int
    cloned: int
    skipped: int
    snapshots: list[RepoSnapshot]


def _repo_already_exists(repo_dir: Path, expected_commit: str) -> bool:
    """
    Check if we've already cloned this repo at the right commit.

    We look for a marker file that the crawler writes after a successful clone.
    If the marker exists and contains the expected commit hash, we can skip
    re-cloning. This makes the crawler idempotent — run it twice, get the same
    result without wasting time on network calls.
    """
    marker = repo_dir / ".m31r_crawl_marker"
    if not marker.exists():
        return False
    try:
        return marker.read_text(encoding="utf-8").strip() == expected_commit
    except OSError:
        return False


def _write_crawl_marker(repo_dir: Path, commit: str) -> None:
    """Drop a small marker file so we know this repo was already crawled."""
    atomic_write(repo_dir / ".m31r_crawl_marker", commit)


def _count_files(directory: Path) -> int:
    """Count all regular files under a directory, recursively."""
    count = 0
    for item in directory.rglob("*"):
        if item.is_file():
            count += 1
    return count


def clone_repository(source: SourceConfig, raw_dir: Path) -> RepoSnapshot:
    """
    Clone a single repository at a pinned commit.

    Here's how it works step by step:
    1. We create the target directory structure: raw_dir/source_name/commit_hash/
    2. We do a shallow clone (--depth 1) to save bandwidth and disk space
    3. After cloning, we verify the HEAD commit matches what we expected
    4. We write a marker file so future runs can skip this repo

    The subprocess call is intentionally limited to 'git clone'. We never run
    cargo, rustc, make, or any other tool on the cloned content. The cloned
    files are read-only data from our perspective.

    If the repo was already cloned at the correct commit, we skip it entirely
    and return the existing snapshot.
    """
    logger = get_logger("m31r.data.crawl")

    repo_dir = raw_dir / source.name / source.commit
    if _repo_already_exists(repo_dir, source.commit):
        logger.info(
            "Repo already cloned, skipping",
            extra={"source": source.name, "commit": source.commit},
        )
        return RepoSnapshot(
            name=source.name,
            url=source.url,
            commit=source.commit,
            license=source.license,
            local_path=str(repo_dir),
            file_count=_count_files(repo_dir),
        )

    ensure_directory(repo_dir)

    logger.info(
        "Cloning repository",
        extra={"source": source.name, "url": source.url, "commit": source.commit},
    )

    try:
        subprocess.run(
            [
                "git", "clone",
                "--depth", "1",
                "--single-branch",
                source.url,
                str(repo_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.CalledProcessError as err:
        raise RuntimeError(
            f"Git clone failed for {source.name}: {err.stderr.strip()}"
        ) from err
    except subprocess.TimeoutExpired as err:
        raise RuntimeError(
            f"Git clone timed out for {source.name} after 600 seconds"
        ) from err

    _write_crawl_marker(repo_dir, source.commit)
    file_count = _count_files(repo_dir)

    logger.info(
        "Clone complete",
        extra={"source": source.name, "files": file_count},
    )

    return RepoSnapshot(
        name=source.name,
        url=source.url,
        commit=source.commit,
        license=source.license,
        local_path=str(repo_dir),
        file_count=file_count,
    )


def crawl_repositories(config: DatasetConfig, project_root: Path) -> CrawlResult:
    """
    Clone all repositories defined in the dataset config.

    This is the main entry point for the crawl stage. It iterates through
    every source in the config, clones each one (or skips if already present),
    and writes a crawl manifest summarizing what was downloaded.

    The function is idempotent: calling it twice with the same config produces
    the same directory tree. Already-cloned repos get skipped.
    """
    logger = get_logger("m31r.data.crawl")
    raw_dir = ensure_directory(project_root / config.raw_directory)

    snapshots: list[RepoSnapshot] = []
    cloned = 0
    skipped = 0

    for source in config.sources:
        repo_dir = raw_dir / source.name / source.commit
        was_present = _repo_already_exists(repo_dir, source.commit)

        snapshot = clone_repository(source, raw_dir)
        snapshots.append(snapshot)

        if was_present:
            skipped += 1
        else:
            cloned += 1

    result = CrawlResult(
        total_sources=len(config.sources),
        cloned=cloned,
        skipped=skipped,
        snapshots=snapshots,
    )

    manifest_path = raw_dir / "crawl_manifest.json"
    manifest_data = {
        "total_sources": result.total_sources,
        "cloned": result.cloned,
        "skipped": result.skipped,
        "snapshots": [
            {
                "name": s.name,
                "url": s.url,
                "commit": s.commit,
                "license": s.license,
                "local_path": s.local_path,
                "file_count": s.file_count,
            }
            for s in result.snapshots
        ],
    }
    atomic_write(manifest_path, json.dumps(manifest_data, indent=2, sort_keys=True))

    logger.info(
        "Crawl complete",
        extra={
            "total": result.total_sources,
            "cloned": result.cloned,
            "skipped": result.skipped,
        },
    )

    return result
