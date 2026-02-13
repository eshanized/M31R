#!/usr/bin/env python3
# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Production-grade Rust dataset harvester for M31R.

Downloads high-quality Rust source code from:
  1. GitHub repositories (via Search API)
  2. crates.io packages (via Registry API)

This is ONLY data acquisition. Filtering, AST parsing, and deduplication
are handled downstream by ``m31r filter``.

Design invariants:
  - Deterministic: same config → identical result set.
  - Resumable: already-downloaded repos/crates are skipped.
  - Safe: never executes downloaded code, sanitizes all paths.
  - Auditable: manifests record name, commit/version, license, checksum.
  - Offline-friendly: after download, no network required.

See docs/05_DATA_ARCHITECTURE.md and docs/19_SECURITY_AND_SAFETY.md.

Usage:
    python scripts/snigdhaos-rust-harvester.py \\
        --github-repos 100 \\
        --crate-count 100 \\
        --min-stars 500 \\
        --output data/raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# ============================================================
# Constants
# ============================================================

_SCRIPT_VERSION: str = "1.0.0"

_GITHUB_API_BASE: str = "https://api.github.com"
_CRATES_IO_API_BASE: str = "https://crates.io/api/v1"

_CRAWL_MARKER_FILENAME: str = ".m31r_crawl_marker"

_ALLOWED_LICENSES: frozenset[str] = frozenset({
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "Unlicense",
    "CC0-1.0",
})

# SPDX identifiers that GitHub may return, mapped to canonical form.
_LICENSE_ALIASES: dict[str, str] = {
    "mit": "MIT",
    "apache-2.0": "Apache-2.0",
    "bsd-2-clause": "BSD-2-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "isc": "ISC",
    "unlicense": "Unlicense",
    "cc0-1.0": "CC0-1.0",
    "0bsd": "BSD-2-Clause",
}

_GIT_CLONE_TIMEOUT_SECONDS: int = 600
_HTTP_TIMEOUT_SECONDS: int = 60
_MAX_RETRIES: int = 3
_INITIAL_BACKOFF_SECONDS: float = 2.0

_STREAM_CHUNK_SIZE: int = 65_536  # 64 KiB


# ============================================================
# Structured Logging Setup
# ============================================================

def _configure_logging(level: str) -> logging.Logger:
    """Configure structured JSON logging. Returns the root harvester logger."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(log_level)

    formatter = logging.Formatter(
        fmt='{"timestamp":"%(asctime)s","level":"%(levelname)s",'
            '"logger":"%(name)s","message":"%(message)s"}',
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger("m31r.harvester")
    logger.setLevel(log_level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# ============================================================
# Data Models
# ============================================================

@dataclass(frozen=True)
class HarvesterConfig:
    """Immutable configuration for a single harvester run."""

    github_repo_count: int
    crate_count: int
    min_stars: int
    output_directory: Path
    github_token: Optional[str]
    log_level: str

    def __post_init__(self) -> None:
        if self.github_repo_count < 0:
            raise ValueError("github_repo_count must be non-negative")
        if self.crate_count < 0:
            raise ValueError("crate_count must be non-negative")
        if self.min_stars < 0:
            raise ValueError("min_stars must be non-negative")


@dataclass(frozen=True)
class RepoRecord:
    """A single GitHub repository record for the manifest."""

    name: str
    full_name: str
    url: str
    clone_url: str
    default_branch: str
    commit: str
    license_spdx: str
    stars: int
    local_path: str
    file_count: int
    checksum_sha256: str


@dataclass(frozen=True)
class CrateRecord:
    """A single crates.io package record for the manifest."""

    name: str
    version: str
    download_url: str
    license_spdx: str
    downloads: int
    local_path: str
    file_count: int
    checksum_sha256: str


@dataclass
class HarvestStats:
    """Mutable accumulator for run statistics."""

    github_queried: int = 0
    github_cloned: int = 0
    github_skipped: int = 0
    github_filtered: int = 0
    crates_queried: int = 0
    crates_downloaded: int = 0
    crates_skipped: int = 0
    crates_filtered: int = 0
    errors: list[str] = field(default_factory=list)


# ============================================================
# HTTP Utilities
# ============================================================

def _build_github_headers(token: Optional[str]) -> dict[str, str]:
    """Build HTTP headers for GitHub API requests."""
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "M31R-Rust-Harvester/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _build_crates_headers() -> dict[str, str]:
    """Build HTTP headers for crates.io API requests."""
    return {
        "Accept": "application/json",
        "User-Agent": "M31R-Rust-Harvester/1.0 (https://github.com/m31r-project)",
    }


def _http_get_json(
    url: str,
    headers: dict[str, str],
    logger: logging.Logger,
) -> tuple[Any, dict[str, str]]:
    """
    Perform an HTTP GET with retries and exponential backoff.

    Returns parsed JSON body and response headers dict.
    Never executes downloaded content — treats everything as data only.
    """
    last_error: Optional[Exception] = None
    backoff = _INITIAL_BACKOFF_SECONDS

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            req = Request(url, headers=headers, method="GET")
            with urlopen(req, timeout=_HTTP_TIMEOUT_SECONDS) as resp:
                resp_headers = {k.lower(): v for k, v in resp.getheaders()}
                body = json.loads(resp.read().decode("utf-8"))
                return body, resp_headers
        except HTTPError as exc:
            last_error = exc
            status = exc.code

            if status == 403:
                # Rate limit — check reset header
                reset_at = exc.headers.get("X-RateLimit-Reset", "")
                if reset_at:
                    wait = max(0, int(reset_at) - int(time.time())) + 1
                    logger.warning(
                        "GitHub rate limit hit, waiting %d seconds "
                        "(attempt %d/%d)",
                        wait, attempt, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue

            if status in (429, 500, 502, 503):
                logger.warning(
                    "HTTP %d from %s, retrying in %.1f seconds "
                    "(attempt %d/%d)",
                    status, url, backoff, attempt, _MAX_RETRIES,
                )
                time.sleep(backoff)
                backoff *= 2
                continue

            raise

        except (URLError, OSError) as exc:
            last_error = exc
            logger.warning(
                "Network error: %s, retrying in %.1f seconds (attempt %d/%d)",
                exc, backoff, attempt, _MAX_RETRIES,
            )
            time.sleep(backoff)
            backoff *= 2

    raise RuntimeError(
        f"HTTP GET failed after {_MAX_RETRIES} attempts: {url}"
    ) from last_error


def _stream_download(
    url: str,
    target_path: Path,
    headers: dict[str, str],
    logger: logging.Logger,
) -> str:
    """
    Stream-download a file to disk and return its SHA256 hex digest.

    Uses chunked reads to avoid loading large files fully into memory.
    Writes to a temp file first, then renames atomically.
    """
    hasher = hashlib.sha256()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_fd = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=str(target_path.parent),
        prefix=".m31r_dl_",
        suffix=".tmp",
        delete=False,
    )
    tmp_path = Path(tmp_fd.name)

    try:
        req = Request(url, headers=headers, method="GET")
        with urlopen(req, timeout=_HTTP_TIMEOUT_SECONDS) as resp:
            while True:
                chunk = resp.read(_STREAM_CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
                tmp_fd.write(chunk)

        tmp_fd.flush()
        tmp_fd.close()
        tmp_path.rename(target_path)
        logger.info("Downloaded %s", target_path.name)

    except BaseException:
        tmp_fd.close()
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return hasher.hexdigest()


# ============================================================
# Path Security
# ============================================================

def _is_safe_tar_member(member: tarfile.TarInfo, extract_dir: Path) -> bool:
    """
    Validate a tar member against path traversal attacks.

    Rejects absolute paths, paths with '..' components, symlinks
    pointing outside the extraction directory, and excessively large files.
    """
    # Reject absolute paths
    if member.name.startswith("/") or member.name.startswith("\\"):
        return False

    # Reject path traversal
    if ".." in member.name.split("/"):
        return False

    # Resolve and check the target stays inside extract_dir
    resolved = (extract_dir / member.name).resolve()
    try:
        resolved.relative_to(extract_dir.resolve())
    except ValueError:
        return False

    # Reject symlinks/hardlinks (could point outside)
    if member.issym() or member.islnk():
        return False

    # Reject excessively large files (> 100 MB — generated/binary likely)
    if member.isfile() and member.size > 100 * 1024 * 1024:
        return False

    return True


def _safe_extract_tarball(
    tarball_path: Path,
    extract_dir: Path,
    logger: logging.Logger,
) -> int:
    """
    Safely extract a tarball into extract_dir.

    Validates every member against path traversal before extraction.
    Returns the count of extracted files.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    file_count = 0

    with tarfile.open(tarball_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not _is_safe_tar_member(member, extract_dir):
                logger.warning(
                    "Skipping unsafe tar member: %s", member.name
                )
                continue

            tar.extract(member, path=extract_dir, set_attrs=False)
            if member.isfile():
                file_count += 1

    return file_count


# ============================================================
# Resume / Marker Utilities
# ============================================================

def _crawl_marker_path(repo_dir: Path) -> Path:
    """Return the path to the crawl marker file for a directory."""
    return repo_dir / _CRAWL_MARKER_FILENAME


def _is_already_harvested(target_dir: Path, pin: str) -> bool:
    """
    Check if a repo/crate has already been downloaded at the correct pin.

    The marker file contains the commit hash or version string. If it matches
    the expected pin, we skip re-downloading — this is idempotent resume.
    """
    marker = _crawl_marker_path(target_dir)
    if not marker.exists():
        return False
    try:
        return marker.read_text(encoding="utf-8").strip() == pin
    except OSError:
        return False


def _write_crawl_marker(target_dir: Path, pin: str) -> None:
    """Write a marker file indicating successful download at a given pin."""
    marker = _crawl_marker_path(target_dir)
    marker.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = marker.parent / f".m31r_marker_{os.getpid()}.tmp"
    try:
        tmp_path.write_text(pin, encoding="utf-8")
        tmp_path.rename(marker)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


# ============================================================
# File counting
# ============================================================

def _count_files(directory: Path) -> int:
    """Count all regular files under a directory, recursively."""
    count = 0
    for item in directory.rglob("*"):
        if item.is_file() and item.name != _CRAWL_MARKER_FILENAME:
            count += 1
    return count


# ============================================================
# Checksum Utilities
# ============================================================

def _sha256_directory(directory: Path) -> str:
    """
    Compute a deterministic SHA256 over all file paths in a directory.

    We hash the sorted list of relative file paths, not file contents,
    because hashing terabytes of content at acquisition time is wasteful.
    The per-file content hashes are computed downstream by ``m31r filter``.
    """
    hasher = hashlib.sha256()
    paths: list[str] = sorted(
        str(p.relative_to(directory))
        for p in directory.rglob("*")
        if p.is_file() and p.name != _CRAWL_MARKER_FILENAME
    )
    for path_str in paths:
        hasher.update(path_str.encode("utf-8"))
    return hasher.hexdigest()


# ============================================================
# GitHub Harvester
# ============================================================

def _normalize_license(raw: Optional[str]) -> Optional[str]:
    """Normalize a license identifier to its canonical SPDX form."""
    if raw is None:
        return None
    lower = raw.strip().lower()
    canonical = _LICENSE_ALIASES.get(lower)
    if canonical is not None:
        return canonical
    # Try exact match against allowed set (case-insensitive)
    for allowed in _ALLOWED_LICENSES:
        if allowed.lower() == lower:
            return allowed
    return raw


def _is_permissive_license(license_spdx: Optional[str]) -> bool:
    """Check if a license is in the allowed permissive set."""
    if license_spdx is None:
        return False
    normalized = _normalize_license(license_spdx)
    return normalized is not None and normalized in _ALLOWED_LICENSES


def query_github_repos(
    config: HarvesterConfig,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """
    Query GitHub Search API for Rust repositories.

    Returns a deterministically-ordered list of repository metadata dicts.
    Handles pagination and rate limits. Excludes forks and archived repos.
    """
    if config.github_repo_count <= 0:
        return []

    headers = _build_github_headers(config.github_token)
    collected: list[dict[str, Any]] = []
    page = 1
    per_page = min(100, config.github_repo_count)

    while len(collected) < config.github_repo_count:
        query = (
            f"language:Rust stars:>={config.min_stars} "
            f"fork:false archived:false"
        )
        url = (
            f"{_GITHUB_API_BASE}/search/repositories"
            f"?q={query.replace(' ', '+')}"
            f"&sort=stars&order=desc"
            f"&per_page={per_page}&page={page}"
        )

        logger.info(
            "Querying GitHub Search API page %d (collected %d/%d)",
            page, len(collected), config.github_repo_count,
        )

        body, resp_headers = _http_get_json(url, headers, logger)
        items = body.get("items", [])

        if not items:
            logger.info("No more results from GitHub API")
            break

        for repo in items:
            if len(collected) >= config.github_repo_count:
                break

            # Double-check fork/archived (API should already filter)
            if repo.get("fork", False):
                continue
            if repo.get("archived", False):
                continue

            # License filtering
            license_info = repo.get("license") or {}
            license_key = license_info.get("spdx_id")
            if not _is_permissive_license(license_key):
                logger.info(
                    "Skipping %s — license: %s",
                    repo.get("full_name", "unknown"),
                    license_key,
                )
                continue

            collected.append(repo)

        # Check remaining rate limit
        remaining = resp_headers.get("x-ratelimit-remaining", "")
        if remaining and int(remaining) < 5:
            reset_at = resp_headers.get("x-ratelimit-reset", "0")
            wait = max(0, int(reset_at) - int(time.time())) + 1
            logger.warning(
                "GitHub rate limit low (%s remaining), waiting %d seconds",
                remaining, wait,
            )
            time.sleep(wait)

        page += 1

    # Sort deterministically by full_name for reproducibility
    collected.sort(key=lambda r: r.get("full_name", ""))

    logger.info(
        "GitHub query complete: %d repositories collected", len(collected)
    )
    return collected


def _get_repo_head_commit(
    full_name: str,
    default_branch: str,
    headers: dict[str, str],
    logger: logging.Logger,
) -> str:
    """Fetch the HEAD commit SHA for a repository's default branch."""
    url = (
        f"{_GITHUB_API_BASE}/repos/{full_name}"
        f"/commits/{default_branch}"
    )
    body, _ = _http_get_json(url, headers, logger)
    sha = body.get("sha", "")
    if not sha:
        raise RuntimeError(
            f"Could not resolve HEAD commit for {full_name}/{default_branch}"
        )
    return sha


def clone_github_repo(
    repo_meta: dict[str, Any],
    github_dir: Path,
    config: HarvesterConfig,
    stats: HarvestStats,
    logger: logging.Logger,
) -> Optional[RepoRecord]:
    """
    Shallow-clone a single GitHub repository at a pinned commit.

    Returns a RepoRecord on success, None on failure.
    Skips if already downloaded (idempotent resume).
    """
    full_name: str = repo_meta["full_name"]
    clone_url: str = repo_meta["clone_url"]
    default_branch: str = repo_meta.get("default_branch", "main")
    stars: int = repo_meta.get("stargazers_count", 0)

    license_info = repo_meta.get("license") or {}
    raw_license = license_info.get("spdx_id", "unknown")
    license_spdx = _normalize_license(raw_license) or raw_license

    # Sanitize directory name: owner/repo → owner__repo
    safe_name = full_name.replace("/", "__")
    repo_dir = github_dir / safe_name

    # Resolve HEAD commit for pinning
    headers = _build_github_headers(config.github_token)
    try:
        commit = _get_repo_head_commit(
            full_name, default_branch, headers, logger
        )
    except Exception as exc:
        logger.error("Failed to resolve commit for %s: %s", full_name, exc)
        stats.errors.append(f"commit_resolve:{full_name}:{exc}")
        return None

    # Resume check
    if _is_already_harvested(repo_dir, commit):
        logger.info("Already harvested %s at %s, skipping", full_name, commit[:12])
        stats.github_skipped += 1
        return RepoRecord(
            name=repo_meta.get("name", ""),
            full_name=full_name,
            url=repo_meta.get("html_url", ""),
            clone_url=clone_url,
            default_branch=default_branch,
            commit=commit,
            license_spdx=license_spdx,
            stars=stars,
            local_path=str(repo_dir),
            file_count=_count_files(repo_dir),
            checksum_sha256=_sha256_directory(repo_dir),
        )

    # Clean up any partial previous clone
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    repo_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Cloning %s at %s (stars: %d, license: %s)",
        full_name, commit[:12], stars, license_spdx,
    )

    try:
        subprocess.run(
            [
                "git", "clone",
                "--depth", "1",
                "--single-branch",
                "--branch", default_branch,
                "--no-tags",
                clone_url,
                str(repo_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=_GIT_CLONE_TIMEOUT_SECONDS,
        )
    except subprocess.CalledProcessError as exc:
        logger.error(
            "Git clone failed for %s: %s",
            full_name, exc.stderr.strip() if exc.stderr else str(exc),
        )
        stats.errors.append(f"clone:{full_name}:{exc}")
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        return None
    except subprocess.TimeoutExpired:
        logger.error("Git clone timed out for %s", full_name)
        stats.errors.append(f"clone_timeout:{full_name}")
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        return None

    _write_crawl_marker(repo_dir, commit)
    file_count = _count_files(repo_dir)
    checksum = _sha256_directory(repo_dir)

    stats.github_cloned += 1

    logger.info(
        "Cloned %s: %d files, checksum %s",
        full_name, file_count, checksum[:16],
    )

    return RepoRecord(
        name=repo_meta.get("name", ""),
        full_name=full_name,
        url=repo_meta.get("html_url", ""),
        clone_url=clone_url,
        default_branch=default_branch,
        commit=commit,
        license_spdx=license_spdx,
        stars=stars,
        local_path=str(repo_dir),
        file_count=file_count,
        checksum_sha256=checksum,
    )


def harvest_github(
    config: HarvesterConfig,
    stats: HarvestStats,
    logger: logging.Logger,
) -> list[RepoRecord]:
    """
    Full GitHub harvesting pipeline: query → filter → clone.

    Returns a deterministically-ordered list of RepoRecords.
    """
    if config.github_repo_count <= 0:
        logger.info("GitHub harvesting disabled (count=0)")
        return []

    github_dir = config.output_directory / "github"
    github_dir.mkdir(parents=True, exist_ok=True)

    repos_meta = query_github_repos(config, logger)
    stats.github_queried = len(repos_meta)

    records: list[RepoRecord] = []
    for repo_meta in repos_meta:
        record = clone_github_repo(
            repo_meta, github_dir, config, stats, logger
        )
        if record is not None:
            records.append(record)

    # Sort deterministically
    records.sort(key=lambda r: r.full_name)
    return records


# ============================================================
# Crates.io Harvester
# ============================================================

def query_top_crates(
    config: HarvesterConfig,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """
    Query crates.io for top-downloaded Rust crates.

    Returns a deterministically-ordered list of crate metadata dicts.
    Filters to permissive licenses only.
    """
    if config.crate_count <= 0:
        return []

    headers = _build_crates_headers()
    collected: list[dict[str, Any]] = []
    page = 1
    per_page = min(100, config.crate_count)

    while len(collected) < config.crate_count:
        url = (
            f"{_CRATES_IO_API_BASE}/crates"
            f"?page={page}&per_page={per_page}&sort=downloads"
        )

        logger.info(
            "Querying crates.io page %d (collected %d/%d)",
            page, len(collected), config.crate_count,
        )

        body, _ = _http_get_json(url, headers, logger)
        crates = body.get("crates", [])

        if not crates:
            logger.info("No more results from crates.io")
            break

        for crate in crates:
            if len(collected) >= config.crate_count:
                break

            # License filtering
            raw_license = crate.get("license", "") or ""

            # crates.io may return compound licenses like "MIT OR Apache-2.0"
            # Accept if ANY component is permissive
            license_parts = [
                part.strip()
                for part in raw_license.replace("/", " OR ").split(" OR ")
            ]
            best_license: Optional[str] = None
            for part in license_parts:
                normalized = _normalize_license(part)
                if normalized and normalized in _ALLOWED_LICENSES:
                    best_license = normalized
                    break

            if best_license is None:
                logger.info(
                    "Skipping crate %s — license: %s",
                    crate.get("name", "unknown"),
                    raw_license,
                )
                continue

            crate["_resolved_license"] = best_license
            collected.append(crate)

        # crates.io rate limit: be polite (1 req/sec)
        time.sleep(1.0)
        page += 1

    # Sort deterministically by name for reproducibility
    collected.sort(key=lambda c: c.get("name", ""))

    logger.info(
        "Crates.io query complete: %d crates collected", len(collected)
    )
    return collected


def _get_crate_latest_version(
    crate_name: str,
    headers: dict[str, str],
    logger: logging.Logger,
) -> tuple[str, str]:
    """
    Fetch the latest (newest) non-yanked version of a crate.

    Returns (version_string, download_url).
    """
    url = f"{_CRATES_IO_API_BASE}/crates/{crate_name}/versions"
    body, _ = _http_get_json(url, headers, logger)

    versions = body.get("versions", [])
    for ver in versions:
        if ver.get("yanked", False):
            continue
        version_num = ver.get("num", "")
        dl_path = ver.get("dl_path", "")
        if version_num and dl_path:
            download_url = f"https://crates.io{dl_path}"
            return version_num, download_url

    raise RuntimeError(f"No valid version found for crate: {crate_name}")


def download_crate(
    crate_meta: dict[str, Any],
    crates_dir: Path,
    stats: HarvestStats,
    logger: logging.Logger,
) -> Optional[CrateRecord]:
    """
    Download and safely extract a single crate tarball.

    Returns a CrateRecord on success, None on failure.
    Skips if already downloaded (idempotent resume).
    """
    crate_name: str = crate_meta["name"]
    downloads: int = crate_meta.get("downloads", 0)
    license_spdx: str = crate_meta.get("_resolved_license", "unknown")
    headers = _build_crates_headers()

    # Fetch latest version info
    try:
        version, download_url = _get_crate_latest_version(
            crate_name, headers, logger
        )
    except Exception as exc:
        logger.error(
            "Failed to resolve version for crate %s: %s", crate_name, exc
        )
        stats.errors.append(f"crate_version:{crate_name}:{exc}")
        return None

    # Sanitize directory name
    safe_name = f"{crate_name}-{version}"
    crate_dir = crates_dir / safe_name

    # Resume check
    if _is_already_harvested(crate_dir, version):
        logger.info(
            "Already harvested crate %s v%s, skipping", crate_name, version
        )
        stats.crates_skipped += 1
        return CrateRecord(
            name=crate_name,
            version=version,
            download_url=download_url,
            license_spdx=license_spdx,
            downloads=downloads,
            local_path=str(crate_dir),
            file_count=_count_files(crate_dir),
            checksum_sha256=_sha256_directory(crate_dir),
        )

    # Clean up any partial previous download
    if crate_dir.exists():
        shutil.rmtree(crate_dir)

    crate_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Downloading crate %s v%s (downloads: %d, license: %s)",
        crate_name, version, downloads, license_spdx,
    )

    # Download tarball to temp file, then extract
    tarball_path = crate_dir / f"{safe_name}.tar.gz"
    try:
        checksum_tarball = _stream_download(
            download_url, tarball_path, headers, logger
        )
    except Exception as exc:
        logger.error(
            "Download failed for crate %s: %s", crate_name, exc
        )
        stats.errors.append(f"crate_download:{crate_name}:{exc}")
        if crate_dir.exists():
            shutil.rmtree(crate_dir)
        return None

    # Safe extraction
    try:
        file_count = _safe_extract_tarball(tarball_path, crate_dir, logger)
    except Exception as exc:
        logger.error(
            "Extraction failed for crate %s: %s", crate_name, exc
        )
        stats.errors.append(f"crate_extract:{crate_name}:{exc}")
        if crate_dir.exists():
            shutil.rmtree(crate_dir)
        return None

    # Remove the tarball after extraction to save disk space
    if tarball_path.exists():
        tarball_path.unlink()

    _write_crawl_marker(crate_dir, version)
    checksum = _sha256_directory(crate_dir)

    stats.crates_downloaded += 1

    # Be polite to crates.io
    time.sleep(1.0)

    logger.info(
        "Extracted crate %s v%s: %d files, checksum %s",
        crate_name, version, file_count, checksum[:16],
    )

    return CrateRecord(
        name=crate_name,
        version=version,
        download_url=download_url,
        license_spdx=license_spdx,
        downloads=downloads,
        local_path=str(crate_dir),
        file_count=file_count,
        checksum_sha256=checksum,
    )


def harvest_crates(
    config: HarvesterConfig,
    stats: HarvestStats,
    logger: logging.Logger,
) -> list[CrateRecord]:
    """
    Full crates.io harvesting pipeline: query → filter → download → extract.

    Returns a deterministically-ordered list of CrateRecords.
    """
    if config.crate_count <= 0:
        logger.info("Crates.io harvesting disabled (count=0)")
        return []

    crates_dir = config.output_directory / "crates"
    crates_dir.mkdir(parents=True, exist_ok=True)

    crates_meta = query_top_crates(config, logger)
    stats.crates_queried = len(crates_meta)

    records: list[CrateRecord] = []
    for crate_meta in crates_meta:
        record = download_crate(crate_meta, crates_dir, stats, logger)
        if record is not None:
            records.append(record)

    # Sort deterministically
    records.sort(key=lambda r: r.name)
    return records


# ============================================================
# Manifest Writer
# ============================================================

def _atomic_write_json(path: Path, data: Any) -> None:
    """Atomically write a JSON file with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)

    tmp_path = path.parent / f".m31r_manifest_{os.getpid()}.tmp"
    try:
        tmp_path.write_text(content + "\n", encoding="utf-8")
        tmp_path.rename(path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def write_github_manifest(
    records: list[RepoRecord],
    manifests_dir: Path,
    logger: logging.Logger,
) -> Path:
    """Write the GitHub harvest manifest to data/manifests/github.json."""
    manifest = {
        "version": _SCRIPT_VERSION,
        "source": "github",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_repos": len(records),
        "entries": [asdict(r) for r in records],
    }

    manifest_path = manifests_dir / "github.json"
    _atomic_write_json(manifest_path, manifest)
    logger.info("GitHub manifest written: %s (%d entries)", manifest_path, len(records))
    return manifest_path


def write_crates_manifest(
    records: list[CrateRecord],
    manifests_dir: Path,
    logger: logging.Logger,
) -> Path:
    """Write the crates.io harvest manifest to data/manifests/crates.json."""
    manifest = {
        "version": _SCRIPT_VERSION,
        "source": "crates.io",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_crates": len(records),
        "entries": [asdict(r) for r in records],
    }

    manifest_path = manifests_dir / "crates.json"
    _atomic_write_json(manifest_path, manifest)
    logger.info("Crates manifest written: %s (%d entries)", manifest_path, len(records))
    return manifest_path


# ============================================================
# CLI Entrypoint
# ============================================================

def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="snigdhaos-rust-harvester",
        description=(
            "M31R Rust Dataset Harvester — production-grade acquisition tool "
            "for downloading Rust source code from GitHub and crates.io."
        ),
        epilog=(
            "Environment variables:\n"
            "  GITHUB_TOKEN   Optional GitHub personal access token for "
            "higher rate limits.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--github-repos",
        type=int,
        default=100,
        metavar="N",
        help="Number of GitHub repositories to harvest (default: 100)",
    )
    parser.add_argument(
        "--crate-count",
        type=int,
        default=100,
        metavar="N",
        help="Number of crates.io packages to harvest (default: 100)",
    )
    parser.add_argument(
        "--min-stars",
        type=int,
        default=500,
        metavar="N",
        help=(
            "Minimum GitHub star count for repository inclusion "
            "(default: 500)"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        metavar="PATH",
        help=(
            "Output directory for downloaded data, relative to project root "
            "(default: data/raw)"
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level (default: INFO)",
    )

    return parser


def _resolve_project_root() -> Path:
    """Resolve the M31R project root from the script location."""
    script_dir = Path(__file__).resolve().parent
    # Script lives in scripts/ → project root is one level up
    project_root = script_dir.parent
    # Sanity check: project root should contain pyproject.toml
    if not (project_root / "pyproject.toml").exists():
        raise RuntimeError(
            f"Cannot find pyproject.toml at expected project root: "
            f"{project_root}"
        )
    return project_root


def main() -> int:
    """
    Main entry point for the Rust dataset harvester.

    Returns 0 on success, 1 on partial failure, 3 on fatal error.
    Exit codes follow M31R CLI spec (12_CLI_AND_TOOLING_SPEC.md §6).
    """
    parser = _build_argument_parser()
    args = parser.parse_args()

    logger = _configure_logging(args.log_level)

    logger.info(
        "M31R Rust Dataset Harvester v%s starting", _SCRIPT_VERSION
    )

    try:
        project_root = _resolve_project_root()
    except RuntimeError as exc:
        logger.critical("Project root resolution failed: %s", exc)
        return 3

    # Build immutable config
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        logger.info("GITHUB_TOKEN detected, using authenticated requests")
    else:
        logger.warning(
            "No GITHUB_TOKEN set — GitHub API rate limits will be strict "
            "(60 req/hour unauthenticated)"
        )

    output_dir = project_root / args.output

    try:
        config = HarvesterConfig(
            github_repo_count=args.github_repos,
            crate_count=args.crate_count,
            min_stars=args.min_stars,
            output_directory=output_dir,
            github_token=github_token,
            log_level=args.log_level,
        )
    except ValueError as exc:
        logger.critical("Invalid configuration: %s", exc)
        return 2

    logger.info(
        "Config: github_repos=%d, crate_count=%d, min_stars=%d, output=%s",
        config.github_repo_count, config.crate_count,
        config.min_stars, config.output_directory,
    )

    stats = HarvestStats()

    # Phase 1: GitHub repositories
    github_records: list[RepoRecord] = []
    try:
        github_records = harvest_github(config, stats, logger)
    except Exception as exc:
        logger.error("GitHub harvesting failed: %s", exc)
        stats.errors.append(f"github_fatal:{exc}")

    # Phase 2: crates.io packages
    crate_records: list[CrateRecord] = []
    try:
        crate_records = harvest_crates(config, stats, logger)
    except Exception as exc:
        logger.error("Crates.io harvesting failed: %s", exc)
        stats.errors.append(f"crates_fatal:{exc}")

    # Phase 3: Write manifests
    manifests_dir = output_dir.parent / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    try:
        if github_records or config.github_repo_count > 0:
            write_github_manifest(github_records, manifests_dir, logger)
    except Exception as exc:
        logger.error("Failed to write GitHub manifest: %s", exc)
        stats.errors.append(f"manifest_github:{exc}")

    try:
        if crate_records or config.crate_count > 0:
            write_crates_manifest(crate_records, manifests_dir, logger)
    except Exception as exc:
        logger.error("Failed to write crates manifest: %s", exc)
        stats.errors.append(f"manifest_crates:{exc}")

    # Summary
    logger.info(
        "Harvest complete — "
        "GitHub: queried=%d cloned=%d skipped=%d | "
        "Crates: queried=%d downloaded=%d skipped=%d | "
        "Errors: %d",
        stats.github_queried, stats.github_cloned, stats.github_skipped,
        stats.crates_queried, stats.crates_downloaded, stats.crates_skipped,
        len(stats.errors),
    )

    if stats.errors:
        for err in stats.errors:
            logger.warning("Error detail: %s", err)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
