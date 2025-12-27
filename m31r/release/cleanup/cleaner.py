# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Project cleanup system.

Per 12_CLI_AND_TOOLING_SPEC.md §25:
- `m31r clean` removes temporary artifacts
- Must not delete checkpoints by default

Per 22_MAINTENANCE_AND_SUPPORT.md §21:
- Logs: rotatable, deletable
- Experiments: may be archived
- Releases: never deleted

This module is intentionally conservative — it only removes items
that are definitively safe to delete. When in doubt, preserve.
"""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from m31r.logging.logger import get_logger

_logger: logging.Logger = get_logger(__name__)

# Directories that are ALWAYS safe to remove
_SAFE_TO_REMOVE_DIRS: frozenset[str] = frozenset({
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
})

# Directory name patterns at project root that are safe to remove
_ROOT_SAFE_DIRS: frozenset[str] = frozenset({
    "build",
    "dist",
})

# File patterns that are safe to remove
_SAFE_TO_REMOVE_PATTERNS: frozenset[str] = frozenset({
    "*.egg-info",
})

# Directories that must NEVER be deleted
_PROTECTED_DIRS: frozenset[str] = frozenset({
    "release",
    "releases",
    "checkpoints",
    "experiments",
    "data",
    "docs",
    "configs",
    "m31r",
    "tests",
    "scripts",
    "tools",
    "benchmarks",
    ".git",
})


@dataclass(frozen=True)
class CleanResult:
    """Outcome of a cleanup operation."""

    removed_dirs: int
    removed_files: int
    freed_bytes: int
    protected_dirs: list[str]
    errors: list[str]


def _is_protected(path: Path, project_root: Path) -> bool:
    """Check if a path is protected from deletion."""
    # Never delete the project root itself
    if path == project_root:
        return True

    # Check if the top-level directory containing this path is protected
    try:
        rel_path = path.relative_to(project_root)
        if rel_path.parts and rel_path.parts[0] in _PROTECTED_DIRS:
            return True
    except ValueError:
        # Path is not inside project_root
        return False

    return False


def _count_dir_size(path: Path) -> int:
    """Recursively sum the size of all files in a directory."""
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def clean_project(
    project_root: Path,
    preserve_releases: bool = True,
    remove_logs: bool = False,
) -> CleanResult:
    """
    Remove temporary files and caches from the project.

    Safe by default — only removes build artifacts, caches, and temp files.
    Never removes source code, configs, docs, checkpoints, experiments,
    or releases.

    Args:
        project_root: Root of the M31R project.
        preserve_releases: If True (default), never touch release directories.
        remove_logs: If True, also remove log files from logs/.

    Returns:
        CleanResult with counts of removed items.
    """
    if not project_root.is_dir():
        raise FileNotFoundError(f"Project root not found: {project_root}")

    removed_dirs = 0
    removed_files = 0
    freed_bytes = 0
    protected: list[str] = []
    errors: list[str] = []

    _logger.info(
        "Starting cleanup",
        extra={
            "project_root": str(project_root),
            "preserve_releases": preserve_releases,
            "remove_logs": remove_logs,
        },
    )

    # Remove __pycache__, .pytest_cache, etc. recursively
    for dirname in _SAFE_TO_REMOVE_DIRS:
        for match in sorted(project_root.rglob(dirname)):
            if not match.is_dir():
                continue
            if _is_protected(match, project_root):
                protected.append(str(match))
                continue
            try:
                size = _count_dir_size(match)
                shutil.rmtree(match)
                removed_dirs += 1
                freed_bytes += size
                _logger.debug("Removed directory", extra={"path": str(match)})
            except OSError as err:
                errors.append(f"Failed to remove {match}: {err}")

    # Remove build/dist at project root
    for dirname in _ROOT_SAFE_DIRS:
        target = project_root / dirname
        if target.is_dir():
            try:
                size = _count_dir_size(target)
                shutil.rmtree(target)
                removed_dirs += 1
                freed_bytes += size
                _logger.debug("Removed root directory", extra={"path": str(target)})
            except OSError as err:
                errors.append(f"Failed to remove {target}: {err}")

    # Remove *.egg-info directories
    for match in sorted(project_root.glob("*.egg-info")):
        if match.is_dir():
            try:
                size = _count_dir_size(match)
                shutil.rmtree(match)
                removed_dirs += 1
                freed_bytes += size
            except OSError as err:
                errors.append(f"Failed to remove {match}: {err}")

    # Remove .m31r_tmp_* temp files (from atomic writes)
    for tmp_file in sorted(project_root.rglob(".m31r_tmp_*")):
        if tmp_file.is_file():
            try:
                size = tmp_file.stat().st_size
                tmp_file.unlink()
                removed_files += 1
                freed_bytes += size
            except OSError as err:
                errors.append(f"Failed to remove {tmp_file}: {err}")

    # Optionally remove log files
    if remove_logs:
        logs_dir = project_root / "logs"
        if logs_dir.is_dir():
            for log_file in sorted(logs_dir.glob("*.log")):
                if log_file.is_file():
                    try:
                        size = log_file.stat().st_size
                        log_file.unlink()
                        removed_files += 1
                        freed_bytes += size
                    except OSError as err:
                        errors.append(f"Failed to remove {log_file}: {err}")

    freed_mb = freed_bytes / (1024 * 1024)
    _logger.info(
        "Cleanup complete",
        extra={
            "removed_dirs": removed_dirs,
            "removed_files": removed_files,
            "freed_mb": f"{freed_mb:.1f}",
            "errors": len(errors),
        },
    )

    return CleanResult(
        removed_dirs=removed_dirs,
        removed_files=removed_files,
        freed_bytes=freed_bytes,
        protected_dirs=protected,
        errors=errors,
    )
