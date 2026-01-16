# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Pre-flight environment validation.

Before training or serving, must check:
- Python version
- torch version
- CUDA availability
- disk space
- memory

Per 18_RELEASE_PROCESS.md and 19_SECURITY_AND_SAFETY.md:
Fail early with clear errors. No cryptic failures 30 minutes into a run.
"""

import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from m31r.logging.logger import get_logger

_logger: logging.Logger = get_logger(__name__)

# Minimum requirements
MIN_PYTHON_MAJOR: int = 3
MIN_PYTHON_MINOR: int = 11
MIN_DISK_SPACE_BYTES: int = 1_073_741_824  # 1 GB
MIN_MEMORY_BYTES: int = 4_294_967_296  # 4 GB


@dataclass(frozen=True)
class EnvironmentCheck:
    """Result of a single environment check."""

    name: str
    passed: bool
    message: str
    value: str


def check_python_version() -> EnvironmentCheck:
    """Verify Python >= 3.11."""
    major, minor, micro = sys.version_info[:3]
    version_str = f"{major}.{minor}.{micro}"
    passed = major > MIN_PYTHON_MAJOR or (major == MIN_PYTHON_MAJOR and minor >= MIN_PYTHON_MINOR)
    if passed:
        msg = f"Python {version_str} meets minimum {MIN_PYTHON_MAJOR}.{MIN_PYTHON_MINOR}"
    else:
        msg = (
            f"Python {version_str} does NOT meet minimum " f"{MIN_PYTHON_MAJOR}.{MIN_PYTHON_MINOR}"
        )
    return EnvironmentCheck(name="python_version", passed=passed, message=msg, value=version_str)


def check_torch_version() -> EnvironmentCheck:
    """Check PyTorch availability and version."""
    try:
        import torch

        version = torch.__version__
        return EnvironmentCheck(
            name="torch_version",
            passed=True,
            message=f"PyTorch {version} available",
            value=version,
        )
    except ImportError:
        return EnvironmentCheck(
            name="torch_version",
            passed=False,
            message="PyTorch not installed",
            value="not_installed",
        )


def check_cuda_availability() -> EnvironmentCheck:
    """Check CUDA GPU availability."""
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda or "unknown"
            return EnvironmentCheck(
                name="cuda",
                passed=True,
                message=f"CUDA available: {device_name} (CUDA {cuda_version})",
                value=f"{device_name} / CUDA {cuda_version}",
            )
        else:
            return EnvironmentCheck(
                name="cuda",
                passed=True,  # CUDA is optional — CPU-only is valid
                message="CUDA not available — CPU-only mode",
                value="not_available",
            )
    except ImportError:
        return EnvironmentCheck(
            name="cuda",
            passed=True,
            message="PyTorch not installed — CUDA check skipped",
            value="skipped",
        )


def check_disk_space(path: Path | None = None) -> EnvironmentCheck:
    """
    Check available disk space at the given path.

    Args:
        path: Directory to check. Defaults to current working directory.
    """
    check_path = path or Path.cwd()
    try:
        usage = shutil.disk_usage(str(check_path))
        free_gb = usage.free / (1024**3)
        passed = usage.free >= MIN_DISK_SPACE_BYTES
        if passed:
            msg = f"{free_gb:.1f} GB free (minimum {MIN_DISK_SPACE_BYTES / (1024**3):.0f} GB)"
        else:
            msg = (
                f"Only {free_gb:.1f} GB free — need at least "
                f"{MIN_DISK_SPACE_BYTES / (1024**3):.0f} GB"
            )
        return EnvironmentCheck(
            name="disk_space",
            passed=passed,
            message=msg,
            value=f"{free_gb:.1f}GB",
        )
    except OSError as err:
        return EnvironmentCheck(
            name="disk_space",
            passed=False,
            message=f"Cannot check disk space: {err}",
            value="error",
        )


def check_memory() -> EnvironmentCheck:
    """Check available system memory."""
    try:
        import os

        # Use os.sysconf on Linux (the target platform)
        if hasattr(os, "sysconf"):
            page_size = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_PHYS_PAGES")
            total_bytes = page_size * pages
        else:
            # Fallback: try psutil or just report unknown
            total_bytes = 0

        if total_bytes > 0:
            total_gb = total_bytes / (1024**3)
            passed = total_bytes >= MIN_MEMORY_BYTES
            if passed:
                msg = (
                    f"{total_gb:.1f} GB total RAM "
                    f"(minimum {MIN_MEMORY_BYTES / (1024**3):.0f} GB)"
                )
            else:
                msg = (
                    f"Only {total_gb:.1f} GB RAM — need at least "
                    f"{MIN_MEMORY_BYTES / (1024**3):.0f} GB"
                )
            return EnvironmentCheck(
                name="memory",
                passed=passed,
                message=msg,
                value=f"{total_gb:.1f}GB",
            )
        else:
            return EnvironmentCheck(
                name="memory",
                passed=True,
                message="Cannot determine total memory — skipping check",
                value="unknown",
            )
    except Exception as err:
        return EnvironmentCheck(
            name="memory",
            passed=True,
            message=f"Memory check unavailable: {err}",
            value="error",
        )


def validate_environment(
    check_path: Path | None = None,
) -> list[EnvironmentCheck]:
    """
    Run all pre-flight environment checks.

    Returns a list of check results. Callers should inspect the `passed`
    field of each check to decide whether to proceed.

    Args:
        check_path: Optional path for disk space check.

    Returns:
        List of EnvironmentCheck results, one per check.
    """
    checks = [
        check_python_version(),
        check_torch_version(),
        check_cuda_availability(),
        check_disk_space(check_path),
        check_memory(),
    ]

    passed_count = sum(1 for c in checks if c.passed)
    failed_count = len(checks) - passed_count

    for check in checks:
        log_fn = _logger.info if check.passed else _logger.error
        log_fn(
            "Environment check",
            extra={
                "check": check.name,
                "passed": check.passed,
                "check_message": check.message,
            },
        )

    _logger.info(
        "Environment validation complete",
        extra={"passed": passed_count, "failed": failed_count},
    )

    return checks
