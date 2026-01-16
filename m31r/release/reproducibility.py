# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Reproducible build tooling.

Per 18_RELEASE_PROCESS.md §10: Must verify same inputs → identical hashes.
Per 19_SECURITY_AND_SAFETY.md §13: If build is not reproducible, it is insecure.

This module captures and compares environment snapshots to ensure
deterministic, reproducible builds. It freezes:
- Python version
- Platform info
- pip package versions
- torch version + CUDA version
- System architecture

Two builds with the same snapshot should produce identical outputs.
"""

import json
import logging
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from m31r.logging.logger import get_logger

_logger: logging.Logger = get_logger(__name__)


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """Complete snapshot of the build environment for reproducibility."""

    python_version: str
    platform_system: str
    platform_machine: str
    platform_release: str
    packages: dict[str, str] = field(default_factory=dict)
    torch_version: str = "not_installed"
    cuda_version: str = "not_available"
    timestamp: str = ""


@dataclass(frozen=True)
class Difference:
    """A single difference between two environment snapshots."""

    field: str
    value_a: str
    value_b: str


def _get_installed_packages() -> dict[str, str]:
    """Get all installed pip packages and versions."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--local"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            return {}

        packages: dict[str, str] = {}
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            if "==" in line:
                name, version = line.split("==", maxsplit=1)
                packages[name.strip().lower()] = version.strip()
            elif " @ " in line:
                # Handle editable installs like "m31r @ file:///..."
                name = line.split(" @ ")[0].strip().lower()
                packages[name] = "editable"
        return dict(sorted(packages.items()))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        _logger.warning("Could not run pip freeze")
        return {}


def _get_torch_info() -> tuple[str, str]:
    """Get PyTorch and CUDA versions if available."""
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda or "not_available"
        return torch_version, cuda_version
    except ImportError:
        return "not_installed", "not_available"


def freeze_environment() -> EnvironmentSnapshot:
    """
    Capture a complete snapshot of the current build environment.

    This is used to verify that two builds happened in the same conditions.
    The snapshot is deterministic — calling it twice in the same environment
    produces the same result (except for timestamp).

    Returns:
        Frozen EnvironmentSnapshot with all reproducibility-relevant info.
    """
    torch_ver, cuda_ver = _get_torch_info()
    packages = _get_installed_packages()

    snapshot = EnvironmentSnapshot(
        python_version=platform.python_version(),
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        platform_release=platform.release(),
        packages=packages,
        torch_version=torch_ver,
        cuda_version=cuda_ver,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
    )

    _logger.info(
        "Environment frozen",
        extra={
            "python": snapshot.python_version,
            "torch": snapshot.torch_version,
            "cuda": snapshot.cuda_version,
            "package_count": len(snapshot.packages),
        },
    )
    return snapshot


def compare_environments(
    snap_a: EnvironmentSnapshot,
    snap_b: EnvironmentSnapshot,
) -> list[Difference]:
    """
    Compare two environment snapshots and find differences.

    Ignores timestamp differences (those are expected).
    Focuses on Python version, platform, packages, and torch/CUDA.

    Args:
        snap_a: First snapshot.
        snap_b: Second snapshot.

    Returns:
        List of Difference objects. Empty list means environments match.
    """
    diffs: list[Difference] = []

    # Check scalar fields
    for field_name in [
        "python_version",
        "platform_system",
        "platform_machine",
        "torch_version",
        "cuda_version",
    ]:
        val_a = getattr(snap_a, field_name)
        val_b = getattr(snap_b, field_name)
        if val_a != val_b:
            diffs.append(Difference(field=field_name, value_a=val_a, value_b=val_b))

    # Check packages
    all_packages = set(snap_a.packages.keys()) | set(snap_b.packages.keys())
    for pkg in sorted(all_packages):
        ver_a = snap_a.packages.get(pkg, "not_installed")
        ver_b = snap_b.packages.get(pkg, "not_installed")
        if ver_a != ver_b:
            diffs.append(
                Difference(
                    field=f"package:{pkg}",
                    value_a=ver_a,
                    value_b=ver_b,
                )
            )

    if diffs:
        _logger.warning(
            "Environment differences detected",
            extra={"difference_count": len(diffs)},
        )
    else:
        _logger.info("Environments match")

    return diffs


def write_lockfile(snapshot: EnvironmentSnapshot, path: Path) -> None:
    """
    Write an environment snapshot as a JSON lockfile.

    This file captures everything needed to reproduce the environment.

    Args:
        snapshot: The environment snapshot to serialize.
        path: Target file path (e.g., requirements.lock).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(snapshot)
    content = json.dumps(data, indent=2, sort_keys=True) + "\n"
    path.write_text(content, encoding="utf-8")

    _logger.info(
        "Lockfile written",
        extra={"path": str(path), "package_count": len(snapshot.packages)},
    )


def load_lockfile(path: Path) -> EnvironmentSnapshot:
    """
    Load an environment snapshot from a JSON lockfile.

    Args:
        path: Path to the lockfile.

    Returns:
        Parsed EnvironmentSnapshot.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Lockfile not found: {path}")

    content = path.read_text(encoding="utf-8")
    data = json.loads(content)

    snapshot = EnvironmentSnapshot(
        python_version=str(data.get("python_version", "")),
        platform_system=str(data.get("platform_system", "")),
        platform_machine=str(data.get("platform_machine", "")),
        platform_release=str(data.get("platform_release", "")),
        packages=data.get("packages", {}),
        torch_version=str(data.get("torch_version", "not_installed")),
        cuda_version=str(data.get("cuda_version", "not_available")),
        timestamp=str(data.get("timestamp", "")),
    )

    _logger.debug("Lockfile loaded", extra={"path": str(path)})
    return snapshot
