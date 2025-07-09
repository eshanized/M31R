# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Environment validation for M31R.

Checks that the machine meets the minimum requirements before we do anything.
If the environment is wrong, we fail immediately rather than hitting cryptic
errors 30 minutes into a training run.
"""

import platform
import sys
from typing import NamedTuple

MINIMUM_PYTHON_MAJOR = 3
MINIMUM_PYTHON_MINOR = 11


class SystemInfo(NamedTuple):
    """Snapshot of the current system environment."""

    python_version: str
    platform: str
    architecture: str
    hostname: str


def get_python_version() -> tuple[int, int, int]:
    """Return the current Python version as a (major, minor, micro) tuple."""
    return sys.version_info[:3]


def check_minimum_python() -> None:
    """
    Verify we're running Python 3.11+.

    The spec requires Python 3.11+ (see 13_CODING_STANDARDS.md section 2).
    We enforce this early so you don't get halfway through and hit a syntax
    or stdlib incompatibility.

    Raises:
        RuntimeError: If Python version is below 3.11.
    """
    major, minor, _ = get_python_version()
    if major < MINIMUM_PYTHON_MAJOR or (
        major == MINIMUM_PYTHON_MAJOR and minor < MINIMUM_PYTHON_MINOR
    ):
        raise RuntimeError(
            f"M31R requires Python >= {MINIMUM_PYTHON_MAJOR}.{MINIMUM_PYTHON_MINOR}, "
            f"but you're running {major}.{minor}. Please upgrade."
        )


def get_system_info() -> SystemInfo:
    """Collect basic system information for logging and diagnostics."""
    return SystemInfo(
        python_version=platform.python_version(),
        platform=platform.system(),
        architecture=platform.machine(),
        hostname=platform.node(),
    )
