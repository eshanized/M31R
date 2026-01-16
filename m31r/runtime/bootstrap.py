# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Runtime bootstrap for M31R.

This module handles the one-time setup that happens before any real work begins.
The bootstrap sequence is:
  1. Validate the environment (Python version, etc.)
  2. Set deterministic seeds
  3. Initialize the logger
  4. Ensure required directories exist

After bootstrap completes, the system is in a known, deterministic state.
Every command goes through this before doing anything else.
"""

import os
import random
from pathlib import Path

from m31r.config.schema import GlobalConfig
from m31r.logging.logger import get_logger
from m31r.runtime.environment import check_minimum_python, get_system_info
from m31r.utils.paths import ensure_directory, resolve_project_root


def set_deterministic_seed(seed: int) -> None:
    """
    Lock down all sources of randomness to the given seed.

    This sets:
      - Python's random module seed
      - PYTHONHASHSEED environment variable (controls hash randomization)
      - PyTorch CPU and CUDA seeds (when available)
      - cuDNN deterministic mode (when available)

    Args:
        seed: Integer seed value. Must be >= 0.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Seed torch if available
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except ImportError:
        pass


def _ensure_project_directories(project_root: Path, config: GlobalConfig) -> None:
    """Create the standard project directories if they don't exist."""
    dirs = config.directories
    ensure_directory(project_root / dirs.data)
    ensure_directory(project_root / dirs.checkpoints)
    ensure_directory(project_root / dirs.logs)
    ensure_directory(project_root / dirs.experiments)


def bootstrap(config: GlobalConfig) -> None:
    """
    Run the full bootstrap sequence for M31R.

    This is called once at the start of every CLI command. It puts the system
    into a known state: environment validated, seeds set, directories ready,
    startup info logged.

    Args:
        config: The validated global configuration.
    """
    check_minimum_python()
    set_deterministic_seed(config.seed)

    log_file = None
    if config.log_file is not None:
        log_file = Path(config.log_file)

    logger = get_logger("m31r.runtime", log_level=config.log_level, log_file=log_file)

    system_info = get_system_info()
    logger.info(
        "M31R bootstrap complete",
        extra={
            "seed": config.seed,
            "python_version": system_info.python_version,
            "platform": system_info.platform,
            "architecture": system_info.architecture,
        },
    )

    try:
        project_root = resolve_project_root()
        _ensure_project_directories(project_root, config)
    except RuntimeError:
        logger.warning(
            "Could not resolve project root — skipping directory creation. "
            "This is fine in test environments."
        )
