# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Experiment directory setup for M31R.

Per 07_TRAINING_ARCHITECTURE.md §17 and 20_OBSERVABILITY_AND_LOGGING.md §3:
  Experiment directory structure:
    experiments/<run_id>/
      ├── config.yaml       — frozen config snapshot
      ├── train.log         — structured JSON log file
      ├── metrics/           — training metrics files
      └── checkpoints/       — saved checkpoints
          └── step_NNNNN/

  run_id format: YYYYMMDD_HHMMSS_<seed>
"""

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from m31r.config.schema import M31RConfig
from m31r.logging.logger import get_logger

logger: logging.Logger = get_logger(__name__)


def create_experiment_dir(
    experiments_root: Path,
    config: M31RConfig,
    seed: int,
) -> Path:
    """
    Create a new experiment directory with the required structure.

    The run_id is generated from the current UTC timestamp and seed.
    The full config is snapshotted into the experiment dir for reproducibility.

    Args:
        experiments_root: Root directory for experiments (e.g. experiments/).
        config: The full validated M31R config.
        seed: The training seed.

    Returns:
        Path to the created experiment directory.
    """
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{seed}"
    experiment_dir = experiments_root / run_id

    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "metrics").mkdir(exist_ok=True)

    # Snapshot the config
    config_snapshot = config.model_dump(by_alias=True)
    (experiment_dir / "config.json").write_text(
        json.dumps(config_snapshot, indent=2, default=str),
        encoding="utf-8",
    )

    logger.info(
        "Experiment directory created",
        extra={"run_id": run_id, "path": str(experiment_dir)},
    )
    return experiment_dir


def find_experiment_dir(
    experiments_root: Path,
    run_id: str | None = None,
) -> Path | None:
    """
    Find an experiment directory by run_id, or return the most recent.

    Args:
        experiments_root: Root directory for experiments.
        run_id: Specific run ID to find. None means find the latest.

    Returns:
        Path to the experiment directory, or None if not found.
    """
    if not experiments_root.is_dir():
        return None

    if run_id is not None:
        target = experiments_root / run_id
        return target if target.is_dir() else None

    # Find the latest by sorting directory names (timestamp-based)
    dirs = sorted(
        (d for d in experiments_root.iterdir() if d.is_dir()),
        reverse=True,
    )
    return dirs[0] if dirs else None
