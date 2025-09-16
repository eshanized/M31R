# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Atomic checkpoint save/load system for M31R.

Per 07_TRAINING_ARCHITECTURE.md §16:
  Checkpoints must include:
    - Model weights
    - Optimizer state
    - Global step
    - Random seed state (Python, torch)
    - Config snapshot

  Saves must be atomic: write to temp file, then rename. This prevents
  corrupted checkpoints from partial writes or crashes.

  Checkpoints are versioned by step number in the experiment directory.
"""

import json
import logging
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from m31r.logging.logger import get_logger

logger: logging.Logger = get_logger(__name__)


@dataclass(frozen=True)
class CheckpointMetadata:
    """Metadata stored alongside the checkpoint for tracing."""

    global_step: int
    seed: int
    config_snapshot: dict[str, object]
    tokens_seen: int
    loss: float


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: CheckpointMetadata,
    checkpoint_dir: Path,
) -> Path:
    """
    Save a checkpoint atomically.

    The save procedure:
      1. Create a temp directory alongside the final checkpoint dir
      2. Write model weights, optimizer state, metadata, and RNG states
      3. Atomically rename temp dir to final dir

    This ensures no partially-written checkpoint can corrupt a resume.

    Args:
        model: The model to checkpoint.
        optimizer: The optimizer to checkpoint.
        metadata: Step, seed, config, tokens_seen, loss.
        checkpoint_dir: Final directory for this checkpoint (e.g. step_01000/).

    Returns:
        Path to the saved checkpoint directory.
    """
    parent = checkpoint_dir.parent
    parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp directory first, then atomic rename
    tmp_dir = Path(tempfile.mkdtemp(dir=parent, prefix=".ckpt_tmp_"))
    try:
        # Model weights
        torch.save(model.state_dict(), tmp_dir / "model.pt")

        # Optimizer state
        torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")

        # RNG states for full reproducibility
        rng_state = {
            "python": random.getstate(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_hash_seed": os.environ.get("PYTHONHASHSEED", ""),
        }
        if torch.cuda.is_available():
            rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
        torch.save(rng_state, tmp_dir / "rng_state.pt")

        # Metadata as JSON for human readability
        meta_dict = {
            "global_step": metadata.global_step,
            "seed": metadata.seed,
            "config_snapshot": metadata.config_snapshot,
            "tokens_seen": metadata.tokens_seen,
            "loss": metadata.loss,
        }
        (tmp_dir / "metadata.json").write_text(
            json.dumps(meta_dict, indent=2, default=str),
            encoding="utf-8",
        )

        # Atomic rename: temp dir → final dir
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        tmp_dir.rename(checkpoint_dir)

        logger.info(
            "Checkpoint saved",
            extra={
                "step": metadata.global_step,
                "path": str(checkpoint_dir),
                "loss": metadata.loss,
            },
        )
        return checkpoint_dir

    except Exception:
        # Clean up temp dir on failure
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise


def load_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> CheckpointMetadata:
    """
    Load a checkpoint and restore model/optimizer/RNG state.

    Args:
        checkpoint_dir: Path to the checkpoint directory.
        model: The model to load weights into.
        optimizer: Optional optimizer to restore state to.
        device: Device to map tensors to.

    Returns:
        CheckpointMetadata with the restored step, seed, config, etc.

    Raises:
        FileNotFoundError: If checkpoint_dir doesn't exist.
        RuntimeError: If any checkpoint file is missing or corrupt.
    """
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    map_location = device if device is not None else "cpu"

    # Restore model weights
    model_path = checkpoint_dir / "model.pt"
    if not model_path.is_file():
        raise RuntimeError(f"model.pt not found in {checkpoint_dir}")
    model.load_state_dict(torch.load(model_path, map_location=map_location, weights_only=True))

    # Restore optimizer state
    if optimizer is not None:
        opt_path = checkpoint_dir / "optimizer.pt"
        if opt_path.is_file():
            optimizer.load_state_dict(torch.load(opt_path, map_location=map_location, weights_only=True))

    # Restore RNG states
    rng_path = checkpoint_dir / "rng_state.pt"
    if rng_path.is_file():
        rng_state = torch.load(rng_path, map_location="cpu", weights_only=False)
        random.setstate(rng_state["python"])
        torch.random.set_rng_state(rng_state["torch_cpu"])
        if rng_state.get("torch_hash_seed"):
            os.environ["PYTHONHASHSEED"] = rng_state["torch_hash_seed"]
        if torch.cuda.is_available() and "torch_cuda" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])

    # Load metadata
    meta_path = checkpoint_dir / "metadata.json"
    if not meta_path.is_file():
        raise RuntimeError(f"metadata.json not found in {checkpoint_dir}")

    meta_dict = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata = CheckpointMetadata(
        global_step=meta_dict["global_step"],
        seed=meta_dict["seed"],
        config_snapshot=meta_dict.get("config_snapshot", {}),
        tokens_seen=meta_dict.get("tokens_seen", 0),
        loss=meta_dict.get("loss", 0.0),
    )

    logger.info(
        "Checkpoint loaded",
        extra={
            "step": metadata.global_step,
            "path": str(checkpoint_dir),
        },
    )
    return metadata


def find_latest_checkpoint(experiment_dir: Path) -> Path | None:
    """
    Find the most recent checkpoint in an experiment directory.

    Checkpoints are named step_NNNNN/ and we pick the one with the
    highest step number.

    Args:
        experiment_dir: The experiment directory to search.

    Returns:
        Path to the latest checkpoint directory, or None if none found.
    """
    checkpoints_dir = experiment_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        return None

    ckpt_dirs = sorted(
        (d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("step_")),
        key=lambda d: int(d.name.split("_")[1]),
    )

    if not ckpt_dirs:
        return None

    return ckpt_dirs[-1]
