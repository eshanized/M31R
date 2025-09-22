# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Model export bundle creation for M31R.

Per 12_CLI_AND_TOOLING_SPEC.md, `m31r export` creates a self-contained
release bundle from a trained checkpoint. The bundle includes:
  - Model weights (state_dict only, no optimizer)
  - Model config
  - Tokenizer bundle reference
  - Training metadata (step, loss, seed)
  - SHA256 checksum of the weights file

The export is designed for deployment and inference — it strips optimizer
state and RNG state since those are training-only artifacts.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from m31r.logging.logger import get_logger
from m31r.utils.hashing import compute_sha256

logger: logging.Logger = get_logger(__name__)


@dataclass(frozen=True)
class ExportResult:
    """Result of an export operation."""

    output_dir: str
    weights_hash: str
    model_config: dict[str, object]
    step: int


def export_model(
    checkpoint_dir: Path,
    output_dir: Path,
    tokenizer_dir: Path | None = None,
) -> ExportResult:
    """
    Create a release bundle from a checkpoint.

    Reads the checkpoint, extracts model weights and metadata, computes
    a SHA256 checksum of the weights file, and writes everything to the
    output directory.

    Args:
        checkpoint_dir: Path to the checkpoint directory (step_NNNNN/).
        output_dir: Where to write the export bundle.
        tokenizer_dir: Optional path to tokenizer bundle to include.

    Returns:
        ExportResult with the output path and verification hash.

    Raises:
        FileNotFoundError: If checkpoint_dir doesn't exist.
        RuntimeError: If checkpoint files are missing.
    """
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    model_path = checkpoint_dir / "model.pt"
    meta_path = checkpoint_dir / "metadata.json"

    if not model_path.is_file():
        raise RuntimeError(f"model.pt not found in {checkpoint_dir}")
    if not meta_path.is_file():
        raise RuntimeError(f"metadata.json not found in {checkpoint_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy model weights
    weights_output = output_dir / "model.pt"
    import shutil
    shutil.copy2(model_path, weights_output)

    # Compute checksum
    weights_hash = compute_sha256(weights_output)

    # Load and write metadata
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    config_snapshot = metadata.get("config_snapshot", {})

    export_meta = {
        "step": metadata.get("global_step", 0),
        "loss": metadata.get("loss", 0.0),
        "seed": metadata.get("seed", 0),
        "tokens_seen": metadata.get("tokens_seen", 0),
        "weights_sha256": weights_hash,
        "model_config": config_snapshot.get("model", {}),
        "tokenizer_dir": str(tokenizer_dir) if tokenizer_dir else None,
    }
    (output_dir / "export_metadata.json").write_text(
        json.dumps(export_meta, indent=2, default=str),
        encoding="utf-8",
    )

    # Write checksum file
    (output_dir / "checksums.sha256").write_text(
        f"{weights_hash}  model.pt\n",
        encoding="utf-8",
    )

    logger.info(
        "Export complete",
        extra={
            "output_dir": str(output_dir),
            "weights_hash": weights_hash[:16] + "...",
            "step": metadata.get("global_step", 0),
        },
    )

    return ExportResult(
        output_dir=str(output_dir),
        weights_hash=weights_hash,
        model_config=config_snapshot.get("model", {}),
        step=metadata.get("global_step", 0),
    )
