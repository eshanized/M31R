# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for the model export system.

Per 15_TESTING_STRATEGY.md — export bundle must contain correct files.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn

from m31r.training.checkpoint.core import CheckpointMetadata, save_checkpoint
from m31r.training.export.core import export_model


class _TinyModel(nn.Module):
    """Minimal model for export testing."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _create_test_checkpoint(tmp_path: Path) -> Path:
    """Create a checkpoint to export from."""
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_dir = tmp_path / "checkpoints" / "step_000100"

    metadata = CheckpointMetadata(
        global_step=100,
        seed=42,
        config_snapshot={"model": {"dim": 8}, "train": {"lr": 1e-3}},
        tokens_seen=5000,
        loss=1.5,
    )
    save_checkpoint(model, optimizer, metadata, ckpt_dir)
    return ckpt_dir


class TestExport:
    """Tests for the export system."""

    def test_export_creates_files(self, tmp_path: Path) -> None:
        """Export must create model.pt, metadata, and checksums."""
        ckpt_dir = _create_test_checkpoint(tmp_path)
        output_dir = tmp_path / "export"

        result = export_model(ckpt_dir, output_dir)

        assert (output_dir / "model.pt").is_file()
        assert (output_dir / "export_metadata.json").is_file()
        assert (output_dir / "checksums.sha256").is_file()

    def test_export_metadata(self, tmp_path: Path) -> None:
        """Export metadata must contain step and checksum."""
        ckpt_dir = _create_test_checkpoint(tmp_path)
        output_dir = tmp_path / "export"

        result = export_model(ckpt_dir, output_dir)

        meta = json.loads((output_dir / "export_metadata.json").read_text())
        assert meta["step"] == 100
        assert meta["loss"] == 1.5
        assert "weights_sha256" in meta
        assert len(meta["weights_sha256"]) == 64  # SHA256 hex length

    def test_export_checksum_file(self, tmp_path: Path) -> None:
        """Checksum file must reference the weights file."""
        ckpt_dir = _create_test_checkpoint(tmp_path)
        output_dir = tmp_path / "export"

        result = export_model(ckpt_dir, output_dir)

        checksum_content = (output_dir / "checksums.sha256").read_text()
        assert "model.pt" in checksum_content
        assert len(checksum_content.split("  ")[0]) == 64

    def test_export_result(self, tmp_path: Path) -> None:
        """ExportResult must have correct fields."""
        ckpt_dir = _create_test_checkpoint(tmp_path)
        output_dir = tmp_path / "export"

        result = export_model(ckpt_dir, output_dir)

        assert result.step == 100
        assert len(result.weights_hash) == 64
        assert result.output_dir == str(output_dir)

    def test_export_no_optimizer(self, tmp_path: Path) -> None:
        """Export bundle must NOT contain optimizer state."""
        ckpt_dir = _create_test_checkpoint(tmp_path)
        output_dir = tmp_path / "export"

        export_model(ckpt_dir, output_dir)

        exported_files = {f.name for f in output_dir.iterdir()}
        assert "optimizer.pt" not in exported_files
        assert "rng_state.pt" not in exported_files
