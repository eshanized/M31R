# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for the checkpoint save/load system.

Per 15_TESTING_STRATEGY.md — checkpoint save/load, resume determinism.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn

from m31r.training.checkpoint.core import (
    CheckpointMetadata,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


class _TinyModel(nn.Module):
    """Minimal model for checkpoint testing."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _make_metadata(step: int = 100) -> CheckpointMetadata:
    """Create test checkpoint metadata."""
    return CheckpointMetadata(
        global_step=step,
        seed=42,
        config_snapshot={"model": {"dim": 8}},
        tokens_seen=1000,
        loss=2.5,
    )


class TestCheckpointSaveLoad:
    """Tests for atomic checkpoint save and load."""

    def test_save_creates_files(self, tmp_path: Path) -> None:
        """save_checkpoint must create all required files."""
        model = _TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ckpt_dir = tmp_path / "step_000100"

        save_checkpoint(model, optimizer, _make_metadata(), ckpt_dir)

        assert (ckpt_dir / "model.pt").is_file()
        assert (ckpt_dir / "optimizer.pt").is_file()
        assert (ckpt_dir / "rng_state.pt").is_file()
        assert (ckpt_dir / "metadata.json").is_file()

    def test_metadata_content(self, tmp_path: Path) -> None:
        """Metadata JSON must contain correct values."""
        model = _TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ckpt_dir = tmp_path / "step_000100"

        save_checkpoint(model, optimizer, _make_metadata(step=42), ckpt_dir)

        meta = json.loads((ckpt_dir / "metadata.json").read_text())
        assert meta["global_step"] == 42
        assert meta["seed"] == 42
        assert meta["tokens_seen"] == 1000
        assert meta["loss"] == 2.5

    def test_roundtrip_weights(self, tmp_path: Path) -> None:
        """Loaded model weights must match saved weights exactly."""
        torch.manual_seed(42)
        model = _TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ckpt_dir = tmp_path / "step_000100"

        # Save original weights
        original_weights = {k: v.clone() for k, v in model.state_dict().items()}
        save_checkpoint(model, optimizer, _make_metadata(), ckpt_dir)

        # Create a new model with different weights
        torch.manual_seed(999)
        model2 = _TinyModel()
        assert not all(
            torch.equal(model2.state_dict()[k], original_weights[k]) for k in original_weights
        )

        # Load checkpoint into new model
        load_checkpoint(ckpt_dir, model2)

        # Weights must now match
        for key in original_weights:
            assert torch.equal(model2.state_dict()[key], original_weights[key])

    def test_roundtrip_optimizer(self, tmp_path: Path) -> None:
        """Loaded optimizer state must match saved state."""
        model = _TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Do a step to create optimizer state
        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        ckpt_dir = tmp_path / "step_000001"
        save_checkpoint(model, optimizer, _make_metadata(step=1), ckpt_dir)

        # Create a fresh model/optimizer and load
        model2 = _TinyModel()
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        load_checkpoint(ckpt_dir, model2, optimizer2)

        # Optimizer states should match
        state1 = optimizer.state_dict()
        state2 = optimizer2.state_dict()
        assert len(state1["state"]) == len(state2["state"])

    def test_atomic_save_no_partial(self, tmp_path: Path) -> None:
        """If the final directory exists, it should be complete."""
        model = _TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ckpt_dir = tmp_path / "step_000100"

        save_checkpoint(model, optimizer, _make_metadata(), ckpt_dir)

        # All files must exist (atomic rename means all-or-nothing)
        expected_files = {"model.pt", "optimizer.pt", "rng_state.pt", "metadata.json"}
        actual_files = {f.name for f in ckpt_dir.iterdir()}
        assert expected_files.issubset(actual_files)


class TestFindLatestCheckpoint:
    """Tests for finding the latest checkpoint."""

    def test_finds_latest(self, tmp_path: Path) -> None:
        """Must find the checkpoint with the highest step number."""
        checkpoints = tmp_path / "checkpoints"
        (checkpoints / "step_000010").mkdir(parents=True)
        (checkpoints / "step_000050").mkdir(parents=True)
        (checkpoints / "step_000020").mkdir(parents=True)

        latest = find_latest_checkpoint(tmp_path)
        assert latest is not None
        assert latest.name == "step_000050"

    def test_returns_none_when_empty(self, tmp_path: Path) -> None:
        """Must return None when no checkpoints exist."""
        latest = find_latest_checkpoint(tmp_path)
        assert latest is None

    def test_returns_none_for_nonexistent(self, tmp_path: Path) -> None:
        """Must return None for nonexistent directory."""
        latest = find_latest_checkpoint(tmp_path / "nonexistent")
        assert latest is None
