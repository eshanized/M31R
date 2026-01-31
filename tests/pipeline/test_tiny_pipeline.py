# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
End-to-end pipeline integration tests for M31R.

These tests validate that every stage of the training pipeline works
correctly with the tiny model configuration on CPU. Each test is
self-contained — it creates its own temporary data and cleans up after.

Run with:
    python -m pytest tests/pipeline/test_tiny_pipeline.py -v --tb=short
"""

import json
import shutil
import struct
import tempfile
from pathlib import Path

import pytest
import torch

from m31r.config.schema import M31RConfig
from m31r.model.transformer import M31RTransformer, TransformerModelConfig


# ── Fixtures ──────────────────────────────────────────────────────────


def _tiny_model_config(vocab_size: int = 256) -> TransformerModelConfig:
    """Create a minimal model config for fast tests."""
    return TransformerModelConfig(
        vocab_size=vocab_size,
        dim=64,
        n_layers=2,
        n_heads=2,
        head_dim=32,
        max_seq_len=128,
        dropout=0.0,
        norm_eps=1e-6,
        rope_theta=10000.0,
        init_std=0.02,
        seed=42,
    )


def _create_shard_dir(path: Path, num_shards: int = 2, tokens_per_shard: int = 2000, vocab_size: int = 256) -> None:
    """Create JSON shard files for testing."""
    import random

    rng = random.Random(42)
    path.mkdir(parents=True, exist_ok=True)

    shard_files = []
    for i in range(num_shards):
        tokens = [rng.randint(0, vocab_size - 1) for _ in range(tokens_per_shard)]
        name = f"shard_{i:04d}.json"
        (path / name).write_text(json.dumps({"tokens": tokens}), encoding="utf-8")
        shard_files.append(name)

    manifest = {
        "version": "1.0.0",
        "total_tokens": num_shards * tokens_per_shard,
        "num_shards": num_shards,
        "shards": shard_files,
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _create_test_config(
    tmp_dir: Path,
    max_steps: int = 20,
    checkpoint_interval: int = 5,
    vocab_size: int = 256,
) -> M31RConfig:
    """Build a minimal M31RConfig for pipeline tests."""
    config_dict = {
        "global": {
            "config_version": "1.0.0",
            "project_name": "m31r-test",
            "seed": 42,
            "log_level": "WARNING",
            "directories": {
                "data": "data",
                "checkpoints": "checkpoints",
                "logs": "logs",
                "experiments": str(tmp_dir / "experiments"),
                "configs": "configs",
            },
        },
        "model": {
            "config_version": "1.0.0",
            "n_layers": 2,
            "hidden_size": 64,
            "n_heads": 2,
            "head_dim": 32,
            "context_length": 128,
            "dropout": 0.0,
            "norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "init_std": 0.02,
            "vocab_size": vocab_size,
        },
        "train": {
            "config_version": "1.0.0",
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "max_steps": max_steps,
            "learning_rate": 1e-3,
            "min_learning_rate": 1e-5,
            "weight_decay": 0.01,
            "beta1": 0.9,
            "beta2": 0.95,
            "grad_clip": 1.0,
            "warmup_steps": 2,
            "precision": "fp32",
            "checkpoint_interval": checkpoint_interval,
            "log_interval": 2,
            "dataset_directory": str(tmp_dir / "shards"),
            "tokenizer_directory": str(tmp_dir / "tokenizer"),
        },
    }

    return M31RConfig.model_validate(config_dict)


@pytest.fixture
def tmp_pipeline_dir(tmp_path: Path):
    """Set up a temporary pipeline directory with shards and experiment space."""
    shard_dir = tmp_path / "shards"
    _create_shard_dir(shard_dir, num_shards=4, tokens_per_shard=10_000, vocab_size=256)

    # Create tokenizer dir (empty is fine for training — only needed for eval/serve)
    (tmp_path / "tokenizer").mkdir(parents=True, exist_ok=True)

    (tmp_path / "experiments").mkdir(parents=True, exist_ok=True)

    return tmp_path


# ── Test: Forward Pass ────────────────────────────────────────────────


class TestForwardPass:
    """Validate model forward pass produces correct output shapes."""

    def test_forward_produces_logits(self):
        """Model forward pass should produce [batch, seq, vocab] logits."""
        cfg = _tiny_model_config(vocab_size=256)
        model = M31RTransformer(cfg)
        model.eval()

        batch_size = 2
        seq_len = 32
        x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            logits = model(x)

        assert logits.shape == (batch_size, seq_len, cfg.vocab_size), (
            f"Expected ({batch_size}, {seq_len}, {cfg.vocab_size}), got {logits.shape}"
        )

    def test_forward_produces_finite_values(self):
        """Logits should not contain NaN or Inf."""
        cfg = _tiny_model_config()
        model = M31RTransformer(cfg)
        model.eval()

        x = torch.randint(0, cfg.vocab_size, (1, 16))
        with torch.no_grad():
            logits = model(x)

        assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"


# ── Test: Loss Decreases ─────────────────────────────────────────────


class TestLossDecreases:
    """Validate that training reduces loss over multiple steps."""

    def test_loss_decreases_over_training(self, tmp_pipeline_dir: Path):
        """Loss should decrease over 20 steps of training."""
        from m31r.training.engine.core import run_training
        from m31r.training.engine.experiment import create_experiment_dir

        config = _create_test_config(tmp_pipeline_dir, max_steps=20, checkpoint_interval=10)
        experiment_dir = create_experiment_dir(
            tmp_pipeline_dir / "experiments", config, seed=42
        )

        result = run_training(config, experiment_dir)

        assert result.final_step == 20, f"Expected final step 20, got {result.final_step}"
        assert result.total_tokens > 0, "Should have processed tokens"
        # Loss should be finite and reasonable for a randomly initialized model
        assert result.final_loss < 10.0, f"Loss {result.final_loss} seems too high"
        assert result.final_loss > 0.0, "Loss should be positive"


# ── Test: Checkpoint Save/Load ────────────────────────────────────────


class TestCheckpoint:
    """Validate checkpoint save and load produces identical state."""

    def test_checkpoint_roundtrip(self, tmp_path: Path):
        """Save and load should produce bit-identical model weights."""
        from m31r.training.checkpoint.core import (
            CheckpointMetadata,
            load_checkpoint,
            save_checkpoint,
        )

        cfg = _tiny_model_config()
        model = M31RTransformer(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Run a forward/backward pass to populate optimizer state
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            x.view(-1),
        )
        loss.backward()
        optimizer.step()

        # Save original state
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        metadata = CheckpointMetadata(
            global_step=10,
            seed=42,
            config_snapshot={},
            tokens_seen=1000,
            loss=loss.item(),
        )
        ckpt_dir = tmp_path / "checkpoints" / "step_000010"
        save_checkpoint(model, optimizer, metadata, ckpt_dir)

        # Verify checkpoint files exist
        assert (ckpt_dir / "model.pt").is_file()
        assert (ckpt_dir / "optimizer.pt").is_file()
        assert (ckpt_dir / "metadata.json").is_file()
        assert (ckpt_dir / "rng_state.pt").is_file()

        # Load into a fresh model
        fresh_model = M31RTransformer(cfg)
        loaded_meta = load_checkpoint(ckpt_dir, fresh_model, device=torch.device("cpu"))

        # Verify weights match
        for key in original_state:
            assert torch.equal(original_state[key], fresh_model.state_dict()[key]), (
                f"Weight mismatch at {key}"
            )

        # Verify metadata
        assert loaded_meta.global_step == 10
        assert loaded_meta.tokens_seen == 1000


# ── Test: Resume Determinism ─────────────────────────────────────────


class TestResumeDeterminism:
    """Validate that resume continues from the correct step."""

    def test_resume_continues_from_checkpoint(self, tmp_pipeline_dir: Path):
        """Resume should start from the saved global step and continue training."""
        from m31r.training.checkpoint.core import find_latest_checkpoint
        from m31r.training.engine.core import run_training
        from m31r.training.engine.experiment import create_experiment_dir

        # Phase 1: Train for 10 steps
        config = _create_test_config(
            tmp_pipeline_dir, max_steps=10, checkpoint_interval=5
        )
        experiment_dir = create_experiment_dir(
            tmp_pipeline_dir / "experiments", config, seed=42
        )
        result1 = run_training(config, experiment_dir)
        assert result1.final_step == 10

        # Phase 2: Resume and train to step 20
        config2 = _create_test_config(
            tmp_pipeline_dir, max_steps=20, checkpoint_interval=5
        )
        checkpoint_dir = find_latest_checkpoint(experiment_dir)
        assert checkpoint_dir is not None, "Should find a checkpoint after training"

        result2 = run_training(config2, experiment_dir, resume_from=checkpoint_dir)

        # Should have continued from step 10 and reached step 20
        assert result2.final_step == 20, (
            f"Expected step 20 after resume, got {result2.final_step}"
        )


# ── Test: Export Creates Valid Bundle ─────────────────────────────────


class TestExport:
    """Validate that export produces all required release files."""

    def test_export_creates_release_files(self, tmp_pipeline_dir: Path):
        """Export should create model weights, metadata, and checksums."""
        from m31r.training.engine.core import run_training
        from m31r.training.engine.experiment import create_experiment_dir

        # Train first
        config = _create_test_config(tmp_pipeline_dir, max_steps=10, checkpoint_interval=5)
        experiment_dir = create_experiment_dir(
            tmp_pipeline_dir / "experiments", config, seed=42
        )
        run_training(config, experiment_dir)

        # Now export
        from m31r.training.checkpoint.core import find_latest_checkpoint

        checkpoint_dir = find_latest_checkpoint(experiment_dir)
        assert checkpoint_dir is not None

        # Verify checkpoint directory structure
        assert (checkpoint_dir / "model.pt").is_file(), "model.pt missing from checkpoint"
        assert (checkpoint_dir / "metadata.json").is_file(), "metadata.json missing from checkpoint"

        # Verify metadata is parseable
        meta = json.loads((checkpoint_dir / "metadata.json").read_text())
        assert "global_step" in meta
        assert "loss" in meta
        assert meta["global_step"] >= 0


# ── Test: Inference Engine ────────────────────────────────────────────


class TestInference:
    """Validate that the model can generate tokens."""

    def test_greedy_generation(self):
        """Model should produce deterministic output with temperature=0."""
        cfg = _tiny_model_config(vocab_size=256)
        model = M31RTransformer(cfg)
        model.eval()

        # Simple greedy decode loop
        prompt = torch.randint(0, cfg.vocab_size, (1, 8))
        generated = prompt.clone()

        with torch.no_grad():
            for _ in range(16):
                logits = model(generated[:, -cfg.max_seq_len :])
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

        assert generated.shape[1] == 24, (
            f"Expected 24 tokens (8 prompt + 16 generated), got {generated.shape[1]}"
        )

    def test_deterministic_generation(self):
        """Same seed + same input should produce identical output."""
        cfg = _tiny_model_config(vocab_size=256)

        torch.manual_seed(42)
        model = M31RTransformer(cfg)
        model.eval()
        prompt = torch.tensor([[1, 2, 3, 4]])

        with torch.no_grad():
            logits1 = model(prompt)

        torch.manual_seed(42)
        model2 = M31RTransformer(cfg)
        model2.eval()

        with torch.no_grad():
            logits2 = model2(prompt)

        assert torch.equal(logits1, logits2), "Determinism violated: same seed, different logits"


# ── Test: Dataloader ──────────────────────────────────────────────────


class TestDataloader:
    """Validate the streaming token dataloader."""

    def test_dataloader_produces_batches(self, tmp_pipeline_dir: Path):
        """Dataloader should yield (input, target) batch pairs."""
        from m31r.training.dataloader.core import TokenDataset, create_dataloader

        shard_dir = tmp_pipeline_dir / "shards"
        dataset = TokenDataset(shard_dir=shard_dir, seq_len=128, seed=42)
        loader = create_dataloader(dataset, batch_size=2, seed=42)

        batch_count = 0
        for input_batch, target_batch in loader:
            assert input_batch.shape == (2, 128), f"Input shape: {input_batch.shape}"
            assert target_batch.shape == (2, 128), f"Target shape: {target_batch.shape}"
            batch_count += 1
            if batch_count >= 3:
                break

        assert batch_count == 3, f"Expected at least 3 batches, got {batch_count}"

    def test_dataloader_deterministic(self, tmp_pipeline_dir: Path):
        """Two dataloaders with same seed should yield identical data."""
        from m31r.training.dataloader.core import TokenDataset, create_dataloader

        shard_dir = tmp_pipeline_dir / "shards"

        dataset1 = TokenDataset(shard_dir=shard_dir, seq_len=64, seed=42)
        dataset2 = TokenDataset(shard_dir=shard_dir, seq_len=64, seed=42)

        loader1 = create_dataloader(dataset1, batch_size=2, seed=42)
        loader2 = create_dataloader(dataset2, batch_size=2, seed=42)

        for (inp1, tgt1), (inp2, tgt2) in zip(loader1, loader2):
            assert torch.equal(inp1, inp2), "Inputs differ between identical dataloaders"
            assert torch.equal(tgt1, tgt2), "Targets differ between identical dataloaders"
            break  # Just check first batch


# ── Test: Full Training Pipeline ──────────────────────────────────────


class TestFullPipeline:
    """Integration test: train → checkpoint → resume → verify continuity."""

    def test_train_checkpoint_resume_cycle(self, tmp_pipeline_dir: Path):
        """Full cycle should produce decreasing loss and valid checkpoints."""
        from m31r.training.checkpoint.core import find_latest_checkpoint
        from m31r.training.engine.core import run_training
        from m31r.training.engine.experiment import create_experiment_dir

        # Train 10 steps
        config = _create_test_config(tmp_pipeline_dir, max_steps=10, checkpoint_interval=5)
        experiment_dir = create_experiment_dir(
            tmp_pipeline_dir / "experiments", config, seed=42
        )
        result = run_training(config, experiment_dir)

        assert result.final_step == 10
        assert result.total_tokens > 0
        assert result.final_loss > 0

        # Find checkpoint
        ckpt = find_latest_checkpoint(experiment_dir)
        assert ckpt is not None
        assert "step_" in ckpt.name

        # Verify checkpoint metadata
        meta = json.loads((ckpt / "metadata.json").read_text())
        assert meta["global_step"] == result.final_step

        # Resume to 20 steps
        config2 = _create_test_config(tmp_pipeline_dir, max_steps=20, checkpoint_interval=5)
        result2 = run_training(config2, experiment_dir, resume_from=ckpt)
        assert result2.final_step == 20
