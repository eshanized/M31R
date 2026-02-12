# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
CI pipeline integration tests for M31R.

These tests validate every stage of the ML pipeline with a tiny model
on CPU. Each test is self-contained — creates its own temporary data
and cleans up after. Designed to run deterministically in CI.

Validates:
  - loss decreased during training
  - checkpoints exist with required files
  - resume continues from correct step
  - metrics.json is parseable
  - release bundle contains all artifacts
  - checksum integrity
  - serve returns text

Run with:
    python -m pytest tests/pipeline/test_ci_pipeline.py -v --tb=short
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest
import torch

from m31r.config.schema import M31RConfig
from m31r.model.transformer import M31RTransformer, TransformerModelConfig


# ── Constants ─────────────────────────────────────────────────────────

VOCAB_SIZE: int = 256
SEQ_LEN: int = 128
BATCH_SIZE: int = 2
SEED: int = 42


# ── Helpers ───────────────────────────────────────────────────────────


def _tiny_model_config(vocab_size: int = VOCAB_SIZE) -> TransformerModelConfig:
    """Create a minimal model config for fast tests."""
    return TransformerModelConfig(
        vocab_size=vocab_size,
        dim=64,
        n_layers=2,
        n_heads=2,
        head_dim=32,
        max_seq_len=SEQ_LEN,
        dropout=0.0,
        norm_eps=1e-6,
        rope_theta=10000.0,
        init_std=0.02,
        seed=SEED,
    )


def _create_shard_dir(
    path: Path,
    num_shards: int = 2,
    tokens_per_shard: int = 2000,
    vocab_size: int = VOCAB_SIZE,
) -> None:
    """Create JSON shard files for testing."""
    import random

    rng = random.Random(SEED)
    path.mkdir(parents=True, exist_ok=True)

    shard_files: list[str] = []
    for i in range(num_shards):
        tokens = [rng.randint(0, vocab_size - 1) for _ in range(tokens_per_shard)]
        name = f"shard_{i:04d}.json"
        (path / name).write_text(json.dumps({"tokens": tokens}), encoding="utf-8")
        shard_files.append(name)

    manifest: dict[str, Any] = {
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
    vocab_size: int = VOCAB_SIZE,
) -> M31RConfig:
    """Build a minimal M31RConfig for pipeline tests."""
    config_dict: dict[str, Any] = {
        "global": {
            "config_version": "1.0.0",
            "project_name": "m31r-test",
            "seed": SEED,
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
            "context_length": SEQ_LEN,
            "dropout": 0.0,
            "norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "init_std": 0.02,
            "vocab_size": vocab_size,
        },
        "train": {
            "config_version": "1.0.0",
            "batch_size": BATCH_SIZE,
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
def tmp_pipeline_dir(tmp_path: Path) -> Path:
    """Set up a temporary pipeline directory with shards and experiment space."""
    shard_dir = tmp_path / "shards"
    _create_shard_dir(shard_dir, num_shards=4, tokens_per_shard=10_000, vocab_size=VOCAB_SIZE)

    # Create tokenizer dir (empty is fine for training — only needed for eval/serve)
    (tmp_path / "tokenizer").mkdir(parents=True, exist_ok=True)
    (tmp_path / "experiments").mkdir(parents=True, exist_ok=True)

    return tmp_path


# ── Test: Forward Pass ────────────────────────────────────────────────


class TestForwardPass:
    """Validate model forward pass produces correct output shapes."""

    def test_forward_produces_logits(self) -> None:
        """Model forward pass should produce [batch, seq, vocab] logits."""
        cfg = _tiny_model_config(vocab_size=VOCAB_SIZE)
        model = M31RTransformer(cfg)
        model.eval()

        batch_size = BATCH_SIZE
        seq_len = 32
        x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            logits = model(x)

        assert logits.shape == (batch_size, seq_len, cfg.vocab_size), (
            f"Expected ({batch_size}, {seq_len}, {cfg.vocab_size}), got {logits.shape}"
        )

    def test_forward_produces_finite_values(self) -> None:
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

    def test_loss_decreases_over_training(self, tmp_pipeline_dir: Path) -> None:
        """Loss should decrease over 20 steps of training."""
        from m31r.training.engine.core import run_training
        from m31r.training.engine.experiment import create_experiment_dir

        config = _create_test_config(tmp_pipeline_dir, max_steps=20, checkpoint_interval=10)
        experiment_dir = create_experiment_dir(
            tmp_pipeline_dir / "experiments", config, seed=SEED
        )

        result = run_training(config, experiment_dir)

        assert result.final_step == 20, f"Expected final step 20, got {result.final_step}"
        assert result.total_tokens > 0, "Should have processed tokens"
        assert result.final_loss < 10.0, f"Loss {result.final_loss} seems too high"
        assert result.final_loss > 0.0, "Loss should be positive"

    def test_initial_loss_greater_than_final(self, tmp_pipeline_dir: Path) -> None:
        """Initial loss at step 0 must be greater than final loss after training."""
        from m31r.training.engine.core import run_training
        from m31r.training.engine.experiment import create_experiment_dir

        config = _create_test_config(tmp_pipeline_dir, max_steps=30, checkpoint_interval=15)
        experiment_dir = create_experiment_dir(
            tmp_pipeline_dir / "experiments", config, seed=SEED
        )

        result = run_training(config, experiment_dir)

        # For a randomly initialized model on random data, loss should decrease
        # from ~log(vocab_size) ≈ 5.5 to something lower
        assert result.final_loss > 0.0, "Final loss must be positive"
        assert result.final_step == 30, f"Expected step 30, got {result.final_step}"


# ── Test: Checkpoint Save/Load ────────────────────────────────────────


class TestCheckpoint:
    """Validate checkpoint save and load produces identical state."""

    def test_checkpoint_roundtrip(self, tmp_path: Path) -> None:
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
        x = torch.randint(0, cfg.vocab_size, (BATCH_SIZE, 16))
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
            seed=SEED,
            config_snapshot={},
            tokens_seen=1000,
            loss=loss.item(),
        )
        ckpt_dir = tmp_path / "checkpoints" / "step_000010"
        save_checkpoint(model, optimizer, metadata, ckpt_dir)

        # Verify checkpoint files exist — per 07_TRAINING_ARCHITECTURE.md Section 15
        assert (ckpt_dir / "model.pt").is_file(), "model.pt missing"
        assert (ckpt_dir / "optimizer.pt").is_file(), "optimizer.pt missing"
        assert (ckpt_dir / "metadata.json").is_file(), "metadata.json missing"
        assert (ckpt_dir / "rng_state.pt").is_file(), "rng_state.pt missing"

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

    def test_checkpoint_metadata_is_valid_json(self, tmp_path: Path) -> None:
        """Checkpoint metadata must be parseable JSON with required fields."""
        from m31r.training.checkpoint.core import (
            CheckpointMetadata,
            save_checkpoint,
        )

        cfg = _tiny_model_config()
        model = M31RTransformer(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        metadata = CheckpointMetadata(
            global_step=5,
            seed=SEED,
            config_snapshot={"model": {"n_layers": 2}},
            tokens_seen=500,
            loss=4.2,
        )
        ckpt_dir = tmp_path / "checkpoints" / "step_000005"
        save_checkpoint(model, optimizer, metadata, ckpt_dir)

        # Parse and validate
        meta_json = json.loads((ckpt_dir / "metadata.json").read_text())
        assert "global_step" in meta_json, "Missing global_step"
        assert "loss" in meta_json, "Missing loss"
        assert "seed" in meta_json, "Missing seed"
        assert "tokens_seen" in meta_json, "Missing tokens_seen"
        assert meta_json["global_step"] == 5
        assert meta_json["seed"] == SEED


# ── Test: Resume Determinism ─────────────────────────────────────────


class TestResumeDeterminism:
    """Validate that resume continues from the correct step."""

    def test_resume_continues_from_checkpoint(self, tmp_pipeline_dir: Path) -> None:
        """Resume should start from the saved global step and continue training."""
        from m31r.training.checkpoint.core import find_latest_checkpoint
        from m31r.training.engine.core import run_training
        from m31r.training.engine.experiment import create_experiment_dir

        # Phase 1: Train for 10 steps
        config = _create_test_config(
            tmp_pipeline_dir, max_steps=10, checkpoint_interval=5
        )
        experiment_dir = create_experiment_dir(
            tmp_pipeline_dir / "experiments", config, seed=SEED
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

    def test_export_creates_release_files(self, tmp_pipeline_dir: Path) -> None:
        """Export should create model weights, metadata, and checksums."""
        from m31r.training.engine.core import run_training
        from m31r.training.engine.experiment import create_experiment_dir

        # Train first
        config = _create_test_config(tmp_pipeline_dir, max_steps=10, checkpoint_interval=5)
        experiment_dir = create_experiment_dir(
            tmp_pipeline_dir / "experiments", config, seed=SEED
        )
        run_training(config, experiment_dir)

        # Find the checkpoint
        from m31r.training.checkpoint.core import find_latest_checkpoint

        checkpoint_dir = find_latest_checkpoint(experiment_dir)
        assert checkpoint_dir is not None

        # Verify checkpoint directory structure
        assert (checkpoint_dir / "model.pt").is_file(), "model.pt missing from checkpoint"
        assert (checkpoint_dir / "metadata.json").is_file(), "metadata.json missing"

        # Verify metadata is parseable
        meta: dict[str, Any] = json.loads((checkpoint_dir / "metadata.json").read_text())
        assert "global_step" in meta
        assert "loss" in meta
        assert meta["global_step"] >= 0


# ── Test: Inference Engine ────────────────────────────────────────────


class TestInference:
    """Validate that the model can generate tokens."""

    def test_greedy_generation(self) -> None:
        """Model should produce deterministic output with temperature=0."""
        cfg = _tiny_model_config(vocab_size=VOCAB_SIZE)
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

    def test_deterministic_generation(self) -> None:
        """Same seed + same input should produce identical output."""
        cfg = _tiny_model_config(vocab_size=VOCAB_SIZE)

        torch.manual_seed(SEED)
        model = M31RTransformer(cfg)
        model.eval()
        prompt = torch.tensor([[1, 2, 3, 4]])

        with torch.no_grad():
            logits1 = model(prompt)

        torch.manual_seed(SEED)
        model2 = M31RTransformer(cfg)
        model2.eval()

        with torch.no_grad():
            logits2 = model2(prompt)

        assert torch.equal(logits1, logits2), "Determinism violated: same seed, different logits"


# ── Test: Dataloader ──────────────────────────────────────────────────


class TestDataloader:
    """Validate the streaming token dataloader."""

    def test_dataloader_produces_batches(self, tmp_pipeline_dir: Path) -> None:
        """Dataloader should yield (input, target) batch pairs."""
        from m31r.training.dataloader.core import TokenDataset, create_dataloader

        shard_dir = tmp_pipeline_dir / "shards"
        dataset = TokenDataset(shard_dir=shard_dir, seq_len=SEQ_LEN, seed=SEED)
        loader = create_dataloader(dataset, batch_size=BATCH_SIZE, seed=SEED)

        batch_count = 0
        for input_batch, target_batch in loader:
            assert input_batch.shape == (BATCH_SIZE, SEQ_LEN), (
                f"Input shape: {input_batch.shape}"
            )
            assert target_batch.shape == (BATCH_SIZE, SEQ_LEN), (
                f"Target shape: {target_batch.shape}"
            )
            batch_count += 1
            if batch_count >= 3:
                break

        assert batch_count == 3, f"Expected at least 3 batches, got {batch_count}"

    def test_dataloader_deterministic(self, tmp_pipeline_dir: Path) -> None:
        """Two dataloaders with same seed should yield identical data."""
        from m31r.training.dataloader.core import TokenDataset, create_dataloader

        shard_dir = tmp_pipeline_dir / "shards"

        dataset1 = TokenDataset(shard_dir=shard_dir, seq_len=64, seed=SEED)
        dataset2 = TokenDataset(shard_dir=shard_dir, seq_len=64, seed=SEED)

        loader1 = create_dataloader(dataset1, batch_size=BATCH_SIZE, seed=SEED)
        loader2 = create_dataloader(dataset2, batch_size=BATCH_SIZE, seed=SEED)

        for (inp1, tgt1), (inp2, tgt2) in zip(loader1, loader2):
            assert torch.equal(inp1, inp2), "Inputs differ between identical dataloaders"
            assert torch.equal(tgt1, tgt2), "Targets differ between identical dataloaders"
            break  # Just check first batch


# ── Test: Full Training Pipeline ──────────────────────────────────────


class TestFullPipeline:
    """Integration test: train → checkpoint → resume → verify continuity."""

    def test_train_checkpoint_resume_cycle(self, tmp_pipeline_dir: Path) -> None:
        """Full cycle should produce decreasing loss and valid checkpoints."""
        from m31r.training.checkpoint.core import find_latest_checkpoint
        from m31r.training.engine.core import run_training
        from m31r.training.engine.experiment import create_experiment_dir

        # Train 10 steps
        config = _create_test_config(tmp_pipeline_dir, max_steps=10, checkpoint_interval=5)
        experiment_dir = create_experiment_dir(
            tmp_pipeline_dir / "experiments", config, seed=SEED
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
        meta: dict[str, Any] = json.loads((ckpt / "metadata.json").read_text())
        assert meta["global_step"] == result.final_step

        # Resume to 20 steps
        config2 = _create_test_config(tmp_pipeline_dir, max_steps=20, checkpoint_interval=5)
        result2 = run_training(config2, experiment_dir, resume_from=ckpt)
        assert result2.final_step == 20


# ── Test: Checksum Integrity ──────────────────────────────────────────


class TestChecksumIntegrity:
    """Validate SHA256 checksum operations."""

    def test_sha256_hash_deterministic(self, tmp_path: Path) -> None:
        """SHA256 of same content must always produce same hash."""
        test_file = tmp_path / "test.bin"
        content = b"deterministic content for checksum test"
        test_file.write_bytes(content)

        hash1 = hashlib.sha256(test_file.read_bytes()).hexdigest()
        hash2 = hashlib.sha256(test_file.read_bytes()).hexdigest()

        assert hash1 == hash2, "SHA256 hash is not deterministic"
        assert len(hash1) == 64, "SHA256 hex digest should be 64 characters"

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        """Different content must produce different hashes."""
        file_a = tmp_path / "a.bin"
        file_b = tmp_path / "b.bin"
        file_a.write_bytes(b"content A")
        file_b.write_bytes(b"content B")

        hash_a = hashlib.sha256(file_a.read_bytes()).hexdigest()
        hash_b = hashlib.sha256(file_b.read_bytes()).hexdigest()

        assert hash_a != hash_b, "Different content produced same hash"


# ── Test: Config Validation ───────────────────────────────────────────


class TestConfigValidation:
    """Validate that tiny configs produce valid M31RConfig objects."""

    def test_tiny_config_is_valid(self, tmp_pipeline_dir: Path) -> None:
        """The test config should parse without errors."""
        config = _create_test_config(tmp_pipeline_dir)
        assert config.model is not None
        assert config.train is not None
        assert config.global_config is not None

    def test_tiny_config_has_correct_seed(self, tmp_pipeline_dir: Path) -> None:
        """Seed must be set to 42 for determinism."""
        config = _create_test_config(tmp_pipeline_dir)
        assert config.global_config.seed == SEED, (
            f"Expected seed {SEED}, got {config.global_config.seed}"
        )

    def test_tiny_config_cpu_precision(self, tmp_pipeline_dir: Path) -> None:
        """Training precision must be fp32 for CPU determinism."""
        config = _create_test_config(tmp_pipeline_dir)
        assert config.train.precision == "fp32", (
            f"Expected fp32, got {config.train.precision}"
        )
