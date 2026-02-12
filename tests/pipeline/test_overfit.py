# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Overfit sanity test suite for M31R.

SYSTEM CORRECTNESS TEST. Proves the model + trainer can learn
by intentionally overfitting a tiny synthetic dataset.

If these tests pass, the training engine is correct.
If they fail, training is broken.

Validates:
  - model can overfit repeated data (final loss < 0.5)
  - checkpoints are saved with correct metadata
  - trained model generates coherent output (not random noise)
  - training is deterministic (same seed → same loss)

Run with:
    python -m pytest tests/pipeline/test_overfit.py -v --tb=short -x
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from m31r.config.schema import M31RConfig
from m31r.model.transformer import M31RTransformer, TransformerModelConfig


# ── Constants ─────────────────────────────────────────────────────────

VOCAB_SIZE: int = 256
SEQ_LEN: int = 128
BATCH_SIZE: int = 4
SEED: int = 42
OVERFIT_STEPS: int = 500
LOSS_THRESHOLD: float = 0.5
HIGH_LR: float = 1e-2


# ── Synthetic Data ────────────────────────────────────────────────────

RUST_FUNCTIONS: list[str] = [
    "fn add(a: i32, b: i32) -> i32 { a + b }",
    "fn sub(a: i32, b: i32) -> i32 { a - b }",
    "fn mul(a: i32, b: i32) -> i32 { a * b }",
    "fn div(a: i32, b: i32) -> i32 { a / b }",
    "fn square(x: i32) -> i32 { x * x }",
    "fn double(x: i32) -> i32 { x + x }",
    "fn negate(x: i32) -> i32 { -x }",
    "fn max(a: i32, b: i32) -> i32 { if a > b { a } else { b } }",
    "fn min(a: i32, b: i32) -> i32 { if a < b { a } else { b } }",
    "fn abs(x: i32) -> i32 { if x < 0 { -x } else { x } }",
]


# ── Helpers ───────────────────────────────────────────────────────────


def _overfit_model_config() -> TransformerModelConfig:
    """Create the minimal model config for overfit testing."""
    return TransformerModelConfig(
        vocab_size=VOCAB_SIZE,
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


def _create_repeated_shards(
    shard_dir: Path,
    num_shards: int = 4,
    tokens_per_shard: int = 80_000,
) -> None:
    """
    Create deterministic tokenized shards with cyclically repeated data.

    Uses a simple modular encoding: each character maps to its ordinal
    value mod VOCAB_SIZE. This is NOT a real tokenizer — it's a
    deterministic encoder that guarantees the same input → same output.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_files: list[str] = []

    corpus = "\n".join(RUST_FUNCTIONS * 200)
    # Simple character-level encoding: ord(c) % VOCAB_SIZE
    all_token_ids = [ord(c) % VOCAB_SIZE for c in corpus]

    for i in range(num_shards):
        # Cycle the same tokens — intentional for overfit
        start = (i * tokens_per_shard) % len(all_token_ids)
        tokens: list[int] = []
        for j in range(tokens_per_shard):
            tokens.append(all_token_ids[(start + j) % len(all_token_ids)])

        shard_name = f"shard_{i:04d}.json"
        (shard_dir / shard_name).write_text(
            json.dumps({"tokens": tokens}), encoding="utf-8"
        )
        shard_files.append(shard_name)

    manifest: dict[str, Any] = {
        "version": "1.0.0",
        "total_tokens": num_shards * tokens_per_shard,
        "num_shards": num_shards,
        "shards": shard_files,
    }
    (shard_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _create_overfit_config(
    tmp_dir: Path,
    max_steps: int = OVERFIT_STEPS,
) -> M31RConfig:
    """Build a config tuned for aggressive overfitting."""
    config_dict: dict[str, Any] = {
        "global": {
            "config_version": "1.0.0",
            "project_name": "m31r-overfit-test",
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
            "vocab_size": VOCAB_SIZE,
        },
        "train": {
            "config_version": "1.0.0",
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": 1,
            "max_steps": max_steps,
            "learning_rate": HIGH_LR,
            "min_learning_rate": 1e-4,
            "weight_decay": 0.01,
            "beta1": 0.9,
            "beta2": 0.95,
            "grad_clip": 1.0,
            "warmup_steps": 5,
            "precision": "fp32",
            "checkpoint_interval": 100,
            "log_interval": 50,
            "dataset_directory": str(tmp_dir / "shards"),
            "tokenizer_directory": str(tmp_dir / "tokenizer"),
        },
    }
    return M31RConfig.model_validate(config_dict)


@pytest.fixture
def overfit_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with synthetic overfit data."""
    shard_dir = tmp_path / "shards"
    _create_repeated_shards(shard_dir, num_shards=4, tokens_per_shard=80_000)
    (tmp_path / "tokenizer").mkdir(parents=True, exist_ok=True)
    (tmp_path / "experiments").mkdir(parents=True, exist_ok=True)
    return tmp_path


# ── Test: Overfit Loss ────────────────────────────────────────────────


class TestOverfitLoss:
    """Prove the model can learn by driving loss below threshold."""

    def test_loss_below_threshold(self, overfit_dir: Path) -> None:
        """
        Train 500 steps with high LR on repeated data.
        Final loss MUST be < 0.5.
        If not, the training engine is broken.
        """
        from m31r.training.engine.core import run_training
        from m31r.training.engine.experiment import create_experiment_dir

        config = _create_overfit_config(overfit_dir, max_steps=OVERFIT_STEPS)
        experiment_dir = create_experiment_dir(
            overfit_dir / "experiments", config, seed=SEED
        )

        result = run_training(config, experiment_dir)

        assert result.final_step == OVERFIT_STEPS, (
            f"Expected {OVERFIT_STEPS} steps, got {result.final_step}"
        )
        assert result.final_loss < LOSS_THRESHOLD, (
            f"SYSTEM BROKEN: Loss {result.final_loss:.4f} >= {LOSS_THRESHOLD} "
            f"after {OVERFIT_STEPS} steps — model did NOT learn"
        )
        assert result.final_loss > 0.0, "Loss must be positive"
        assert result.total_tokens > 0, "Must have processed tokens"

    def test_loss_decreases_significantly(self, overfit_dir: Path) -> None:
        """
        Loss must decrease by at least 50% from its initial value.
        A model that doesn't decrease loss is not learning.
        """
        from m31r.training.engine.core import run_training
        from m31r.training.engine.experiment import create_experiment_dir

        config = _create_overfit_config(overfit_dir, max_steps=OVERFIT_STEPS)
        experiment_dir = create_experiment_dir(
            overfit_dir / "experiments", config, seed=SEED
        )

        result = run_training(config, experiment_dir)

        # For vocab_size=256 random init, initial loss ≈ log(256) ≈ 5.5
        # After overfit, should be well below 2.75 (50% of 5.5)
        initial_expected = 5.0  # conservative estimate
        assert result.final_loss < initial_expected * 0.5, (
            f"Loss {result.final_loss:.4f} did not decrease enough "
            f"(expected < {initial_expected * 0.5:.2f})"
        )


# ── Test: Checkpoint Integrity ────────────────────────────────────────


class TestOverfitCheckpoint:
    """Verify checkpoints are saved correctly during overfit training."""

    def test_checkpoint_saved_with_metadata(self, overfit_dir: Path) -> None:
        """Checkpoint must contain model.pt + metadata.json with correct fields."""
        from m31r.training.checkpoint.core import find_latest_checkpoint
        from m31r.training.engine.core import run_training
        from m31r.training.engine.experiment import create_experiment_dir

        config = _create_overfit_config(overfit_dir, max_steps=OVERFIT_STEPS)
        experiment_dir = create_experiment_dir(
            overfit_dir / "experiments", config, seed=SEED
        )

        result = run_training(config, experiment_dir)

        # Find checkpoint
        ckpt = find_latest_checkpoint(experiment_dir)
        assert ckpt is not None, "No checkpoint found after training"
        assert (ckpt / "model.pt").is_file(), "model.pt missing from checkpoint"
        assert (ckpt / "metadata.json").is_file(), "metadata.json missing"

        # Validate metadata
        meta: dict[str, Any] = json.loads((ckpt / "metadata.json").read_text())
        assert "global_step" in meta, "Missing global_step in metadata"
        assert "loss" in meta, "Missing loss in metadata"
        assert "seed" in meta, "Missing seed in metadata"
        assert meta["seed"] == SEED, f"Seed mismatch: {meta['seed']} != {SEED}"
        assert meta["loss"] < LOSS_THRESHOLD, (
            f"Checkpoint loss {meta['loss']:.4f} >= {LOSS_THRESHOLD}"
        )


# ── Test: Generation Quality ─────────────────────────────────────────


class TestOverfitGeneration:
    """Verify trained model produces coherent output, not random noise."""

    def test_generation_not_random(self, overfit_dir: Path) -> None:
        """
        After overfitting, greedy decode from training-like tokens
        should produce consistent, non-random output.
        """
        from m31r.training.engine.core import run_training
        from m31r.training.engine.experiment import create_experiment_dir

        config = _create_overfit_config(overfit_dir, max_steps=OVERFIT_STEPS)
        experiment_dir = create_experiment_dir(
            overfit_dir / "experiments", config, seed=SEED
        )

        result = run_training(config, experiment_dir)
        assert result.final_loss < LOSS_THRESHOLD

        # Load trained model from the path stored in result
        from m31r.training.checkpoint.core import find_latest_checkpoint

        ckpt = find_latest_checkpoint(experiment_dir)
        assert ckpt is not None

        model_cfg = _overfit_model_config()
        model = M31RTransformer(model_cfg)
        state = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()

        # Encode prompt using the same character-level scheme as training
        prompt = "fn add("
        prompt_ids = [ord(c) % VOCAB_SIZE for c in prompt]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)

        # Greedy decode 32 tokens
        with torch.no_grad():
            for _ in range(32):
                logits = model(input_ids[:, -SEQ_LEN:])
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)

        generated_ids = input_ids[0].tolist()[len(prompt_ids):]

        # The model should NOT produce uniform random output.
        # An overfitted model will have low entropy in its output distribution.
        # Check that at least some tokens repeat (sign of pattern learning).
        unique_tokens = len(set(generated_ids))
        total_tokens = len(generated_ids)
        uniqueness_ratio = unique_tokens / total_tokens

        assert uniqueness_ratio < 0.95, (
            f"Output looks random: {unique_tokens}/{total_tokens} unique tokens "
            f"(ratio {uniqueness_ratio:.2f}) — model did not learn patterns"
        )

    def test_deterministic_generation(self, overfit_dir: Path) -> None:
        """Same seed + same weights → same generated output."""
        from m31r.training.engine.core import run_training
        from m31r.training.engine.experiment import create_experiment_dir

        config = _create_overfit_config(overfit_dir, max_steps=200)
        experiment_dir = create_experiment_dir(
            overfit_dir / "experiments", config, seed=SEED
        )

        run_training(config, experiment_dir)

        from m31r.training.checkpoint.core import find_latest_checkpoint

        ckpt = find_latest_checkpoint(experiment_dir)
        assert ckpt is not None

        model_cfg = _overfit_model_config()

        # Run 1
        model1 = M31RTransformer(model_cfg)
        state = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=True)
        model1.load_state_dict(state)
        model1.eval()

        prompt_ids = [ord(c) % VOCAB_SIZE for c in "fn add("]
        input1 = torch.tensor([prompt_ids], dtype=torch.long)
        with torch.no_grad():
            for _ in range(16):
                logits = model1(input1[:, -SEQ_LEN:])
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                input1 = torch.cat([input1, next_token], dim=1)

        # Run 2 — should produce identical output
        model2 = M31RTransformer(model_cfg)
        model2.load_state_dict(state)
        model2.eval()

        input2 = torch.tensor([prompt_ids], dtype=torch.long)
        with torch.no_grad():
            for _ in range(16):
                logits = model2(input2[:, -SEQ_LEN:])
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                input2 = torch.cat([input2, next_token], dim=1)

        assert torch.equal(input1, input2), (
            "Determinism violated: same weights + same input → different output"
        )


# ── Test: Training Determinism ────────────────────────────────────────


class TestOverfitDeterminism:
    """Two identical training runs must produce identical results."""

    def test_same_seed_same_loss(self, tmp_path: Path) -> None:
        """Identical config + seed → identical final loss (bit-exact)."""
        from m31r.training.engine.core import run_training
        from m31r.training.engine.experiment import create_experiment_dir

        losses: list[float] = []

        for trial in range(2):
            trial_dir = tmp_path / f"trial_{trial}"
            shard_dir = trial_dir / "shards"
            _create_repeated_shards(shard_dir, num_shards=2, tokens_per_shard=40_000)
            (trial_dir / "tokenizer").mkdir(parents=True, exist_ok=True)
            (trial_dir / "experiments").mkdir(parents=True, exist_ok=True)

            config = _create_overfit_config(trial_dir, max_steps=50)
            experiment_dir = create_experiment_dir(
                trial_dir / "experiments", config, seed=SEED
            )

            result = run_training(config, experiment_dir)
            losses.append(result.final_loss)

        assert losses[0] == losses[1], (
            f"Determinism violated: trial 0 loss={losses[0]:.6f}, "
            f"trial 1 loss={losses[1]:.6f}"
        )
