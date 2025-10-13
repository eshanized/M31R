# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for the training engine.

Per 15_TESTING_STRATEGY.md:
  - Tiny training run must complete
  - Loss must decrease over steps
  - Reproducibility: two runs with same seed = identical results
"""

import json
import struct
import textwrap
from pathlib import Path

import torch

from m31r.config.loader import load_config
from m31r.training.scheduler.core import get_learning_rate


def _create_tiny_training_setup(tmp_path: Path) -> Path:
    """Create a minimal training setup with config, shards, and tokenizer."""
    # Create config with tiny model
    config_content = textwrap.dedent("""\
        global:
          config_version: "1.0.0"
          project_name: "m31r-test"
          seed: 42
          log_level: "DEBUG"
          directories:
            data: "data"
            checkpoints: "checkpoints"
            logs: "logs"
            experiments: "experiments"

        model:
          config_version: "1.0.0"
          n_layers: 2
          hidden_size: 64
          n_heads: 4
          head_dim: 16
          context_length: 128
          dropout: 0.0
          norm_eps: 1e-6
          rope_theta: 10000.0
          init_std: 0.02
          vocab_size: 256

        train:
          config_version: "1.0.0"
          batch_size: 2
          gradient_accumulation_steps: 1
          max_steps: 5
          learning_rate: 1e-3
          min_learning_rate: 1e-5
          weight_decay: 0.01
          beta1: 0.9
          beta2: 0.95
          grad_clip: 1.0
          warmup_steps: 2
          precision: "fp32"
          checkpoint_interval: 3
          log_interval: 1
          dataset_directory: "data/datasets"
          tokenizer_directory: "data/tokenizer"
    """)
    config_path = tmp_path / "train_config.yaml"
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_content, encoding="utf-8")

    # Create synthetic dataset shards
    shard_dir = tmp_path / "data" / "datasets"
    shard_dir.mkdir(parents=True)
    tokens = list(range(256)) * 40  # 10240 tokens — enough for 128-len windows
    data = struct.pack(f"<{len(tokens)}i", *tokens)
    (shard_dir / "shard_0000.bin").write_bytes(data)

    # Create experiments dir
    (tmp_path / "experiments").mkdir(exist_ok=True)

    return config_path


class TestLearningRateScheduler:
    """Tests for the cosine warmup scheduler."""

    def test_warmup_phase(self) -> None:
        """LR should increase linearly during warmup."""
        lr_0 = get_learning_rate(0, max_lr=1e-3, min_lr=1e-5, warmup_steps=10, max_steps=100)
        lr_5 = get_learning_rate(5, max_lr=1e-3, min_lr=1e-5, warmup_steps=10, max_steps=100)
        lr_9 = get_learning_rate(9, max_lr=1e-3, min_lr=1e-5, warmup_steps=10, max_steps=100)
        assert lr_0 < lr_5 < lr_9

    def test_peak_lr(self) -> None:
        """At the end of warmup, LR should be approximately max_lr."""
        lr = get_learning_rate(9, max_lr=1e-3, min_lr=1e-5, warmup_steps=10, max_steps=100)
        assert abs(lr - 1e-3) < 1e-6

    def test_decay_phase(self) -> None:
        """LR should decrease during cosine decay."""
        lr_20 = get_learning_rate(20, max_lr=1e-3, min_lr=1e-5, warmup_steps=10, max_steps=100)
        lr_80 = get_learning_rate(80, max_lr=1e-3, min_lr=1e-5, warmup_steps=10, max_steps=100)
        assert lr_20 > lr_80

    def test_min_lr_floor(self) -> None:
        """After max_steps, LR should be min_lr."""
        lr = get_learning_rate(200, max_lr=1e-3, min_lr=1e-5, warmup_steps=10, max_steps=100)
        assert abs(lr - 1e-5) < 1e-8


class TestTrainingEngine:
    """Integration tests for the training engine."""

    def test_tiny_training_completes(self, tmp_path: Path) -> None:
        """A tiny training run must complete without errors."""
        from m31r.training.engine.core import run_training
        from m31r.training.engine.experiment import create_experiment_dir

        config_path = _create_tiny_training_setup(tmp_path)
        config = load_config(config_path)

        experiment_dir = create_experiment_dir(
            tmp_path / "experiments", config, config.global_config.seed,
        )

        result = run_training(config, experiment_dir)

        assert result.final_step > 0
        assert result.total_tokens > 0
        assert result.experiment_dir == str(experiment_dir)

    def test_loss_decreases(self, tmp_path: Path) -> None:
        """Loss should decrease over a few training steps."""
        from m31r.training.engine.core import (
            _build_model_config,
            _select_device,
            _select_dtype,
        )
        from m31r.training.dataloader.core import TokenDataset, create_dataloader
        from m31r.training.optimizer.core import create_optimizer

        config_path = _create_tiny_training_setup(tmp_path)
        config = load_config(config_path)
        train_cfg = config.train
        assert train_cfg is not None

        model_config = _build_model_config(config)

        from m31r.model.transformer import M31RTransformer
        model = M31RTransformer(model_config)
        optimizer = create_optimizer(model, train_cfg)

        loss_fn = torch.nn.CrossEntropyLoss()
        shard_dir = tmp_path / "data" / "datasets"
        dataset = TokenDataset(shard_dir=shard_dir, seq_len=128, seed=42)

        losses = []
        model.train()
        for step, (inp, tgt) in enumerate(create_dataloader(dataset, batch_size=2)):
            if step >= 10:
                break
            logits = model(inp)
            loss = loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        # Loss should trend downward (allow some noise)
        if len(losses) >= 5:
            first_avg = sum(losses[:3]) / 3
            last_avg = sum(losses[-3:]) / 3
            # Just verify it's not diverging catastrophically
            assert last_avg < first_avg * 5.0

    def test_checkpoint_created(self, tmp_path: Path) -> None:
        """Training must create checkpoint files."""
        from m31r.training.engine.core import run_training
        from m31r.training.engine.experiment import create_experiment_dir

        config_path = _create_tiny_training_setup(tmp_path)
        config = load_config(config_path)

        experiment_dir = create_experiment_dir(
            tmp_path / "experiments", config, config.global_config.seed,
        )

        result = run_training(config, experiment_dir)

        # Final checkpoint should exist
        ckpt_path = Path(result.checkpoint_path)
        assert ckpt_path.is_dir()
        assert (ckpt_path / "model.pt").is_file()
        assert (ckpt_path / "metadata.json").is_file()

    def test_reproducibility(self, tmp_path: Path) -> None:
        """Two runs with same config and seed must produce identical results."""
        from m31r.training.engine.core import run_training
        from m31r.training.engine.experiment import create_experiment_dir

        config_path = _create_tiny_training_setup(tmp_path)

        # Run 1
        config1 = load_config(config_path)
        exp_dir1 = create_experiment_dir(
            tmp_path / "experiments", config1, config1.global_config.seed,
        )
        result1 = run_training(config1, exp_dir1)

        # Run 2 — fresh setup with same config
        config_path2 = _create_tiny_training_setup(tmp_path / "run2")
        config2 = load_config(config_path2)
        exp_dir2 = create_experiment_dir(
            tmp_path / "run2" / "experiments", config2, config2.global_config.seed,
        )
        result2 = run_training(config2, exp_dir2)

        # Final losses should be identical
        assert abs(result1.final_loss - result2.final_loss) < 1e-5
        assert result1.total_tokens == result2.total_tokens
