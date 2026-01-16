# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Core training engine for M31R.

Per 07_TRAINING_ARCHITECTURE.md §4, the training loop must be explicit:
  1. Forward pass
  2. Loss computation (cross-entropy for next-token prediction)
  3. Backward pass (scaled for gradient accumulation)
  4. Gradient clipping (after accumulation completes)
  5. Optimizer step
  6. Zero gradients
  7. Update learning rate schedule
  8. Log metrics
  9. Checkpoint at intervals

No hidden autograd magic, no trainer frameworks, no silent side effects.
Every step is explicit, traceable, and deterministic.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn

from m31r.config.schema import M31RConfig
from m31r.logging.logger import get_logger
from m31r.model.transformer import M31RTransformer, TransformerModelConfig
from m31r.training.checkpoint.core import (
    CheckpointMetadata,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from m31r.training.dataloader.core import TokenDataset, create_dataloader
from m31r.training.metrics.core import MetricsTracker
from m31r.training.optimizer.core import create_optimizer
from m31r.training.scheduler.core import get_learning_rate, set_learning_rate

logger: logging.Logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    """Final result of a training run."""

    final_step: int
    final_loss: float
    total_tokens: int
    experiment_dir: str
    checkpoint_path: str


def _select_device() -> torch.device:
    """Select the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _select_dtype(precision: str) -> torch.dtype:
    """Map precision string to torch dtype."""
    dtypes = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return dtypes.get(precision, torch.float32)


def _build_model_config(config: M31RConfig) -> TransformerModelConfig:
    """Construct a TransformerModelConfig from the validated M31R config."""
    model_cfg = config.model
    if model_cfg is None:
        raise RuntimeError("Model config is required for training")

    return TransformerModelConfig(
        vocab_size=model_cfg.vocab_size,
        dim=model_cfg.hidden_size,
        n_layers=model_cfg.n_layers,
        n_heads=model_cfg.n_heads,
        head_dim=model_cfg.head_dim,
        max_seq_len=model_cfg.context_length,
        dropout=model_cfg.dropout,
        norm_eps=model_cfg.norm_eps,
        rope_theta=model_cfg.rope_theta,
        init_std=model_cfg.init_std,
        seed=config.global_config.seed,
    )


def run_training(
    config: M31RConfig,
    experiment_dir: Path,
    resume_from: Path | None = None,
) -> TrainingResult:
    """
    Execute the full training loop.

    This is the core function that implements the spec's explicit training
    loop. Every operation is visible, every side effect is logged, and the
    entire execution is deterministic given the same config and data.

    Args:
        config: Validated M31R configuration with model and train sections.
        experiment_dir: Experiment output directory.
        resume_from: Optional checkpoint directory to resume from.

    Returns:
        TrainingResult with final metrics and paths.

    Raises:
        RuntimeError: If train or model config sections are missing.
    """
    train_cfg = config.train
    if train_cfg is None:
        raise RuntimeError("Training config is required")

    device = _select_device()
    dtype = _select_dtype(train_cfg.precision)

    logger.info(
        "Training setup",
        extra={
            "device": str(device),
            "precision": train_cfg.precision,
            "max_steps": train_cfg.max_steps,
            "batch_size": train_cfg.batch_size,
            "grad_accum": train_cfg.gradient_accumulation_steps,
        },
    )

    # ── Build model ──
    model_config = _build_model_config(config)
    model = M31RTransformer(model_config)
    model = model.to(device)

    param_count = model.count_parameters()
    logger.info(
        "Model created",
        extra={"parameters": param_count, "layers": model_config.n_layers},
    )

    # ── Build optimizer ──
    optimizer = create_optimizer(model, train_cfg)

    # ── Resume from checkpoint ──
    global_step = 0
    tokens_seen = 0
    latest_loss = 0.0

    if resume_from is not None:
        metadata = load_checkpoint(resume_from, model, optimizer, device)
        global_step = metadata.global_step
        tokens_seen = metadata.tokens_seen
        latest_loss = metadata.loss
        logger.info(
            "Resumed from checkpoint",
            extra={"step": global_step, "tokens_seen": tokens_seen},
        )

    # ── Build dataloader ──
    project_root = experiment_dir.parent.parent  # experiments/<run>/.. = project root
    shard_dir = project_root / train_cfg.dataset_directory
    dataset = TokenDataset(
        shard_dir=shard_dir,
        seq_len=model_config.max_seq_len,
        seed=config.global_config.seed,
        start_offset=tokens_seen // model_config.max_seq_len if tokens_seen > 0 else 0,
    )

    # ── Metrics tracker ──
    metrics_tracker = MetricsTracker(log_interval=train_cfg.log_interval)

    # ── Loss function ──
    loss_fn = nn.CrossEntropyLoss()

    # ── Mixed precision context ──
    use_amp = train_cfg.precision in ("bf16", "fp16") and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=(train_cfg.precision == "fp16" and use_amp))

    # ── Training loop ──
    model.train()
    optimizer.zero_grad()
    accumulation_step = 0
    checkpoint_path = ""

    dataloader: Iterator[tuple[torch.Tensor, torch.Tensor]] = create_dataloader(
        dataset,
        train_cfg.batch_size,
        seed=config.global_config.seed,
    )

    for input_batch, target_batch in dataloader:
        if global_step >= train_cfg.max_steps:
            break

        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        tokens_in_batch = input_batch.numel()
        metrics_tracker.begin_step(tokens_in_batch)

        # ── Step 1-2: Forward pass + loss ──
        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            logits = model(input_batch)
            # Reshape for cross-entropy: (batch * seq_len, vocab_size) vs (batch * seq_len,)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                target_batch.view(-1),
            )
            # Scale loss for gradient accumulation
            scaled_loss = loss / train_cfg.gradient_accumulation_steps

        # ── Step 3: Backward pass ──
        scaler.scale(scaled_loss).backward()
        metrics_tracker.record_loss(loss.item())

        accumulation_step += 1
        tokens_seen += tokens_in_batch

        # ── Steps 4-7: Gradient clipping + optimizer step (after full accumulation) ──
        if accumulation_step % train_cfg.gradient_accumulation_steps == 0:
            # Unscale before clipping
            scaler.unscale_(optimizer)

            # Step 4: Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                train_cfg.grad_clip,
            ).item()

            # Step 5: Optimizer step
            scaler.step(optimizer)
            scaler.update()

            # Step 6: Zero gradients
            optimizer.zero_grad()

            # Step 7: Update learning rate
            lr = get_learning_rate(
                step=global_step,
                max_lr=train_cfg.learning_rate,
                min_lr=train_cfg.min_learning_rate,
                warmup_steps=train_cfg.warmup_steps,
                max_steps=train_cfg.max_steps,
            )
            set_learning_rate(optimizer, lr)

            # ── Step 8: Log metrics ──
            step_metrics = metrics_tracker.end_step(
                step=global_step,
                learning_rate=lr,
                grad_norm=grad_norm,
                tokens_seen=tokens_seen,
            )
            latest_loss = step_metrics.loss

            # ── Step 9: Checkpoint ──
            if global_step > 0 and global_step % train_cfg.checkpoint_interval == 0:
                ckpt_dir = experiment_dir / "checkpoints" / f"step_{global_step:06d}"
                config_snapshot = config.model_dump(by_alias=True)
                metadata = CheckpointMetadata(
                    global_step=global_step,
                    seed=config.global_config.seed,
                    config_snapshot=config_snapshot,
                    tokens_seen=tokens_seen,
                    loss=latest_loss,
                )
                checkpoint_path = str(save_checkpoint(model, optimizer, metadata, ckpt_dir))

            global_step += 1

    # ── Final checkpoint ──
    ckpt_dir = experiment_dir / "checkpoints" / f"step_{global_step:06d}"
    config_snapshot = config.model_dump(by_alias=True)
    metadata = CheckpointMetadata(
        global_step=global_step,
        seed=config.global_config.seed,
        config_snapshot=config_snapshot,
        tokens_seen=tokens_seen,
        loss=latest_loss,
    )
    checkpoint_path = str(save_checkpoint(model, optimizer, metadata, ckpt_dir))

    logger.info(
        "Training complete",
        extra={
            "final_step": global_step,
            "final_loss": latest_loss,
            "total_tokens": tokens_seen,
        },
    )

    return TrainingResult(
        final_step=global_step,
        final_loss=latest_loss,
        total_tokens=tokens_seen,
        experiment_dir=str(experiment_dir),
        checkpoint_path=checkpoint_path,
    )
