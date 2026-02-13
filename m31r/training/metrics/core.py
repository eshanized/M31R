# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Structured training metrics for M31R.

Per 20_OBSERVABILITY_AND_LOGGING.md §7:
  Mandatory training metrics:
    - step, loss, learning_rate, tokens_per_sec
    - gradient_norm, memory_usage_mb
    - checkpoint events

All metrics are logged as structured JSON via the standard logger.
This module collects and formats them.
"""

import logging
import time
from dataclasses import dataclass, field

import torch

from m31r.logging.logger import get_logger

logger: logging.Logger = get_logger(__name__)


@dataclass
class StepMetrics:
    """Metrics collected for a single training step."""

    step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    tokens_per_sec: float = 0.0
    memory_usage_mb: float = 0.0
    tokens_seen: int = 0


@dataclass
class MetricsTracker:
    """
    Accumulates and logs training metrics.

    Tracks per-step values and maintains running averages.
    Logs are emitted at configured intervals via the structured logger.

    Args:
        log_interval: Log metrics every N steps.
    """

    log_interval: int = 10
    _loss_accumulator: float = field(default=0.0, init=False)
    _accumulation_count: int = field(default=0, init=False)
    _step_start_time: float = field(default=0.0, init=False)
    _step_tokens: int = field(default=0, init=False)

    def begin_step(self, tokens_in_step: int) -> None:
        """Mark the beginning of a training step for timing."""
        self._step_start_time = time.monotonic()
        self._step_tokens = tokens_in_step

    def record_loss(self, loss: float) -> None:
        """Accumulate a loss value (for gradient accumulation)."""
        self._loss_accumulator += loss
        self._accumulation_count += 1

    def end_step(
        self,
        step: int,
        learning_rate: float,
        grad_norm: float,
        tokens_seen: int,
    ) -> StepMetrics:
        """
        Finalize step metrics and optionally log them.

        Args:
            step: Current global step.
            learning_rate: Current LR.
            grad_norm: Gradient norm before clipping.
            tokens_seen: Cumulative tokens processed.

        Returns:
            StepMetrics for this step.
        """
        elapsed = time.monotonic() - self._step_start_time
        tokens_per_sec = self._step_tokens / elapsed if elapsed > 0 else 0.0

        avg_loss = (
            self._loss_accumulator / self._accumulation_count
            if self._accumulation_count > 0
            else 0.0
        )

        # Memory usage
        memory_mb = 0.0
        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        metrics = StepMetrics(
            step=step,
            loss=avg_loss,
            learning_rate=learning_rate,
            grad_norm=grad_norm,
            tokens_per_sec=tokens_per_sec,
            memory_usage_mb=memory_mb,
            tokens_seen=tokens_seen,
        )

        # Reset accumulators
        self._loss_accumulator = 0.0
        self._accumulation_count = 0

        # Log at configured interval
        if step % self.log_interval == 0:
            self._log_metrics(metrics)

        return metrics

    def _log_metrics(self, metrics: StepMetrics) -> None:
        """Emit structured log entry for training metrics."""
        log_data = {
            "step": metrics.step,
            "loss": round(metrics.loss, 6),
            "lr": metrics.learning_rate,
            "grad_norm": round(metrics.grad_norm, 4),
            "tokens_per_sec": round(metrics.tokens_per_sec, 1),
            "memory_mb": round(metrics.memory_usage_mb, 1),
            "tokens_seen": metrics.tokens_seen,
        }

        logger.info("Training step", extra=log_data)

        # Broadcast to dashboard if available
        try:
            import asyncio
            from m31r.dashboard import broadcast_metrics

            # Run async broadcast in sync context
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(broadcast_metrics(log_data))
            except RuntimeError:
                # No event loop running, skip broadcast
                pass
        except ImportError:
            # Dashboard not available
            pass
