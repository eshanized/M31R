# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Serving metrics collector.

Tracks the numbers that tell you whether inference is fast enough:
ms per token, tokens per second, peak memory, and per-request timing.
Everything stays local — no telemetry, no external reporting.
"""

import logging
import time
from dataclasses import dataclass, field

import torch

from m31r.logging.logger import get_logger

logger: logging.Logger = get_logger(__name__)


@dataclass
class RequestMetrics:
    """Stats captured for a single generation request."""

    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_time_ms: float = 0.0
    first_token_ms: float = 0.0
    tokens_per_second: float = 0.0
    peak_memory_mb: float = 0.0


class ServingMetrics:
    """
    Accumulates performance metrics across requests.

    This isn't fancy — it just keeps running totals and lets you
    query averages. The idea is to catch latency regressions early
    without needing a full monitoring stack.
    """

    def __init__(self) -> None:
        self._requests: list[RequestMetrics] = []
        self._start_time: float = time.monotonic()

    @property
    def total_requests(self) -> int:
        return len(self._requests)

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self._start_time

    def record(self, metrics: RequestMetrics) -> None:
        """Save metrics from a completed request and log them."""
        self._requests.append(metrics)
        logger.info(
            "Request completed",
            extra={
                "prompt_tokens": metrics.prompt_tokens,
                "generated_tokens": metrics.generated_tokens,
                "total_time_ms": round(metrics.total_time_ms, 2),
                "tokens_per_second": round(metrics.tokens_per_second, 2),
                "peak_memory_mb": round(metrics.peak_memory_mb, 2),
            },
        )

    def average_tokens_per_second(self) -> float:
        if not self._requests:
            return 0.0
        total = sum(r.tokens_per_second for r in self._requests)
        return total / len(self._requests)

    def average_ms_per_token(self) -> float:
        tps = self.average_tokens_per_second()
        if tps <= 0:
            return 0.0
        return 1000.0 / tps

    def peak_memory_mb(self) -> float:
        if not self._requests:
            return 0.0
        return max(r.peak_memory_mb for r in self._requests)

    def summary(self) -> dict[str, object]:
        """Structured summary suitable for logging or API responses."""
        return {
            "total_requests": self.total_requests,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "avg_tokens_per_second": round(self.average_tokens_per_second(), 2),
            "avg_ms_per_token": round(self.average_ms_per_token(), 2),
            "peak_memory_mb": round(self.peak_memory_mb(), 2),
        }

    @staticmethod
    def get_gpu_memory_mb() -> float:
        """
        Ask PyTorch how much GPU memory we're using right now.

        Returns 0.0 if CUDA isn't available — this is fine, it just
        means we're running on CPU and GPU memory isn't relevant.
        """
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        return 0.0

    @staticmethod
    def get_memory_allocated_mb() -> float:
        """Current GPU memory allocated (not peak). Zero on CPU."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0
