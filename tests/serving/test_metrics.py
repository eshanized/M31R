# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""Tests for serving metrics tracking."""

import time

from m31r.serving.metrics.core import RequestMetrics, ServingMetrics


class TestRequestMetrics:

    def test_defaults_are_zero(self) -> None:
        m = RequestMetrics()
        assert m.prompt_tokens == 0
        assert m.generated_tokens == 0
        assert m.total_time_ms == 0.0


class TestServingMetrics:

    def test_starts_empty(self) -> None:
        metrics = ServingMetrics()
        assert metrics.total_requests == 0
        assert metrics.average_tokens_per_second() == 0.0

    def test_records_requests(self) -> None:
        metrics = ServingMetrics()
        metrics.record(RequestMetrics(
            prompt_tokens=10,
            generated_tokens=20,
            total_time_ms=100.0,
            tokens_per_second=200.0,
            peak_memory_mb=50.0,
        ))
        assert metrics.total_requests == 1
        assert metrics.average_tokens_per_second() == 200.0

    def test_averages_across_requests(self) -> None:
        metrics = ServingMetrics()
        metrics.record(RequestMetrics(tokens_per_second=100.0, peak_memory_mb=100.0))
        metrics.record(RequestMetrics(tokens_per_second=200.0, peak_memory_mb=200.0))

        assert metrics.average_tokens_per_second() == 150.0
        assert metrics.peak_memory_mb() == 200.0

    def test_average_ms_per_token(self) -> None:
        metrics = ServingMetrics()
        metrics.record(RequestMetrics(tokens_per_second=100.0))
        assert metrics.average_ms_per_token() == 10.0

    def test_uptime_increases(self) -> None:
        metrics = ServingMetrics()
        time.sleep(0.01)
        assert metrics.uptime_seconds > 0

    def test_summary_structure(self) -> None:
        metrics = ServingMetrics()
        metrics.record(RequestMetrics(tokens_per_second=50.0, peak_memory_mb=100.0))
        summary = metrics.summary()

        expected_keys = {
            "total_requests", "uptime_seconds",
            "avg_tokens_per_second", "avg_ms_per_token", "peak_memory_mb",
        }
        assert set(summary.keys()) == expected_keys
