# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for the metrics engine and pass@k computation.

These verify the math is right. The pass@k formula in particular has some
tricky edge cases (all pass, all fail, k > n) that we need to nail.
"""

import math

import pytest

from m31r.evaluation.benchmarks.models import (
    CompileResult,
    EvalMetrics,
    TaskResult,
    TestResult,
)
from m31r.evaluation.metrics.engine import compute_metrics
from m31r.evaluation.metrics.passatk import compute_pass_at_k


class TestPassAtK:
    """
    The pass@k formula is: 1 - C(n-c, k) / C(n, k)

    These tests verify known values. The formula gives the probability
    that at least one of k randomly chosen samples from n total is
    correct, given c correct samples.
    """

    def test_all_pass(self) -> None:
        assert compute_pass_at_k(10, 10, 1) == 1.0

    def test_none_pass(self) -> None:
        assert compute_pass_at_k(10, 0, 1) == 0.0

    def test_all_pass_k5(self) -> None:
        assert compute_pass_at_k(10, 10, 5) == 1.0

    def test_known_value_pass_at_1(self) -> None:
        # 5 out of 10 correct, k=1
        # pass@1 = 1 - C(5,1)/C(10,1) = 1 - 5/10 = 0.5
        result = compute_pass_at_k(10, 5, 1)
        assert abs(result - 0.5) < 1e-10

    def test_known_value_pass_at_2(self) -> None:
        # 1 out of 10 correct, k=2
        # pass@2 = 1 - C(9,2)/C(10,2) = 1 - 36/45 = 0.2
        result = compute_pass_at_k(10, 1, 2)
        assert abs(result - 0.2) < 1e-10

    def test_k_greater_than_n(self) -> None:
        # k gets clamped to n
        result = compute_pass_at_k(5, 3, 100)
        assert result == 1.0

    def test_n_minus_c_less_than_k(self) -> None:
        # Not enough failures to fill k slots, so pass@k = 1.0
        result = compute_pass_at_k(5, 4, 3)
        assert result == 1.0

    def test_zero_n(self) -> None:
        assert compute_pass_at_k(0, 0, 1) == 0.0

    def test_negative_k(self) -> None:
        assert compute_pass_at_k(10, 5, -1) == 0.0

    def test_single_sample_passes(self) -> None:
        assert compute_pass_at_k(1, 1, 1) == 1.0

    def test_single_sample_fails(self) -> None:
        assert compute_pass_at_k(1, 0, 1) == 0.0


def _make_result(
    task_id: str = "test/task_001",
    category: str = "completion",
    attempt: int = 0,
    compiled: bool = True,
    passed: bool = True,
) -> TaskResult:
    """Build a TaskResult for testing without needing real compile/test data."""
    cr = CompileResult(
        success=compiled, exit_code=0 if compiled else 1, stdout="", stderr="", elapsed_seconds=0.1
    )
    tr = None
    if compiled:
        tr = TestResult(
            success=passed, exit_code=0 if passed else 1, stdout="", stderr="", elapsed_seconds=0.05
        )

    return TaskResult(
        task_id=task_id,
        category=category,
        attempt_index=attempt,
        compiled=compiled,
        tests_passed=passed,
        compile_result=cr,
        test_result=tr,
        generation_time_seconds=0.2,
    )


class TestComputeMetrics:
    def test_empty_results(self) -> None:
        metrics = compute_metrics([], [1, 5], seed=42)
        assert metrics.total_tasks == 0
        assert metrics.compile_success_rate == 0.0

    def test_all_pass(self) -> None:
        results = [
            _make_result("t/1", attempt=0),
            _make_result("t/1", attempt=1),
            _make_result("t/2", attempt=0),
            _make_result("t/2", attempt=1),
        ]
        metrics = compute_metrics(results, [1], seed=42)

        assert metrics.compile_success_rate == 1.0
        assert metrics.pass_at_k[1] == 1.0
        assert metrics.total_tasks == 2
        assert metrics.total_attempts == 4

    def test_all_fail(self) -> None:
        results = [
            _make_result("t/1", compiled=False, passed=False),
            _make_result("t/1", attempt=1, compiled=False, passed=False),
        ]
        metrics = compute_metrics(results, [1], seed=42)

        assert metrics.compile_success_rate == 0.0
        assert metrics.pass_at_k[1] == 0.0

    def test_mixed_results(self) -> None:
        results = [
            _make_result("t/1", attempt=0, passed=True),
            _make_result("t/1", attempt=1, passed=False),
            _make_result("t/2", attempt=0, passed=False),
            _make_result("t/2", attempt=1, passed=False),
        ]
        metrics = compute_metrics(results, [1, 2], seed=42)

        assert metrics.total_tasks == 2
        assert 0.0 < metrics.pass_at_k[1] < 1.0

    def test_category_breakdown(self) -> None:
        results = [
            _make_result("c/1", category="completion"),
            _make_result("f/1", category="functions", compiled=False, passed=False),
        ]
        metrics = compute_metrics(results, [1], seed=42)

        assert metrics.category_compile_rates["completion"] == 1.0
        assert metrics.category_compile_rates["functions"] == 0.0
        assert metrics.category_pass_rates["completion"] == 1.0
        assert metrics.category_pass_rates["functions"] == 0.0

    def test_timing_averages(self) -> None:
        results = [_make_result(), _make_result("t/2")]
        metrics = compute_metrics(results, [1], seed=42)

        assert metrics.avg_generation_time_seconds > 0
        assert metrics.avg_compile_time_seconds > 0
        assert metrics.avg_test_time_seconds > 0
