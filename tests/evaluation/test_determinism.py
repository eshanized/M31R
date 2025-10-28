# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for deterministic evaluation.

The whole point of our evaluation system is reproducibility.
Same inputs + same seed = same outputs, every single time.
These tests verify that guarantee.
"""

from m31r.evaluation.benchmarks.models import CompileResult, TaskResult, TestResult
from m31r.evaluation.metrics.engine import compute_metrics
from m31r.evaluation.metrics.passatk import compute_pass_at_k


def _build_results() -> list[TaskResult]:
    """A fixed set of results we can compute metrics on repeatedly."""
    cr_ok = CompileResult(success=True, exit_code=0, stdout="", stderr="", elapsed_seconds=0.1)
    cr_fail = CompileResult(success=False, exit_code=1, stdout="", stderr="error", elapsed_seconds=0.1)
    tr_ok = TestResult(success=True, exit_code=0, stdout="", stderr="", elapsed_seconds=0.05)
    tr_fail = TestResult(success=False, exit_code=1, stdout="", stderr="fail", elapsed_seconds=0.05)

    return [
        TaskResult(task_id="a/1", category="completion", attempt_index=0, compiled=True, tests_passed=True, compile_result=cr_ok, test_result=tr_ok, generation_time_seconds=0.2),
        TaskResult(task_id="a/1", category="completion", attempt_index=1, compiled=True, tests_passed=False, compile_result=cr_ok, test_result=tr_fail, generation_time_seconds=0.2),
        TaskResult(task_id="b/1", category="functions", attempt_index=0, compiled=False, tests_passed=False, compile_result=cr_fail, generation_time_seconds=0.3),
        TaskResult(task_id="b/1", category="functions", attempt_index=1, compiled=True, tests_passed=True, compile_result=cr_ok, test_result=tr_ok, generation_time_seconds=0.25),
    ]


class TestDeterminism:
    def test_metrics_are_deterministic(self) -> None:
        """Running compute_metrics twice on the same data must give identical results."""
        results = _build_results()
        k_values = [1, 2, 5]

        metrics1 = compute_metrics(results, k_values, seed=42)
        metrics2 = compute_metrics(results, k_values, seed=42)

        assert metrics1.compile_success_rate == metrics2.compile_success_rate
        assert metrics1.pass_at_k == metrics2.pass_at_k
        assert metrics1.category_compile_rates == metrics2.category_compile_rates
        assert metrics1.category_pass_rates == metrics2.category_pass_rates
        assert metrics1.total_tasks == metrics2.total_tasks
        assert metrics1.total_compiled == metrics2.total_compiled
        assert metrics1.total_passed == metrics2.total_passed

    def test_pass_at_k_is_deterministic(self) -> None:
        """Pure math, no randomness — should always give the same answer."""
        for _ in range(100):
            assert compute_pass_at_k(10, 3, 5) == compute_pass_at_k(10, 3, 5)

    def test_different_seeds_same_metrics(self) -> None:
        """
        Seed in metrics is metadata, not a randomness source.
        Changing it shouldn't change the computed values.
        """
        results = _build_results()
        m1 = compute_metrics(results, [1], seed=1)
        m2 = compute_metrics(results, [1], seed=999)

        assert m1.compile_success_rate == m2.compile_success_rate
        assert m1.pass_at_k == m2.pass_at_k
