# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Metrics computation engine.

Takes raw task results and computes all the numbers that determine whether
a model ships or gets rejected. The primary metrics are:

  - compile_success_rate: what fraction of generated code actually compiles
  - pass@k: probability at least one of K attempts solves the task
  - per-category breakdowns: same metrics but sliced by task type

Everything here is a pure function — given the same inputs, you always
get the same outputs. No randomness, no side effects, no global state.
"""

from collections import defaultdict

from m31r.evaluation.benchmarks.models import EvalMetrics, TaskResult
from m31r.evaluation.metrics.passatk import compute_pass_at_k
from m31r.logging.logger import get_logger

logger = get_logger(__name__)


def compute_metrics(
    results: list[TaskResult],
    k_values: list[int],
    seed: int = 42,
) -> EvalMetrics:
    """
    Crunch all the numbers from an evaluation run.

    This takes the raw list of TaskResult objects (one per attempt per task)
    and computes everything: overall compile rate, pass@k for each K value,
    per-category breakdowns, and timing averages.

    The results are grouped by task_id first, since a single task can have
    multiple attempts (that's how pass@k works). Within each task group,
    we count how many attempts compiled and how many passed tests.

    Args:
        results: All TaskResult objects from the evaluation run.
        k_values: Which K values to compute pass@k for (e.g., [1, 5, 10]).
        seed: The seed used for this run, stored in metrics for traceability.

    Returns:
        EvalMetrics with all computed values filled in.
    """
    if not results:
        return EvalMetrics(seed=seed)

    # Group results by task_id so we can compute per-task statistics.
    # Each task might have K attempts, and we need all of them together
    # to compute pass@k correctly.
    task_groups: dict[str, list[TaskResult]] = defaultdict(list)
    for r in results:
        task_groups[r.task_id].append(r)

    total_attempts = len(results)
    total_compiled = sum(1 for r in results if r.compiled)
    total_passed = sum(1 for r in results if r.tests_passed)

    compile_success_rate = total_compiled / total_attempts if total_attempts > 0 else 0.0

    # Pass@k: for each task, we look at how many of its attempts passed,
    # then use the combinatorial formula to estimate the probability that
    # at least one of K random picks would be correct. We average across
    # all tasks to get the final number.
    pass_at_k: dict[int, float] = {}
    for k in sorted(k_values):
        task_scores: list[float] = []
        for task_id in sorted(task_groups.keys()):
            group = task_groups[task_id]
            n = len(group)
            c = sum(1 for r in group if r.tests_passed)
            score = compute_pass_at_k(n, c, k)
            task_scores.append(score)

        pass_at_k[k] = (
            sum(task_scores) / len(task_scores) if task_scores else 0.0
        )

    # Per-category breakdowns let us see which types of tasks the model
    # struggles with. A model might ace function completion but fail
    # ownership tasks — this is exactly the kind of thing we need to know.
    category_groups: dict[str, list[TaskResult]] = defaultdict(list)
    for r in results:
        category_groups[r.category].append(r)

    category_compile_rates: dict[str, float] = {}
    category_pass_rates: dict[str, float] = {}
    for cat in sorted(category_groups.keys()):
        cat_results = category_groups[cat]
        cat_total = len(cat_results)
        cat_compiled = sum(1 for r in cat_results if r.compiled)
        cat_passed = sum(1 for r in cat_results if r.tests_passed)

        category_compile_rates[cat] = cat_compiled / cat_total if cat_total > 0 else 0.0
        category_pass_rates[cat] = cat_passed / cat_total if cat_total > 0 else 0.0

    # Timing averages for performance tracking
    gen_times = [r.generation_time_seconds for r in results if r.generation_time_seconds > 0]
    compile_times = [
        r.compile_result.elapsed_seconds
        for r in results
        if r.compile_result is not None
    ]
    test_times = [
        r.test_result.elapsed_seconds
        for r in results
        if r.test_result is not None
    ]

    metrics = EvalMetrics(
        compile_success_rate=compile_success_rate,
        pass_at_k=pass_at_k,
        category_compile_rates=category_compile_rates,
        category_pass_rates=category_pass_rates,
        total_tasks=len(task_groups),
        total_attempts=total_attempts,
        total_compiled=total_compiled,
        total_passed=total_passed,
        avg_generation_time_seconds=(
            sum(gen_times) / len(gen_times) if gen_times else 0.0
        ),
        avg_compile_time_seconds=(
            sum(compile_times) / len(compile_times) if compile_times else 0.0
        ),
        avg_test_time_seconds=(
            sum(test_times) / len(test_times) if test_times else 0.0
        ),
        seed=seed,
    )

    logger.info(
        "Metrics computed",
        extra={
            "compile_success_rate": round(metrics.compile_success_rate, 4),
            "pass_at_k": {str(k): round(v, 4) for k, v in metrics.pass_at_k.items()},
            "total_tasks": metrics.total_tasks,
        },
    )

    return metrics
