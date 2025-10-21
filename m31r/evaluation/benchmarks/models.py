# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Data models for the benchmark system.

These are the core types that everything in the evaluation pipeline passes
around. They're all frozen dataclasses because benchmark data should never
be mutated after loading — if something changes a task mid-evaluation,
that's a bug.
"""

from dataclasses import dataclass, field


# These are the valid benchmark categories from 16_BENCHMARK_SUITE.md §4.
# If you add a new category, add it here and create the matching directory
# under benchmarks/.
VALID_CATEGORIES: frozenset[str] = frozenset({
    "completion",
    "fim",
    "functions",
    "bugs",
    "refactor",
    "ownership",
    "projects",
    "stdlib",
})


@dataclass(frozen=True)
class BenchmarkTask:
    """
    A single benchmark task that the model needs to solve.

    Each task lives in its own directory with a standard layout:
      prompt.rs     — the incomplete code the model sees
      solution.rs   — a known-good solution (for reference, never shown to model)
      tests.rs      — test code that validates correctness
      Cargo.toml    — project manifest for compilation
      metadata.yaml — category, difficulty, tags

    The task is "solved" if the model's output compiles and passes the tests.
    That's it. No partial credit, no style points.
    """

    task_id: str
    category: str
    difficulty: str
    prompt: str
    solution: str
    test_code: str
    cargo_toml: str
    context_files: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    expected_runtime_seconds: float = 10.0


@dataclass(frozen=True)
class CompileResult:
    """What came back from trying to compile generated code."""

    success: bool
    exit_code: int
    stdout: str
    stderr: str
    elapsed_seconds: float


@dataclass(frozen=True)
class TestResult:
    """What came back from running cargo test on generated code."""

    success: bool
    exit_code: int
    stdout: str
    stderr: str
    elapsed_seconds: float


@dataclass(frozen=True)
class TaskResult:
    """
    The outcome of one attempt at solving a benchmark task.

    This captures the full pipeline result: did it compile? did tests pass?
    How long did each step take? The binary verdict is simple:
    passed = compiled AND tests passed. Everything else is a fail.
    """

    task_id: str
    category: str
    attempt_index: int
    compiled: bool
    tests_passed: bool
    compile_result: CompileResult | None = None
    test_result: TestResult | None = None
    generation_time_seconds: float = 0.0
    generated_code: str = ""


@dataclass(frozen=True)
class BenchmarkSuite:
    """
    The complete set of tasks to evaluate a model against.

    Tasks are always sorted by task_id for deterministic iteration.
    The version string tracks which benchmark set this is — changing
    tasks requires a version bump per 16_BENCHMARK_SUITE.md §22.
    """

    version: str
    tasks: list[BenchmarkTask]
    categories: list[str]


@dataclass
class EvalMetrics:
    """
    All the numbers that come out of an evaluation run.

    The primary metrics (Tier 1 from 14_EVALUATION_METHODOLOGY.md) are
    what actually determines if a model ships or not. Everything else
    is diagnostic.
    """

    compile_success_rate: float = 0.0
    pass_at_k: dict[int, float] = field(default_factory=dict)
    category_compile_rates: dict[str, float] = field(default_factory=dict)
    category_pass_rates: dict[str, float] = field(default_factory=dict)
    total_tasks: int = 0
    total_attempts: int = 0
    total_compiled: int = 0
    total_passed: int = 0
    avg_generation_time_seconds: float = 0.0
    avg_compile_time_seconds: float = 0.0
    avg_test_time_seconds: float = 0.0
    seed: int = 42
