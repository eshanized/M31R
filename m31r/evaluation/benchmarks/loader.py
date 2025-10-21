# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Benchmark suite loader.

Reads the benchmarks/ directory and turns it into a BenchmarkSuite object
that the evaluation runner can iterate over. The directory structure follows
16_BENCHMARK_SUITE.md §14:

    benchmarks/
    ├── completion/task_001/
    ├── functions/task_001/
    └── ...

Each task folder must have: prompt.rs, solution.rs, tests.rs, Cargo.toml,
and metadata.yaml. Missing any of these is a hard error — we don't guess.
"""

from pathlib import Path

import yaml

from m31r.evaluation.benchmarks.models import (
    VALID_CATEGORIES,
    BenchmarkSuite,
    BenchmarkTask,
)
from m31r.logging.logger import get_logger

logger = get_logger(__name__)

# These files must exist in every task directory. If any are missing,
# the task is invalid and we refuse to load it.
_REQUIRED_FILES: tuple[str, ...] = (
    "prompt.rs",
    "solution.rs",
    "tests.rs",
    "Cargo.toml",
    "metadata.yaml",
)


def _read_file(path: Path) -> str:
    """Read a text file and return its contents, stripping trailing whitespace."""
    return path.read_text(encoding="utf-8").rstrip()


def _load_single_task(task_dir: Path, category: str) -> BenchmarkTask:
    """
    Load one benchmark task from its directory.

    We validate that all required files are present, parse the metadata,
    and bundle everything into a BenchmarkTask. The metadata.yaml tells us
    the difficulty, tags, and expected runtime — stuff we need for filtering
    and reporting but not for the actual pass/fail decision.
    """
    for filename in _REQUIRED_FILES:
        filepath = task_dir / filename
        if not filepath.is_file():
            raise FileNotFoundError(
                f"Task {task_dir.name} is missing required file: {filename}"
            )

    metadata_raw = yaml.safe_load((task_dir / "metadata.yaml").read_text(encoding="utf-8"))
    if not isinstance(metadata_raw, dict):
        raise ValueError(f"metadata.yaml in {task_dir.name} is not a valid YAML mapping")

    # Pick up any extra .rs files as context (things like lib.rs, utils.rs)
    context_files: dict[str, str] = {}
    for rs_file in sorted(task_dir.glob("*.rs")):
        if rs_file.name not in ("prompt.rs", "solution.rs", "tests.rs"):
            context_files[rs_file.name] = _read_file(rs_file)

    return BenchmarkTask(
        task_id=f"{category}/{task_dir.name}",
        category=category,
        difficulty=str(metadata_raw.get("difficulty", "medium")),
        prompt=_read_file(task_dir / "prompt.rs"),
        solution=_read_file(task_dir / "solution.rs"),
        test_code=_read_file(task_dir / "tests.rs"),
        cargo_toml=_read_file(task_dir / "Cargo.toml"),
        context_files=context_files,
        tags=list(metadata_raw.get("tags", [])),
        expected_runtime_seconds=float(metadata_raw.get("expected_runtime", 10.0)),
    )


def load_benchmark_suite(benchmark_dir: Path) -> BenchmarkSuite:
    """
    Load the entire benchmark suite from disk.

    Walks through each category directory, loads every task inside it,
    and assembles them into a sorted BenchmarkSuite. The sorting is by
    task_id so evaluation order is always deterministic — same benchmarks
    in, same order out. No shuffling, no randomness.

    Raises FileNotFoundError if the benchmark directory doesn't exist or
    has no valid tasks.
    """
    if not benchmark_dir.is_dir():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")

    tasks: list[BenchmarkTask] = []
    found_categories: set[str] = set()

    # We only look at directories that match known categories.
    # Unknown directories get a warning but don't blow up the whole run.
    for category_dir in sorted(benchmark_dir.iterdir()):
        if not category_dir.is_dir():
            continue

        category_name = category_dir.name
        if category_name not in VALID_CATEGORIES:
            logger.warning(
                "Skipping unknown benchmark category",
                extra={"category": category_name, "path": str(category_dir)},
            )
            continue

        found_categories.add(category_name)

        for task_dir in sorted(category_dir.iterdir()):
            if not task_dir.is_dir():
                continue

            try:
                task = _load_single_task(task_dir, category_name)
                tasks.append(task)
                logger.debug(
                    "Loaded benchmark task",
                    extra={"task_id": task.task_id, "category": category_name},
                )
            except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
                logger.error(
                    "Failed to load benchmark task",
                    extra={"task_dir": str(task_dir), "error": str(exc)},
                )
                raise

    if not tasks:
        raise FileNotFoundError(
            f"No valid benchmark tasks found in {benchmark_dir}"
        )

    # Sort for deterministic ordering — this is a hard requirement
    tasks.sort(key=lambda t: t.task_id)

    logger.info(
        "Benchmark suite loaded",
        extra={
            "total_tasks": len(tasks),
            "categories": sorted(found_categories),
        },
    )

    return BenchmarkSuite(
        version="v1",
        tasks=tasks,
        categories=sorted(found_categories),
    )
