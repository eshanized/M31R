# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for the benchmark loader.

Verifies that the loader can parse task directories, handle missing files
gracefully, maintain deterministic ordering, and correctly identify categories.
"""

import textwrap
from pathlib import Path

import pytest

from m31r.evaluation.benchmarks.loader import load_benchmark_suite, _load_single_task
from m31r.evaluation.benchmarks.models import VALID_CATEGORIES


def _create_task(task_dir: Path, category: str = "completion", difficulty: str = "easy") -> None:
    """Helper that sets up a minimal valid benchmark task directory."""
    task_dir.mkdir(parents=True, exist_ok=True)

    (task_dir / "metadata.yaml").write_text(
        f"difficulty: {difficulty}\ncategory: {category}\ntags: [test]\n",
        encoding="utf-8",
    )
    (task_dir / "prompt.rs").write_text(
        "fn foo() -> i32 {\n    // TODO\n",
        encoding="utf-8",
    )
    (task_dir / "solution.rs").write_text(
        "fn foo() -> i32 { 42 }\nfn main() {}\n",
        encoding="utf-8",
    )
    (task_dir / "tests.rs").write_text(
        '#[test]\nfn test_foo() { assert_eq!(foo(), 42); }\n',
        encoding="utf-8",
    )
    (task_dir / "Cargo.toml").write_text(
        '[package]\nname = "test_task"\nversion = "0.1.0"\nedition = "2021"\n',
        encoding="utf-8",
    )


class TestLoadBenchmarkSuite:
    def test_loads_valid_suite(self, tmp_path: Path) -> None:
        bench_dir = tmp_path / "benchmarks"
        _create_task(bench_dir / "completion" / "task_001")
        _create_task(bench_dir / "functions" / "task_001", category="functions")

        suite = load_benchmark_suite(bench_dir)

        assert len(suite.tasks) == 2
        assert suite.version == "v1"
        assert "completion" in suite.categories
        assert "functions" in suite.categories

    def test_deterministic_ordering(self, tmp_path: Path) -> None:
        """Same directory, same task order — every time."""
        bench_dir = tmp_path / "benchmarks"
        _create_task(bench_dir / "functions" / "task_002", category="functions")
        _create_task(bench_dir / "completion" / "task_001")
        _create_task(bench_dir / "functions" / "task_001", category="functions")

        suite1 = load_benchmark_suite(bench_dir)
        suite2 = load_benchmark_suite(bench_dir)

        ids1 = [t.task_id for t in suite1.tasks]
        ids2 = [t.task_id for t in suite2.tasks]
        assert ids1 == ids2
        assert ids1 == sorted(ids1)

    def test_error_on_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_benchmark_suite(tmp_path / "nonexistent")

    def test_error_on_empty_suite(self, tmp_path: Path) -> None:
        bench_dir = tmp_path / "benchmarks"
        bench_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No valid benchmark tasks"):
            load_benchmark_suite(bench_dir)

    def test_skips_unknown_categories(self, tmp_path: Path) -> None:
        bench_dir = tmp_path / "benchmarks"
        _create_task(bench_dir / "completion" / "task_001")
        # This should be skipped, not crash
        (bench_dir / "unknown_category").mkdir()

        suite = load_benchmark_suite(bench_dir)
        assert len(suite.tasks) == 1

    def test_error_on_missing_required_file(self, tmp_path: Path) -> None:
        task_dir = tmp_path / "task_bad"
        task_dir.mkdir(parents=True)
        (task_dir / "prompt.rs").write_text("fn foo() {}", encoding="utf-8")
        # Missing solution.rs, tests.rs, Cargo.toml, metadata.yaml

        with pytest.raises(FileNotFoundError, match="missing required file"):
            _load_single_task(task_dir, "completion")


class TestLoadSingleTask:
    def test_loads_all_fields(self, tmp_path: Path) -> None:
        task_dir = tmp_path / "task_001"
        _create_task(task_dir, category="bugs", difficulty="hard")

        task = _load_single_task(task_dir, "bugs")

        assert task.task_id == "bugs/task_001"
        assert task.category == "bugs"
        assert task.difficulty == "hard"
        assert "foo" in task.prompt
        assert task.cargo_toml.startswith("[package]")

    def test_picks_up_context_files(self, tmp_path: Path) -> None:
        task_dir = tmp_path / "task_ctx"
        _create_task(task_dir)
        (task_dir / "utils.rs").write_text("fn helper() {}", encoding="utf-8")

        task = _load_single_task(task_dir, "completion")

        assert "utils.rs" in task.context_files
        assert "helper" in task.context_files["utils.rs"]

    def test_invalid_metadata_raises(self, tmp_path: Path) -> None:
        task_dir = tmp_path / "task_bad_meta"
        _create_task(task_dir)
        (task_dir / "metadata.yaml").write_text("just a string\n", encoding="utf-8")

        with pytest.raises(ValueError, match="not a valid YAML mapping"):
            _load_single_task(task_dir, "completion")
