# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for the compile/test harness and sandbox.

These test the subprocess isolation, timeout handling, and sandbox
lifecycle without needing a real Rust project — we use tiny programs
that either compile or don't.
"""

import os
from pathlib import Path

import pytest

from m31r.evaluation.benchmarks.models import BenchmarkTask
from m31r.evaluation.compiler.sandbox import (
    SandboxContext,
    _validate_sandbox_path,
    cleanup_sandbox,
    create_sandbox,
)


def _make_task(**kwargs) -> BenchmarkTask:
    """Build a minimal BenchmarkTask with sensible defaults for testing."""
    defaults = {
        "task_id": "test/task_001",
        "category": "completion",
        "difficulty": "easy",
        "prompt": "fn main() {}",
        "solution": "fn main() {}",
        "test_code": "",
        "cargo_toml": '[package]\nname = "test"\nversion = "0.1.0"\nedition = "2021"\n',
    }
    defaults.update(kwargs)
    return BenchmarkTask(**defaults)


class TestSandbox:
    def test_creates_directory_with_source(self, tmp_path: Path) -> None:
        task = _make_task()
        sandbox = create_sandbox(task, "fn main() { println!(\"hi\"); }", base_dir=tmp_path)

        try:
            assert sandbox.is_dir()
            assert (sandbox / "Cargo.toml").is_file()
            assert (sandbox / "src" / "main.rs").is_file()
            assert "println" in (sandbox / "src" / "main.rs").read_text()
        finally:
            cleanup_sandbox(sandbox)

    def test_writes_test_code(self, tmp_path: Path) -> None:
        task = _make_task(test_code='#[test]\nfn it_works() { assert!(true); }')
        sandbox = create_sandbox(task, "fn main() {}", base_dir=tmp_path)

        try:
            tests_file = sandbox / "src" / "tests.rs"
            assert tests_file.is_file()
            assert "it_works" in tests_file.read_text()
        finally:
            cleanup_sandbox(sandbox)

    def test_writes_context_files(self, tmp_path: Path) -> None:
        task = _make_task(context_files={"utils.rs": "pub fn helper() -> i32 { 1 }"})
        sandbox = create_sandbox(task, "fn main() {}", base_dir=tmp_path)

        try:
            assert (sandbox / "src" / "utils.rs").is_file()
        finally:
            cleanup_sandbox(sandbox)

    def test_cleanup_removes_directory(self, tmp_path: Path) -> None:
        task = _make_task()
        sandbox = create_sandbox(task, "fn main() {}", base_dir=tmp_path)
        assert sandbox.is_dir()

        cleanup_sandbox(sandbox)
        assert not sandbox.exists()

    def test_cleanup_nonexistent_is_safe(self, tmp_path: Path) -> None:
        cleanup_sandbox(tmp_path / "does_not_exist")

    def test_context_manager_cleans_up(self, tmp_path: Path) -> None:
        task = _make_task()
        sandbox_dir = None

        with SandboxContext(task, "fn main() {}", base_dir=tmp_path) as d:
            sandbox_dir = d
            assert d.is_dir()

        assert not sandbox_dir.exists()

    def test_context_manager_cleans_up_on_error(self, tmp_path: Path) -> None:
        task = _make_task()
        sandbox_dir = None

        with pytest.raises(RuntimeError):
            with SandboxContext(task, "fn main() {}", base_dir=tmp_path) as d:
                sandbox_dir = d
                raise RuntimeError("boom")

        assert not sandbox_dir.exists()


class TestPathValidation:
    def test_valid_path(self, tmp_path: Path) -> None:
        _validate_sandbox_path(tmp_path / "src" / "main.rs", tmp_path)

    def test_rejects_traversal(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="escapes sandbox"):
            _validate_sandbox_path(tmp_path / ".." / "etc" / "passwd", tmp_path)


class TestCompilerHarness:
    """
    These tests check the harness's error handling paths.
    We don't assume cargo is installed in the test environment.
    """

    def test_compile_result_on_missing_cargo(self, tmp_path: Path) -> None:
        """If cargo isn't found, we should get a clean failure, not a crash."""
        from m31r.evaluation.compiler.harness import compile_rust

        result = compile_rust(tmp_path, timeout_seconds=5)

        # It either fails because of missing source or missing cargo.
        # Either way, success should be False — no crash.
        assert result.success is False
        assert result.elapsed_seconds >= 0

    def test_test_result_on_missing_cargo(self, tmp_path: Path) -> None:
        from m31r.evaluation.compiler.harness import test_rust

        result = test_rust(tmp_path, timeout_seconds=5)
        assert result.success is False
        assert result.elapsed_seconds >= 0
