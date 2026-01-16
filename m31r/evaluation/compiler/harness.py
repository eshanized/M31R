# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Compile and test harness for Rust code.

This is the part that actually runs `cargo build` and `cargo test` on
generated code. It's deliberately simple: run the subprocess, capture
everything, enforce a timeout, return the result. No fancy build
caching or incremental compilation — we want clean, isolated builds
every time so results are reproducible.

Per 19_SECURITY_AND_SAFETY.md §16: only `cargo build` and `cargo test`
are allowed. No arbitrary commands, no shell=True, no eval().
"""

import subprocess
import time
from pathlib import Path

from m31r.evaluation.benchmarks.models import CompileResult, TestResult
from m31r.logging.logger import get_logger

logger = get_logger(__name__)


def compile_rust(source_dir: Path, timeout_seconds: int = 10) -> CompileResult:
    """
    Run `cargo build` in the given directory and capture the result.

    We use subprocess.run with a hard timeout so a hung compilation can't
    block the entire evaluation run. The output (stdout + stderr) is captured
    for diagnostics but the only thing that matters for scoring is the
    exit code: 0 means it compiled, anything else means it didn't.

    The --manifest-path flag ensures cargo looks at the right Cargo.toml
    even if our working directory is somewhere else.
    """
    start = time.monotonic()

    try:
        result = subprocess.run(
            [
                "cargo",
                "build",
                "--manifest-path",
                str(source_dir / "Cargo.toml"),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(source_dir),
            env=_build_restricted_env(),
        )

        elapsed = time.monotonic() - start
        success = result.returncode == 0

        logger.debug(
            "Compilation finished",
            extra={
                "success": success,
                "exit_code": result.returncode,
                "elapsed_seconds": round(elapsed, 3),
                "source_dir": str(source_dir),
            },
        )

        return CompileResult(
            success=success,
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            elapsed_seconds=elapsed,
        )

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        logger.warning(
            "Compilation timed out",
            extra={
                "timeout_seconds": timeout_seconds,
                "source_dir": str(source_dir),
            },
        )
        return CompileResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr=f"Compilation timed out after {timeout_seconds}s",
            elapsed_seconds=elapsed,
        )

    except FileNotFoundError:
        elapsed = time.monotonic() - start
        logger.error(
            "cargo not found — is Rust installed?",
            extra={"source_dir": str(source_dir)},
        )
        return CompileResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr="cargo executable not found",
            elapsed_seconds=elapsed,
        )


def test_rust(source_dir: Path, timeout_seconds: int = 10) -> TestResult:
    """
    Run `cargo test` in the given directory and capture the result.

    Same approach as compile_rust: subprocess with timeout, capture output,
    return structured result. Only called after a successful compilation
    since there's no point running tests on code that doesn't compile.
    """
    start = time.monotonic()

    try:
        result = subprocess.run(
            [
                "cargo",
                "test",
                "--manifest-path",
                str(source_dir / "Cargo.toml"),
                "--",
                "--test-threads=1",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(source_dir),
            env=_build_restricted_env(),
        )

        elapsed = time.monotonic() - start
        success = result.returncode == 0

        logger.debug(
            "Tests finished",
            extra={
                "success": success,
                "exit_code": result.returncode,
                "elapsed_seconds": round(elapsed, 3),
                "source_dir": str(source_dir),
            },
        )

        return TestResult(
            success=success,
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            elapsed_seconds=elapsed,
        )

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        logger.warning(
            "Tests timed out",
            extra={
                "timeout_seconds": timeout_seconds,
                "source_dir": str(source_dir),
            },
        )
        return TestResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr=f"Tests timed out after {timeout_seconds}s",
            elapsed_seconds=elapsed,
        )

    except FileNotFoundError:
        elapsed = time.monotonic() - start
        return TestResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr="cargo executable not found",
            elapsed_seconds=elapsed,
        )


def _build_restricted_env() -> dict[str, str]:
    """
    Build a minimal environment for subprocess execution.

    We start with the current environment but could restrict it further
    in the future (e.g., removing network access variables). For now,
    inheriting the parent env is fine because we're running in isolated
    temp directories anyway.
    """
    import os

    env = dict(os.environ)
    # Make sure cargo doesn't try to phone home for crate downloads
    env["CARGO_NET_OFFLINE"] = "true"
    return env
