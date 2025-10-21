# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Sandboxed execution environment for generated code.

Every piece of generated code runs inside its own temporary directory.
This prevents tasks from contaminating each other and protects the host
filesystem. The sandbox creates a minimal Cargo project, drops the
generated source and test files in, and cleans up after itself.

Per 19_SECURITY_AND_SAFETY.md: generated code is untrusted.
We never auto-execute arbitrary commands — only `cargo build` and
`cargo test` are allowed, and only inside the sandbox directory.
"""

import shutil
import tempfile
from pathlib import Path
from types import TracebackType

from m31r.evaluation.benchmarks.models import BenchmarkTask
from m31r.logging.logger import get_logger

logger = get_logger(__name__)


def _validate_sandbox_path(path: Path, sandbox_root: Path) -> None:
    """
    Make sure a path doesn't escape the sandbox.

    This catches path traversal attacks where someone sneaks a `../../`
    into a filename. We resolve both paths to their absolute forms and
    check that the target is still under the sandbox root.
    """
    resolved = path.resolve()
    root_resolved = sandbox_root.resolve()
    if not str(resolved).startswith(str(root_resolved)):
        raise ValueError(
            f"Path escapes sandbox: {path} resolves outside {sandbox_root}"
        )


def create_sandbox(
    task: BenchmarkTask,
    generated_code: str,
    base_dir: Path | None = None,
) -> Path:
    """
    Set up an isolated directory with a Cargo project for compilation.

    Creates a temp directory, writes the Cargo.toml, drops the generated
    code into src/main.rs (or src/lib.rs depending on the task), and puts
    the tests alongside it. Returns the path to the sandbox so the caller
    can run cargo commands against it.

    If the task has context files (like a lib.rs or utils.rs), those get
    written into src/ as well.
    """
    sandbox_dir = Path(tempfile.mkdtemp(
        prefix="m31r_eval_",
        dir=str(base_dir) if base_dir else None,
    ))

    try:
        src_dir = sandbox_dir / "src"
        src_dir.mkdir()

        (sandbox_dir / "Cargo.toml").write_text(
            task.cargo_toml, encoding="utf-8",
        )

        # The generated code goes into src/main.rs. For library-style tasks,
        # the Cargo.toml should already be configured to use lib.rs instead.
        main_file = src_dir / "main.rs"
        _validate_sandbox_path(main_file, sandbox_dir)
        main_file.write_text(generated_code, encoding="utf-8")

        # Tests go in a separate file that gets included by the test harness
        tests_file = src_dir / "tests.rs"
        _validate_sandbox_path(tests_file, sandbox_dir)
        tests_file.write_text(task.test_code, encoding="utf-8")

        for filename, content in sorted(task.context_files.items()):
            context_path = src_dir / filename
            _validate_sandbox_path(context_path, sandbox_dir)
            context_path.write_text(content, encoding="utf-8")

        logger.debug(
            "Sandbox created",
            extra={"path": str(sandbox_dir), "task_id": task.task_id},
        )
        return sandbox_dir

    except Exception:
        shutil.rmtree(sandbox_dir, ignore_errors=True)
        raise


def cleanup_sandbox(sandbox_dir: Path) -> None:
    """Remove a sandbox directory and everything inside it."""
    if sandbox_dir.is_dir():
        shutil.rmtree(sandbox_dir, ignore_errors=True)
        logger.debug("Sandbox cleaned up", extra={"path": str(sandbox_dir)})


class SandboxContext:
    """
    Context manager that creates a sandbox on enter and cleans it up on exit.

    Usage:
        with SandboxContext(task, generated_code) as sandbox_path:
            # sandbox_path is a Path to the temp directory
            # run cargo build / cargo test here
        # directory is automatically deleted when you leave the block

    This is the preferred way to use sandboxes — it guarantees cleanup
    even if something crashes midway through.
    """

    def __init__(
        self,
        task: BenchmarkTask,
        generated_code: str,
        base_dir: Path | None = None,
    ) -> None:
        self._task = task
        self._generated_code = generated_code
        self._base_dir = base_dir
        self._sandbox_dir: Path | None = None

    def __enter__(self) -> Path:
        self._sandbox_dir = create_sandbox(
            self._task, self._generated_code, self._base_dir,
        )
        return self._sandbox_dir

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._sandbox_dir is not None:
            cleanup_sandbox(self._sandbox_dir)
