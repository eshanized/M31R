# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Path utilities for M31R.

All path helpers live here. The rules from the spec:
  - paths must be relative to project root (for portability)
  - path traversal attacks must be prevented
  - directory creation must be explicit
"""

from pathlib import Path


def resolve_project_root() -> Path:
    """
    Walk up from this file's location to find the project root.

    The project root is identified by the presence of pyproject.toml.
    This approach works regardless of where the user's working directory is,
    as long as we're running from within the installed package.

    Returns:
        Absolute path to the project root directory.

    Raises:
        RuntimeError: If no pyproject.toml is found in any ancestor directory.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError(
        "Cannot find project root. No pyproject.toml found in any ancestor directory."
    )


def ensure_directory(path: Path) -> Path:
    """
    Create a directory (and parents) if it doesn't exist. Returns the path for chaining.

    Args:
        path: Directory path to create.

    Returns:
        The same path, now guaranteed to exist.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_path_within_project(target: Path, project_root: Path) -> Path:
    """
    Make sure a path doesn't escape the project directory.

    This is a security measure against path traversal. Any path that resolves
    to somewhere outside the project root is rejected. We resolve both paths
    to their absolute forms before comparing, so tricks like ../../etc/passwd
    get caught.

    Args:
        target: The path to validate.
        project_root: The project root directory.

    Returns:
        The resolved absolute path if it's safe.

    Raises:
        ValueError: If the path escapes the project root.
    """
    resolved_target = target.resolve()
    resolved_root = project_root.resolve()

    if not str(resolved_target).startswith(str(resolved_root)):
        raise ValueError(
            f"Path '{target}' resolves to '{resolved_target}' which is outside "
            f"the project root '{resolved_root}'. This is not allowed."
        )

    return resolved_target
