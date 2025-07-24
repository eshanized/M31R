# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Rust file detection and extension filtering.

Pretty simple module — it just checks whether a file has the right extension
to be considered a Rust source file. The "allowed extensions" list comes from
config so it's not hardcoded here, but .rs is the obvious default.
"""

from pathlib import PurePosixPath


def is_rust_file(file_path: PurePosixPath) -> bool:
    """Check if a file ends with .rs — the Rust source extension."""
    return file_path.suffix.lower() == ".rs"


def is_allowed_extension(file_path: PurePosixPath, allowed: list[str]) -> bool:
    """
    Check if a file's extension is in the allowed list.

    The allowed list typically comes from FilterConfig.allowed_extensions.
    We do a case-insensitive comparison because some systems are
    case-insensitive and we don't want to miss files on Linux that happen
    to have an uppercase extension.
    """
    suffix = file_path.suffix.lower()
    return suffix in {ext.lower() for ext in allowed}
