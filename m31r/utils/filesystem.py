# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Safe filesystem operations for M31R.

Per the architecture spec (17_FAILURE_MODEL in 04_SYSTEM_ARCHITECTURE.md):
  - writes must be atomic (no partial files on failure)
  - failures must not corrupt artifacts
  - resources must be properly closed

Atomic writes work by writing to a temporary file in the same directory as
the target, then renaming. Rename on the same filesystem is atomic on POSIX.
If the process crashes mid-write, you get a leftover temp file instead of a
corrupted target file.
"""

import tempfile
from pathlib import Path


def atomic_write(target_path: Path, content: str, encoding: str = "utf-8") -> None:
    """
    Write content to a file atomically.

    The trick: we write to a temp file in the same directory, then rename it
    to the target path. os.rename (which Path.rename uses) is atomic on POSIX
    as long as source and destination are on the same filesystem. Writing to
    the same directory guarantees that.

    If anything goes wrong during the write (disk full, permissions, crash),
    the target file is never touched — you either get the full new content or
    the old content, never a partial mess.

    Args:
        target_path: Where the final file should end up.
        content: The string content to write.
        encoding: Text encoding to use.

    Raises:
        OSError: If the write or rename fails.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # delete=False because we need the file to survive closing so we can rename it.
    # dir= same directory as target so rename is atomic (same filesystem).
    temp_fd = tempfile.NamedTemporaryFile(
        mode="w",
        encoding=encoding,
        dir=str(target_path.parent),
        prefix=".m31r_tmp_",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(temp_fd.name)

    try:
        temp_fd.write(content)
        temp_fd.flush()
        temp_fd.close()
        temp_path.rename(target_path)
    except BaseException:
        temp_fd.close()
        # Clean up the temp file if anything goes wrong.
        if temp_path.exists():
            temp_path.unlink()
        raise


def atomic_write_bytes(target_path: Path, data: bytes) -> None:
    """
    Write binary data to a file atomically. Same approach as atomic_write.

    Args:
        target_path: Where the final file should end up.
        data: The raw bytes to write.

    Raises:
        OSError: If the write or rename fails.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)

    temp_fd = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=str(target_path.parent),
        prefix=".m31r_tmp_",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(temp_fd.name)

    try:
        temp_fd.write(data)
        temp_fd.flush()
        temp_fd.close()
        temp_path.rename(target_path)
    except BaseException:
        temp_fd.close()
        if temp_path.exists():
            temp_path.unlink()
        raise


def safe_read(file_path: Path, encoding: str = "utf-8") -> str:
    """
    Read a text file with proper error context.

    Args:
        file_path: Path to the file to read.
        encoding: Text encoding, defaults to UTF-8.

    Returns:
        The file contents as a string.

    Raises:
        FileNotFoundError: If the path doesn't exist.
        IsADirectoryError: If the path is a directory.
        OSError: For other I/O errors.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise IsADirectoryError(f"Expected a file, got a directory: {file_path}")
    return file_path.read_text(encoding=encoding)


def safe_delete(file_path: Path) -> bool:
    """
    Delete a file if it exists. Returns whether anything was actually deleted.

    This never throws on a missing file — that's the "safe" part.

    Args:
        file_path: Path to the file to delete.

    Returns:
        True if the file existed and was deleted, False if it didn't exist.

    Raises:
        OSError: If the file exists but can't be deleted (permissions, etc).
    """
    if file_path.exists():
        file_path.unlink()
        return True
    return False
