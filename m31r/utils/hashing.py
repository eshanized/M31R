# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Hashing utilities for M31R.

Per 19_SECURITY_AND_SAFETY.md, all artifacts must include SHA256 checksums.
Every load must verify the hash. Corrupted artifacts must abort.

These functions are intentionally simple — hash a file, hash bytes, verify
a checksum. No fancy caching, no streaming for now.
"""

import hashlib
from pathlib import Path

HASH_ALGORITHM = "sha256"
HASH_BUFFER_SIZE = 65536  # 64 KiB — keeps memory usage low for large files


def compute_sha256(file_path: Path) -> str:
    """
    Compute the SHA256 hex digest of a file.

    Reads the file in chunks to handle large files without loading everything
    into memory. The chunk size is 64 KiB which balances I/O efficiency and
    memory usage.

    Args:
        file_path: Path to the file to hash.

    Returns:
        Lowercase hex string of the SHA256 digest.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        OSError: If the file can't be read.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(HASH_BUFFER_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_sha256_bytes(data: bytes) -> str:
    """
    Compute the SHA256 hex digest of raw bytes.

    Args:
        data: The bytes to hash.

    Returns:
        Lowercase hex string of the SHA256 digest.
    """
    return hashlib.sha256(data).hexdigest()


def verify_checksum(file_path: Path, expected_hash: str) -> bool:
    """
    Check whether a file's SHA256 matches the expected hash.

    Args:
        file_path: Path to the file to verify.
        expected_hash: Expected lowercase hex SHA256 digest.

    Returns:
        True if the hash matches, False otherwise.
    """
    actual_hash = compute_sha256(file_path)
    return actual_hash == expected_hash.lower()
