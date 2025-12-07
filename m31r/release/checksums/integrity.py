# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Release checksum generation and verification.

Per 18_RELEASE_PROCESS.md §15 and 19_SECURITY_AND_SAFETY.md §10:
- All artifacts must include SHA256 checksums
- Verification required on load
- Corrupted artifacts rejected
- Silent corruption forbidden

Checksum file format (checksum.txt):
    <sha256hex>  <filename>
    <sha256hex>  <filename>

One line per file, space-separated, sorted by filename for determinism.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from m31r.logging.logger import get_logger
from m31r.utils.hashing import compute_sha256

_logger: logging.Logger = get_logger(__name__)

# Files that are part of the checksum system itself — never checksum these
_CHECKSUM_EXCLUDED_FILES: frozenset[str] = frozenset({"checksum.txt"})


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of a checksum verification run."""

    is_valid: bool
    checked_count: int
    mismatches: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def generate_checksums(release_dir: Path) -> dict[str, str]:
    """
    Compute SHA256 checksums for every file in a release directory.

    Skips checksum.txt itself to avoid self-referential hashing.
    Returns a dict mapping filename → hex digest, sorted by filename
    for deterministic output.

    Args:
        release_dir: Path to the release directory.

    Returns:
        Ordered dict of {filename: sha256_hex}.

    Raises:
        FileNotFoundError: If release_dir doesn't exist.
    """
    if not release_dir.is_dir():
        raise FileNotFoundError(f"Release directory not found: {release_dir}")

    checksums: dict[str, str] = {}
    for file_path in sorted(release_dir.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.name in _CHECKSUM_EXCLUDED_FILES:
            continue
        digest = compute_sha256(file_path)
        checksums[file_path.name] = digest
        _logger.debug(
            "Computed checksum",
            extra={"file": file_path.name, "sha256": digest[:16] + "..."},
        )

    _logger.info(
        "Checksums generated",
        extra={"file_count": len(checksums), "release_dir": str(release_dir)},
    )
    return checksums


def write_checksum_file(release_dir: Path, checksums: dict[str, str]) -> Path:
    """
    Write checksums to checksum.txt in the release directory.

    Format: "<sha256>  <filename>" — two spaces between hash and name,
    matching the GNU coreutils sha256sum format.

    Args:
        release_dir: Directory to write checksum.txt into.
        checksums: Dict of {filename: sha256_hex}.

    Returns:
        Path to the written checksum.txt file.
    """
    checksum_path = release_dir / "checksum.txt"
    lines: list[str] = []
    for filename in sorted(checksums.keys()):
        lines.append(f"{checksums[filename]}  {filename}")

    content = "\n".join(lines) + "\n"
    checksum_path.write_text(content, encoding="utf-8")

    _logger.info(
        "Checksum file written",
        extra={"path": str(checksum_path), "entries": len(lines)},
    )
    return checksum_path


def parse_checksum_file(checksum_path: Path) -> dict[str, str]:
    """
    Parse a checksum.txt file into a dict of {filename: sha256_hex}.

    Args:
        checksum_path: Path to the checksum.txt file.

    Returns:
        Dict mapping filename to expected SHA256 hex digest.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file format is invalid.
    """
    if not checksum_path.is_file():
        raise FileNotFoundError(f"Checksum file not found: {checksum_path}")

    checksums: dict[str, str] = {}
    content = checksum_path.read_text(encoding="utf-8")

    for line_num, line in enumerate(content.strip().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid checksum format at line {line_num}: expected "
                f"'<sha256>  <filename>', got: {line!r}"
            )
        sha256_hex, filename = parts
        if len(sha256_hex) != 64:
            raise ValueError(
                f"Invalid SHA256 hash length at line {line_num}: "
                f"expected 64 chars, got {len(sha256_hex)}"
            )
        checksums[filename] = sha256_hex

    return checksums


def verify_checksums(release_dir: Path) -> VerificationResult:
    """
    Verify all checksums in a release directory against checksum.txt.

    Fail-fast: reports ALL mismatches and missing files, not just the first.

    Args:
        release_dir: Path to the release directory containing checksum.txt.

    Returns:
        VerificationResult with pass/fail status and details.
    """
    checksum_path = release_dir / "checksum.txt"
    if not checksum_path.is_file():
        return VerificationResult(
            is_valid=False,
            checked_count=0,
            errors=["checksum.txt not found in release directory"],
        )

    try:
        expected = parse_checksum_file(checksum_path)
    except ValueError as err:
        return VerificationResult(
            is_valid=False,
            checked_count=0,
            errors=[f"Failed to parse checksum.txt: {err}"],
        )

    mismatches: list[str] = []
    missing_files: list[str] = []
    checked = 0

    for filename, expected_hash in sorted(expected.items()):
        file_path = release_dir / filename
        if not file_path.is_file():
            missing_files.append(filename)
            _logger.error(
                "File missing during verification",
                extra={"file": filename},
            )
            continue

        actual_hash = compute_sha256(file_path)
        checked += 1

        if actual_hash != expected_hash:
            mismatches.append(filename)
            _logger.error(
                "Checksum mismatch",
                extra={
                    "file": filename,
                    "expected": expected_hash[:16] + "...",
                    "actual": actual_hash[:16] + "...",
                },
            )
        else:
            _logger.debug(
                "Checksum verified",
                extra={"file": filename},
            )

    is_valid = len(mismatches) == 0 and len(missing_files) == 0

    if is_valid:
        _logger.info(
            "All checksums verified",
            extra={"checked_count": checked},
        )
    else:
        _logger.error(
            "Checksum verification failed",
            extra={
                "mismatches": len(mismatches),
                "missing": len(missing_files),
            },
        )

    return VerificationResult(
        is_valid=is_valid,
        checked_count=checked,
        mismatches=mismatches,
        missing_files=missing_files,
    )
