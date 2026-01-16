# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Release verification — validates a release directory is complete and intact.

Per 18_RELEASE_PROCESS.md §19 (Validation Step):
After packaging, must:
- load model (check file exists and is readable)
- verify checksums
- verify manifest completeness

Per §28 (Acceptance Criteria):
Release accepted only if: reproducible, tested, benchmarked, versioned,
packaged, validated. Otherwise rejected.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from m31r.logging.logger import get_logger
from m31r.release.checksums.integrity import verify_checksums
from m31r.release.manifests.manifest import _REQUIRED_MANIFEST_FIELDS

_logger: logging.Logger = get_logger(__name__)

# Every valid release must contain exactly these files
REQUIRED_FILES: frozenset[str] = frozenset(
    {
        "model.safetensors",
        "tokenizer.json",
        "config.yaml",
        "metadata.json",
        "checksum.txt",
        "README.txt",
    }
)


@dataclass(frozen=True)
class VerificationReport:
    """Complete outcome of a release verification."""

    is_valid: bool
    release_dir: str
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _check_required_files(release_dir: Path) -> tuple[bool, list[str]]:
    """Check that all required release files are present."""
    missing: list[str] = []
    for filename in sorted(REQUIRED_FILES):
        file_path = release_dir / filename
        if not file_path.is_file():
            missing.append(filename)

    return len(missing) == 0, missing


def _check_metadata_valid(release_dir: Path) -> tuple[bool, list[str]]:
    """Validate that metadata.json is parseable and has all required fields."""
    metadata_path = release_dir / "metadata.json"
    errors: list[str] = []

    if not metadata_path.is_file():
        return False, ["metadata.json not found"]

    try:
        content = metadata_path.read_text(encoding="utf-8")
        data = json.loads(content)
    except (json.JSONDecodeError, OSError) as err:
        return False, [f"Failed to parse metadata.json: {err}"]

    if not isinstance(data, dict):
        return False, ["metadata.json root is not a JSON object"]

    missing = _REQUIRED_MANIFEST_FIELDS - set(data.keys())
    if missing:
        errors.append(f"Missing manifest fields: {', '.join(sorted(missing))}")

    # Validate field types
    if "training_seed" in data and not isinstance(data["training_seed"], int):
        errors.append("training_seed must be an integer")

    if "version" in data and not isinstance(data["version"], str):
        errors.append("version must be a string")

    return len(errors) == 0, errors


def _check_config_readable(release_dir: Path) -> tuple[bool, list[str]]:
    """Verify config.yaml can be read."""
    config_path = release_dir / "config.yaml"
    if not config_path.is_file():
        return False, ["config.yaml not found"]

    try:
        content = config_path.read_text(encoding="utf-8")
        if len(content.strip()) == 0:
            return False, ["config.yaml is empty"]
    except OSError as err:
        return False, [f"Cannot read config.yaml: {err}"]

    return True, []


def _check_model_readable(release_dir: Path) -> tuple[bool, list[str]]:
    """Verify model weights file exists and is non-empty."""
    model_path = release_dir / "model.safetensors"
    if not model_path.is_file():
        return False, ["model.safetensors not found"]

    if model_path.stat().st_size == 0:
        return False, ["model.safetensors is empty (0 bytes)"]

    return True, []


def verify_release(release_dir: Path) -> VerificationReport:
    """
    Run full verification suite on a release directory.

    Checks performed (in order):
    1. Directory exists
    2. All required files present
    3. Model weights readable and non-empty
    4. Config readable and non-empty
    5. Metadata.json valid with all required fields
    6. All checksums match

    Args:
        release_dir: Path to the release directory to verify.

    Returns:
        VerificationReport with complete pass/fail details.
    """
    if not release_dir.is_dir():
        return VerificationReport(
            is_valid=False,
            release_dir=str(release_dir),
            checks_failed=["directory_exists"],
            errors=[f"Release directory not found: {release_dir}"],
        )

    passed: list[str] = []
    failed: list[str] = []
    all_errors: list[str] = []

    # Check 1: Required files
    ok, missing = _check_required_files(release_dir)
    if ok:
        passed.append("required_files")
    else:
        failed.append("required_files")
        all_errors.extend([f"Missing required file: {f}" for f in missing])

    # Check 2: Model readable
    ok, errors = _check_model_readable(release_dir)
    if ok:
        passed.append("model_readable")
    else:
        failed.append("model_readable")
        all_errors.extend(errors)

    # Check 3: Config readable
    ok, errors = _check_config_readable(release_dir)
    if ok:
        passed.append("config_readable")
    else:
        failed.append("config_readable")
        all_errors.extend(errors)

    # Check 4: Metadata valid
    ok, errors = _check_metadata_valid(release_dir)
    if ok:
        passed.append("metadata_valid")
    else:
        failed.append("metadata_valid")
        all_errors.extend(errors)

    # Check 5: Checksums match
    checksum_result = verify_checksums(release_dir)
    if checksum_result.is_valid:
        passed.append("checksums_valid")
    else:
        failed.append("checksums_valid")
        if checksum_result.mismatches:
            all_errors.extend([f"Checksum mismatch: {f}" for f in checksum_result.mismatches])
        if checksum_result.missing_files:
            all_errors.extend(
                [f"Missing file in checksums: {f}" for f in checksum_result.missing_files]
            )
        all_errors.extend(checksum_result.errors)

    is_valid = len(failed) == 0

    if is_valid:
        _logger.info(
            "Release verification passed",
            extra={
                "release_dir": str(release_dir),
                "checks_passed": len(passed),
            },
        )
    else:
        _logger.error(
            "Release verification FAILED",
            extra={
                "release_dir": str(release_dir),
                "checks_passed": len(passed),
                "checks_failed": len(failed),
                "errors": all_errors,
            },
        )

    return VerificationReport(
        is_valid=is_valid,
        release_dir=str(release_dir),
        checks_passed=passed,
        checks_failed=failed,
        errors=all_errors,
    )
