# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Release packager — bundles model, tokenizer, config, metadata, and checksums
into a single immutable release directory.

Per 18_RELEASE_PROCESS.md §13, a valid release directory must contain:

    release/<version>/
    ├─ model.safetensors
    ├─ tokenizer.json
    ├─ config.yaml
    ├─ metadata.json
    ├─ checksum.txt
    └─ README.txt

Per §18: Release must be created via `m31r export`. Manual packaging forbidden.
Per §21: Once released, artifacts must never change. If change required → new version.
"""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from m31r.logging.logger import get_logger
from m31r.release.checksums.integrity import generate_checksums, write_checksum_file
from m31r.release.manifests.manifest import (
    ArtifactManifest,
    create_manifest,
    write_manifest,
)
from m31r.utils.hashing import compute_sha256

_logger: logging.Logger = get_logger(__name__)

# These are the only files allowed in a release. Anything else is a violation.
REQUIRED_RELEASE_FILES: frozenset[str] = frozenset(
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
class ReleaseResult:
    """Outcome of a successful release creation."""

    version: str
    output_dir: str
    manifest: ArtifactManifest
    file_count: int
    checksum_count: int


def _find_model_weights(checkpoint_dir: Path) -> Path:
    """
    Locate model weights file in a checkpoint directory.

    Looks for safetensors first, then falls back to .pt/.bin files.
    """
    # Prefer safetensors format
    for pattern in ["*.safetensors", "model.safetensors"]:
        matches = list(checkpoint_dir.glob(pattern))
        if matches:
            return matches[0]

    # Fallback to PyTorch formats
    for pattern in ["*.pt", "*.bin", "model_weights.*"]:
        matches = list(checkpoint_dir.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"No model weights found in {checkpoint_dir}. " f"Expected .safetensors, .pt, or .bin file."
    )


def _find_tokenizer(tokenizer_dir: Path) -> Path:
    """Locate the tokenizer.json file."""
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    if tokenizer_path.is_file():
        return tokenizer_path

    # Check one level deep
    for sub in tokenizer_dir.iterdir():
        if sub.is_dir():
            candidate = sub / "tokenizer.json"
            if candidate.is_file():
                return candidate

    raise FileNotFoundError(f"No tokenizer.json found in {tokenizer_dir}")


def _find_config(config_path: Path | None, checkpoint_dir: Path) -> Path:
    """
    Locate the config file — explicit path or config snapshot from checkpoint.
    """
    if config_path is not None and config_path.is_file():
        return config_path

    # Check for config snapshot in checkpoint directory
    snapshot = checkpoint_dir / "config_snapshot.yaml"
    if snapshot.is_file():
        return snapshot

    # Check parent experiment directory
    parent_config = checkpoint_dir.parent / "config_snapshot.yaml"
    if parent_config.is_file():
        return parent_config

    raise FileNotFoundError(
        "No config file found. Provide --config or ensure config_snapshot.yaml "
        "exists in the checkpoint/experiment directory."
    )


def _generate_readme(version: str, manifest: ArtifactManifest) -> str:
    """Generate a README.txt for the release."""
    return (
        f"M31R Release v{version}\n"
        f"{'=' * 40}\n"
        f"\n"
        f"Version:        {version}\n"
        f"M31R Version:   {manifest.m31r_version}\n"
        f"Git Commit:     {manifest.git_commit}\n"
        f"Training Seed:  {manifest.training_seed}\n"
        f"Created:        {manifest.timestamp}\n"
        f"\n"
        f"Files:\n"
        f"  model.safetensors  — Trained model weights\n"
        f"  tokenizer.json     — Tokenizer vocabulary\n"
        f"  config.yaml        — Frozen training configuration\n"
        f"  metadata.json      — Release manifest with hashes\n"
        f"  checksum.txt       — SHA256 checksums for verification\n"
        f"\n"
        f"Verification:\n"
        f"  m31r verify --release-dir <this-directory>\n"
        f"\n"
        f"This release is immutable. Any modifications invalidate the checksums.\n"
        f"To make changes, create a new release version.\n"
    )


def create_release(
    version: str,
    checkpoint_dir: Path,
    tokenizer_dir: Path,
    output_root: Path,
    config_path: Path | None = None,
    training_seed: int = 42,
    m31r_version: str = "0.1.0",
    dataset_hash: str = "unknown",
    metrics_summary: dict[str, object] | None = None,
) -> ReleaseResult:
    """
    Create a complete, immutable release bundle.

    This is the core packaging function called by `m31r export`.
    It assembles all artifacts, generates checksums and metadata,
    and produces a self-contained release directory.

    Args:
        version: Semantic version string (e.g., "1.0.0").
        checkpoint_dir: Path to the checkpoint with model weights.
        tokenizer_dir: Path to the tokenizer bundle directory.
        output_root: Root directory for releases (release/<version>/ created inside).
        config_path: Optional explicit config file path.
        training_seed: Seed used during training.
        m31r_version: Version of the m31r package.
        dataset_hash: SHA256 of the training dataset.
        metrics_summary: Optional evaluation metrics.

    Returns:
        ReleaseResult with details of the created release.

    Raises:
        FileNotFoundError: If required source files are missing.
        FileExistsError: If the release version already exists (immutability).
    """
    release_dir = output_root / version
    if release_dir.exists():
        raise FileExistsError(
            f"Release v{version} already exists at {release_dir}. "
            f"Releases are immutable — use a new version number."
        )

    _logger.info(
        "Creating release",
        extra={"version": version, "output": str(release_dir)},
    )

    # Find source files
    model_path = _find_model_weights(checkpoint_dir)
    tokenizer_path = _find_tokenizer(tokenizer_dir)
    config_source = _find_config(config_path, checkpoint_dir)

    # Create release directory
    release_dir.mkdir(parents=True, exist_ok=False)

    try:
        # Copy model weights
        dst_model = release_dir / "model.safetensors"
        shutil.copy2(str(model_path), str(dst_model))
        _logger.info("Copied model weights", extra={"source": str(model_path)})

        # Copy tokenizer
        dst_tokenizer = release_dir / "tokenizer.json"
        shutil.copy2(str(tokenizer_path), str(dst_tokenizer))
        _logger.info("Copied tokenizer", extra={"source": str(tokenizer_path)})

        # Copy config
        dst_config = release_dir / "config.yaml"
        shutil.copy2(str(config_source), str(dst_config))
        _logger.info("Copied config", extra={"source": str(config_source)})

        # Compute hashes for manifest
        model_hash = compute_sha256(dst_model)
        tokenizer_hash = compute_sha256(dst_tokenizer)
        config_hash = compute_sha256(dst_config)

        # Create and write manifest (metadata.json)
        manifest = create_manifest(
            version=version,
            dataset_hash=dataset_hash,
            tokenizer_hash=tokenizer_hash,
            model_hash=model_hash,
            config_hash=config_hash,
            training_seed=training_seed,
            m31r_version=m31r_version,
            metrics_summary=metrics_summary,
        )
        write_manifest(manifest, release_dir / "metadata.json")

        # Write README
        readme_content = _generate_readme(version, manifest)
        (release_dir / "README.txt").write_text(readme_content, encoding="utf-8")

        # Generate and write checksums (covers all files including metadata.json and README.txt)
        checksums = generate_checksums(release_dir)
        write_checksum_file(release_dir, checksums)

        file_count = len(list(release_dir.iterdir()))

        _logger.info(
            "Release created successfully",
            extra={
                "version": version,
                "output_dir": str(release_dir),
                "file_count": file_count,
                "model_hash": model_hash[:16] + "...",
            },
        )

        return ReleaseResult(
            version=version,
            output_dir=str(release_dir),
            manifest=manifest,
            file_count=file_count,
            checksum_count=len(checksums),
        )

    except Exception:
        # Clean up partial release on failure — never leave a half-built release
        if release_dir.exists():
            shutil.rmtree(release_dir)
            _logger.warning(
                "Cleaned up partial release after failure",
                extra={"release_dir": str(release_dir)},
            )
        raise
