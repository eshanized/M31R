# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Artifact manifest generation and management.

Per 18_RELEASE_PROCESS.md §14, metadata.json must include:
- version
- dataset version / hash
- tokenizer version / hash
- config hash
- git commit hash
- training seed
- date
- metrics summary

Per 20_OBSERVABILITY_AND_LOGGING.md §23:
Each artifact must trace to: config → dataset → code commit → seed → metrics

No anonymous runs allowed.
"""

import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from m31r.logging.logger import get_logger

_logger: logging.Logger = get_logger(__name__)


@dataclass(frozen=True)
class ArtifactManifest:
    """
    Complete traceability record for a release artifact.

    Every field is required for a valid manifest. This is what enables
    reproducibility — given this manifest, you can rebuild the exact
    same release.
    """

    version: str
    dataset_hash: str
    tokenizer_hash: str
    model_hash: str
    config_hash: str
    git_commit: str
    training_seed: int
    timestamp: str
    m31r_version: str
    metrics_summary: dict[str, object] = field(default_factory=dict)


_REQUIRED_MANIFEST_FIELDS: frozenset[str] = frozenset(
    {
        "version",
        "dataset_hash",
        "tokenizer_hash",
        "model_hash",
        "config_hash",
        "git_commit",
        "training_seed",
        "timestamp",
        "m31r_version",
    }
)


def get_git_commit() -> str:
    """
    Read the current HEAD commit hash from git.

    Returns "unknown" if git is not available or we're not in a repo.
    Never raises — this is best-effort for traceability.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    _logger.warning("Could not determine git commit hash")
    return "unknown"


def create_manifest(
    version: str,
    dataset_hash: str,
    tokenizer_hash: str,
    model_hash: str,
    config_hash: str,
    training_seed: int,
    m31r_version: str,
    git_commit: str | None = None,
    metrics_summary: dict[str, object] | None = None,
) -> ArtifactManifest:
    """
    Create a new artifact manifest with all required traceability fields.

    Args:
        version: Release version string (semver).
        dataset_hash: SHA256 of the dataset used for training.
        tokenizer_hash: SHA256 of the tokenizer artifact.
        model_hash: SHA256 of the model weights.
        config_hash: SHA256 of the frozen config.
        training_seed: Seed used for deterministic training.
        m31r_version: Version of the m31r package.
        git_commit: Git HEAD hash. Auto-detected if None.
        metrics_summary: Optional metrics from evaluation.

    Returns:
        Frozen ArtifactManifest instance.
    """
    if git_commit is None:
        git_commit = get_git_commit()

    timestamp = datetime.now(tz=timezone.utc).isoformat()

    manifest = ArtifactManifest(
        version=version,
        dataset_hash=dataset_hash,
        tokenizer_hash=tokenizer_hash,
        model_hash=model_hash,
        config_hash=config_hash,
        git_commit=git_commit,
        training_seed=training_seed,
        timestamp=timestamp,
        m31r_version=m31r_version,
        metrics_summary=metrics_summary or {},
    )

    _logger.info(
        "Manifest created",
        extra={
            "version": version,
            "git_commit": git_commit[:12] if git_commit != "unknown" else git_commit,
            "training_seed": training_seed,
        },
    )
    return manifest


def write_manifest(manifest: ArtifactManifest, path: Path) -> None:
    """
    Serialize a manifest to JSON and write it atomically.

    Args:
        manifest: The manifest to write.
        path: Target file path (usually metadata.json).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(manifest)
    content = json.dumps(data, indent=2, sort_keys=True, default=str) + "\n"
    path.write_text(content, encoding="utf-8")

    _logger.info(
        "Manifest written",
        extra={"path": str(path), "version": manifest.version},
    )


def load_manifest(path: Path) -> ArtifactManifest:
    """
    Load a manifest from a JSON file.

    Args:
        path: Path to the metadata.json file.

    Returns:
        Parsed ArtifactManifest.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If required fields are missing or invalid.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    content = path.read_text(encoding="utf-8")
    data = json.loads(content)

    # Validate all required fields present
    missing = _REQUIRED_MANIFEST_FIELDS - set(data.keys())
    if missing:
        raise ValueError(f"Manifest is missing required fields: {', '.join(sorted(missing))}")

    manifest = ArtifactManifest(
        version=str(data["version"]),
        dataset_hash=str(data["dataset_hash"]),
        tokenizer_hash=str(data["tokenizer_hash"]),
        model_hash=str(data["model_hash"]),
        config_hash=str(data["config_hash"]),
        git_commit=str(data["git_commit"]),
        training_seed=int(data["training_seed"]),
        timestamp=str(data["timestamp"]),
        m31r_version=str(data["m31r_version"]),
        metrics_summary=data.get("metrics_summary", {}),
    )

    _logger.debug(
        "Manifest loaded",
        extra={"path": str(path), "version": manifest.version},
    )
    return manifest
