# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tokenizer artifact bundle creator.

After training a tokenizer, we need to package it into a self-contained
directory with everything needed to reproduce or use the result. This is
the tokenizer equivalent of the dataset builder's versioned output.

The bundle contains:
  - tokenizer.json     — the full serialized tokenizer (what you load to encode)
  - vocab.txt          — human-readable vocabulary listing
  - merges.txt         — BPE merge rules (empty for Unigram)
  - config_snapshot.yaml — frozen copy of the config used for training
  - metadata.json      — version hash, timestamps, vocab stats
  - checksum.txt       — SHA256 hashes of every file for integrity verification
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import yaml
from tokenizers import Tokenizer

from m31r.config.schema import TokenizerConfig
from m31r.logging.logger import get_logger
from m31r.tokenizer.vocab.core import extract_vocab, write_merges_file, write_vocab_file
from m31r.utils.filesystem import atomic_write
from m31r.utils.hashing import compute_sha256, compute_sha256_bytes
from m31r.utils.paths import ensure_directory


class BundleResult(NamedTuple):
    """What you get back after writing a tokenizer bundle."""

    output_directory: str
    version_hash: str
    vocab_size: int
    file_count: int


def _write_tokenizer_json(tokenizer: Tokenizer, output_path: Path) -> None:
    """Save the full tokenizer to its canonical JSON format."""
    tokenizer.save(str(output_path))


def _write_config_snapshot(config: TokenizerConfig, output_path: Path) -> None:
    """
    Freeze the training config as a YAML snapshot alongside the tokenizer.

    This way you can always look at a tokenizer bundle and know exactly
    what settings produced it. The snapshot is a plain dump of the pydantic
    model — no transformations, no filtering.
    """
    config_dict = config.model_dump()
    content = yaml.dump(config_dict, default_flow_style=False, sort_keys=True)
    atomic_write(output_path, content)


def _write_metadata(
    config: TokenizerConfig,
    tokenizer: Tokenizer,
    version_hash: str,
    dataset_hash: str,
    output_path: Path,
) -> None:
    """
    Write the metadata file that ties everything together.

    This is the manifest for the tokenizer bundle. It records what produced
    the tokenizer (config hash, dataset hash, seed) and when, so you can
    trace any tokenizer artifact back to its exact inputs.
    """
    config_bytes = json.dumps(config.model_dump(), sort_keys=True).encode("utf-8")
    config_hash = compute_sha256_bytes(config_bytes)

    metadata = {
        "version_hash": version_hash,
        "dataset_hash": dataset_hash,
        "config_hash": config_hash,
        "tokenizer_type": config.tokenizer_type,
        "vocab_size": tokenizer.get_vocab_size(),
        "target_vocab_size": config.vocab_size,
        "seed": config.seed,
        "special_tokens": list(config.special_tokens),
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    content = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    atomic_write(output_path, content)


def _write_checksums(output_dir: Path, files_to_hash: list[str]) -> None:
    """
    Compute and write SHA256 checksums for all bundle files.

    The format matches the BSD checksum convention: "hash  filename" on
    each line. This is the same format used by the dataset pipeline's
    checksums.txt, so the verification tool works on both.
    """
    lines: list[str] = []
    for filename in sorted(files_to_hash):
        file_path = output_dir / filename
        if file_path.is_file():
            file_hash = compute_sha256(file_path)
            lines.append(f"{file_hash}  {filename}")

    content = "\n".join(lines) + "\n"
    atomic_write(output_dir / "checksum.txt", content)


def _compute_version_hash(output_dir: Path, files: list[str]) -> str:
    """
    Derive a single version hash from all the bundle's content files.

    We hash each file individually, concatenate those hashes in sorted
    filename order, then hash the concatenation. This gives us a
    content-addressable identifier for the entire bundle — if any file
    changes, even by a single byte, the version hash changes too.
    """
    file_hashes: list[str] = []
    for filename in sorted(files):
        file_path = output_dir / filename
        if file_path.is_file():
            file_hashes.append(compute_sha256(file_path))

    combined = "\n".join(file_hashes).encode("utf-8")
    return compute_sha256_bytes(combined)


def create_bundle(
    tokenizer: Tokenizer,
    config: TokenizerConfig,
    project_root: Path,
    dataset_hash: str = "",
) -> BundleResult:
    """
    Write a complete tokenizer artifact bundle to disk.

    This is the top-level function that creates the whole package. Here's
    what happens step by step:

    1. Create the output directory (data/tokenizer/)
    2. Save tokenizer.json — the serialized tokenizer
    3. Extract and write vocab.txt — human-readable token list
    4. Write merges.txt — BPE merge rules (empty for Unigram)
    5. Snapshot the config as YAML — so you know what settings were used
    6. Compute the version hash from all content files
    7. Write metadata.json — ties version, config, and dataset together
    8. Write checksum.txt — SHA256 of every file for integrity checks

    The result is an immutable, self-contained directory that you can
    move around, verify, and load without needing anything else.
    """
    logger = get_logger("m31r.tokenizer.artifacts")

    output_dir = ensure_directory(project_root / config.output_directory)

    logger.info(
        "Creating tokenizer bundle",
        extra={"output_dir": str(output_dir)},
    )

    _write_tokenizer_json(tokenizer, output_dir / "tokenizer.json")

    vocab = extract_vocab(tokenizer)
    write_vocab_file(vocab, output_dir / "vocab.txt")
    write_merges_file(tokenizer, output_dir / "merges.txt")
    _write_config_snapshot(config, output_dir / "config_snapshot.yaml")

    content_files = ["tokenizer.json", "vocab.txt", "merges.txt", "config_snapshot.yaml"]
    version_hash = _compute_version_hash(output_dir, content_files)

    _write_metadata(config, tokenizer, version_hash, dataset_hash, output_dir / "metadata.json")

    all_files = content_files + ["metadata.json"]
    _write_checksums(output_dir, all_files)

    file_count = len(all_files) + 1  # +1 for checksum.txt itself

    logger.info(
        "Tokenizer bundle created",
        extra={
            "version_hash": version_hash,
            "vocab_size": tokenizer.get_vocab_size(),
            "file_count": file_count,
            "output_dir": str(output_dir),
        },
    )

    return BundleResult(
        output_directory=str(output_dir),
        version_hash=version_hash,
        vocab_size=tokenizer.get_vocab_size(),
        file_count=file_count,
    )
