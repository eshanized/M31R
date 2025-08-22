# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Streaming corpus reader for tokenizer training.

The whole point of this module is to let the tokenizer trainer consume
dataset shards without loading everything into memory. Shards are JSONL
files where each line is a JSON object with a "content" field containing
source code text. We read them lazily, one line at a time.
"""

import json
from pathlib import Path
from typing import Iterator

from m31r.logging.logger import get_logger


def _find_shard_files(dataset_dir: Path) -> list[Path]:
    """
    Find all JSONL shard files in the dataset directory and its subdirectories.

    We sort the results so that the tokenizer always sees data in the same
    order across runs. Filesystem iteration order can vary between platforms,
    so explicit sorting is the only way to guarantee determinism here.
    """
    shard_files: list[Path] = []

    if not dataset_dir.is_dir():
        return shard_files

    for child in sorted(dataset_dir.iterdir()):
        if child.is_dir():
            for shard in sorted(child.glob("*.jsonl")):
                if shard.is_file():
                    shard_files.append(shard)
        elif child.is_file() and child.suffix == ".jsonl":
            shard_files.append(child)

    return shard_files


def stream_corpus(dataset_dir: Path) -> Iterator[str]:
    """
    Lazily yield text content from all dataset shards in deterministic order.

    This is the bridge between the dataset pipeline (Phase 2) and the
    tokenizer trainer. Each shard is a JSONL file where every line looks like:

        {"source": "...", "path": "...", "content": "fn main() { ... }", ...}

    We extract just the "content" field and yield it as a string. The caller
    (the tokenizer trainer) never needs to think about shards, files, or JSON —
    it just gets a stream of source code text.

    Why streaming matters: a full dataset can be 100GB+. Loading that into
    memory would kill the process. By reading line-by-line and yielding
    immediately, we keep memory usage constant regardless of dataset size.
    """
    logger = get_logger("m31r.tokenizer.streaming")
    shard_files = _find_shard_files(dataset_dir)

    if not shard_files:
        logger.warning(
            "No shard files found in dataset directory",
            extra={"dataset_dir": str(dataset_dir)},
        )
        return

    logger.info(
        "Starting corpus stream",
        extra={"shard_count": len(shard_files), "dataset_dir": str(dataset_dir)},
    )

    lines_yielded = 0
    for shard_path in shard_files:
        with open(shard_path, "r", encoding="utf-8") as shard_file:
            for raw_line in shard_file:
                stripped = raw_line.strip()
                if not stripped:
                    continue

                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    continue

                content = record.get("content")
                if content and isinstance(content, str):
                    lines_yielded += 1
                    yield content

    logger.info(
        "Corpus stream complete",
        extra={"lines_yielded": lines_yielded, "shard_count": len(shard_files)},
    )
