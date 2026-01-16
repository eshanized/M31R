# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Streaming deterministic token dataloader for M31R.

Per 07_TRAINING_ARCHITECTURE.md §5:
  - Data loading must be deterministic and reproducible
  - Sequence packing: concatenate tokenized files into continuous token stream
  - Fixed context-length windows sliced from the stream
  - Shuffling controlled by seed
  - Must support resume from arbitrary position

The dataloader reads pre-tokenized shards, concatenates them into a flat
token buffer, and yields fixed-length windows. A seeded RNG controls
shard ordering. Resume is handled by tracking the global token offset.
"""

import json
import logging
from pathlib import Path
from typing import Iterator

import torch

from m31r.logging.logger import get_logger

logger: logging.Logger = get_logger(__name__)


class TokenDataset:
    """
    A streaming dataset that yields fixed-length token windows.

    Loads tokenized shards from disk, concatenates into a flat token buffer,
    and serves (input, target) pairs where target is input shifted by one
    position (next-token prediction).

    Determinism: shard ordering is controlled by a seeded generator.
    Resume: set `start_offset` to skip tokens already consumed.

    Args:
        shard_dir: Directory containing tokenized shard files (*.bin).
        seq_len: Length of each token window.
        seed: Random seed for deterministic shard ordering.
        start_offset: Number of windows to skip (for resume).
    """

    def __init__(
        self,
        shard_dir: Path,
        seq_len: int,
        seed: int = 42,
        start_offset: int = 0,
    ) -> None:
        self.shard_dir = shard_dir
        self.seq_len = seq_len
        self.seed = seed
        self.start_offset = start_offset
        self._shard_files: list[Path] = []
        self._discover_shards()

    def _discover_shards(self) -> None:
        """Find all shard files and sort deterministically."""
        if not self.shard_dir.is_dir():
            logger.warning(
                "Shard directory not found",
                extra={"shard_dir": str(self.shard_dir)},
            )
            return

        # Look for binary shard files or JSON shard manifests
        bin_shards = sorted(self.shard_dir.glob("*.bin"))
        json_shards = sorted(self.shard_dir.glob("shard_*.json"))

        if bin_shards:
            self._shard_files = bin_shards
        elif json_shards:
            self._shard_files = json_shards

        logger.debug(
            "Discovered shards",
            extra={"count": len(self._shard_files), "dir": str(self.shard_dir)},
        )

    def _load_tokens_from_shard(self, shard_path: Path) -> list[int]:
        """Load token IDs from a single shard file."""
        if shard_path.suffix == ".bin":
            data = shard_path.read_bytes()
            # Assume 4-byte little-endian integers
            import struct

            count = len(data) // 4
            return list(struct.unpack(f"<{count}i", data[: count * 4]))
        elif shard_path.suffix == ".json":
            content = json.loads(shard_path.read_text(encoding="utf-8"))
            if isinstance(content, dict) and "tokens" in content:
                return content["tokens"]
            if isinstance(content, list):
                return content
        return []

    def _build_token_stream(self) -> list[int]:
        """
        Load all shards in deterministic order and concatenate tokens.

        The shard order is shuffled by a seeded RNG so every run with
        the same seed produces the same token stream.
        """
        # Deterministic shuffle of shard order
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        indices = torch.randperm(len(self._shard_files), generator=generator).tolist()

        all_tokens: list[int] = []
        for idx in indices:
            shard_path = self._shard_files[idx]
            tokens = self._load_tokens_from_shard(shard_path)
            all_tokens.extend(tokens)

        logger.info(
            "Token stream built",
            extra={"total_tokens": len(all_tokens), "shards": len(self._shard_files)},
        )
        return all_tokens

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """
        Yield (input, target) pairs of shape (seq_len,).

        Target is the input shifted right by one position (next-token prediction).
        Skips `start_offset` windows for resume support.
        """
        tokens = self._build_token_stream()

        if len(tokens) < self.seq_len + 1:
            logger.warning(
                "Not enough tokens for even one window",
                extra={"total_tokens": len(tokens), "seq_len": self.seq_len},
            )
            return

        # Calculate total number of windows
        n_windows = (len(tokens) - 1) // self.seq_len

        for window_idx in range(self.start_offset, n_windows):
            start = window_idx * self.seq_len
            end = start + self.seq_len

            input_ids = torch.tensor(tokens[start:end], dtype=torch.long)
            target_ids = torch.tensor(tokens[start + 1 : end + 1], dtype=torch.long)
            yield input_ids, target_ids

    def total_windows(self) -> int:
        """Calculate total number of token windows available."""
        tokens = self._build_token_stream()
        if len(tokens) < self.seq_len + 1:
            return 0
        return (len(tokens) - 1) // self.seq_len


def create_dataloader(
    dataset: TokenDataset,
    batch_size: int,
    seed: int = 42,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """
    Create a batched dataloader from a TokenDataset.

    Collects individual windows into mini-batches of shape (batch_size, seq_len).
    Drops the last incomplete batch to maintain deterministic batch sizes.

    Args:
        dataset: The TokenDataset to draw from.
        batch_size: Number of sequences per batch.
        seed: Random seed (unused currently, reserved for future shuffling).

    Yields:
        Tuple of (input_batch, target_batch), each (batch_size, seq_len).
    """
    batch_inputs: list[torch.Tensor] = []
    batch_targets: list[torch.Tensor] = []

    for input_ids, target_ids in dataset:
        batch_inputs.append(input_ids)
        batch_targets.append(target_ids)

        if len(batch_inputs) == batch_size:
            yield (
                torch.stack(batch_inputs),
                torch.stack(batch_targets),
            )
            batch_inputs = []
            batch_targets = []
