# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Vocabulary management — extracting, sorting, and writing vocab files.

The vocab and merges files are human-readable artifacts that let you
inspect what the tokenizer learned without needing to load the full
tokenizer.json. They're also useful for debugging fragmentation issues.
"""

import json
from pathlib import Path

from tokenizers import Tokenizer

from m31r.utils.filesystem import atomic_write


def extract_vocab(tokenizer: Tokenizer) -> dict[str, int]:
    """
    Pull the full vocabulary out of a trained tokenizer.

    Returns a dict mapping token strings to their integer IDs, sorted
    by ID. The sorting is important for determinism — Python dicts
    preserve insertion order, so sorting by ID here means the vocab
    file will always come out the same way.
    """
    raw_vocab = tokenizer.get_vocab()
    return dict(sorted(raw_vocab.items(), key=lambda item: item[1]))


def write_vocab_file(vocab: dict[str, int], output_path: Path) -> None:
    """
    Write a plain-text vocab file with one token per line.

    Format is "token\\tID" on each line, sorted by ID. This is the
    standard format that most tokenizer tools expect.
    """
    lines: list[str] = []
    for token, token_id in vocab.items():
        lines.append(f"{token}\t{token_id}")

    content = "\n".join(lines) + "\n"
    atomic_write(output_path, content)


def write_merges_file(tokenizer: Tokenizer, output_path: Path) -> None:
    """
    Write the BPE merges file if the tokenizer was trained with BPE.

    The merges file records the order in which byte pairs were merged
    during training. It's the core of how BPE works — the model applies
    these merges in order to tokenize new text. This file is only
    meaningful for BPE tokenizers; Unigram models don't have merges.

    We extract merges from the tokenizer's JSON representation because
    the tokenizers library doesn't expose a direct merges accessor.
    """
    tok_json = json.loads(tokenizer.to_str())
    model_data = tok_json.get("model", {})
    merges = model_data.get("merges", [])

    if not merges:
        atomic_write(output_path, "")
        return

    content = "\n".join(merges) + "\n"
    atomic_write(output_path, content)
