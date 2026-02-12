#!/usr/bin/env python3
# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Generate a deterministic synthetic Rust dataset for overfit sanity testing.

Produces repeated simple Rust functions as tokenized JSON shards in
datasets/dev-overfit/. The data is entirely synthetic — no randomness,
no external sources, no downloads. The same invocation always produces
the identical output.

Usage:
    python scripts/generate_synthetic_rust.py [--output-dir datasets/dev-overfit]
"""

import argparse
import json
import sys
from pathlib import Path


# ── Synthetic Rust corpus ────────────────────────────────────────────

RUST_FUNCTIONS: list[str] = [
    "fn add(a: i32, b: i32) -> i32 { a + b }",
    "fn sub(a: i32, b: i32) -> i32 { a - b }",
    "fn mul(a: i32, b: i32) -> i32 { a * b }",
    "fn div(a: i32, b: i32) -> i32 { a / b }",
    "fn square(x: i32) -> i32 { x * x }",
    "fn double(x: i32) -> i32 { x + x }",
    "fn negate(x: i32) -> i32 { -x }",
    "fn max(a: i32, b: i32) -> i32 { if a > b { a } else { b } }",
    "fn min(a: i32, b: i32) -> i32 { if a < b { a } else { b } }",
    "fn abs(x: i32) -> i32 { if x < 0 { -x } else { x } }",
]


def build_corpus_text(target_bytes: int) -> str:
    """
    Build a deterministic corpus by cycling through RUST_FUNCTIONS
    until we reach approximately target_bytes of text.

    Each function is separated by a newline. The corpus is deterministic:
    same target_bytes always produces the same output.
    """
    parts: list[str] = []
    total_size: int = 0
    idx: int = 0
    num_functions: int = len(RUST_FUNCTIONS)

    while total_size < target_bytes:
        fn_text = RUST_FUNCTIONS[idx % num_functions]
        parts.append(fn_text)
        total_size += len(fn_text) + 1  # +1 for newline separator
        idx += 1

    return "\n".join(parts)


def create_tokenizer_bundle(output_dir: Path, vocab_size: int = 256) -> object:
    """
    Create a minimal BPE tokenizer trained on our synthetic Rust corpus.

    Returns the tokenizer object so we can use it for encoding shards.
    """
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    except ImportError:
        print("ERROR: tokenizers package required. Install with: pip install tokenizers",
              file=sys.stderr)
        sys.exit(1)

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Train on the synthetic corpus — deterministic, same vocab every time
    corpus = [fn for fn in RUST_FUNCTIONS] * 100

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=1,
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_dir / "tokenizer.json"))

    metadata = {
        "vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_type": "bpe",
        "version_hash": "overfit_sanity_test",
        "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"  Tokenizer bundle created: {output_dir}")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")
    return tokenizer


def create_shards(
    output_dir: Path,
    tokenizer: object,
    num_shards: int = 4,
    target_bytes_per_shard: int = 800_000,
) -> None:
    """
    Generate tokenized shard files from the synthetic Rust corpus.

    Each shard is a JSON file with {"tokens": [...]} containing integer
    token IDs. The data is purely deterministic — no randomness anywhere.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total_tokens: int = 0
    shard_files: list[str] = []

    for i in range(num_shards):
        # Each shard encodes the same cycled corpus — intentional for overfit
        corpus_text = build_corpus_text(target_bytes_per_shard)
        encoding = tokenizer.encode(corpus_text)
        tokens: list[int] = encoding.ids

        shard_name = f"shard_{i:04d}.json"
        shard_path = output_dir / shard_name
        shard_path.write_text(
            json.dumps({"tokens": tokens}), encoding="utf-8"
        )

        shard_files.append(shard_name)
        total_tokens += len(tokens)
        print(f"  Shard {shard_name}: {len(tokens):,} tokens ({len(corpus_text):,} bytes source)")

    # Write manifest
    manifest = {
        "version": "1.0.0",
        "total_tokens": total_tokens,
        "num_shards": num_shards,
        "shards": shard_files,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"\n  Total: {total_tokens:,} tokens across {num_shards} shards")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Rust dataset for overfit sanity testing"
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/dev-overfit",
        help="Output directory for dataset shards (default: datasets/dev-overfit)",
    )
    parser.add_argument(
        "--tokenizer-dir",
        default="data/tokenizer-overfit",
        help="Output directory for tokenizer bundle (default: data/tokenizer-overfit)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=4,
        help="Number of shard files to create (default: 4)",
    )
    parser.add_argument(
        "--target-bytes-per-shard",
        type=int,
        default=800_000,
        help="Approximate bytes of source text per shard (default: 800000)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=256,
        help="Tokenizer vocabulary size (default: 256)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / args.output_dir
    tokenizer_dir = project_root / args.tokenizer_dir

    print("=" * 60)
    print("M31R Synthetic Rust Dataset Generator (Overfit Test)")
    print("=" * 60)

    # Step 1: Create tokenizer
    print(f"\n[1/2] Creating tokenizer bundle at {tokenizer_dir}")
    tokenizer = create_tokenizer_bundle(tokenizer_dir, vocab_size=args.vocab_size)

    # Step 2: Create shards
    print(f"\n[2/2] Creating dataset shards at {dataset_dir}")
    create_shards(
        output_dir=dataset_dir,
        tokenizer=tokenizer,
        num_shards=args.num_shards,
        target_bytes_per_shard=args.target_bytes_per_shard,
    )

    print("\n" + "=" * 60)
    print("Done. Ready for overfit training.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
