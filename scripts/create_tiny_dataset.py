#!/usr/bin/env python3
# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Create a tiny synthetic dataset for pipeline smoke tests.

Generates deterministic tokenized shards (JSON format) and a minimal
BPE tokenizer bundle so the full pipeline can run without real data.

Usage:
    python scripts/create_tiny_dataset.py [--output-dir datasets/dev-small]
"""

import argparse
import json
import random
import struct
import sys
from pathlib import Path


def create_tokenizer_bundle(output_dir: Path, vocab_size: int = 16384) -> None:
    """
    Create a minimal BPE tokenizer using the `tokenizers` library.

    This produces the tokenizer.json that the training engine and serving
    loader expect under data/tokenizer/.
    """
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers

        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # Train on a small synthetic Rust-like corpus for vocab
        corpus = [
            "fn main() { println!(\"Hello, world!\"); }",
            "pub struct Point { x: f64, y: f64 }",
            "impl Point { pub fn new(x: f64, y: f64) -> Self { Self { x, y } } }",
            "let mut vec: Vec<i32> = Vec::new();",
            "for i in 0..10 { vec.push(i); }",
            "match result { Ok(val) => val, Err(e) => panic!(\"{}\", e) }",
            "#[derive(Debug, Clone)] pub enum Token { Ident(String), Number(i64) }",
            "use std::collections::HashMap;",
            "pub trait Iterator { type Item; fn next(&mut self) -> Option<Self::Item>; }",
            "async fn fetch(url: &str) -> Result<String, Box<dyn std::error::Error>> { todo!() }",
        ] * 50  # Repeat to give the trainer enough data

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
            min_frequency=1,
        )
        tokenizer.train_from_iterator(corpus, trainer=trainer)

        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(output_dir / "tokenizer.json"))

        # Write metadata
        metadata = {
            "vocab_size": tokenizer.get_vocab_size(),
            "tokenizer_type": "bpe",
            "version_hash": "synthetic_smoke_test",
            "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"],
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        print(f"  Tokenizer bundle created: {output_dir}")
        print(f"  Vocab size: {tokenizer.get_vocab_size()}")

    except ImportError:
        print("  WARNING: tokenizers package not available, creating stub tokenizer.json")
        # Create a minimal stub that the loader can parse
        stub = {
            "version": "1.0",
            "model": {"type": "BPE", "vocab": {}, "merges": []},
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "tokenizer.json").write_text(
            json.dumps(stub, indent=2), encoding="utf-8"
        )
        metadata = {
            "vocab_size": vocab_size,
            "tokenizer_type": "bpe",
            "version_hash": "synthetic_stub",
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )


def create_shards(
    output_dir: Path,
    num_shards: int = 4,
    tokens_per_shard: int = 50_000,
    vocab_size: int = 16384,
    seed: int = 42,
) -> None:
    """
    Generate deterministic tokenized shard files.

    Each shard is a JSON file with a "tokens" key containing a list of
    integer token IDs in range [0, vocab_size). The dataloader reads
    these directly.
    """
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    shard_files = []

    for i in range(num_shards):
        tokens = [rng.randint(0, vocab_size - 1) for _ in range(tokens_per_shard)]
        shard_name = f"shard_{i:04d}.json"
        shard_path = output_dir / shard_name

        shard_path.write_text(
            json.dumps({"tokens": tokens}), encoding="utf-8"
        )

        shard_files.append(shard_name)
        total_tokens += len(tokens)
        print(f"  Shard {shard_name}: {len(tokens):,} tokens")

    # Write manifest
    manifest = {
        "version": "1.0.0",
        "total_tokens": total_tokens,
        "num_shards": num_shards,
        "vocab_size": vocab_size,
        "seed": seed,
        "shards": shard_files,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"\n  Total: {total_tokens:,} tokens across {num_shards} shards")
    print(f"  Manifest: {output_dir / 'manifest.json'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create synthetic dataset for M31R pipeline smoke tests"
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/dev-small",
        help="Output directory for dataset shards (default: datasets/dev-small)",
    )
    parser.add_argument(
        "--tokenizer-dir",
        default="data/tokenizer",
        help="Output directory for tokenizer bundle (default: data/tokenizer)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=4,
        help="Number of shard files to create (default: 4)",
    )
    parser.add_argument(
        "--tokens-per-shard",
        type=int,
        default=50_000,
        help="Tokens per shard (default: 50000)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=16384,
        help="Vocabulary size (default: 16384)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / args.output_dir
    tokenizer_dir = project_root / args.tokenizer_dir

    print("=" * 60)
    print("M31R Tiny Dataset Generator")
    print("=" * 60)

    # Step 1: Create tokenizer bundle
    print(f"\n[1/2] Creating tokenizer bundle at {tokenizer_dir}")
    create_tokenizer_bundle(tokenizer_dir, vocab_size=args.vocab_size)

    # Step 2: Create dataset shards
    print(f"\n[2/2] Creating dataset shards at {dataset_dir}")
    create_shards(
        output_dir=dataset_dir,
        num_shards=args.num_shards,
        tokens_per_shard=args.tokens_per_shard,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("Done. Ready for: m31r train --config configs/train_tiny.yaml")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
