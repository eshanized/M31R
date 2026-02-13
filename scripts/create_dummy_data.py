#!/usr/bin/env python3
"""Create dummy training data for testing with correct vocab size."""

import json
import struct
from pathlib import Path

# Create dummy token data
vocab_size = 256  # Match the model vocab size
seq_length = 256
num_sequences = 100

# Generate random token sequences
tokens = []
for i in range(num_sequences):
    # Simple pattern: repeating sequence within vocab range
    seq = [(i + j) % vocab_size for j in range(seq_length)]
    tokens.extend(seq)

# Create dataset directory
dataset_dir = Path("data/datasets")
dataset_dir.mkdir(parents=True, exist_ok=True)

# Write as binary shard
shard_path = dataset_dir / "shard_000.bin"
with open(shard_path, "wb") as f:
    f.write(struct.pack(f"<{len(tokens)}i", *tokens))

# Create manifest
manifest = {
    "version": "test_v1",
    "num_tokens": len(tokens),
    "num_sequences": num_sequences,
    "vocab_size": vocab_size,
    "shards": ["shard_000.bin"],
}

manifest_path = dataset_dir / "manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Created dummy dataset:")
print(f"  Tokens: {len(tokens)}")
print(f"  Sequences: {num_sequences}")
print(f"  Vocab size: {vocab_size}")
print(f"  Shard: {shard_path}")
print(f"  Manifest: {manifest_path}")
