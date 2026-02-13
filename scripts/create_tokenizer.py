#!/usr/bin/env python3
"""Create a simple tokenizer for testing."""

import json
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

# Create a simple tokenizer
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = ByteLevel()

# Train on dummy text
trainer = BpeTrainer(
    vocab_size=1000,
    special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    show_progress=True,
)

# Simple training text
texts = [
    'fn main() { println!("Hello, World!"); }',
    "let x = 42;",
    "pub struct Point { x: f64, y: f64 }",
    "impl Point { fn new(x: f64, y: f64) -> Self { Self { x, y } } }",
    "use std::collections::HashMap;",
    "pub fn add(a: i32, b: i32) -> i32 { a + b }",
    "match result { Ok(val) => val, Err(e) => panic!(e) }",
    'for i in 0..10 { println!("{}", i); }',
]

tokenizer.train_from_iterator(texts, trainer=trainer)

# Create tokenizer directory
tokenizer_dir = Path("data/tokenizer")
tokenizer_dir.mkdir(parents=True, exist_ok=True)

# Save tokenizer
tokenizer_path = tokenizer_dir / "tokenizer.json"
tokenizer.save(str(tokenizer_path))

# Create metadata
metadata = {
    "vocab_size": tokenizer.get_vocab_size(),
    "tokenizer_type": "bpe",
    "version_hash": "test_v1",
    "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"],
}

metadata_path = tokenizer_dir / "metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Created tokenizer:")
print(f"  Vocab size: {tokenizer.get_vocab_size()}")
print(f"  Path: {tokenizer_path}")
print(f"  Metadata: {metadata_path}")

# Test encoding
test_text = "fn main() {}"
encoding = tokenizer.encode(test_text)
print(f"\nTest encode '{test_text}':")
print(f"  Tokens: {encoding.ids}")
print(f"  Text: {encoding.tokens}")
