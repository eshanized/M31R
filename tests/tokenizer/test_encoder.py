# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for the encode/decode roundtrip.

The golden rule: decode(encode(text)) should give you back the original
text. We test this with various inputs including edge cases.
"""

from m31r.config.schema import TokenizerConfig
from m31r.tokenizer.decoder.core import decode, decode_batch
from m31r.tokenizer.encoder.core import encode, encode_batch
from m31r.tokenizer.trainer.core import train_tokenizer


def _train_test_tokenizer() -> "tokenizers.Tokenizer":
    """Train a small tokenizer for encoding/decoding tests."""
    config = TokenizerConfig(
        config_version="1.0.0",
        vocab_size=256,
        tokenizer_type="bpe",
        seed=42,
        min_frequency=1,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        dataset_directory="data/datasets",
        output_directory="data/tokenizer",
    )

    corpus = [
        'fn main() { println!("Hello, world!"); }',
        "pub struct Point { x: f64, y: f64 }",
        "let values: Vec<i32> = vec![1, 2, 3, 4, 5];",
        "fn add(a: i32, b: i32) -> i32 { a + b }",
        "use std::collections::HashMap;",
    ] * 20

    return train_tokenizer(config, iter(corpus))


def test_encode_produces_ids() -> None:
    """encode() should return a list of integers."""
    tokenizer = _train_test_tokenizer()
    ids = encode(tokenizer, "fn main() {}")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0


def test_roundtrip() -> None:
    """The fundamental property: decode(encode(text)) gets the original back."""
    tokenizer = _train_test_tokenizer()
    original = 'fn main() { println!("Hello"); }'
    ids = encode(tokenizer, original)
    recovered = decode(tokenizer, ids)
    assert recovered == original


def test_roundtrip_with_various_text() -> None:
    """Roundtrip should handle typical Rust code constructs."""
    tokenizer = _train_test_tokenizer()
    original = "fn add(a: i32, b: i32) -> i32 { a + b }"
    ids = encode(tokenizer, original)
    recovered = decode(tokenizer, ids)
    assert recovered == original


def test_empty_string() -> None:
    """Encoding an empty string should produce an empty list."""
    tokenizer = _train_test_tokenizer()
    ids = encode(tokenizer, "")
    assert ids == []

    text = decode(tokenizer, [])
    assert text == ""


def test_batch_encoding() -> None:
    """Batch encoding should produce the same results as individual encoding."""
    tokenizer = _train_test_tokenizer()
    texts = ["fn foo() {}", "let x = 42;", "struct Bar;"]

    batch_results = encode_batch(tokenizer, texts)
    individual_results = [encode(tokenizer, t) for t in texts]

    assert batch_results == individual_results


def test_batch_decoding() -> None:
    """Batch decoding should produce the same results as individual decoding."""
    tokenizer = _train_test_tokenizer()
    texts = ["fn foo() {}", "let x = 42;"]

    ids_batch = encode_batch(tokenizer, texts)
    batch_decoded = decode_batch(tokenizer, ids_batch)
    individual_decoded = [decode(tokenizer, ids) for ids in ids_batch]

    assert batch_decoded == individual_decoded


def test_encode_determinism() -> None:
    """Encoding the same text multiple times must give the same IDs."""
    tokenizer = _train_test_tokenizer()
    text = "fn factorial(n: u64) -> u64 { n * factorial(n - 1) }"

    ids_first = encode(tokenizer, text)
    ids_second = encode(tokenizer, text)

    assert ids_first == ids_second
