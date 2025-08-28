# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for the tokenizer trainer.

The big thing we're verifying here is determinism — training the same
data with the same config twice must produce an identical tokenizer.
We also check that BPE and Unigram modes both work and that the vocab
size is in the right ballpark.
"""

import json
from pathlib import Path

import pytest

from m31r.config.schema import NormalizationConfig, TokenizerConfig
from m31r.tokenizer.streaming.reader import stream_corpus
from m31r.tokenizer.trainer.core import train_tokenizer


def _make_config(**overrides) -> TokenizerConfig:
    """Build a TokenizerConfig with sensible test defaults."""
    defaults = {
        "config_version": "1.0.0",
        "vocab_size": 256,
        "tokenizer_type": "bpe",
        "seed": 42,
        "min_frequency": 1,
        "max_token_length": 64,
        "pre_tokenizer_type": "byte_level",
        "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"],
        "dataset_directory": "data/datasets",
        "output_directory": "data/tokenizer",
    }
    defaults.update(overrides)
    return TokenizerConfig(**defaults)


def _create_test_corpus() -> list[str]:
    """A small corpus of Rust-ish text for training tests."""
    return [
        "fn main() { println!(\"Hello, world!\"); }",
        "pub struct Point { x: f64, y: f64 }",
        "impl Point { fn new(x: f64, y: f64) -> Self { Point { x, y } } }",
        "let values: Vec<i32> = vec![1, 2, 3, 4, 5];",
        "fn add(a: i32, b: i32) -> i32 { a + b }",
        "use std::collections::HashMap;",
        "fn factorial(n: u64) -> u64 { if n <= 1 { 1 } else { n * factorial(n - 1) } }",
        "#[derive(Debug, Clone)]",
        "enum Color { Red, Green, Blue }",
        "trait Drawable { fn draw(&self); }",
    ] * 10  # repeat for a bigger corpus


def test_bpe_training_produces_valid_tokenizer() -> None:
    """A BPE train should produce a tokenizer that can encode and decode."""
    config = _make_config(tokenizer_type="bpe")
    corpus = iter(_create_test_corpus())
    tokenizer = train_tokenizer(config, corpus)

    assert tokenizer.get_vocab_size() > 0
    assert tokenizer.get_vocab_size() <= config.vocab_size

    result = tokenizer.encode("fn main() {}")
    assert len(result.ids) > 0


def test_unigram_training_produces_valid_tokenizer() -> None:
    """A Unigram train should also work and produce a usable tokenizer."""
    config = _make_config(tokenizer_type="unigram")
    corpus = iter(_create_test_corpus())
    tokenizer = train_tokenizer(config, corpus)

    assert tokenizer.get_vocab_size() > 0

    result = tokenizer.encode("fn main() {}")
    assert len(result.ids) > 0


def test_determinism() -> None:
    """
    Training the same corpus twice with the same config must produce
    the exact same tokenizer — same JSON bytes, same vocab, same everything.

    This is the most critical test in the entire tokenizer suite. If this
    fails, we can't trust that our training is reproducible.
    """
    config = _make_config(tokenizer_type="bpe")

    tokenizer_a = train_tokenizer(config, iter(_create_test_corpus()))
    tokenizer_b = train_tokenizer(config, iter(_create_test_corpus()))

    json_a = tokenizer_a.to_str()
    json_b = tokenizer_b.to_str()

    assert json_a == json_b, "Tokenizer training is not deterministic"


def test_special_tokens_in_vocab() -> None:
    """Special tokens should always be in the vocabulary after training."""
    special = ["<pad>", "<unk>", "<bos>", "<eos>"]
    config = _make_config(special_tokens=special)
    corpus = iter(_create_test_corpus())
    tokenizer = train_tokenizer(config, corpus)

    vocab = tokenizer.get_vocab()
    for token in special:
        assert token in vocab, f"Special token {token!r} missing from vocab"


def test_vocab_size_respects_config() -> None:
    """The resulting vocab shouldn't exceed the configured limit."""
    config = _make_config(vocab_size=300)
    corpus = iter(_create_test_corpus())
    tokenizer = train_tokenizer(config, corpus)

    assert tokenizer.get_vocab_size() <= config.vocab_size


def test_train_from_streaming_reader(tmp_path: Path) -> None:
    """Integration test: train directly from the streaming corpus reader."""
    version_dir = tmp_path / "v1"
    version_dir.mkdir(parents=True)

    entries = [{"content": text, "source": "test", "path": f"test/{i}.rs"}
               for i, text in enumerate(_create_test_corpus())]
    lines = [json.dumps(e, sort_keys=True) for e in entries]
    (version_dir / "shard_000000.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    config = _make_config(vocab_size=256)
    corpus = stream_corpus(version_dir)
    tokenizer = train_tokenizer(config, corpus)

    assert tokenizer.get_vocab_size() > 0
    assert tokenizer.get_vocab_size() <= 256
