# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for the tokenizer quality metrics.

We train a small tokenizer and check that the metrics computation
returns sensible values — vocab size matches, unk rate is low for
in-distribution text, and all the fields are populated.
"""

from m31r.config.schema import TokenizerConfig
from m31r.tokenizer.metrics.core import TokenizerMetrics, compute_metrics
from m31r.tokenizer.trainer.core import train_tokenizer


def _train_test_tokenizer():
    """Train a tiny tokenizer for metrics testing."""
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
        'fn main() { println!("Hello"); }',
        "struct Point { x: f64, y: f64 }",
        "let v: Vec<i32> = vec![1, 2, 3];",
    ] * 20

    return train_tokenizer(config, iter(corpus))


def test_metrics_returns_named_tuple() -> None:
    """compute_metrics should return a TokenizerMetrics NamedTuple."""
    tokenizer = _train_test_tokenizer()
    sample = ["fn main() {}", "let x = 42;"]
    metrics = compute_metrics(tokenizer, sample)

    assert isinstance(metrics, TokenizerMetrics)


def test_vocab_size_matches() -> None:
    """Reported vocab_size should match the tokenizer's actual vocabulary."""
    tokenizer = _train_test_tokenizer()
    metrics = compute_metrics(tokenizer, ["test"])

    assert metrics.vocab_size == tokenizer.get_vocab_size()


def test_unk_rate_low_for_training_data() -> None:
    """Text similar to training data should have a very low unknown token rate."""
    tokenizer = _train_test_tokenizer()
    sample = [
        'fn main() { println!("Hello"); }',
        "struct Point { x: f64, y: f64 }",
    ]
    metrics = compute_metrics(tokenizer, sample)

    assert metrics.unk_rate < 0.1


def test_avg_tokens_per_line_positive() -> None:
    """Average tokens per line should be greater than zero for non-empty input."""
    tokenizer = _train_test_tokenizer()
    metrics = compute_metrics(tokenizer, ["fn main() {}"])

    assert metrics.avg_tokens_per_line > 0


def test_empty_sample_returns_zero_metrics() -> None:
    """Passing an empty sample list should return zero counts without crashing."""
    tokenizer = _train_test_tokenizer()
    metrics = compute_metrics(tokenizer, [])

    assert metrics.total_lines == 0
    assert metrics.total_tokens == 0
    assert metrics.avg_tokens_per_line == 0.0


def test_total_lines_matches_input() -> None:
    """The total_lines field should match the number of sample texts provided."""
    tokenizer = _train_test_tokenizer()
    sample = ["line one", "line two", "line three"]
    metrics = compute_metrics(tokenizer, sample)

    assert metrics.total_lines == 3
