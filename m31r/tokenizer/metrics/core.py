# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tokenizer quality metrics.

After training a tokenizer, you want to know: Is this vocabulary any good?
These metrics help answer that by measuring things like how many unknown
tokens appear, how badly text gets fragmented, and what percentage of
the corpus is actually covered by the vocabulary.
"""

from typing import NamedTuple

from tokenizers import Tokenizer

from m31r.logging.logger import get_logger


class TokenizerMetrics(NamedTuple):
    """All the numbers you'd want to check after training a tokenizer."""

    vocab_size: int
    avg_tokens_per_line: float
    unk_rate: float
    avg_token_length: float
    total_lines: int
    total_tokens: int


def compute_metrics(
    tokenizer: Tokenizer,
    sample_texts: list[str],
) -> TokenizerMetrics:
    """
    Compute quality metrics over a set of sample texts.

    This gives you a quick health check on the tokenizer. You want to see:
      - avg_tokens_per_line: lower is better (less fragmentation)
      - unk_rate: should be near zero (good coverage)
      - avg_token_length: higher means more efficient encoding

    The sample_texts should be representative of the actual training data.
    Passing in a few hundred lines from the corpus is usually enough to
    get stable numbers.
    """
    logger = get_logger("m31r.tokenizer.metrics")

    if not sample_texts:
        logger.warning("No sample texts provided for metrics computation")
        return TokenizerMetrics(
            vocab_size=tokenizer.get_vocab_size(),
            avg_tokens_per_line=0.0,
            unk_rate=0.0,
            avg_token_length=0.0,
            total_lines=0,
            total_tokens=0,
        )

    unk_id = tokenizer.token_to_id("<unk>")

    total_tokens = 0
    total_unk = 0
    total_chars = 0

    encodings = tokenizer.encode_batch(sample_texts)

    for encoding in encodings:
        token_count = len(encoding.ids)
        total_tokens += token_count

        if unk_id is not None:
            total_unk += encoding.ids.count(unk_id)

    for text in sample_texts:
        total_chars += len(text)

    total_lines = len(sample_texts)
    avg_tokens_per_line = total_tokens / total_lines if total_lines > 0 else 0.0
    unk_rate = total_unk / total_tokens if total_tokens > 0 else 0.0
    avg_token_length = total_chars / total_tokens if total_tokens > 0 else 0.0

    metrics = TokenizerMetrics(
        vocab_size=tokenizer.get_vocab_size(),
        avg_tokens_per_line=round(avg_tokens_per_line, 4),
        unk_rate=round(unk_rate, 6),
        avg_token_length=round(avg_token_length, 4),
        total_lines=total_lines,
        total_tokens=total_tokens,
    )

    logger.info(
        "Tokenizer metrics computed",
        extra={
            "vocab_size": metrics.vocab_size,
            "avg_tokens_per_line": metrics.avg_tokens_per_line,
            "unk_rate": metrics.unk_rate,
            "avg_token_length": metrics.avg_token_length,
        },
    )

    return metrics
