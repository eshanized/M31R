# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Encoding engine — turns text into token IDs.

Every call to encode() with the same tokenizer and the same text will
produce the same token IDs, always. No randomness, no side effects,
safe to call from multiple threads.
"""

from tokenizers import Tokenizer


def encode(tokenizer: Tokenizer, text: str) -> list[int]:
    """
    Encode a single text string into a list of token IDs.

    This is the fundamental operation that turns human-readable text into
    the integer sequences the model actually sees. The tokenizer applies
    its normalizer, pre-tokenizer, and BPE/Unigram model in sequence.
    """
    encoding = tokenizer.encode(text)
    return encoding.ids


def encode_batch(tokenizer: Tokenizer, texts: list[str]) -> list[list[int]]:
    """
    Encode multiple texts in one call for better throughput.

    The underlying Rust implementation parallelizes batch encoding,
    so this is significantly faster than calling encode() in a loop
    when you have many texts to process.
    """
    encodings = tokenizer.encode_batch(texts)
    return [enc.ids for enc in encodings]
