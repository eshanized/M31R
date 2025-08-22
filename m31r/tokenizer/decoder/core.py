# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Decoding engine — turns token IDs back into text.

Mirror of the encoder. decode(encode(text)) should give you back the
original text (or something very close, depending on normalization).
"""

from tokenizers import Tokenizer


def decode(tokenizer: Tokenizer, ids: list[int]) -> str:
    """
    Decode a list of token IDs back into a text string.

    The tokenizer reverses the encoding process — it looks up each ID in
    the vocabulary, applies any byte-level decoding, and stitches the
    pieces back together into readable text.
    """
    return tokenizer.decode(ids)


def decode_batch(tokenizer: Tokenizer, ids_batch: list[list[int]]) -> list[str]:
    """
    Decode multiple token ID sequences in one call.

    Same performance benefit as encode_batch — the Rust backend can
    parallelize the work across sequences.
    """
    return tokenizer.decode_batch(ids_batch)
