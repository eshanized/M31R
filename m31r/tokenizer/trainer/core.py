# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Deterministic tokenizer trainer.

This module wires up the HuggingFace `tokenizers` library to train a BPE
or Unigram vocabulary from a streaming text corpus. The key guarantee is
determinism: given the same config and the same data in the same order,
you'll get the exact same tokenizer.json output, byte for byte.

The HuggingFace tokenizers lib is Rust-backed, so the actual training
loop runs at native speed. We just configure it and feed it data.
"""

from typing import Iterator

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE, Unigram
from tokenizers.trainers import BpeTrainer, UnigramTrainer

from m31r.config.schema import TokenizerConfig
from m31r.logging.logger import get_logger


def _build_normalizer(config: TokenizerConfig) -> normalizers.Normalizer:
    """
    Stack up the text normalizers based on config settings.

    Normalizers run before anything else — they clean up the raw text so that
    equivalent inputs (like different Unicode representations of the same char)
    get treated identically. NFKC is almost always what you want for code
    because it collapses things like fullwidth characters into their ASCII
    equivalents.
    """
    steps: list[normalizers.Normalizer] = []

    if config.normalization.nfkc:
        steps.append(normalizers.NFKC())

    if config.normalization.lowercase:
        steps.append(normalizers.Lowercase())

    if not steps:
        return normalizers.Sequence([])

    return normalizers.Sequence(steps)


def _build_pre_tokenizer(config: TokenizerConfig) -> pre_tokenizers.PreTokenizer:
    """
    Choose how raw text gets split into chunks before BPE/Unigram merging.

    ByteLevel is the standard choice — it operates on bytes rather than
    Unicode codepoints, which means it can handle any text without unknown
    characters. Whitespace is simpler but can't represent arbitrary byte
    sequences.
    """
    if config.pre_tokenizer_type == "whitespace":
        return pre_tokenizers.Whitespace()
    return pre_tokenizers.ByteLevel(add_prefix_space=False)


def _build_decoder(config: TokenizerConfig) -> decoders.Decoder:
    """Match the decoder to the pre-tokenizer so round-tripping works."""
    if config.pre_tokenizer_type == "whitespace":
        return decoders.WordPiece()
    return decoders.ByteLevel()


def _create_bpe_tokenizer(config: TokenizerConfig) -> tuple[Tokenizer, BpeTrainer]:
    """
    Set up a fresh BPE tokenizer and its trainer.

    BPE (Byte-Pair Encoding) works by starting with individual characters
    and repeatedly merging the most frequent pair. The result is a vocabulary
    where common subwords get their own tokens, keeping the total vocab
    small while still covering all possible text.
    """
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = _build_normalizer(config)
    tokenizer.pre_tokenizer = _build_pre_tokenizer(config)
    tokenizer.decoder = _build_decoder(config)

    trainer = BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=list(config.special_tokens),
        max_token_length=config.max_token_length,
        show_progress=False,
    )

    return tokenizer, trainer


def _create_unigram_tokenizer(config: TokenizerConfig) -> tuple[Tokenizer, UnigramTrainer]:
    """
    Set up a fresh Unigram tokenizer and its trainer.

    Unigram starts with a large candidate vocabulary and iteratively prunes
    tokens that contribute least to the corpus likelihood. It tends to produce
    a vocabulary with better coverage than BPE for the same size, but the
    training is a bit more expensive.
    """
    tokenizer = Tokenizer(Unigram())
    tokenizer.normalizer = _build_normalizer(config)
    tokenizer.pre_tokenizer = _build_pre_tokenizer(config)
    tokenizer.decoder = _build_decoder(config)

    trainer = UnigramTrainer(
        vocab_size=config.vocab_size,
        special_tokens=list(config.special_tokens),
        unk_token="<unk>",
        show_progress=False,
    )

    return tokenizer, trainer


def train_tokenizer(
    config: TokenizerConfig,
    corpus_iterator: Iterator[str],
) -> Tokenizer:
    """
    Train a tokenizer from a streaming text corpus.

    This is the main entrypoint for tokenizer training. It takes a config
    that specifies all the knobs (vocab size, algorithm, normalization, etc.)
    and an iterator that yields text samples one at a time.

    The corpus_iterator is consumed fully during training — the tokenizers
    library handles batching and internal state. Because the iterator is
    deterministic (sorted shard order from the streaming reader) and the
    library's internal algorithms are seeded, the result is identical across
    runs.

    Returns a trained Tokenizer object that can encode and decode text.
    """
    logger = get_logger("m31r.tokenizer.trainer")

    logger.info(
        "Starting tokenizer training",
        extra={
            "tokenizer_type": config.tokenizer_type,
            "vocab_size": config.vocab_size,
            "seed": config.seed,
        },
    )

    if config.tokenizer_type == "unigram":
        tokenizer, trainer = _create_unigram_tokenizer(config)
    else:
        tokenizer, trainer = _create_bpe_tokenizer(config)

    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)

    actual_vocab_size = tokenizer.get_vocab_size()
    logger.info(
        "Tokenizer training complete",
        extra={
            "actual_vocab_size": actual_vocab_size,
            "target_vocab_size": config.vocab_size,
        },
    )

    return tokenizer
