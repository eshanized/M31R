# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Streaming token emitter.

Instead of waiting for the model to finish generating all tokens before
showing anything, we yield each token as soon as it's decoded. This
dramatically reduces perceived latency — the user sees output appearing
character by character instead of staring at a blank screen.

The streamer works the same way for both CLI output (printing to stdout)
and the HTTP server (Server-Sent Events). Both consumers just iterate
over the stream and handle each chunk however they want.
"""

import time
from collections.abc import Generator

from m31r.serving.api.schema import StreamChunk


class TokenStreamer:
    """
    Wraps a sequence of generated token IDs and emits them as StreamChunks.

    You feed it tokens one at a time via `emit()`, and it packages each
    one up with timing info and a position counter. The consumer decides
    what to do with each chunk (print it, send it over HTTP, etc.).
    """

    def __init__(self) -> None:
        self._position: int = 0
        self._start_time: float = time.monotonic()
        self._chunks: list[StreamChunk] = []

    @property
    def token_count(self) -> int:
        return self._position

    @property
    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._start_time) * 1000.0

    def emit(self, token_text: str, token_id: int, done: bool = False) -> StreamChunk:
        """
        Create a chunk for a newly generated token.

        Each call bumps the position counter and records how long it's
        been since streaming started. The 'done' flag signals the final
        token so consumers know when to stop listening.
        """
        chunk = StreamChunk(
            token_text=token_text,
            token_id=token_id,
            position=self._position,
            elapsed_ms=self.elapsed_ms,
            done=done,
        )
        self._chunks.append(chunk)
        self._position += 1
        return chunk

    def all_chunks(self) -> list[StreamChunk]:
        """Everything emitted so far, in order."""
        return list(self._chunks)

    def reset(self) -> None:
        """Start fresh for a new generation sequence."""
        self._position = 0
        self._start_time = time.monotonic()
        self._chunks.clear()


def stream_tokens(
    token_ids: list[int],
    decode_fn: callable,
) -> Generator[StreamChunk, None, None]:
    """
    Turn a list of token IDs into a stream of decoded chunks.

    This is a convenience function for the simple case where you
    already have all the tokens and just want to yield them one
    by one with timing info. For real-time generation, the engine
    calls TokenStreamer.emit() directly during the generation loop.

    Args:
        token_ids: The token IDs to stream.
        decode_fn: A function that turns a single token ID into text.

    Yields:
        StreamChunk objects, one per token.
    """
    streamer = TokenStreamer()

    for i, token_id in enumerate(token_ids):
        is_last = i == len(token_ids) - 1
        text = decode_fn(token_id)
        yield streamer.emit(text, token_id, done=is_last)
