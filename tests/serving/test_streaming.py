# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""Tests for the streaming token emitter."""

from m31r.serving.streaming.core import TokenStreamer, stream_tokens


class TestTokenStreamer:

    def test_emits_chunks_in_order(self) -> None:
        streamer = TokenStreamer()
        chunks = []
        for i in range(5):
            chunks.append(streamer.emit(f"tok_{i}", i))

        positions = [c.position for c in chunks]
        assert positions == [0, 1, 2, 3, 4]

    def test_marks_last_chunk_as_done(self) -> None:
        streamer = TokenStreamer()
        streamer.emit("hello", 10)
        last = streamer.emit("world", 20, done=True)
        assert last.done is True

    def test_tracks_token_count(self) -> None:
        streamer = TokenStreamer()
        for i in range(7):
            streamer.emit(f"t{i}", i)
        assert streamer.token_count == 7

    def test_elapsed_time_increases(self) -> None:
        streamer = TokenStreamer()
        first = streamer.emit("a", 0)
        second = streamer.emit("b", 1)
        assert second.elapsed_ms >= first.elapsed_ms

    def test_reset_clears_state(self) -> None:
        streamer = TokenStreamer()
        streamer.emit("x", 0)
        streamer.emit("y", 1)
        streamer.reset()
        assert streamer.token_count == 0
        assert len(streamer.all_chunks()) == 0

    def test_all_chunks_returns_copy(self) -> None:
        streamer = TokenStreamer()
        streamer.emit("a", 0)
        chunks = streamer.all_chunks()
        streamer.emit("b", 1)
        assert len(chunks) == 1


class TestStreamTokens:

    def test_yields_all_tokens(self) -> None:
        ids = [10, 20, 30]
        decode = lambda tid: f"[{tid}]"

        chunks = list(stream_tokens(ids, decode))
        assert len(chunks) == 3
        assert chunks[0].token_text == "[10]"
        assert chunks[-1].done is True

    def test_empty_list_yields_nothing(self) -> None:
        chunks = list(stream_tokens([], lambda x: ""))
        assert len(chunks) == 0
