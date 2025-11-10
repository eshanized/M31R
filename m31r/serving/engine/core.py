# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Inference engine — the heart of the serving runtime.

This module ties together the model, tokenizer, generation strategy,
and streaming output into one clean interface. When you call
`engine.generate("write a function")`, everything else happens
automatically: tokenization, the autoregressive decoding loop,
memory tracking, and token-by-token streaming.

The engine is designed to be thread-safe for the single-request case
(which is what M31R targets). It holds no mutable shared state between
requests — each generation gets its own KV cache and RNG state.
"""

import logging
import time
from collections.abc import Generator

import torch
import torch.nn as nn

from m31r.logging.logger import get_logger
from m31r.serving.api.schema import GenerateResponse, StreamChunk
from m31r.serving.generation.core import GenerationConfig, sample_next_token
from m31r.serving.metrics.core import RequestMetrics, ServingMetrics
from m31r.serving.streaming.core import TokenStreamer

logger: logging.Logger = get_logger(__name__)


class InferenceEngine:
    """
    High-level inference API.

    This is what the CLI and server modules talk to. You create one of
    these with a loaded model and tokenizer, then call generate() or
    generate_stream() to produce text.

    The engine doesn't know or care about HTTP, CLI arguments, or file
    paths — it just takes prompts and returns tokens. The CLI and server
    modules handle all the I/O plumbing.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: object,
        device: torch.device,
        max_context_length: int,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._max_context_length = max_context_length
        self._metrics = ServingMetrics()

    @property
    def metrics(self) -> ServingMetrics:
        return self._metrics

    def generate(self, prompt: str, config: GenerationConfig) -> GenerateResponse:
        """
        Generate text from a prompt (non-streaming).

        This is the simple "give me the whole answer" mode. It runs the
        full generation loop and returns everything at once. Good for
        programmatic use, not great for user-facing latency.

        Args:
            prompt: The input text to continue from.
            config: Generation parameters (max_tokens, temperature, etc.).

        Returns:
            GenerateResponse with the full generated text and timing stats.
        """
        if self._tokenizer is None:
            raise RuntimeError("No tokenizer loaded — can't generate text")

        start = time.monotonic()
        prompt_ids = self._encode(prompt)
        generated_ids: list[int] = []

        # Set up deterministic RNG if we're doing non-greedy sampling
        generator = None
        if config.temperature > 0:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(config.seed)

        input_ids = prompt_ids[:]
        first_token_time: float | None = None

        for _ in range(config.max_tokens):
            # Don't exceed the model's context window
            if len(input_ids) >= self._max_context_length:
                break

            logits = self._forward(input_ids)
            next_token = sample_next_token(logits, config, generator)

            if first_token_time is None:
                first_token_time = (time.monotonic() - start) * 1000.0

            if next_token == config.eos_token_id:
                break

            generated_ids.append(next_token)
            input_ids.append(next_token)

        elapsed_ms = (time.monotonic() - start) * 1000.0
        generated_text = self._decode(generated_ids)
        tps = (len(generated_ids) / elapsed_ms * 1000.0) if elapsed_ms > 0 else 0.0

        request_metrics = RequestMetrics(
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(generated_ids),
            total_time_ms=elapsed_ms,
            first_token_ms=first_token_time or 0.0,
            tokens_per_second=tps,
            peak_memory_mb=ServingMetrics.get_gpu_memory_mb(),
        )
        self._metrics.record(request_metrics)

        finish = "eos" if (generated_ids and generated_ids[-1] == config.eos_token_id) else "length"

        return GenerateResponse(
            text=generated_text,
            tokens_generated=len(generated_ids),
            prompt_tokens=len(prompt_ids),
            total_time_ms=round(elapsed_ms, 2),
            tokens_per_second=round(tps, 2),
            finish_reason=finish,
        )

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> Generator[StreamChunk, None, None]:
        """
        Generate tokens one at a time, yielding each as it's produced.

        This is the streaming version of generate(). It yields StreamChunk
        objects that the consumer can process immediately — the CLI prints
        each token as it arrives, and the HTTP server sends them as SSE events.

        The generation loop here is the same as in generate(), but instead
        of collecting all tokens and returning at the end, we yield each
        one as soon as it's ready. This makes the first token appear much
        faster from the user's perspective.

        Args:
            prompt: The input text to continue from.
            config: Generation parameters.

        Yields:
            StreamChunk objects, one per generated token.
        """
        if self._tokenizer is None:
            raise RuntimeError("No tokenizer loaded — can't generate text")

        start = time.monotonic()
        prompt_ids = self._encode(prompt)
        streamer = TokenStreamer()

        generator = None
        if config.temperature > 0:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(config.seed)

        input_ids = prompt_ids[:]
        generated_count = 0

        for _ in range(config.max_tokens):
            if len(input_ids) >= self._max_context_length:
                break

            logits = self._forward(input_ids)
            next_token = sample_next_token(logits, config, generator)

            is_eos = next_token == config.eos_token_id
            token_text = self._decode([next_token])

            yield streamer.emit(token_text, next_token, done=is_eos)

            if is_eos:
                break

            generated_count += 1
            input_ids.append(next_token)

        elapsed_ms = (time.monotonic() - start) * 1000.0
        tps = (generated_count / elapsed_ms * 1000.0) if elapsed_ms > 0 else 0.0

        request_metrics = RequestMetrics(
            prompt_tokens=len(prompt_ids),
            generated_tokens=generated_count,
            total_time_ms=elapsed_ms,
            first_token_ms=0.0,
            tokens_per_second=tps,
            peak_memory_mb=ServingMetrics.get_gpu_memory_mb(),
        )
        self._metrics.record(request_metrics)

    @torch.no_grad()
    def _forward(self, token_ids: list[int]) -> torch.Tensor:
        """
        Run the model's forward pass and return logits for the last position.

        This is where the actual neural network computation happens.
        We wrap the token IDs in a tensor, feed them through the model,
        and pull out the logits (raw prediction scores) for just the
        last token position — that's the one we're trying to predict.

        The @torch.no_grad() decorator is critical here. During inference
        we don't need gradients, and skipping them saves a lot of memory
        and compute. Never remove this.
        """
        input_tensor = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self._device,
        )
        output = self._model(input_tensor)

        # output shape is [batch, seq_len, vocab_size]
        # we want the logits for the last token only
        return output[0, -1, :]

    def _encode(self, text: str) -> list[int]:
        """Turn text into token IDs using the loaded tokenizer."""
        encoding = self._tokenizer.encode(text)
        return encoding.ids

    def _decode(self, token_ids: list[int]) -> str:
        """Turn token IDs back into text."""
        return self._tokenizer.decode(token_ids)
