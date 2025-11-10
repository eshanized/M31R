# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Request and response schemas for the M31R inference API.

These are plain dataclasses — no pydantic here, since these are
runtime data structures rather than config validation. Every field
is typed and documented so you know exactly what goes over the wire.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GenerateRequest:
    """What the client sends when they want the model to produce text."""

    prompt: str
    max_tokens: int = 512
    temperature: float = 0.0
    top_k: int = 0
    seed: int = 42
    stream: bool = False


@dataclass(frozen=True)
class GenerateResponse:
    """What comes back after generation finishes (non-streaming case)."""

    text: str
    tokens_generated: int
    prompt_tokens: int
    total_time_ms: float
    tokens_per_second: float
    finish_reason: str = "length"


@dataclass(frozen=True)
class StreamChunk:
    """
    A single piece of a streaming response.

    Each chunk carries one newly decoded token plus running stats.
    The consumer (CLI or HTTP client) assembles these into the full output.
    """

    token_text: str
    token_id: int
    position: int
    elapsed_ms: float
    done: bool = False


@dataclass(frozen=True)
class CompletionRequest:
    """Code completion request — same idea as generate but framed differently."""

    prefix: str
    suffix: str = ""
    max_tokens: int = 256
    temperature: float = 0.0
    top_k: int = 0
    seed: int = 42
    stream: bool = False


@dataclass(frozen=True)
class FIMRequest:
    """
    Fill-in-the-middle request.

    The model sees the prefix and suffix and tries to fill the gap.
    This is the bread and butter of editor integration.
    """

    prefix: str
    suffix: str
    max_tokens: int = 256
    temperature: float = 0.0
    top_k: int = 0
    seed: int = 42
    stream: bool = False


@dataclass
class ServerStatus:
    """Quick health check info for the /status endpoint."""

    model_loaded: bool = False
    model_path: str = ""
    device: str = "cpu"
    quantization: str = "none"
    max_context_length: int = 2048
    uptime_seconds: float = 0.0
    requests_served: int = 0
    errors: list[str] = field(default_factory=list)
