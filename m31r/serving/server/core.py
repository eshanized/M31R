# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Local HTTP server for M31R inference.

This is a minimal, no-dependency HTTP server built on Python's standard
library. No Flask, no FastAPI, no aiohttp — just http.server and json.
That keeps the dependency footprint tiny and means this works anywhere
Python runs.

The server binds to localhost only by default. This is a deliberate
security choice — you don't want random machines on your network sending
prompts to your model. If someone needs remote access, they can put a
reverse proxy in front.

Endpoints:
  POST /generate     — generate text from a prompt
  POST /completion   — code completion (prefix + optional suffix)
  POST /fim          — fill-in-the-middle
  GET  /status       — health check and server info
  POST /shutdown     — gracefully stop the server
"""

import json
import logging
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from m31r.logging.logger import get_logger
from m31r.serving.api.schema import (
    GenerateRequest,
    GenerateResponse,
    ServerStatus,
    StreamChunk,
)
from m31r.serving.engine.core import InferenceEngine
from m31r.serving.generation.core import GenerationConfig

logger: logging.Logger = get_logger(__name__)


class M31RRequestHandler(BaseHTTPRequestHandler):
    """
    Handles incoming HTTP requests and routes them to the inference engine.

    Each request goes through the same lifecycle:
      1. Parse and validate the JSON body
      2. Check payload size limits
      3. Route to the right handler based on the URL path
      4. Return a JSON response with appropriate status code

    We override log_message to suppress the default stderr output —
    all logging goes through our structured logger instead.
    """

    def log_message(self, format: str, *args: Any) -> None:
        """Silence the default request logger — we use structured logging."""
        pass

    def do_GET(self) -> None:
        if self.path == "/status":
            self._handle_status()
        else:
            self._send_error(404, "Not found")

    def do_POST(self) -> None:
        routes = {
            "/generate": self._handle_generate,
            "/completion": self._handle_completion,
            "/fim": self._handle_fim,
            "/shutdown": self._handle_shutdown,
        }

        handler = routes.get(self.path)
        if handler is None:
            self._send_error(404, f"Unknown endpoint: {self.path}")
            return

        handler()

    def _read_body(self) -> dict[str, Any] | None:
        """
        Read and parse the request body as JSON.

        Returns None if the body is missing, too large, or not valid JSON.
        In each case we've already sent the appropriate error response.
        """
        content_length = int(self.headers.get("Content-Length", "0"))
        max_size = self.server.max_request_size_bytes

        if content_length <= 0:
            self._send_error(400, "Request body is empty")
            return None

        if content_length > max_size:
            self._send_error(
                413,
                f"Payload too large: {content_length} bytes exceeds limit of {max_size}",
            )
            return None

        try:
            raw = self.rfile.read(content_length)
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as err:
            self._send_error(400, f"Invalid JSON: {err}")
            return None

    def _handle_generate(self) -> None:
        body = self._read_body()
        if body is None:
            return

        prompt = body.get("prompt", "")
        if not prompt:
            self._send_error(400, "Missing required field: prompt")
            return

        try:
            request = GenerateRequest(
                prompt=prompt,
                max_tokens=body.get("max_tokens", 512),
                temperature=body.get("temperature", 0.0),
                top_k=body.get("top_k", 0),
                seed=body.get("seed", 42),
                stream=body.get("stream", False),
            )

            gen_config = GenerationConfig(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                seed=request.seed,
            )

            engine: InferenceEngine = self.server.engine

            if request.stream:
                self._stream_response(engine, request.prompt, gen_config)
            else:
                response = engine.generate(request.prompt, gen_config)
                self._send_json(
                    200,
                    {
                        "text": response.text,
                        "tokens_generated": response.tokens_generated,
                        "prompt_tokens": response.prompt_tokens,
                        "total_time_ms": response.total_time_ms,
                        "tokens_per_second": response.tokens_per_second,
                        "finish_reason": response.finish_reason,
                    },
                )

        except Exception as err:
            logger.error("Generation failed", extra={"error": str(err)}, exc_info=True)
            self._send_error(500, f"Generation error: {err}")

    def _handle_completion(self) -> None:
        body = self._read_body()
        if body is None:
            return

        prefix = body.get("prefix", "")
        if not prefix:
            self._send_error(400, "Missing required field: prefix")
            return

        try:
            gen_config = GenerationConfig(
                max_tokens=body.get("max_tokens", 256),
                temperature=body.get("temperature", 0.0),
                top_k=body.get("top_k", 0),
                seed=body.get("seed", 42),
            )

            engine: InferenceEngine = self.server.engine
            response = engine.generate(prefix, gen_config)
            self._send_json(
                200,
                {
                    "text": response.text,
                    "tokens_generated": response.tokens_generated,
                    "prompt_tokens": response.prompt_tokens,
                    "total_time_ms": response.total_time_ms,
                    "tokens_per_second": response.tokens_per_second,
                    "finish_reason": response.finish_reason,
                },
            )

        except Exception as err:
            logger.error("Completion failed", extra={"error": str(err)}, exc_info=True)
            self._send_error(500, f"Completion error: {err}")

    def _handle_fim(self) -> None:
        body = self._read_body()
        if body is None:
            return

        prefix = body.get("prefix", "")
        suffix = body.get("suffix", "")
        if not prefix:
            self._send_error(400, "Missing required field: prefix")
            return

        try:
            prompt = f"{prefix}<|fim|>{suffix}"
            gen_config = GenerationConfig(
                max_tokens=body.get("max_tokens", 256),
                temperature=body.get("temperature", 0.0),
                top_k=body.get("top_k", 0),
                seed=body.get("seed", 42),
            )

            engine: InferenceEngine = self.server.engine
            response = engine.generate(prompt, gen_config)
            self._send_json(
                200,
                {
                    "text": response.text,
                    "tokens_generated": response.tokens_generated,
                    "prompt_tokens": response.prompt_tokens,
                    "total_time_ms": response.total_time_ms,
                    "tokens_per_second": response.tokens_per_second,
                    "finish_reason": response.finish_reason,
                },
            )

        except Exception as err:
            logger.error("FIM failed", extra={"error": str(err)}, exc_info=True)
            self._send_error(500, f"FIM error: {err}")

    def _handle_status(self) -> None:
        status: ServerStatus = self.server.status
        self._send_json(
            200,
            {
                "model_loaded": status.model_loaded,
                "model_path": status.model_path,
                "device": status.device,
                "quantization": status.quantization,
                "max_context_length": status.max_context_length,
                "uptime_seconds": round(time.monotonic() - self.server.start_time, 2),
                "requests_served": (
                    self.server.engine.metrics.total_requests if self.server.engine else 0
                ),
            },
        )

    def _handle_shutdown(self) -> None:
        self._send_json(200, {"message": "Server shutting down"})
        logger.info("Shutdown requested via API")
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def _stream_response(
        self,
        engine: InferenceEngine,
        prompt: str,
        config: GenerationConfig,
    ) -> None:
        """
        Send tokens as Server-Sent Events (SSE).

        Each token gets its own event in the SSE stream. The client reads
        them one by one and assembles the output progressively. When the
        last token arrives (done=true), the client knows to stop listening.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        try:
            for chunk in engine.generate_stream(prompt, config):
                event_data = json.dumps(
                    {
                        "token": chunk.token_text,
                        "token_id": chunk.token_id,
                        "position": chunk.position,
                        "elapsed_ms": round(chunk.elapsed_ms, 2),
                        "done": chunk.done,
                    }
                )
                self.wfile.write(f"data: {event_data}\n\n".encode("utf-8"))
                self.wfile.flush()

        except (BrokenPipeError, ConnectionResetError):
            logger.info("Client disconnected during streaming")

    def _send_json(self, status_code: int, data: dict[str, Any]) -> None:
        payload = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_error(self, status_code: int, message: str) -> None:
        self._send_json(status_code, {"error": message})


class M31RServer(HTTPServer):
    """
    Extended HTTPServer that carries the inference engine and config.

    We stash the engine and status on the server object so the request
    handler can access them via self.server. This avoids global state.
    """

    def __init__(
        self,
        address: tuple[str, int],
        engine: InferenceEngine,
        status: ServerStatus,
        max_request_size_bytes: int,
    ) -> None:
        super().__init__(address, M31RRequestHandler)
        self.engine = engine
        self.status = status
        self.max_request_size_bytes = max_request_size_bytes
        self.start_time = time.monotonic()


def run_server(
    engine: InferenceEngine,
    host: str,
    port: int,
    status: ServerStatus,
    max_request_size_bytes: int,
) -> None:
    """
    Start the inference server and block until it shuts down.

    The server runs in the calling thread. To stop it, either send a
    POST to /shutdown or hit Ctrl+C. Both paths call server.shutdown()
    which cleanly winds everything down.
    """
    if host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning(
            "Server binding to non-localhost address — this is a security risk",
            extra={"host": host},
        )

    server = M31RServer(
        (host, port),
        engine=engine,
        status=status,
        max_request_size_bytes=max_request_size_bytes,
    )

    logger.info(
        "M31R inference server started",
        extra={"host": host, "port": port},
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down server (keyboard interrupt)")
    finally:
        server.server_close()
        logger.info("Server stopped")
