# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""Tests for the local HTTP server."""

import json
import threading
import time
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
import torch

from m31r.serving.api.schema import GenerateResponse, ServerStatus
from m31r.serving.engine.core import InferenceEngine
from m31r.serving.generation.core import GenerationConfig
from m31r.serving.server.core import M31RServer, M31RRequestHandler


class TestServerStatus:

    def test_status_defaults(self) -> None:
        status = ServerStatus()
        assert status.model_loaded is False
        assert status.device == "cpu"


class TestM31RServer:

    def test_server_creation(self) -> None:
        mock_engine = MagicMock(spec=InferenceEngine)
        mock_engine.metrics = MagicMock()
        mock_engine.metrics.total_requests = 0

        status = ServerStatus(model_loaded=True, device="cpu")

        server = M31RServer(
            ("127.0.0.1", 0),  # port 0 picks a random free port
            engine=mock_engine,
            status=status,
            max_request_size_bytes=1_048_576,
        )

        assert server.engine is mock_engine
        assert server.status.model_loaded is True
        server.server_close()

    def test_server_binds_to_localhost(self) -> None:
        mock_engine = MagicMock(spec=InferenceEngine)
        mock_engine.metrics = MagicMock()
        status = ServerStatus()

        server = M31RServer(
            ("127.0.0.1", 0),
            engine=mock_engine,
            status=status,
            max_request_size_bytes=1_048_576,
        )

        host, port = server.server_address
        assert host == "127.0.0.1"
        assert port > 0
        server.server_close()


class TestAPISchemaValidation:

    def test_generate_request_requires_prompt(self) -> None:
        from m31r.serving.api.schema import GenerateRequest

        req = GenerateRequest(prompt="hello world")
        assert req.prompt == "hello world"
        assert req.max_tokens == 512

    def test_generate_response_fields(self) -> None:
        resp = GenerateResponse(
            text="output",
            tokens_generated=10,
            prompt_tokens=5,
            total_time_ms=100.0,
            tokens_per_second=100.0,
        )
        assert resp.finish_reason == "length"
