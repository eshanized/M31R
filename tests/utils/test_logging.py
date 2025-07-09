# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Tests for the structured JSON logger.

We verify:
  - output is valid JSON
  - all mandatory fields are present (ts, level, module, msg)
  - log levels filter correctly
  - extra context fields get merged into the JSON
"""

import json
import logging
from pathlib import Path

import pytest

from m31r.logging.logger import get_logger


@pytest.fixture(autouse=True)
def _reset_loggers() -> None:
    """
    Clear all logger handlers between tests so get_logger's handler-stacking
    guard doesn't interfere with test isolation.
    """
    yield  # type: ignore[misc]
    for name in list(logging.Logger.manager.loggerDict):
        if name.startswith("m31r.test"):
            logger = logging.getLogger(name)
            logger.handlers.clear()


class TestJsonOutput:
    def test_output_is_valid_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        logger = get_logger("m31r.test.json", log_level="INFO")
        logger.info("hello")
        captured = capsys.readouterr()

        parsed = json.loads(captured.out.strip())
        assert isinstance(parsed, dict)

    def test_mandatory_fields_are_present(self, capsys: pytest.CaptureFixture[str]) -> None:
        logger = get_logger("m31r.test.fields", log_level="INFO")
        logger.info("test message")
        captured = capsys.readouterr()

        parsed = json.loads(captured.out.strip())
        assert "ts" in parsed
        assert "level" in parsed
        assert "module" in parsed
        assert "msg" in parsed
        assert parsed["level"] == "INFO"
        assert parsed["module"] == "m31r.test.fields"
        assert parsed["msg"] == "test message"

    def test_extra_fields_are_included(self, capsys: pytest.CaptureFixture[str]) -> None:
        logger = get_logger("m31r.test.extra", log_level="DEBUG")
        logger.info("step done", extra={"step": 100, "loss": 1.23})
        captured = capsys.readouterr()

        parsed = json.loads(captured.out.strip())
        assert parsed["step"] == 100
        assert parsed["loss"] == 1.23


class TestLogLevelFiltering:
    def test_debug_messages_hidden_at_info_level(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        logger = get_logger("m31r.test.level_filter", log_level="INFO")
        logger.debug("this should not appear")
        captured = capsys.readouterr()
        assert captured.out.strip() == ""

    def test_info_messages_shown_at_info_level(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        logger = get_logger("m31r.test.level_show", log_level="INFO")
        logger.info("this should appear")
        captured = capsys.readouterr()
        assert "this should appear" in captured.out


class TestFileOutput:
    def test_logs_are_written_to_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        logger = get_logger("m31r.test.file_output", log_level="INFO", log_file=log_file)
        logger.info("file log test")

        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        parsed = json.loads(content.strip())
        assert parsed["msg"] == "file log test"


class TestInvalidLogLevel:
    def test_invalid_level_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid log level"):
            get_logger("m31r.test.invalid", log_level="INVALID")
