# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Structured JSON logger for M31R.

Per 20_OBSERVABILITY_AND_LOGGING.md, every log entry must be structured (JSON),
timestamped, leveled, and include the source module. Human-only text logs and
print() are both forbidden.

How this works:
  - We use Python's standard `logging` module under the hood, but replace the
    default formatter with JsonFormatter, which serializes every log record
    into a single JSON line.
  - Two handlers are always attached: one for stdout, one optionally for a file.
  - The factory function `get_logger` is the only way to create loggers. Direct
    construction of logging.Logger is not allowed elsewhere in the codebase.

The JSON structure looks like:
  {"ts": "2026-...", "level": "INFO", "module": "m31r.config.loader", "msg": "loaded config", ...}
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class JsonFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects.

    Each log entry contains the four mandatory fields from the spec:
      ts     — ISO 8601 UTC timestamp
      level  — log level name
      module — the logger name (usually the Python module path)
      msg    — the formatted message string

    If the log call includes `extra` keyword args, those get merged into the
    JSON object as additional context fields. This is how subsystems attach
    structured data like step counts, loss values, etc.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "msg": record.getMessage(),
        }

        # Merge any extra fields the caller passed via the `extra` kwarg.
        # We skip internal LogRecord attributes to avoid dumping noise.
        standard_attrs = {
            "name",
            "msg",
            "args",
            "created",
            "relativeCreated",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "pathname",
            "filename",
            "module",
            "levelno",
            "levelname",
            "processName",
            "process",
            "threadName",
            "thread",
            "message",
            "msecs",
            "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                entry[key] = value

        return json.dumps(entry, default=str)


_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _resolve_log_level(level_name: str) -> int:
    """Turn a level name string into the corresponding logging constant."""
    upper = level_name.upper()
    if upper not in _VALID_LOG_LEVELS:
        raise ValueError(
            f"Invalid log level '{level_name}'. Must be one of: {', '.join(sorted(_VALID_LOG_LEVELS))}"
        )
    return getattr(logging, upper)


def get_logger(
    name: str,
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Create a structured JSON logger.

    This is the only sanctioned way to get a logger in M31R. Every module
    should call this once at the top and use the returned logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module.
        log_level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
        log_file: Optional path to a log file. If provided, logs go to both
                  stdout and the file.

    Returns:
        A configured logging.Logger that outputs structured JSON.
    """
    logger = logging.getLogger(name)
    level = _resolve_log_level(log_level)
    logger.setLevel(level)

    # Avoid stacking handlers if get_logger is called multiple times for the
    # same name (happens in tests).
    if logger.handlers:
        return logger

    formatter = JsonFormatter()

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Don't propagate to root logger — we handle all output ourselves.
    logger.propagate = False

    return logger
