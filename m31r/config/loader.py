# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Config loader — reads YAML from disk and produces a validated, frozen M31RConfig.

The loading pipeline is deliberately simple and linear:
  1. Read raw bytes from the file
  2. Parse as YAML into a plain dict
  3. Hand the dict to pydantic for schema validation
  4. Return the frozen, immutable config object

If anything goes wrong at any step, we fail immediately with a clear error.
There is no retry logic, no fallback defaults, no recovery. This is intentional:
a broken config should stop the system before it does anything wrong.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from m31r.config.exceptions import ConfigLoadError, ConfigValidationError
from m31r.config.schema import M31RConfig


def _read_yaml_file(config_path: Path) -> dict[str, Any]:
    """
    Read a YAML file and return the parsed dict.

    We explicitly check for file existence and readability before parsing,
    because yaml.safe_load gives cryptic errors on missing files.

    Args:
        config_path: Absolute or relative path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        ConfigLoadError: If the file doesn't exist, isn't readable, or isn't valid YAML.
    """
    if not config_path.exists():
        raise ConfigLoadError(f"Config file not found: {config_path}")

    if not config_path.is_file():
        raise ConfigLoadError(f"Config path is not a file: {config_path}")

    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as err:
        raise ConfigLoadError(f"Cannot read config file {config_path}: {err}") from err

    try:
        parsed = yaml.safe_load(raw_text)
    except yaml.YAMLError as err:
        raise ConfigLoadError(f"Invalid YAML in {config_path}: {err}") from err

    if not isinstance(parsed, dict):
        raise ConfigLoadError(
            f"Config file must contain a YAML mapping (dict), got {type(parsed).__name__}"
        )

    return parsed


def load_config(config_path: Path) -> M31RConfig:
    """
    Load, validate, and freeze a config file into a M31RConfig object.

    This is the single entry point for config loading in the entire system.
    After this function returns, the config is guaranteed to be:
      - structurally valid (all required fields present)
      - type-safe (all values match their declared types)
      - immutable (frozen pydantic model, no mutation possible)

    Args:
        config_path: Path to a YAML config file.

    Returns:
        A fully validated, frozen M31RConfig instance.

    Raises:
        ConfigLoadError: File I/O or YAML parse failures.
        ConfigValidationError: Schema violations (missing fields, wrong types, unknown keys).
    """
    raw_data = _read_yaml_file(config_path)

    try:
        config = M31RConfig.model_validate(raw_data)
    except ValidationError as err:
        raise ConfigValidationError(
            f"Config validation failed for {config_path}:\n{err}"
        ) from err

    return config
