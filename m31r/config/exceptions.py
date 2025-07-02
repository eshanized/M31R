# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Custom exceptions for the configuration system.

We keep these separate so that CLI and other layers can catch config-specific
failures without importing the entire config machinery.
"""


class ConfigError(Exception):
    """Base for all configuration errors."""


class ConfigLoadError(ConfigError):
    """Raised when a config file cannot be read from disk or parsed as YAML."""


class ConfigValidationError(ConfigError):
    """
    Raised when a config file parses fine but fails schema validation.
    This covers missing required fields, type mismatches, out-of-range values,
    and any other structural problem.
    """


class ConfigSchemaError(ConfigError):
    """Raised when the schema definition itself is malformed (developer bug)."""
