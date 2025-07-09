# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Subcommand handlers for the M31R CLI.

Each function here corresponds to one CLI subcommand. In Phase 1 these are
foundation stubs — they accept the parsed args, load config if provided,
run bootstrap, log that the command was invoked, and exit cleanly.

The actual business logic (crawling, training, serving, etc.) gets added in
later phases. But the contract is already established: every command logs its
start and end, handles config loading failures, and returns proper exit codes.

No print() calls. Everything goes through the structured logger.
"""

import argparse
import logging
from pathlib import Path

from m31r.cli.exit_codes import CONFIG_ERROR, RUNTIME_ERROR, SUCCESS
from m31r.config.exceptions import ConfigError
from m31r.config.loader import load_config
from m31r.logging.logger import get_logger
from m31r.runtime.bootstrap import bootstrap


def _execute_command(args: argparse.Namespace, command_name: str) -> int:
    """
    Shared execution flow for all subcommands.

    Every command follows the same pattern:
      1. If --config is provided, load and validate it
      2. Run bootstrap (seed, env check, directories)
      3. Log that the command started
      4. (In future phases: run actual logic)
      5. Log completion and return exit code

    This is factored out to avoid repeating the same boilerplate 11 times.
    Each subcommand function below is a thin wrapper that calls this with
    the right command name.

    Args:
        args: Parsed CLI arguments from argparse.
        command_name: Name of the subcommand being run (for logging).

    Returns:
        Exit code integer.
    """
    logger = get_logger(f"m31r.cli.{command_name}", log_level=args.log_level)

    config = None
    if args.config is not None:
        try:
            config = load_config(Path(args.config))
        except ConfigError as err:
            logger.error(
                "Configuration error",
                extra={"command": command_name, "error": str(err)},
            )
            return CONFIG_ERROR

    try:
        if config is not None:
            bootstrap(config.global_config)
        else:
            logger.debug(
                "No config provided, running with defaults",
                extra={"command": command_name},
            )

        if args.seed is not None:
            from m31r.runtime.bootstrap import set_deterministic_seed

            set_deterministic_seed(args.seed)

        logger.info(
            "Command started",
            extra={
                "command": command_name,
                "dry_run": args.dry_run,
                "config": args.config,
            },
        )

        # Future phases will add real logic here.
        # For now, we confirm the command runs cleanly through the full
        # bootstrap → log → exit pipeline.

        logger.info("Command completed", extra={"command": command_name})
        return SUCCESS

    except Exception as err:
        logger.error(
            "Runtime error",
            extra={"command": command_name, "error": str(err)},
            exc_info=True,
        )
        return RUNTIME_ERROR


def handle_crawl(args: argparse.Namespace) -> int:
    """Download raw Rust repositories."""
    return _execute_command(args, "crawl")


def handle_filter(args: argparse.Namespace) -> int:
    """Apply filtering and cleaning to raw data."""
    return _execute_command(args, "filter")


def handle_dataset(args: argparse.Namespace) -> int:
    """Build versioned dataset from filtered data."""
    return _execute_command(args, "dataset")


def handle_tokenizer(args: argparse.Namespace) -> int:
    """Train or manage the tokenizer."""
    return _execute_command(args, "tokenizer")


def handle_train(args: argparse.Namespace) -> int:
    """Train the model from scratch."""
    return _execute_command(args, "train")


def handle_eval(args: argparse.Namespace) -> int:
    """Run evaluation suite against a trained model."""
    return _execute_command(args, "eval")


def handle_serve(args: argparse.Namespace) -> int:
    """Start local inference server."""
    return _execute_command(args, "serve")


def handle_generate(args: argparse.Namespace) -> int:
    """Generate tokens from a prompt (single-shot inference)."""
    return _execute_command(args, "generate")


def handle_export(args: argparse.Namespace) -> int:
    """Create a release bundle from a trained model."""
    return _execute_command(args, "export")


def handle_verify(args: argparse.Namespace) -> int:
    """Validate dataset or artifact integrity (checksums, manifests)."""
    return _execute_command(args, "verify")


def handle_info(args: argparse.Namespace) -> int:
    """Display environment and configuration information."""
    logger = get_logger("m31r.cli.info", log_level=args.log_level)

    from m31r import __version__
    from m31r.runtime.environment import get_system_info

    system_info = get_system_info()

    logger.info(
        "System information",
        extra={
            "m31r_version": __version__,
            "python_version": system_info.python_version,
            "platform": system_info.platform,
            "architecture": system_info.architecture,
            "hostname": system_info.hostname,
            "config": args.config,
        },
    )
    return SUCCESS
