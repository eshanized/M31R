# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
CLI entrypoint for M31R.

This is the single root command — every operation is a subcommand of `m31r`.
Per 12_CLI_AND_TOOLING_SPEC.md: no separate executables, no hidden commands,
no interactive prompts by default.

The global options (--config, --log-level, --dry-run, --seed) are inherited
by every subcommand through argparse's parent parser mechanism.

Usage:
    m31r <subcommand> [options]
    m31r crawl --config configs/global.yaml
    m31r info
    m31r train --config configs/train.yaml --seed 123
"""

import argparse
import sys

from m31r.cli.commands import (
    handle_crawl,
    handle_dataset,
    handle_eval,
    handle_export,
    handle_filter,
    handle_generate,
    handle_info,
    handle_serve,
    handle_tokenizer,
    handle_train,
    handle_verify,
)
from m31r.cli.exit_codes import USER_ERROR


def _build_global_parser() -> argparse.ArgumentParser:
    """
    Build the parent parser with global options.

    These options get inherited by every subcommand. We use a separate parent
    parser (with add_help=False) so that help text doesn't collide between the
    parent and the subcommand parsers.
    """
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file.",
    )
    parent.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging verbosity level.",
    )
    parent.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        dest="dry_run",
        help="Simulate the command without making changes.",
    )
    parent.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed (takes precedence over config).",
    )
    return parent


def _register_subcommands(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
    parent: argparse.ArgumentParser,
) -> None:
    """
    Register all subcommands with their handler functions.

    Each subcommand gets the global options from the parent parser and sets
    its handler function via set_defaults(func=...). When the user runs
    `m31r crawl`, argparse populates args.func with handle_crawl.
    """
    commands = [
        ("crawl", "Download raw Rust repositories.", handle_crawl),
        ("filter", "Apply filtering and cleaning to raw data.", handle_filter),
        ("dataset", "Build versioned dataset from filtered data.", handle_dataset),
        ("tokenizer", "Train or manage the tokenizer.", handle_tokenizer),
        ("train", "Train the model from scratch.", handle_train),
        ("eval", "Run evaluation suite.", handle_eval),
        ("serve", "Start local inference server.", handle_serve),
        ("generate", "Generate tokens from a prompt.", handle_generate),
        ("export", "Create release bundle.", handle_export),
        ("verify", "Validate artifact integrity.", handle_verify),
        ("info", "Display environment and config info.", handle_info),
    ]

    for name, help_text, handler in commands:
        parser = subparsers.add_parser(name, parents=[parent], help=help_text)
        parser.set_defaults(func=handler, dataset_dir=None)

    verify_parser = subparsers.choices["verify"]
    verify_parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        dest="dataset_dir",
        help="Path to a dataset directory to verify integrity of.",
    )



def main() -> None:
    """
    Main CLI entrypoint. This is what pyproject.toml's [project.scripts] points to.

    The flow is straightforward:
      1. Build the argument parser with global options and all subcommands
      2. Parse the command line
      3. Call the handler function for the chosen subcommand
      4. Exit with the handler's return code

    If no subcommand is given, we show help and exit with USER_ERROR.
    """
    parent = _build_global_parser()

    root_parser = argparse.ArgumentParser(
        prog="m31r",
        description="M31R — Offline-first Rust-focused SLM platform.",
        parents=[parent],
    )
    subparsers = root_parser.add_subparsers(dest="command")
    _register_subcommands(subparsers, parent)

    args = root_parser.parse_args()

    if not hasattr(args, "func") or args.func is None:
        root_parser.print_help()
        sys.exit(USER_ERROR)

    exit_code = args.func(args)
    sys.exit(exit_code)
