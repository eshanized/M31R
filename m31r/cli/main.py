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
    handle_benchmark,
    handle_clean,
    handle_crawl,
    handle_dataset,
    handle_eval,
    handle_export,
    handle_filter,
    handle_generate,
    handle_info,
    handle_resume,
    handle_serve,
    handle_tokenizer_decode,
    handle_tokenizer_encode,
    handle_tokenizer_info,
    handle_tokenizer_train,
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
        ("train", "Train the model from scratch.", handle_train),
        ("resume", "Resume training from a checkpoint.", handle_resume),
        ("eval", "Run evaluation suite.", handle_eval),
        ("benchmark", "List or inspect benchmark tasks.", handle_benchmark),
        ("serve", "Start local inference server.", handle_serve),
        ("generate", "Generate tokens from a prompt.", handle_generate),
        ("export", "Create release bundle.", handle_export),
        ("verify", "Validate artifact integrity.", handle_verify),
        ("clean", "Remove temporary files and caches.", handle_clean),
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

    # Resume command args
    resume_parser = subparsers.choices["resume"]
    resume_parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        dest="run_id",
        help="Experiment run ID to resume. If omitted, resumes the latest.",
    )

    # Export command args
    export_parser = subparsers.choices["export"]
    export_parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        dest="run_id",
        help="Experiment run ID to export from. If omitted, exports the latest.",
    )
    export_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        dest="output_dir",
        help="Directory to write the release bundle to.",
    )
    export_parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Release version string (default: package version).",
    )

    # Verify command args — add --release-dir
    verify_parser = subparsers.choices["verify"]
    verify_parser.add_argument(
        "--release-dir",
        type=str,
        default=None,
        dest="release_dir",
        help="Path to a release directory to verify.",
    )

    # Clean command args
    clean_parser = subparsers.choices["clean"]
    clean_parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Also clean releases (use with caution).",
    )
    clean_parser.add_argument(
        "--logs",
        action="store_true",
        default=False,
        help="Also remove log files.",
    )

    # Eval command args
    eval_parser = subparsers.choices["eval"]
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint directory to evaluate.",
    )
    eval_parser.add_argument(
        "--benchmark-dir",
        type=str,
        default=None,
        dest="benchmark_dir",
        help="Override the benchmark directory path.",
    )

    # Benchmark command args
    benchmark_parser = subparsers.choices["benchmark"]
    benchmark_parser.add_argument(
        "--benchmark-dir",
        type=str,
        default=None,
        dest="benchmark_dir",
        help="Override the benchmark directory path.",
    )

    # Generate command args
    gen_parser = subparsers.choices["generate"]
    gen_parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt to generate from.",
    )
    gen_parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        dest="max_tokens",
        help="Maximum number of tokens to generate.",
    )
    gen_parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (0.0 = greedy).",
    )
    gen_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        dest="top_k",
        help="Top-k sampling parameter (0 = disabled).",
    )
    gen_parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        dest="quantization_mode",
        choices=["none", "fp16", "int8", "int4"],
        help="Override quantization mode.",
    )
    gen_parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Override device selection.",
    )

    # Serve command args
    serve_parser = subparsers.choices["serve"]
    serve_parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Server bind address (default: 127.0.0.1).",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server bind port (default: 8731).",
    )
    serve_parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        dest="quantization_mode",
        choices=["none", "fp16", "int8", "int4"],
        help="Override quantization mode.",
    )
    serve_parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Override device selection.",
    )

    _register_tokenizer_subcommands(subparsers, parent)


def _register_tokenizer_subcommands(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
    parent: argparse.ArgumentParser,
) -> None:
    """
    Set up the tokenizer subcommand with its own sub-subcommands.

    Instead of a single 'tokenizer' command, we get:
      m31r tokenizer train   — train a new tokenizer
      m31r tokenizer encode  — encode text into token IDs
      m31r tokenizer decode  — decode token IDs into text
      m31r tokenizer info    — show metadata about the trained tokenizer
    """
    tok_parser = subparsers.add_parser(
        "tokenizer",
        help="Train or manage the tokenizer.",
    )
    tok_subs = tok_parser.add_subparsers(dest="tokenizer_command")

    train_p = tok_subs.add_parser("train", parents=[parent], help="Train a new tokenizer.")
    train_p.set_defaults(func=handle_tokenizer_train, dataset_dir=None)

    encode_p = tok_subs.add_parser("encode", parents=[parent], help="Encode text into token IDs.")
    encode_p.add_argument("--text", type=str, default=None, help="Text to encode.")
    encode_p.add_argument(
        "--input-file",
        type=str,
        default=None,
        dest="input_file",
        help="Path to a text file to encode.",
    )
    encode_p.set_defaults(func=handle_tokenizer_encode, dataset_dir=None)

    decode_p = tok_subs.add_parser("decode", parents=[parent], help="Decode token IDs into text.")
    decode_p.add_argument(
        "--ids",
        type=str,
        default=None,
        dest="token_ids",
        help="Comma-separated token IDs to decode.",
    )
    decode_p.set_defaults(func=handle_tokenizer_decode, dataset_dir=None)

    info_p = tok_subs.add_parser("info", parents=[parent], help="Show tokenizer metadata.")
    info_p.set_defaults(func=handle_tokenizer_info, dataset_dir=None)


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
