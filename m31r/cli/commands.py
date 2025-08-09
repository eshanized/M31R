# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Subcommand handlers for the M31R CLI.

Each function here corresponds to one CLI subcommand. Phase 2 adds real
pipeline logic for crawl, filter, and dataset. The other commands remain
stubs until their respective phases.

No print() calls. Everything goes through the structured logger.
"""

import argparse
import logging
from pathlib import Path

from m31r.cli.exit_codes import CONFIG_ERROR, RUNTIME_ERROR, SUCCESS, VALIDATION_ERROR
from m31r.config.exceptions import ConfigError
from m31r.config.loader import load_config
from m31r.config.schema import M31RConfig
from m31r.logging.logger import get_logger
from m31r.runtime.bootstrap import bootstrap


def _load_and_bootstrap(
    args: argparse.Namespace,
    command_name: str,
) -> tuple[int, M31RConfig | None, logging.Logger]:
    """
    The shared setup that every command needs: load config, run bootstrap.

    Returns a tuple of (exit_code, config, logger). If exit_code is not SUCCESS,
    the caller should return it immediately — something went wrong during setup.
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
            return CONFIG_ERROR, None, logger

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

    return SUCCESS, config, logger


def _execute_stub(args: argparse.Namespace, command_name: str) -> int:
    """Placeholder handler for commands that aren't implemented yet."""
    exit_code, config, logger = _load_and_bootstrap(args, command_name)
    if exit_code != SUCCESS:
        return exit_code

    try:
        logger.info(
            "Command started",
            extra={"command": command_name, "dry_run": args.dry_run, "config": args.config},
        )
        logger.info("Command completed", extra={"command": command_name})
        return SUCCESS
    except Exception as err:
        logger.error(
            "Runtime error",
            extra={"command": command_name, "error": str(err)},
            exc_info=True,
        )
        return RUNTIME_ERROR


def _resolve_project_root() -> Path:
    """Find the project root so pipeline outputs land in the right place."""
    from m31r.utils.paths import resolve_project_root

    return resolve_project_root()


def handle_crawl(args: argparse.Namespace) -> int:
    """Clone repositories at pinned commits as defined in the dataset config."""
    exit_code, config, logger = _load_and_bootstrap(args, "crawl")
    if exit_code != SUCCESS:
        return exit_code

    try:
        if config is None or config.dataset is None:
            logger.info(
                "No dataset config provided, nothing to crawl",
                extra={"command": "crawl"},
            )
            return SUCCESS

        logger.info("Starting crawl", extra={"command": "crawl", "dry_run": args.dry_run})

        if args.dry_run:
            logger.info(
                "Dry run — would crawl sources",
                extra={"source_count": len(config.dataset.sources)},
            )
            return SUCCESS

        from m31r.data.crawl.crawler import crawl_repositories

        project_root = _resolve_project_root()
        result = crawl_repositories(config.dataset, project_root)

        logger.info(
            "Crawl finished",
            extra={
                "total": result.total_sources,
                "cloned": result.cloned,
                "skipped": result.skipped,
            },
        )
        return SUCCESS

    except Exception as err:
        logger.error("Crawl failed", extra={"error": str(err)}, exc_info=True)
        return RUNTIME_ERROR


def handle_filter(args: argparse.Namespace) -> int:
    """Run the filter pipeline over crawled data."""
    exit_code, config, logger = _load_and_bootstrap(args, "filter")
    if exit_code != SUCCESS:
        return exit_code

    try:
        if config is None or config.dataset is None:
            logger.info(
                "No dataset config provided, nothing to filter",
                extra={"command": "filter"},
            )
            return SUCCESS

        logger.info("Starting filter pipeline", extra={"command": "filter", "dry_run": args.dry_run})

        if args.dry_run:
            logger.info("Dry run — would filter crawled data")
            return SUCCESS

        from m31r.data.filter.pipeline import run_filter_pipeline

        project_root = _resolve_project_root()
        stats = run_filter_pipeline(config.dataset, project_root)

        logger.info(
            "Filter finished",
            extra={
                "kept": stats.kept,
                "total_scanned": stats.total_files_scanned,
            },
        )
        return SUCCESS

    except Exception as err:
        logger.error("Filter failed", extra={"error": str(err)}, exc_info=True)
        return RUNTIME_ERROR


def handle_dataset(args: argparse.Namespace) -> int:
    """Build versioned, sharded dataset from filtered files."""
    exit_code, config, logger = _load_and_bootstrap(args, "dataset")
    if exit_code != SUCCESS:
        return exit_code

    try:
        if config is None or config.dataset is None:
            logger.info(
                "No dataset config provided, nothing to build",
                extra={"command": "dataset"},
            )
            return SUCCESS

        logger.info("Starting dataset build", extra={"command": "dataset", "dry_run": args.dry_run})

        if args.dry_run:
            logger.info("Dry run — would build dataset from filtered files")
            return SUCCESS

        from m31r.data.dataset.builder import build_dataset

        project_root = _resolve_project_root()
        result = build_dataset(config.dataset, project_root)

        logger.info(
            "Dataset build finished",
            extra={
                "version_hash": result.version_hash,
                "total_files": result.total_files,
                "total_shards": result.total_shards,
            },
        )
        return SUCCESS

    except Exception as err:
        logger.error("Dataset build failed", extra={"error": str(err)}, exc_info=True)
        return RUNTIME_ERROR


def handle_tokenizer(args: argparse.Namespace) -> int:
    """Train or manage the tokenizer."""
    return _execute_stub(args, "tokenizer")


def handle_train(args: argparse.Namespace) -> int:
    """Train the model from scratch."""
    return _execute_stub(args, "train")


def handle_eval(args: argparse.Namespace) -> int:
    """Run evaluation suite against a trained model."""
    return _execute_stub(args, "eval")


def handle_serve(args: argparse.Namespace) -> int:
    """Start local inference server."""
    return _execute_stub(args, "serve")


def handle_generate(args: argparse.Namespace) -> int:
    """Generate tokens from a prompt (single-shot inference)."""
    return _execute_stub(args, "generate")


def handle_export(args: argparse.Namespace) -> int:
    """Create a release bundle from a trained model."""
    return _execute_stub(args, "export")


def handle_verify(args: argparse.Namespace) -> int:
    """Validate dataset or artifact integrity."""
    exit_code, config, logger = _load_and_bootstrap(args, "verify")
    if exit_code != SUCCESS:
        return exit_code

    try:
        logger.info("Starting verification", extra={"command": "verify"})

        if args.dataset_dir is not None:
            from m31r.data.hashing.integrity import verify_dataset_integrity

            dataset_path = Path(args.dataset_dir)
            if not dataset_path.is_dir():
                logger.error("Dataset directory not found", extra={"path": str(dataset_path)})
                return VALIDATION_ERROR

            is_valid = verify_dataset_integrity(dataset_path)
            if not is_valid:
                logger.error("Integrity check failed", extra={"path": str(dataset_path)})
                return VALIDATION_ERROR

            logger.info("Integrity check passed", extra={"path": str(dataset_path)})

        logger.info("Verification complete", extra={"command": "verify"})
        return SUCCESS

    except Exception as err:
        logger.error("Verification failed", extra={"error": str(err)}, exc_info=True)
        return RUNTIME_ERROR


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

