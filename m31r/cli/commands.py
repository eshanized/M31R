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

from m31r.cli.exit_codes import CONFIG_ERROR, RUNTIME_ERROR, SUCCESS, USER_ERROR, VALIDATION_ERROR
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


def handle_tokenizer_train(args: argparse.Namespace) -> int:
    """
    Train a tokenizer from dataset shards and write the artifact bundle.

    This is the most important tokenizer command. It reads the streaming
    corpus from the dataset shards, trains BPE or Unigram vocabulary
    (depending on config), computes quality metrics on a sample, and
    writes the whole package to data/tokenizer/.
    """
    exit_code, config, logger = _load_and_bootstrap(args, "tokenizer_train")
    if exit_code != SUCCESS:
        return exit_code

    try:
        if config is None or config.tokenizer is None:
            logger.error(
                "Tokenizer config is required for training",
                extra={"command": "tokenizer_train"},
            )
            return CONFIG_ERROR

        logger.info(
            "Starting tokenizer training",
            extra={"command": "tokenizer_train", "dry_run": args.dry_run},
        )

        if args.dry_run:
            logger.info(
                "Dry run — would train tokenizer",
                extra={
                    "vocab_size": config.tokenizer.vocab_size,
                    "tokenizer_type": config.tokenizer.tokenizer_type,
                },
            )
            return SUCCESS

        from m31r.tokenizer.artifacts.bundle import create_bundle
        from m31r.tokenizer.metrics.core import compute_metrics
        from m31r.tokenizer.streaming.reader import stream_corpus
        from m31r.tokenizer.trainer.core import train_tokenizer

        project_root = _resolve_project_root()
        dataset_dir = project_root / config.tokenizer.dataset_directory

        latest_dataset = _find_latest_dataset(dataset_dir)
        if latest_dataset is None:
            logger.error(
                "No dataset shards found — run 'm31r dataset' first",
                extra={"dataset_dir": str(dataset_dir)},
            )
            return VALIDATION_ERROR

        corpus = stream_corpus(latest_dataset)
        tokenizer = train_tokenizer(config.tokenizer, corpus)

        # Grab a small sample from the corpus for metrics
        sample_corpus = stream_corpus(latest_dataset)
        sample_texts = []
        for i, text in enumerate(sample_corpus):
            if i >= 500:
                break
            sample_texts.append(text)

        if sample_texts:
            metrics = compute_metrics(tokenizer, sample_texts)
            logger.info(
                "Tokenizer quality metrics",
                extra={
                    "vocab_size": metrics.vocab_size,
                    "avg_tokens_per_line": metrics.avg_tokens_per_line,
                    "unk_rate": metrics.unk_rate,
                },
            )

        from m31r.utils.hashing import compute_sha256

        dataset_hash = ""
        manifest_path = latest_dataset / "manifest.json"
        if manifest_path.is_file():
            dataset_hash = compute_sha256(manifest_path)

        result = create_bundle(tokenizer, config.tokenizer, project_root, dataset_hash)

        logger.info(
            "Tokenizer training complete",
            extra={
                "version_hash": result.version_hash,
                "vocab_size": result.vocab_size,
                "output_dir": result.output_directory,
            },
        )
        return SUCCESS

    except Exception as err:
        logger.error("Tokenizer training failed", extra={"error": str(err)}, exc_info=True)
        return RUNTIME_ERROR


def handle_tokenizer_encode(args: argparse.Namespace) -> int:
    """Encode text into token IDs using a trained tokenizer."""
    exit_code, config, logger = _load_and_bootstrap(args, "tokenizer_encode")
    if exit_code != SUCCESS:
        return exit_code

    try:
        from tokenizers import Tokenizer

        from m31r.tokenizer.encoder.core import encode

        tokenizer_path = _resolve_tokenizer_path(args, config)
        if tokenizer_path is None or not tokenizer_path.is_file():
            logger.error(
                "No tokenizer.json found — run 'm31r tokenizer train' first",
                extra={"path": str(tokenizer_path)},
            )
            return VALIDATION_ERROR

        tokenizer = Tokenizer.from_file(str(tokenizer_path))

        text = args.text if hasattr(args, "text") and args.text else ""
        if hasattr(args, "input_file") and args.input_file:
            text = Path(args.input_file).read_text(encoding="utf-8")

        if not text:
            logger.error("No text provided — use --text or --input-file")
            return USER_ERROR

        token_ids = encode(tokenizer, text)
        logger.info(
            "Encoding complete",
            extra={
                "input_length": len(text),
                "token_count": len(token_ids),
                "tokens": token_ids,
            },
        )
        return SUCCESS

    except Exception as err:
        logger.error("Encoding failed", extra={"error": str(err)}, exc_info=True)
        return RUNTIME_ERROR


def handle_tokenizer_decode(args: argparse.Namespace) -> int:
    """Decode token IDs back into text using a trained tokenizer."""
    exit_code, config, logger = _load_and_bootstrap(args, "tokenizer_decode")
    if exit_code != SUCCESS:
        return exit_code

    try:
        from tokenizers import Tokenizer

        from m31r.tokenizer.decoder.core import decode

        tokenizer_path = _resolve_tokenizer_path(args, config)
        if tokenizer_path is None or not tokenizer_path.is_file():
            logger.error(
                "No tokenizer.json found — run 'm31r tokenizer train' first",
                extra={"path": str(tokenizer_path)},
            )
            return VALIDATION_ERROR

        tokenizer = Tokenizer.from_file(str(tokenizer_path))

        if not hasattr(args, "token_ids") or not args.token_ids:
            logger.error("No token IDs provided — use --ids")
            return USER_ERROR

        ids = [int(x) for x in args.token_ids.split(",")]
        text = decode(tokenizer, ids)

        logger.info(
            "Decoding complete",
            extra={"token_count": len(ids), "text": text},
        )
        return SUCCESS

    except Exception as err:
        logger.error("Decoding failed", extra={"error": str(err)}, exc_info=True)
        return RUNTIME_ERROR


def handle_tokenizer_info(args: argparse.Namespace) -> int:
    """Display metadata about an existing tokenizer bundle."""
    exit_code, config, logger = _load_and_bootstrap(args, "tokenizer_info")
    if exit_code != SUCCESS:
        return exit_code

    try:
        import json

        tokenizer_dir = _resolve_tokenizer_dir(args, config)
        if tokenizer_dir is None or not tokenizer_dir.is_dir():
            logger.error("No tokenizer bundle found", extra={"path": str(tokenizer_dir)})
            return VALIDATION_ERROR

        metadata_path = tokenizer_dir / "metadata.json"
        if not metadata_path.is_file():
            logger.error("No metadata.json in tokenizer bundle")
            return VALIDATION_ERROR

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        logger.info("Tokenizer info", extra=metadata)
        return SUCCESS

    except Exception as err:
        logger.error("Info command failed", extra={"error": str(err)}, exc_info=True)
        return RUNTIME_ERROR


def _find_latest_dataset(dataset_dir: Path) -> Path | None:
    """
    Find the most recently created dataset version directory.

    Dataset versions are subdirectories named by their content hash.
    We pick the last one alphabetically, which works because we just
    need any valid dataset — and if there's only one, that's it.
    """
    if not dataset_dir.is_dir():
        return None

    version_dirs = sorted(
        d for d in dataset_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    )

    if not version_dirs:
        return None

    return version_dirs[-1]


def _resolve_tokenizer_path(
    args: argparse.Namespace,
    config: M31RConfig | None,
) -> Path | None:
    """Figure out where the tokenizer.json file lives."""
    tokenizer_dir = _resolve_tokenizer_dir(args, config)
    if tokenizer_dir is None:
        return None
    return tokenizer_dir / "tokenizer.json"


def _resolve_tokenizer_dir(
    args: argparse.Namespace,
    config: M31RConfig | None,
) -> Path | None:
    """Figure out the tokenizer bundle directory from config or defaults."""
    project_root = _resolve_project_root()

    if config is not None and config.tokenizer is not None:
        return project_root / config.tokenizer.output_directory

    return project_root / "data" / "tokenizer"


def handle_train(args: argparse.Namespace) -> int:
    """Train the model from scratch."""
    exit_code, config, logger = _load_and_bootstrap(args, "train")
    if exit_code != SUCCESS:
        return exit_code

    try:
        if config is None or config.model is None or config.train is None:
            logger.error(
                "Model and training config sections are required",
                extra={"command": "train"},
            )
            return CONFIG_ERROR

        logger.info(
            "Starting training",
            extra={"command": "train", "dry_run": args.dry_run},
        )

        if args.dry_run:
            logger.info(
                "Dry run — would start training",
                extra={
                    "max_steps": config.train.max_steps,
                    "batch_size": config.train.batch_size,
                    "precision": config.train.precision,
                },
            )
            return SUCCESS

        from m31r.training.engine.core import run_training
        from m31r.training.engine.experiment import create_experiment_dir

        project_root = _resolve_project_root()
        experiments_root = project_root / config.global_config.directories.experiments
        experiment_dir = create_experiment_dir(
            experiments_root, config, config.global_config.seed,
        )

        result = run_training(config, experiment_dir)

        logger.info(
            "Training complete",
            extra={
                "final_step": result.final_step,
                "final_loss": result.final_loss,
                "total_tokens": result.total_tokens,
                "experiment_dir": result.experiment_dir,
            },
        )
        return SUCCESS

    except Exception as err:
        logger.error("Training failed", extra={"error": str(err)}, exc_info=True)
        return RUNTIME_ERROR


def handle_resume(args: argparse.Namespace) -> int:
    """Resume training from a checkpoint."""
    exit_code, config, logger = _load_and_bootstrap(args, "resume")
    if exit_code != SUCCESS:
        return exit_code

    try:
        if config is None or config.model is None or config.train is None:
            logger.error(
                "Model and training config sections are required",
                extra={"command": "resume"},
            )
            return CONFIG_ERROR

        logger.info(
            "Resuming training",
            extra={"command": "resume", "dry_run": args.dry_run},
        )

        if args.dry_run:
            logger.info("Dry run — would resume training")
            return SUCCESS

        from m31r.training.checkpoint.core import find_latest_checkpoint
        from m31r.training.engine.core import run_training
        from m31r.training.engine.experiment import find_experiment_dir

        project_root = _resolve_project_root()
        experiments_root = project_root / config.global_config.directories.experiments

        # Find experiment to resume
        run_id = getattr(args, "run_id", None)
        experiment_dir = find_experiment_dir(experiments_root, run_id)
        if experiment_dir is None:
            logger.error(
                "No experiment found to resume",
                extra={"experiments_root": str(experiments_root)},
            )
            return VALIDATION_ERROR

        # Find latest checkpoint
        checkpoint_dir = find_latest_checkpoint(experiment_dir)
        if checkpoint_dir is None:
            logger.error(
                "No checkpoint found in experiment",
                extra={"experiment_dir": str(experiment_dir)},
            )
            return VALIDATION_ERROR

        logger.info(
            "Found checkpoint",
            extra={"checkpoint": str(checkpoint_dir)},
        )

        result = run_training(config, experiment_dir, resume_from=checkpoint_dir)

        logger.info(
            "Resumed training complete",
            extra={
                "final_step": result.final_step,
                "final_loss": result.final_loss,
                "total_tokens": result.total_tokens,
            },
        )
        return SUCCESS

    except Exception as err:
        logger.error("Resume failed", extra={"error": str(err)}, exc_info=True)
        return RUNTIME_ERROR


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
    exit_code, config, logger = _load_and_bootstrap(args, "export")
    if exit_code != SUCCESS:
        return exit_code

    try:
        logger.info(
            "Starting export",
            extra={"command": "export", "dry_run": args.dry_run},
        )

        if args.dry_run:
            logger.info("Dry run — would export model")
            return SUCCESS

        from m31r.training.checkpoint.core import find_latest_checkpoint
        from m31r.training.engine.experiment import find_experiment_dir
        from m31r.training.export.core import export_model

        project_root = _resolve_project_root()

        # Find experiment and checkpoint
        experiments_root = project_root / (
            config.global_config.directories.experiments
            if config is not None
            else "experiments"
        )
        run_id = getattr(args, "run_id", None)
        experiment_dir = find_experiment_dir(experiments_root, run_id)
        if experiment_dir is None:
            logger.error("No experiment found to export from")
            return VALIDATION_ERROR

        checkpoint_dir = find_latest_checkpoint(experiment_dir)
        if checkpoint_dir is None:
            logger.error("No checkpoint found to export")
            return VALIDATION_ERROR

        output_dir = getattr(args, "output_dir", None)
        if output_dir is None:
            output_dir = str(project_root / "exports" / experiment_dir.name)
        output_path = Path(output_dir)

        tokenizer_dir = None
        if config is not None and config.train is not None:
            tokenizer_dir = project_root / config.train.tokenizer_directory

        result = export_model(checkpoint_dir, output_path, tokenizer_dir)

        logger.info(
            "Export complete",
            extra={
                "output_dir": result.output_dir,
                "weights_hash": result.weights_hash[:16] + "...",
                "step": result.step,
            },
        )
        return SUCCESS

    except Exception as err:
        logger.error("Export failed", extra={"error": str(err)}, exc_info=True)
        return RUNTIME_ERROR


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

