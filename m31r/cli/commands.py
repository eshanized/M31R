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
    """
    Run the full evaluation pipeline against a trained model.

    This is the command that answers the question: "Is this model good enough?"
    It loads a checkpoint, runs every benchmark task K times, compiles the output,
    runs tests, and writes a report with pass@k and compile success rates.
    """
    exit_code, config, logger = _load_and_bootstrap(args, "eval")
    if exit_code != SUCCESS:
        return exit_code

    try:
        if config is None or config.eval is None:
            logger.error(
                "Eval config section is required — add it to your config file",
                extra={"command": "eval"},
            )
            return CONFIG_ERROR

        eval_config = config.eval
        logger.info(
            "Starting evaluation",
            extra={
                "command": "eval",
                "dry_run": args.dry_run,
                "k_values": eval_config.k_values,
                "benchmark_dir": eval_config.benchmark_directory,
            },
        )

        if args.dry_run:
            logger.info(
                "Dry run — would evaluate model against benchmarks",
                extra={
                    "k_values": eval_config.k_values,
                    "compile_timeout": eval_config.compile_timeout_seconds,
                    "test_timeout": eval_config.test_timeout_seconds,
                },
            )
            return SUCCESS

        from m31r.evaluation.benchmarks.loader import load_benchmark_suite
        from m31r.evaluation.metrics.engine import compute_metrics
        from m31r.evaluation.reporting.writer import write_report
        from m31r.evaluation.runner.executor import execute_suite

        project_root = _resolve_project_root()

        # Figure out where the benchmarks live
        benchmark_dir = Path(getattr(args, "benchmark_dir", None) or "")
        if not benchmark_dir.is_absolute():
            benchmark_dir = project_root / eval_config.benchmark_directory
        if not benchmark_dir.is_dir():
            logger.error(
                "Benchmark directory not found",
                extra={"path": str(benchmark_dir)},
            )
            return VALIDATION_ERROR

        suite = load_benchmark_suite(benchmark_dir)

        # Load the model and tokenizer from checkpoint
        model, tokenizer = _load_model_for_eval(args, config, project_root, logger)
        if model is None:
            return VALIDATION_ERROR

        # The max K value determines how many attempts each task gets
        max_k = max(eval_config.k_values)

        results = execute_suite(
            model=model,
            tokenizer=tokenizer,
            tasks=suite.tasks,
            seed=eval_config.seed,
            k=max_k,
            compile_timeout=eval_config.compile_timeout_seconds,
            test_timeout=eval_config.test_timeout_seconds,
        )

        metrics = compute_metrics(results, eval_config.k_values, seed=eval_config.seed)

        # Write results to experiments/<run_id>/eval/
        import time
        run_id = f"eval_{int(time.time())}"
        output_dir = project_root / eval_config.output_directory / run_id / "eval"

        config_snapshot = config.model_dump() if hasattr(config, "model_dump") else {}
        write_report(metrics, output_dir, config_snapshot=config_snapshot)

        logger.info(
            "Evaluation complete",
            extra={
                "compile_success_rate": round(metrics.compile_success_rate, 4),
                "pass_at_k": {str(k): round(v, 4) for k, v in metrics.pass_at_k.items()},
                "total_tasks": metrics.total_tasks,
                "output_dir": str(output_dir),
            },
        )
        return SUCCESS

    except FileNotFoundError as err:
        logger.error("Evaluation failed — missing files", extra={"error": str(err)})
        return VALIDATION_ERROR
    except Exception as err:
        logger.error("Evaluation failed", extra={"error": str(err)}, exc_info=True)
        return RUNTIME_ERROR


def _load_model_for_eval(
    args: argparse.Namespace,
    config: M31RConfig,
    project_root: Path,
    logger: logging.Logger,
) -> tuple:
    """
    Load model and tokenizer for evaluation.

    Tries to load from an explicit --checkpoint path first, then falls back
    to finding the latest checkpoint in the experiments directory. Returns
    (model, tokenizer) on success, (None, None) if something goes wrong.
    """
    try:
        import torch

        from m31r.model.transformer import M31RTransformer, TransformerModelConfig
        from m31r.training.checkpoint.core import find_latest_checkpoint, load_checkpoint

        if config.model is None:
            logger.error("Model config section is required for evaluation")
            return None, None

        model_cfg = TransformerModelConfig(
            vocab_size=config.model.vocab_size,
            dim=config.model.dim,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            head_dim=config.model.head_dim,
            max_seq_len=config.model.max_seq_len,
            dropout=config.model.dropout,
            norm_eps=config.model.norm_eps,
            rope_theta=config.model.rope_theta,
            init_std=config.model.init_std,
            seed=config.global_config.seed,
        )
        model = M31RTransformer(model_cfg)

        # Resolve checkpoint path
        checkpoint_path = getattr(args, "checkpoint", None)
        if checkpoint_path is not None:
            checkpoint_dir = Path(checkpoint_path)
        else:
            experiments_root = project_root / config.global_config.directories.experiments
            from m31r.training.engine.experiment import find_experiment_dir
            experiment_dir = find_experiment_dir(experiments_root, None)
            if experiment_dir is None:
                logger.error("No experiment found — train a model first")
                return None, None
            checkpoint_dir = find_latest_checkpoint(experiment_dir)
            if checkpoint_dir is None:
                logger.error(
                    "No checkpoint found in experiment",
                    extra={"experiment_dir": str(experiment_dir)},
                )
                return None, None

        if not checkpoint_dir.is_dir():
            logger.error(
                "Checkpoint directory not found",
                extra={"path": str(checkpoint_dir)},
            )
            return None, None

        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")

        load_checkpoint(checkpoint_dir, model, device=device)
        model = model.to(device)
        model.eval()

        # Load tokenizer if available
        tokenizer = None
        try:
            from tokenizers import Tokenizer as HFTokenizer

            tokenizer_path = project_root / "data" / "tokenizer" / "tokenizer.json"
            if config.train is not None:
                tokenizer_path = project_root / config.train.tokenizer_directory / "tokenizer.json"

            if tokenizer_path.is_file():
                tokenizer = HFTokenizer.from_file(str(tokenizer_path))
                logger.info("Tokenizer loaded", extra={"path": str(tokenizer_path)})
            else:
                logger.warning("No tokenizer found — generation will be limited")
        except ImportError:
            logger.warning("tokenizers package not available")

        logger.info(
            "Model loaded for evaluation",
            extra={
                "checkpoint": str(checkpoint_dir),
                "parameters": model.count_parameters(),
                "device": str(device),
            },
        )
        return model, tokenizer

    except Exception as err:
        logger.error("Failed to load model", extra={"error": str(err)}, exc_info=True)
        return None, None


def handle_benchmark(args: argparse.Namespace) -> int:
    """
    List or inspect benchmark tasks without running evaluation.

    Handy for checking that your benchmark directory is structured correctly
    and all tasks load without errors, before committing to a full eval run.
    """
    exit_code, config, logger = _load_and_bootstrap(args, "benchmark")
    if exit_code != SUCCESS:
        return exit_code

    try:
        from m31r.evaluation.benchmarks.loader import load_benchmark_suite

        project_root = _resolve_project_root()

        benchmark_dir_str = getattr(args, "benchmark_dir", None)
        if benchmark_dir_str:
            benchmark_dir = Path(benchmark_dir_str)
        elif config is not None and config.eval is not None:
            benchmark_dir = project_root / config.eval.benchmark_directory
        else:
            benchmark_dir = project_root / "benchmarks"

        if not benchmark_dir.is_dir():
            logger.error(
                "Benchmark directory not found",
                extra={"path": str(benchmark_dir)},
            )
            return VALIDATION_ERROR

        suite = load_benchmark_suite(benchmark_dir)

        # Tally tasks by category for a nice summary
        from collections import Counter
        category_counts = Counter(t.category for t in suite.tasks)

        logger.info(
            "Benchmark suite summary",
            extra={
                "version": suite.version,
                "total_tasks": len(suite.tasks),
                "categories": dict(category_counts),
            },
        )

        for task in suite.tasks:
            logger.info(
                "Task",
                extra={
                    "task_id": task.task_id,
                    "category": task.category,
                    "difficulty": task.difficulty,
                    "tags": task.tags,
                },
            )

        return SUCCESS

    except FileNotFoundError as err:
        logger.error("Benchmark loading failed", extra={"error": str(err)})
        return VALIDATION_ERROR
    except Exception as err:
        logger.error("Benchmark command failed", extra={"error": str(err)}, exc_info=True)
        return RUNTIME_ERROR


def handle_serve(args: argparse.Namespace) -> int:
    """Start the local inference server."""
    exit_code, config, logger = _load_and_bootstrap(args, "serve")
    if exit_code != SUCCESS:
        return exit_code

    try:
        if config is None or config.model is None or config.runtime is None:
            logger.error(
                "Model and runtime config sections are required for serving",
                extra={"command": "serve"},
            )
            return CONFIG_ERROR

        runtime_cfg = config.runtime
        host = getattr(args, "host", None) or runtime_cfg.host
        port = getattr(args, "port", None) or runtime_cfg.port

        logger.info(
            "Starting inference server",
            extra={
                "host": host,
                "port": port,
                "dry_run": args.dry_run,
                "quantization": runtime_cfg.quantization,
            },
        )

        if args.dry_run:
            logger.info(
                "Dry run — would start server",
                extra={"host": host, "port": port},
            )
            return SUCCESS

        from m31r.serving.api.schema import ServerStatus
        from m31r.serving.engine.core import InferenceEngine
        from m31r.serving.loader.core import load_artifacts
        from m31r.serving.server.core import run_server

        project_root = _resolve_project_root()
        artifacts = load_artifacts(config.model, runtime_cfg, project_root)

        engine = InferenceEngine(
            model=artifacts.model,
            tokenizer=artifacts.tokenizer,
            device=artifacts.device,
            max_context_length=runtime_cfg.max_context_length,
        )

        status = ServerStatus(
            model_loaded=True,
            model_path=str(project_root / runtime_cfg.model_path),
            device=str(artifacts.device),
            quantization=runtime_cfg.quantization,
            max_context_length=runtime_cfg.max_context_length,
        )

        run_server(
            engine=engine,
            host=host,
            port=port,
            status=status,
            max_request_size_bytes=runtime_cfg.max_request_size_bytes,
        )
        return SUCCESS

    except FileNotFoundError as err:
        logger.error("Serve failed — missing files", extra={"error": str(err)})
        return VALIDATION_ERROR
    except Exception as err:
        logger.error("Serve failed", extra={"error": str(err)}, exc_info=True)
        return RUNTIME_ERROR


def handle_generate(args: argparse.Namespace) -> int:
    """Generate tokens from a prompt (single-shot inference)."""
    exit_code, config, logger = _load_and_bootstrap(args, "generate")
    if exit_code != SUCCESS:
        return exit_code

    try:
        if config is None or config.model is None or config.runtime is None:
            logger.error(
                "Model and runtime config sections are required for generation",
                extra={"command": "generate"},
            )
            return CONFIG_ERROR

        prompt = getattr(args, "prompt", None) or ""
        if not prompt:
            logger.error("No prompt provided — use --prompt")
            return USER_ERROR

        runtime_cfg = config.runtime
        max_tokens = getattr(args, "max_tokens", None) or runtime_cfg.max_tokens
        temperature = getattr(args, "temperature", None)
        if temperature is None:
            temperature = runtime_cfg.temperature
        top_k = getattr(args, "top_k", None)
        if top_k is None:
            top_k = runtime_cfg.top_k
        quantization = getattr(args, "quantization_mode", None) or runtime_cfg.quantization

        logger.info(
            "Starting generation",
            extra={
                "dry_run": args.dry_run,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_k": top_k,
            },
        )

        if args.dry_run:
            logger.info(
                "Dry run — would generate text",
                extra={"prompt_length": len(prompt), "max_tokens": max_tokens},
            )
            return SUCCESS

        from m31r.serving.engine.core import InferenceEngine
        from m31r.serving.generation.core import GenerationConfig
        from m31r.serving.loader.core import load_artifacts

        project_root = _resolve_project_root()

        # Override quantization if specified via CLI
        if quantization != runtime_cfg.quantization:
            runtime_cfg = runtime_cfg.model_copy(
                update={"quantization": quantization},
            )

        artifacts = load_artifacts(config.model, runtime_cfg, project_root)

        engine = InferenceEngine(
            model=artifacts.model,
            tokenizer=artifacts.tokenizer,
            device=artifacts.device,
            max_context_length=runtime_cfg.max_context_length,
        )

        gen_config = GenerationConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=runtime_cfg.seed,
        )

        if runtime_cfg.stream:
            import sys as _sys

            for chunk in engine.generate_stream(prompt, gen_config):
                _sys.stdout.write(chunk.token_text)
                _sys.stdout.flush()
            _sys.stdout.write("\n")
            _sys.stdout.flush()
        else:
            response = engine.generate(prompt, gen_config)
            import sys as _sys

            _sys.stdout.write(response.text + "\n")
            _sys.stdout.flush()

        metrics_summary = engine.metrics.summary()
        logger.info("Generation complete", extra=metrics_summary)
        return SUCCESS

    except FileNotFoundError as err:
        logger.error("Generate failed — missing files", extra={"error": str(err)})
        return VALIDATION_ERROR
    except Exception as err:
        logger.error("Generate failed", extra={"error": str(err)}, exc_info=True)
        return RUNTIME_ERROR


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

