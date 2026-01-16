# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Evaluation report writer.

Writes evaluation results to disk in the format specified by
14_EVALUATION_METHODOLOGY.md §25 and 20_OBSERVABILITY_AND_LOGGING.md §16:

    experiments/<run_id>/eval/
    ├── metrics.json      — machine-readable metrics
    ├── report.txt        — human-readable summary
    └── config_snapshot.yaml  — the config used for this run

Everything is structured and deterministic. The metrics.json is the
authoritative output — report.txt is just a convenience view of the
same data.
"""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from m31r.evaluation.benchmarks.models import EvalMetrics
from m31r.logging.logger import get_logger

logger = get_logger(__name__)


def write_report(
    metrics: EvalMetrics,
    output_dir: Path,
    config_snapshot: dict[str, object] | None = None,
) -> Path:
    """
    Write the full evaluation report to disk.

    Creates the output directory if needed, then writes three files:
    metrics.json (the numbers), report.txt (human-readable version),
    and config_snapshot.yaml (what config was used). Returns the path
    to the output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    metrics_dict = asdict(metrics)
    metrics_path.write_text(
        json.dumps(metrics_dict, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    report_path = output_dir / "report.txt"
    report_text = format_report_text(metrics)
    report_path.write_text(report_text, encoding="utf-8")

    if config_snapshot is not None:
        config_path = output_dir / "config_snapshot.yaml"
        config_path.write_text(
            yaml.dump(config_snapshot, default_flow_style=False, sort_keys=True),
            encoding="utf-8",
        )

    logger.info(
        "Evaluation report written",
        extra={"output_dir": str(output_dir)},
    )

    return output_dir


def format_report_text(metrics: EvalMetrics) -> str:
    """
    Format metrics into a human-readable text report.

    This is what you'd look at to get a quick sense of how the model
    did. It's not the authoritative output (that's metrics.json), but
    it's a lot easier on the eyes.
    """
    timestamp = datetime.now(tz=timezone.utc).isoformat()
    lines: list[str] = [
        "=" * 60,
        "M31R EVALUATION REPORT",
        f"Generated: {timestamp}",
        f"Seed: {metrics.seed}",
        "=" * 60,
        "",
        "--- PRIMARY METRICS ---",
        f"Compile Success Rate: {metrics.compile_success_rate:.2%}",
    ]

    for k, score in sorted(metrics.pass_at_k.items()):
        lines.append(f"Pass@{k}: {score:.2%}")

    lines.extend(
        [
            "",
            "--- SUMMARY ---",
            f"Total Tasks: {metrics.total_tasks}",
            f"Total Attempts: {metrics.total_attempts}",
            f"Total Compiled: {metrics.total_compiled}",
            f"Total Passed: {metrics.total_passed}",
            "",
            "--- TIMING ---",
            f"Avg Generation: {metrics.avg_generation_time_seconds:.3f}s",
            f"Avg Compilation: {metrics.avg_compile_time_seconds:.3f}s",
            f"Avg Testing: {metrics.avg_test_time_seconds:.3f}s",
        ]
    )

    if metrics.category_compile_rates:
        lines.extend(["", "--- CATEGORY COMPILE RATES ---"])
        for cat, rate in sorted(metrics.category_compile_rates.items()):
            lines.append(f"  {cat}: {rate:.2%}")

    if metrics.category_pass_rates:
        lines.extend(["", "--- CATEGORY PASS RATES ---"])
        for cat, rate in sorted(metrics.category_pass_rates.items()):
            lines.append(f"  {cat}: {rate:.2%}")

    lines.extend(["", "=" * 60])
    return "\n".join(lines) + "\n"
