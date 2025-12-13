#!/usr/bin/env python3
# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Benchmark regression detection for CI.

Compares metrics from the latest evaluation run against a baseline.
Fails if any metric regresses by more than the allowed threshold.

Usage:
    python scripts/ci/regression_check.py --baseline <baseline.json> --current <current.json>
    python scripts/ci/regression_check.py --baseline <baseline.json> --current <current.json> --threshold 0.05
"""

import argparse
import json
import sys


def load_metrics(path: str) -> dict[str, float]:
    """Load metrics from a JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Flatten pass_at_k into top-level metrics
    metrics: dict[str, float] = {}
    if "pass_at_k" in data:
        for k, v in data["pass_at_k"].items():
            metrics[f"pass_at_{k}"] = float(v)
    if "compile_success_rate" in data:
        metrics["compile_success_rate"] = float(data["compile_success_rate"])

    return metrics


def check_regressions(
    baseline: dict[str, float],
    current: dict[str, float],
    threshold: float,
) -> list[str]:
    """
    Compare current metrics against baseline.

    A regression is when a metric drops by more than `threshold` fraction.
    For example, if baseline pass@1 is 0.50 and current is 0.45,
    the drop is 0.05/0.50 = 0.10 (10%), which exceeds a 5% threshold.

    Returns list of regression error messages. Empty = no regressions.
    """
    errors: list[str] = []

    for metric_name, baseline_value in sorted(baseline.items()):
        if metric_name not in current:
            errors.append(f"Missing metric in current results: {metric_name}")
            continue

        current_value = current[metric_name]
        if baseline_value == 0:
            continue  # Can't compute relative regression from zero baseline

        relative_drop = (baseline_value - current_value) / baseline_value
        if relative_drop > threshold:
            errors.append(
                f"REGRESSION: {metric_name} dropped from "
                f"{baseline_value:.4f} → {current_value:.4f} "
                f"({relative_drop:.1%} regression, threshold: {threshold:.1%})"
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Check for benchmark regressions")
    parser.add_argument("--baseline", required=True, help="Path to baseline metrics JSON")
    parser.add_argument("--current", required=True, help="Path to current metrics JSON")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Maximum allowed relative regression (default: 0.05 = 5%%)",
    )
    args = parser.parse_args()

    print(f"Baseline: {args.baseline}")
    print(f"Current:  {args.current}")
    print(f"Threshold: {args.threshold:.1%}")
    print()

    try:
        baseline = load_metrics(args.baseline)
        current = load_metrics(args.current)
    except (FileNotFoundError, json.JSONDecodeError) as err:
        print(f"ERROR: Failed to load metrics: {err}")
        return 1

    errors = check_regressions(baseline, current, args.threshold)

    if errors:
        print("=== REGRESSIONS DETECTED ===")
        for error in errors:
            print(f"  ✗ {error}")
        return 1
    else:
        print("=== No regressions detected ===")
        for name in sorted(baseline.keys()):
            bv = baseline[name]
            cv = current.get(name, 0)
            print(f"  ✓ {name}: {bv:.4f} → {cv:.4f}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
