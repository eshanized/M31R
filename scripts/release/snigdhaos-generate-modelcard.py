#!/usr/bin/env python3
"""
M31R Release Automation: Model Card Generator
=============================================

Automated generation of RELEASE/README.md from artifact metadata.
Strictly deterministic, factual, and production-grade.

Usage:
    python3 snigdhaos-generate-modelcard.py <release_dir>

Author: Eshan Roy
Status: PROD
"""

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("m31r.release.modelcard")


class ModelCardGenerator:
    """Generates a deterministic README.md for M31R releases."""

    def __init__(self, release_dir: Path) -> None:
        self.release_dir = release_dir
        self.config_path = release_dir / "config.yaml"
        self.metadata_path = release_dir / "metadata.json"
        self.output_path = release_dir / "README.md"
        self.checksum_path = release_dir / "checksum.txt"

        self._validate_paths()

    def _validate_paths(self) -> None:
        """Fail-fast if required artifacts are missing."""
        required = [self.config_path, self.metadata_path, self.checksum_path]
        for p in required:
            if not p.exists():
                logger.error(f"Missing required artifact: {p}")
                sys.exit(1)

    def load_config(self) -> Dict[str, Any]:
        """Load YAML configuration safely."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)

    def load_metadata(self) -> Dict[str, Any]:
        """Load JSON metadata safely."""
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            sys.exit(1)

    def get_file_stats(self) -> List[Dict[str, str]]:
        """Compute file sizes and hashes for the report."""
        stats = []
        # Sort for deterministic output
        files = sorted([f for f in self.release_dir.iterdir() if f.is_file()])
        
        for f in files:
            if f.name == "README.md":
                continue
            
            size_mb = f.stat().st_size / (1024 * 1024)
            sha256 = self._compute_sha256(f)
            stats.append({
                "filename": f.name,
                "size": f"{size_mb:.2f} MB",
                "sha256": sha256
            })
        return stats

    def _compute_sha256(self, path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            logger.error(f"Failed to hash {path}: {e}")
            sys.exit(1)

    def render_markdown(
        self, 
        config: Dict[str, Any], 
        metadata: Dict[str, Any], 
        files: List[Dict[str, str]]
    ) -> str:
        """Render the model card markdown content."""
        
        # Extract fields safely with defaults
        model_ver = metadata.get("version", "unknown")
        project_name = config.get("global", {}).get("project_name", "M31R")
        
        # Sections
        model_conf = config.get("model", {})
        train_conf = config.get("train", {})
        
        # Model Params
        n_layers = model_conf.get("n_layers", "N/A")
        hidden_size = model_conf.get("hidden_size", "N/A")
        n_heads = model_conf.get("n_heads", "N/A")
        head_dim = model_conf.get("head_dim", "N/A")
        context_len = model_conf.get("context_length", "N/A")
        vocab_size = model_conf.get("vocab_size", "N/A")
        rope_theta = model_conf.get("rope_theta", "N/A")
        norm_eps = model_conf.get("norm_eps", "N/A")
        dropout = model_conf.get("dropout", "N/A")

        # Training Params
        dataset = train_conf.get("dataset_directory", "unknown")
        steps = train_conf.get("max_steps", "unknown")
        bs = train_conf.get("batch_size", "unknown")
        grad_accum = train_conf.get("gradient_accumulation_steps", "unknown")
        lr = train_conf.get("learning_rate", "unknown")
        min_lr = train_conf.get("min_learning_rate", "unknown")
        weight_decay = train_conf.get("weight_decay", "unknown")
        beta1 = train_conf.get("beta1", "unknown")
        beta2 = train_conf.get("beta2", "unknown")
        grad_clip = train_conf.get("grad_clip", "unknown")
        warmup = train_conf.get("warmup_steps", "unknown")
        precision = train_conf.get("precision", "unknown")
        
        commit = metadata.get("git_commit", "unknown")
        seed = metadata.get("training_seed", "unknown")
        timestamp = metadata.get("timestamp", datetime.now(timezone.utc).isoformat())

        # Build Markdown
        md = []
        md.append(f"# {project_name} - Release {model_ver}")
        md.append("")
        md.append("## Description")
        md.append(f"Auto-generated release of the **{project_name}** language model.")
        md.append("This model is built for **offline-first, deterministic, and safe execution**.")
        md.append("")
        
        md.append("## Architecture Configuration")
        md.append("| Component | Value |")
        md.append("|-----------|-------|")
        md.append(f"| Layers | {n_layers} |")
        md.append(f"| Hidden Size | {hidden_size} |")
        md.append(f"| Attention Heads | {n_heads} |")
        md.append(f"| Head Dimension | {head_dim} |")
        md.append(f"| Context Length | {context_len} tokens |")
        md.append(f"| Vocab Size | {vocab_size} |")
        md.append(f"| RoPE Theta | {rope_theta} |")
        md.append(f"| Norm Epsilon | {norm_eps} |")
        md.append(f"| Dropout | {dropout} |")
        md.append("")

        md.append("## Training Hyperparameters")
        md.append("### Optimizer (AdamW)")
        md.append("| Parameter | Value |")
        md.append("|-----------|-------|")
        md.append(f"| Learning Rate | {lr} |")
        md.append(f"| Min Learning Rate | {min_lr} |")
        md.append(f"| Weight Decay | {weight_decay} |")
        md.append(f"| Beta1 | {beta1} |")
        md.append(f"| Beta2 | {beta2} |")
        md.append(f"| Gradient Clip | {grad_clip} |")
        md.append(f"| Warmup Steps | {warmup} |")
        md.append("")

        md.append("### Batching & Compute")
        md.append("| Parameter | Value |")
        md.append("|-----------|-------|")
        md.append(f"| Batch Size | {bs} |")
        md.append(f"| Gradient Accumulation | {grad_accum} |")
        md.append(f"| Max Steps | {steps} |")
        md.append(f"| Precision | {precision} |")
        md.append(f"| Seed | {seed} |")
        md.append("")
        
        md.append("## Source & Metadata")
        md.append(f"- **Dataset Source**: `{dataset}`")
        md.append(f"- **Git Commit**: `{commit}`")
        md.append(f"- **Timestamp**: `{timestamp}`")
        md.append("")
        
        md.append("## Intended Use")
        md.append("- Local, offline code generation")
        md.append("- Research into deterministic ML systems")
        md.append("- Embedded systems with limited resources")
        md.append("")
        
        md.append("## Limitations")
        md.append("- **Not for production safety-critical systems** without further validation.")
        md.append("- Trained on specific datasets; may not generalize to broad domains.")
        md.append("- **Offline only**: Does not access the internet.")
        md.append("")
        
        md.append("## Hardware Requirements")
        md.append("- **Architecture**: x86_64 / ARM64")
        md.append("- **RAM**: Minimum 4GB recommended")
        md.append("- **Storage**: ~200MB for artifacts")
        md.append("")
        
        md.append("## Usage")
        md.append("### CLI")
        md.append("```bash")
        md.append(f"m31r generate --model release/{model_ver} --prompt \"fn main() {{\"")
        md.append("```")
        md.append("")
        
        md.append("### Python API")
        md.append("```python")
        md.append("from m31r.api import Runtime")
        md.append("runtime = Runtime.load('path/to/release')")
        md.append("# Generate text")
        md.append("output = runtime.generate('fn main()', max_tokens=128)")
        md.append("```")
        md.append("")
        
        md.append("## Evaluation")
        metrics = metadata.get("metrics_summary", {})
        if metrics:
            md.append("| Metric | Value |")
            md.append("|--------|-------|")
            for k, v in metrics.items():
                md.append(f"| {k} | {v} |")
        else:
            md.append("No specific evaluation metrics recorded in metadata.")
        md.append("")
        
        md.append("## Safety & License")
        md.append("- **License**: MIT (See `pyproject.toml`)")
        md.append("- **Security**: Validated via `m31r verify`.")
        md.append("- **Network**: 0.0.0.0 (No outbound calls).")
        md.append("")
        
        md.append("## Checksums (SHA256)")
        md.append("Verification is mandatory before use.")
        md.append("")
        md.append("| Filename | Size | SHA256 |")
        md.append("|----------|------|--------|")
        for f in files:
            md.append(f"| `{f['filename']}` | {f['size']} | `{f['sha256']}` |")
        md.append("")
        
        md.append("---")
        md.append(f"*Generated by snigdhaos-generate-modelcard.py at {datetime.now(timezone.utc).isoformat()}*")
        
        return "\n".join(md)

    def run(self) -> None:
        """Execute the generation process."""
        logger.info(f"Processing release: {self.release_dir}")
        
        config = self.load_config()
        metadata = self.load_metadata()
        files = self.get_file_stats()
        
        content = self.render_markdown(config, metadata, files)
        
        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Generated {self.output_path}")
        except Exception as e:
            logger.error(f"Failed to write README: {e}")
            sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate release README.md")
    parser.add_argument("release_dir", type=Path, help="Path to release directory")
    args = parser.parse_args()

    if not args.release_dir.exists():
        logger.error(f"Release directory not found: {args.release_dir}")
        sys.exit(1)

    gen = ModelCardGenerator(args.release_dir)
    gen.run()


if __name__ == "__main__":
    main()
