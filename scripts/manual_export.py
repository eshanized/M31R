#!/usr/bin/env python3
"""Manually create model export for serving."""

import json
import shutil
from pathlib import Path
from m31r.utils.hashing import compute_sha256

# Paths
checkpoint_dir = Path("experiments/20260213_025853_42/checkpoints/step_000005")
exports_dir = Path("exports")
exports_dir.mkdir(parents=True, exist_ok=True)

# Copy model weights
model_src = checkpoint_dir / "model.pt"
model_dst = exports_dir / "model.pt"
shutil.copy2(model_src, model_dst)

# Compute checksum
weights_hash = compute_sha256(model_dst)

# Load checkpoint metadata
with open(checkpoint_dir / "metadata.json") as f:
    metadata = json.load(f)

# Create export metadata
export_meta = {
    "step": metadata.get("global_step", 0),
    "loss": metadata.get("loss", 0.0),
    "seed": metadata.get("seed", 0),
    "tokens_seen": metadata.get("tokens_seen", 0),
    "weights_sha256": weights_hash,
    "model_config": metadata.get("config_snapshot", {}).get("model", {}),
}

with open(exports_dir / "metadata.json", "w") as f:
    json.dump(export_meta, f, indent=2)

print(f"Exported model to {exports_dir}")
print(f"  Model: {model_dst}")
print(f"  Metadata: {exports_dir / 'metadata.json'}")
print(f"  Weights hash: {weights_hash}")
print(f"  Step: {export_meta['step']}")
print(f"  Loss: {export_meta['loss']}")
