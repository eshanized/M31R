# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Artifact loader for the inference runtime.

This is the first thing that runs when you start serving — it finds the
model weights, tokenizer, and metadata on disk, verifies everything is
intact via SHA256 checksums, and hands back a ready-to-use bundle.

The loader is intentionally strict. If a checksum doesn't match, if a
file is missing, if the config doesn't line up with what the model
expects — it refuses to proceed. Better to fail loudly at startup than
silently serve garbage predictions.

No internet calls happen here. Everything is local. Once your artifacts
are on disk, you never need a network connection again.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from m31r.config.schema import ModelConfig, RuntimeConfig
from m31r.logging.logger import get_logger
from m31r.model.transformer import M31RTransformer, TransformerModelConfig
from m31r.serving.quantization.core import quantize_model
from m31r.utils.hashing import compute_sha256

logger: logging.Logger = get_logger(__name__)


@dataclass(frozen=True)
class LoadedArtifacts:
    """Everything the inference engine needs, bundled together after loading."""

    model: nn.Module
    tokenizer: object
    model_config: TransformerModelConfig
    device: torch.device
    metadata: dict[str, object]


def resolve_device(device_str: str) -> torch.device:
    """
    Turn the config's device string into an actual torch device.

    "auto" picks CUDA if available, otherwise CPU. This is what you
    want 99% of the time — the model runs on whatever hardware is
    available without you having to think about it.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


def _find_model_weights(model_dir: Path) -> Path:
    """
    Locate the model weights file within the artifact directory.

    We look for model.pt first (the standard export format), falling
    back to any .pt file if the exact name isn't there. If nothing
    works, we raise so the user knows what's missing.
    """
    standard = model_dir / "model.pt"
    if standard.is_file():
        return standard

    pt_files = list(model_dir.glob("*.pt"))
    if pt_files:
        return pt_files[0]

    raise FileNotFoundError(
        f"No model weights (.pt file) found in {model_dir}"
    )


def _verify_checksum(weights_path: Path, expected_hash: str | None) -> None:
    """
    Verify the integrity of the weights file against a known hash.

    If no expected hash is provided (older exports might not have one),
    we skip verification with a warning. But if there IS a hash and it
    doesn't match, that means the file was corrupted or tampered with,
    and we absolutely should not load it.
    """
    if expected_hash is None:
        logger.warning("No checksum available for weights — skipping verification")
        return

    actual_hash = compute_sha256(weights_path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Weights checksum mismatch. "
            f"Expected: {expected_hash[:16]}... "
            f"Got: {actual_hash[:16]}... "
            f"The model file may be corrupted."
        )
    logger.info("Weights checksum verified", extra={"hash": actual_hash[:16] + "..."})


def _load_export_metadata(model_dir: Path) -> dict[str, object]:
    """Load the export metadata JSON if it exists."""
    meta_path = model_dir / "export_metadata.json"
    if not meta_path.is_file():
        meta_path = model_dir / "metadata.json"

    if meta_path.is_file():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def _build_model_config(
    model_cfg: ModelConfig,
    seed: int,
) -> TransformerModelConfig:
    """
    Bridge between the YAML config schema and the actual model constructor.

    The config schema uses human-friendly names like "hidden_size" and
    "context_length", but the transformer expects "dim" and "max_seq_len".
    This function does that translation.
    """
    return TransformerModelConfig(
        vocab_size=model_cfg.vocab_size,
        dim=model_cfg.hidden_size,
        n_layers=model_cfg.n_layers,
        n_heads=model_cfg.n_heads,
        head_dim=model_cfg.head_dim,
        max_seq_len=model_cfg.context_length,
        dropout=model_cfg.dropout,
        norm_eps=model_cfg.norm_eps,
        rope_theta=model_cfg.rope_theta,
        init_std=model_cfg.init_std,
        seed=seed,
    )


def load_artifacts(
    model_cfg: ModelConfig,
    runtime_cfg: RuntimeConfig,
    project_root: Path,
) -> LoadedArtifacts:
    """
    Load everything needed for inference in one shot.

    This is the main entry point for the loader. Here's what happens:

      1. Figure out where the model weights and tokenizer live on disk
      2. Load the export metadata and check for a weights checksum
      3. Build the model architecture from config
      4. Load the saved weights into the model
      5. Apply quantization if requested (fp16, int8, etc.)
      6. Move everything to the target device (CPU or GPU)
      7. Load the tokenizer
      8. Return everything bundled up and ready to use

    If anything goes wrong — missing files, bad checksums, incompatible
    configs — we fail immediately with a clear error message.

    Args:
        model_cfg: Model architecture parameters from config.
        runtime_cfg: Runtime settings (device, quantization, paths).
        project_root: Root directory of the M31R project.

    Returns:
        LoadedArtifacts with model, tokenizer, config, device, and metadata.
    """
    device = resolve_device(runtime_cfg.device)
    model_dir = project_root / runtime_cfg.model_path

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    metadata = _load_export_metadata(model_dir)
    weights_path = _find_model_weights(model_dir)
    expected_hash = metadata.get("weights_sha256")
    _verify_checksum(weights_path, expected_hash)

    transformer_cfg = _build_model_config(model_cfg, runtime_cfg.seed)
    model = M31RTransformer(transformer_cfg)

    # Load weights onto CPU first, then quantize, then move to device.
    # This ordering matters because some quantization modes only work on CPU.
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model = quantize_model(model, runtime_cfg.quantization, device)

    # Only move to GPU if quantization didn't change our plans
    if runtime_cfg.quantization not in ("int8", "int4"):
        model = model.to(device)

    model.eval()

    # Load tokenizer
    tokenizer = _load_tokenizer(project_root / runtime_cfg.tokenizer_path)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        "Artifacts loaded successfully",
        extra={
            "device": str(device),
            "parameters": param_count,
            "quantization": runtime_cfg.quantization,
            "model_path": str(model_dir),
        },
    )

    return LoadedArtifacts(
        model=model,
        tokenizer=tokenizer,
        model_config=transformer_cfg,
        device=device,
        metadata=metadata,
    )


def _load_tokenizer(tokenizer_dir: Path) -> object:
    """
    Load the tokenizer from the bundle directory.

    We use the HuggingFace tokenizers library since that's what the
    training pipeline produces. If the tokenizer files aren't there,
    we return None rather than crashing — some operations (like model
    info) don't actually need a tokenizer.
    """
    tokenizer_path = tokenizer_dir / "tokenizer.json"

    if not tokenizer_path.is_file():
        logger.warning(
            "Tokenizer not found — generation will not work",
            extra={"path": str(tokenizer_path)},
        )
        return None

    try:
        from tokenizers import Tokenizer as HFTokenizer
        tokenizer = HFTokenizer.from_file(str(tokenizer_path))
        logger.info("Tokenizer loaded", extra={"path": str(tokenizer_path)})
        return tokenizer
    except ImportError:
        logger.warning("tokenizers package not installed — can't load tokenizer")
        return None
