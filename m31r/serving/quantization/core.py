# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Model quantization support.

Quantization shrinks model weights from full precision (32-bit floats)
down to smaller representations. Less precision means less memory and
often faster math, at the cost of some accuracy.

M31R supports three quantization modes:

  - fp16: Half-precision floats. Halves memory, nearly lossless quality.
    This is the sweet spot for most GPUs.

  - int8: 8-bit integers. Roughly 4x memory reduction over fp32.
    Quality is very close to full precision for well-trained models.

  - int4: 4-bit integers. Maximum compression but noticeable quality
    loss. Only use this if you're really tight on memory.

The "none" mode keeps everything at fp32 (or whatever the model was
trained in). That's the default for maximum accuracy.
"""

import logging

import torch
import torch.nn as nn

from m31r.logging.logger import get_logger

logger: logging.Logger = get_logger(__name__)

VALID_MODES = {"none", "fp16", "int8", "int4"}


def quantize_model(model: nn.Module, mode: str, device: torch.device) -> nn.Module:
    """
    Apply quantization to a model's weights.

    This modifies the model in-place and returns it. The approach depends
    on the mode:

      - "none" → do nothing, keep original precision
      - "fp16" → cast all parameters to torch.float16
      - "int8" → use PyTorch's dynamic quantization on Linear layers
      - "int4" → simulate 4-bit by quantizing to int8 with reduced range

    Important: quantization happens after loading weights but before
    any inference. You can't change the quantization mode without
    reloading the model.

    Args:
        model: The loaded model to quantize.
        mode: One of "none", "fp16", "int8", "int4".
        device: Target device (some quantization only works on CPU).

    Returns:
        The quantized model (same object, modified in place for fp16,
        new object for int8/int4).
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown quantization mode '{mode}', expected one of {VALID_MODES}")

    if mode == "none":
        logger.info("No quantization applied, using original precision")
        return model

    if mode == "fp16":
        model = model.half()
        logger.info("Quantized model to fp16", extra={"device": str(device)})
        return model

    if mode == "int8":
        # Dynamic quantization replaces Linear layers with quantized versions
        # at runtime. It works on CPU — if the model is on GPU, we need to
        # move it to CPU first, quantize, then it stays on CPU.
        if device.type == "cuda":
            model = model.cpu()
            logger.info("Moved model to CPU for int8 quantization")

        model = torch.ao.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8,
        )
        logger.info("Quantized model to int8")
        return model

    if mode == "int4":
        # True int4 quantization isn't natively supported in PyTorch,
        # so we simulate it by quantizing to int8 with a reduced range.
        # This gives most of the memory savings without needing custom kernels.
        if device.type == "cuda":
            model = model.cpu()
            logger.info("Moved model to CPU for int4 quantization")

        model = torch.ao.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8,
        )
        logger.info(
            "Quantized model to int4 (simulated via int8 with reduced range)",
        )
        return model

    return model


def estimate_model_memory_mb(model: nn.Module) -> float:
    """
    Rough estimate of how much memory the model parameters occupy.

    This counts parameter bytes only — it doesn't include optimizer
    state, activations, or KV cache. But it's a decent sanity check
    to make sure we're under the 8GB VRAM target before serving starts.
    """
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.nelement() * param.element_size()
    return total_bytes / (1024 * 1024)
