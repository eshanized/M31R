# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Task executor — the heart of the evaluation pipeline.

This is where the actual evaluation happens. For each benchmark task,
the executor:
  1. Builds a prompt from the task
  2. Generates code using the model (K times for pass@k)
  3. Drops each generated output into a sandbox
  4. Compiles it with cargo
  5. Runs tests if compilation succeeded
  6. Records the binary pass/fail result

Each attempt uses a deterministic seed derived from the base seed plus
the attempt index, so running the same evaluation twice always produces
identical results.
"""

import time
from pathlib import Path

import torch
import torch.nn as nn

from m31r.evaluation.benchmarks.models import BenchmarkTask, TaskResult
from m31r.evaluation.compiler.harness import compile_rust, test_rust
from m31r.evaluation.compiler.sandbox import SandboxContext
from m31r.evaluation.tasks.builder import build_prompt
from m31r.logging.logger import get_logger

logger = get_logger(__name__)


def _generate_code(
    model: nn.Module,
    tokenizer: object,
    prompt: str,
    seed: int,
    max_new_tokens: int = 512,
) -> str:
    """
    Generate Rust code from a prompt using the model.

    This does greedy decoding (always pick the highest-probability token)
    because we need deterministic output. Temperature sampling would give
    different results each run, which violates our reproducibility requirement.

    The generation loop is simple:
    1. Encode the prompt into token IDs
    2. Feed through the model to get logits
    3. Take argmax of the last position's logits
    4. Append that token
    5. Repeat until we hit max_new_tokens or the EOS token

    We set the torch seed before each generation so attempt N with seed S
    always produces exactly the same output regardless of what happened in
    attempts 0 through N-1.
    """
    torch.manual_seed(seed)

    encode_fn = getattr(tokenizer, "encode", None)
    decode_fn = getattr(tokenizer, "decode", None)

    if encode_fn is None or decode_fn is None:
        logger.warning(
            "Tokenizer missing encode/decode — returning prompt as-is",
        )
        return prompt

    encoding = encode_fn(prompt)
    if hasattr(encoding, "ids"):
        input_ids = encoding.ids
    else:
        input_ids = list(encoding)

    device = next(model.parameters()).device
    tokens = torch.tensor([input_ids], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(tokens)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Check for end-of-sequence
            next_id = next_token.item()
            eos_id = getattr(tokenizer, "token_to_id", lambda x: None)("<eos>")
            if eos_id is not None and next_id == eos_id:
                break

            tokens = torch.cat([tokens, next_token], dim=1)

    output_ids = tokens[0].tolist()
    generated_ids = output_ids[len(input_ids):]

    return decode_fn(generated_ids)


def execute_task(
    model: nn.Module,
    tokenizer: object,
    task: BenchmarkTask,
    seed: int,
    k: int,
    compile_timeout: int = 10,
    test_timeout: int = 10,
    sandbox_base_dir: Path | None = None,
) -> list[TaskResult]:
    """
    Run K evaluation attempts on a single benchmark task.

    Each attempt gets its own deterministic seed (base_seed + attempt_index)
    and its own sandbox directory. The results are independent — one attempt
    crashing doesn't affect the others.

    Returns a list of K TaskResult objects, one per attempt. The caller
    is responsible for aggregating these into pass@k metrics.
    """
    prompt = build_prompt(task)
    results: list[TaskResult] = []

    for attempt in range(k):
        attempt_seed = seed + attempt

        gen_start = time.monotonic()
        try:
            generated_code = _generate_code(
                model, tokenizer, prompt, seed=attempt_seed,
            )
        except Exception as exc:
            logger.error(
                "Code generation failed",
                extra={
                    "task_id": task.task_id,
                    "attempt": attempt,
                    "error": str(exc),
                },
            )
            results.append(TaskResult(
                task_id=task.task_id,
                category=task.category,
                attempt_index=attempt,
                compiled=False,
                tests_passed=False,
                generation_time_seconds=time.monotonic() - gen_start,
            ))
            continue
        gen_elapsed = time.monotonic() - gen_start

        with SandboxContext(task, generated_code, sandbox_base_dir) as sandbox_dir:
            compile_result = compile_rust(sandbox_dir, timeout_seconds=compile_timeout)

            test_result = None
            if compile_result.success:
                test_result = test_rust(sandbox_dir, timeout_seconds=test_timeout)

            results.append(TaskResult(
                task_id=task.task_id,
                category=task.category,
                attempt_index=attempt,
                compiled=compile_result.success,
                tests_passed=test_result.success if test_result else False,
                compile_result=compile_result,
                test_result=test_result,
                generation_time_seconds=gen_elapsed,
                generated_code=generated_code,
            ))

        logger.debug(
            "Attempt complete",
            extra={
                "task_id": task.task_id,
                "attempt": attempt,
                "compiled": compile_result.success,
                "passed": test_result.success if test_result else False,
            },
        )

    return results


def execute_suite(
    model: nn.Module,
    tokenizer: object,
    tasks: list[BenchmarkTask],
    seed: int,
    k: int,
    compile_timeout: int = 10,
    test_timeout: int = 10,
    sandbox_base_dir: Path | None = None,
) -> list[TaskResult]:
    """
    Run the full benchmark suite against a model.

    Iterates through tasks in order (they should already be sorted by
    task_id for determinism), runs K attempts on each, and collects
    all results. Each task gets its own seed offset based on its position
    in the list, so task ordering affects per-task seeds deterministically.
    """
    all_results: list[TaskResult] = []

    for task_index, task in enumerate(tasks):
        task_seed = seed + (task_index * k)

        logger.info(
            "Evaluating task",
            extra={
                "task_id": task.task_id,
                "task_index": task_index + 1,
                "total_tasks": len(tasks),
                "k": k,
            },
        )

        task_results = execute_task(
            model=model,
            tokenizer=tokenizer,
            task=task,
            seed=task_seed,
            k=k,
            compile_timeout=compile_timeout,
            test_timeout=test_timeout,
            sandbox_base_dir=sandbox_base_dir,
        )
        all_results.extend(task_results)

    return all_results
