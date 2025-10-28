# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Prompt builder for benchmark tasks.

Takes a BenchmarkTask and constructs the text prompt that gets fed to the
model for code generation. The prompt includes the incomplete code from
prompt.rs plus any context files the task provides.

We keep prompt construction separate from evaluation logic so it's easy
to experiment with prompt formatting without touching the rest of the
pipeline.
"""

from m31r.evaluation.benchmarks.models import BenchmarkTask


def build_prompt(task: BenchmarkTask) -> str:
    """
    Assemble the full prompt string from a benchmark task.

    The prompt format is straightforward:
    1. Any context files come first (like lib.rs or utility modules)
    2. Then the main prompt.rs content

    The model should complete or fix the code in prompt.rs such that
    the result compiles and passes the tests. We don't tell the model
    about the tests — it has to figure out the right behavior from the
    function signatures and documentation in the prompt.
    """
    parts: list[str] = []

    if task.context_files:
        for filename in sorted(task.context_files.keys()):
            content = task.context_files[filename]
            parts.append(f"// --- {filename} ---")
            parts.append(content)
            parts.append("")

    parts.append(task.prompt)

    return "\n".join(parts)
