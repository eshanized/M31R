# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Content normalization for Rust source files.

The spec (§13) says normalization must include UTF-8 encoding, consistent
line endings, and trailing whitespace trimming. Critically, it must NOT
change semantics — we're cleaning up formatting noise, not rewriting code.

This is a pure function with no side effects: same input always produces
the same output, which is exactly what the determinism requirement demands.
"""


def normalize_content(text: str) -> str:
    """
    Clean up a Rust source file's formatting without changing its meaning.

    What this does, step by step:
    1. Replace Windows line endings (\\r\\n) with Unix ones (\\n)
    2. Replace any leftover carriage returns (old Mac style)
    3. Strip trailing whitespace from every line
    4. Make sure the file ends with exactly one newline
    5. Remove any stray null bytes (shouldn't be there, but safety first)

    What this does NOT do:
    - Reformat code (that's rustfmt's job, not ours)
    - Change indentation
    - Modify string literals
    - Touch anything semantic
    """
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\x00", "")

    lines = text.split("\n")
    lines = [line.rstrip() for line in lines]

    result = "\n".join(lines)

    result = result.strip("\n")
    if result:
        result += "\n"

    return result
