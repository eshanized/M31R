# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
AST-level syntax validation for Rust files.

We need to verify that Rust files are syntactically plausible before including
them in the training corpus. Broken syntax = broken training signal.

Important caveat: we do NOT compile with rustc or run cargo check. That would
mean executing untrusted code (build.rs, proc macros, etc.) which the security
spec explicitly forbids. Instead, we do structural validation — checking that
the file looks like valid Rust based on text-level heuristics.

This catches the obvious garbage (binary files misnamed as .rs, truncated files,
files with null bytes) while staying safe. It won't catch every possible syntax
error, but that's an acceptable trade-off for never executing untrusted code.
"""


def _has_null_bytes(content: str) -> bool:
    """Binary files sometimes sneak in with a .rs extension. Null bytes give them away."""
    return "\x00" in content


def _has_balanced_braces(content: str) -> bool:
    """
    Check that curly braces are balanced in the file.

    This is a quick structural sanity check. Every Rust file that parses
    successfully must have balanced braces. If they're not balanced, the file
    is either truncated, corrupted, or not actually Rust.

    We track a simple counter — increment on '{', decrement on '}'. If it
    goes negative at any point or doesn't end at zero, something's wrong.

    We skip braces inside string literals and comments to avoid false positives.
    The string/comment tracking is intentionally simplified — we're not building
    a full lexer, just catching the obvious cases.
    """
    depth = 0
    in_string = False
    in_line_comment = False
    in_block_comment = False
    prev_char = ""

    for char in content:
        if in_line_comment:
            if char == "\n":
                in_line_comment = False
            prev_char = char
            continue

        if in_block_comment:
            if prev_char == "*" and char == "/":
                in_block_comment = False
            prev_char = char
            continue

        if in_string:
            if char == '"' and prev_char != "\\":
                in_string = False
            prev_char = char
            continue

        if prev_char == "/" and char == "/":
            in_line_comment = True
            prev_char = char
            continue

        if prev_char == "/" and char == "*":
            in_block_comment = True
            prev_char = char
            continue

        if char == '"':
            in_string = True
            prev_char = char
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return False

        prev_char = char

    return depth == 0


def _looks_like_rust(content: str) -> bool:
    """
    A quick smell test for whether content is plausibly Rust code.

    We look for common Rust keywords and patterns. If a file claims to be .rs
    but contains none of the usual suspects (fn, use, struct, impl, mod, etc.),
    it's probably not real Rust code.

    This is intentionally generous — we'd rather keep a few odd files than
    accidentally drop valid Rust. The deduplicator and other filters catch
    the remaining noise.
    """
    rust_indicators = [
        "fn ", "let ", "use ", "mod ", "pub ", "struct ", "impl ",
        "enum ", "trait ", "match ", "crate", "extern ", "macro_rules!",
        "async ", "unsafe ", "where ", "self", "mut ", "const ",
        "type ", "return ", "if ", "for ", "while ", "loop ",
    ]
    content_lower = content.lower()
    return any(indicator in content_lower for indicator in rust_indicators)


def validate_rust_syntax(content: str) -> bool:
    """
    Run all structural checks on a piece of Rust source code.

    Returns True if the content looks like valid, non-corrupt Rust code.
    Returns False if any check fails.

    The checks run in order of cheapness — null byte detection is O(n) with
    a small constant, while brace balancing needs to track state. We bail
    early on the first failure.
    """
    if not content.strip():
        return False

    if _has_null_bytes(content):
        return False

    if not _has_balanced_braces(content):
        return False

    if not _looks_like_rust(content):
        return False

    return True
