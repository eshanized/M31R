# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Filter subsystem tests.

Tests for extension filtering, path exclusions, size limits, AST validation,
license detection, normalization, and the full filter pipeline end-to-end.
"""

import textwrap
from pathlib import Path, PurePosixPath

import pytest

from m31r.data.filter.ast_validator import (
    _has_balanced_braces,
    _has_null_bytes,
    _looks_like_rust,
    validate_rust_syntax,
)
from m31r.data.filter.exclusions import exceeds_size_limits, should_exclude_path
from m31r.data.filter.normalizer import normalize_content
from m31r.data.filter.rust_filter import is_allowed_extension, is_rust_file


class TestRustFilter:
    def test_rust_extension_detected(self) -> None:
        assert is_rust_file(PurePosixPath("main.rs")) is True

    def test_non_rust_extension_rejected(self) -> None:
        assert is_rust_file(PurePosixPath("main.py")) is False

    def test_no_extension_rejected(self) -> None:
        assert is_rust_file(PurePosixPath("Makefile")) is False

    def test_allowed_extension_match(self) -> None:
        assert is_allowed_extension(PurePosixPath("lib.rs"), [".rs"]) is True

    def test_allowed_extension_mismatch(self) -> None:
        assert is_allowed_extension(PurePosixPath("lib.py"), [".rs"]) is False

    def test_allowed_extension_case_insensitive(self) -> None:
        assert is_allowed_extension(PurePosixPath("lib.RS"), [".rs"]) is True


class TestExclusions:
    def test_target_directory_excluded(self) -> None:
        path = PurePosixPath("target/debug/main.rs")
        assert should_exclude_path(path, []) is True

    def test_git_directory_excluded(self) -> None:
        path = PurePosixPath(".git/hooks/pre-commit")
        assert should_exclude_path(path, []) is True

    def test_vendor_directory_excluded(self) -> None:
        path = PurePosixPath("vendor/some_crate/lib.rs")
        assert should_exclude_path(path, []) is True

    def test_normal_path_allowed(self) -> None:
        path = PurePosixPath("src/main.rs")
        assert should_exclude_path(path, []) is False

    def test_custom_exclusion(self) -> None:
        path = PurePosixPath("custom_dir/file.rs")
        assert should_exclude_path(path, ["custom_dir"]) is True

    def test_binary_extension_excluded(self) -> None:
        path = PurePosixPath("lib.so")
        assert should_exclude_path(path, []) is True

    def test_image_extension_excluded(self) -> None:
        path = PurePosixPath("assets/logo.png")
        assert should_exclude_path(path, []) is True


class TestSizeLimits:
    def test_within_limits(self) -> None:
        content = "fn main() {}\n"
        assert exceeds_size_limits(content, max_bytes=1024, max_lines=100) is False

    def test_exceeds_byte_limit(self) -> None:
        content = "x" * 2000
        assert exceeds_size_limits(content, max_bytes=1024, max_lines=100000) is True

    def test_exceeds_line_limit(self) -> None:
        content = "\n".join(["line"] * 200)
        assert exceeds_size_limits(content, max_bytes=1_000_000, max_lines=50) is True


class TestASTValidator:
    def test_valid_rust_code(self) -> None:
        code = textwrap.dedent("""\
            fn main() {
                let x = 42;
            }
        """)
        assert validate_rust_syntax(code) is True

    def test_empty_content_rejected(self) -> None:
        assert validate_rust_syntax("") is False
        assert validate_rust_syntax("   \n  ") is False

    def test_null_bytes_rejected(self) -> None:
        assert validate_rust_syntax("fn main() {\x00}") is False

    def test_unbalanced_braces_rejected(self) -> None:
        assert validate_rust_syntax("fn main() {") is False

    def test_excess_closing_braces_rejected(self) -> None:
        assert validate_rust_syntax("fn main() { } }") is False

    def test_non_rust_content_rejected(self) -> None:
        assert validate_rust_syntax("This is just plain text with no code.") is False

    def test_braces_in_strings_handled(self) -> None:
        code = 'fn main() { let x = "{ unclosed"; }'
        assert validate_rust_syntax(code) is True

    def test_braces_in_comments_handled(self) -> None:
        code = "fn main() {\n// { this is a comment\n}"
        assert validate_rust_syntax(code) is True

    def test_block_comments_handled(self) -> None:
        code = "fn main() {\n/* { block comment */ }"
        assert validate_rust_syntax(code) is True

    def test_complex_valid_rust(self) -> None:
        code = textwrap.dedent("""\
            use std::collections::HashMap;

            pub struct Config {
                name: String,
                values: HashMap<String, i32>,
            }

            impl Config {
                pub fn new(name: &str) -> Self {
                    Self {
                        name: name.to_string(),
                        values: HashMap::new(),
                    }
                }
            }
        """)
        assert validate_rust_syntax(code) is True


class TestNormalizer:
    def test_windows_line_endings_converted(self) -> None:
        result = normalize_content("fn main() {\r\n}\r\n")
        assert "\r" not in result
        assert result == "fn main() {\n}\n"

    def test_trailing_whitespace_stripped(self) -> None:
        result = normalize_content("fn main()   \n{  \n}  \n")
        assert result == "fn main()\n{\n}\n"

    def test_null_bytes_removed(self) -> None:
        result = normalize_content("fn\x00 main() {}")
        assert "\x00" not in result

    def test_ends_with_single_newline(self) -> None:
        result = normalize_content("fn main() {}\n\n\n")
        assert result.endswith("\n")
        assert not result.endswith("\n\n")

    def test_empty_input_stays_empty(self) -> None:
        assert normalize_content("") == ""

    def test_deterministic_output(self) -> None:
        input_text = "fn main() {\r\n    let x = 42;  \r\n}\r\n\r\n"
        result_a = normalize_content(input_text)
        result_b = normalize_content(input_text)
        assert result_a == result_b
