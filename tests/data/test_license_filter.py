# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
License detection and filtering tests.

Verifies that we correctly identify permissive licenses from LICENSE files
and Cargo.toml, and that the filtering logic matches the spec.
"""

import textwrap
from pathlib import Path

import pytest

from m31r.data.filter.license_filter import (
    _extract_license_from_cargo_toml,
    _identify_license_from_content,
    detect_license,
    is_permissive_license,
)


class TestLicenseIdentification:
    def test_mit_license_detected(self) -> None:
        content = "MIT License\n\nPermission is hereby granted, free of charge"
        assert _identify_license_from_content(content) == "MIT"

    def test_apache_license_detected(self) -> None:
        content = "Apache License\nVersion 2.0"
        assert _identify_license_from_content(content) == "Apache-2.0"

    def test_bsd3_license_detected(self) -> None:
        content = "BSD 3-Clause License"
        assert _identify_license_from_content(content) == "BSD-3-Clause"

    def test_unknown_content_returns_unknown(self) -> None:
        assert _identify_license_from_content("This is some random text") == "unknown"


class TestCargoTomlExtraction:
    def test_mit_from_cargo_toml(self) -> None:
        content = '[package]\nname = "my_crate"\nlicense = "MIT"\n'
        assert _extract_license_from_cargo_toml(content) == "MIT"

    def test_dual_license_picks_first_permissive(self) -> None:
        content = 'license = "MIT OR Apache-2.0"\n'
        result = _extract_license_from_cargo_toml(content)
        assert result in {"MIT", "Apache-2.0"}

    def test_missing_license_field(self) -> None:
        content = '[package]\nname = "no_license"\n'
        assert _extract_license_from_cargo_toml(content) == "unknown"


class TestDetectLicense:
    def test_detect_from_license_file(self, tmp_path: Path) -> None:
        license_file = tmp_path / "LICENSE"
        license_file.write_text(
            "MIT License\n\nPermission is hereby granted, free of charge",
            encoding="utf-8",
        )
        assert detect_license(tmp_path) == "MIT"

    def test_detect_from_cargo_toml_fallback(self, tmp_path: Path) -> None:
        cargo_file = tmp_path / "Cargo.toml"
        cargo_file.write_text(
            '[package]\nname = "test"\nlicense = "Apache-2.0"\n',
            encoding="utf-8",
        )
        assert detect_license(tmp_path) == "Apache-2.0"

    def test_no_license_found(self, tmp_path: Path) -> None:
        assert detect_license(tmp_path) == "unknown"

    def test_license_file_takes_priority(self, tmp_path: Path) -> None:
        license_file = tmp_path / "LICENSE"
        license_file.write_text(
            "MIT License\n\nPermission is hereby granted, free of charge",
            encoding="utf-8",
        )
        cargo_file = tmp_path / "Cargo.toml"
        cargo_file.write_text('license = "Apache-2.0"\n', encoding="utf-8")
        assert detect_license(tmp_path) == "MIT"


class TestPermissiveCheck:
    def test_mit_is_permissive(self) -> None:
        allowed = ["MIT", "Apache-2.0"]
        assert is_permissive_license("MIT", allowed) is True

    def test_gpl_is_not_permissive(self) -> None:
        allowed = ["MIT", "Apache-2.0"]
        assert is_permissive_license("GPL-3.0", allowed) is False

    def test_unknown_is_not_permissive(self) -> None:
        allowed = ["MIT"]
        assert is_permissive_license("unknown", allowed) is False
