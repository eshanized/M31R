# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
License detection and filtering.

The data architecture spec (§14) is very clear: only permissively licensed code
goes into the training corpus. GPL, AGPL, unknown, and proprietary licenses are
all rejected. This protects us legally and keeps the model's output clean.

License detection works by scanning common license file locations in each
repository. We look for SPDX identifiers in file content, which is the
industry standard way to tag licenses in open source projects.
"""

from pathlib import Path

PERMISSIVE_LICENSES: frozenset[str] = frozenset(
    {
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "Unlicense",
        "CC0-1.0",
        "0BSD",
        "BSL-1.0",
        "Zlib",
    }
)

LICENSE_FILE_NAMES: tuple[str, ...] = (
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "LICENSE-MIT",
    "LICENSE-APACHE",
    "LICENCE",
    "LICENCE.md",
    "LICENCE.txt",
    "COPYING",
    "COPYING.md",
    "COPYING.txt",
)

SPDX_PATTERNS: dict[str, list[str]] = {
    "MIT": ["MIT License", "Permission is hereby granted, free of charge"],
    "Apache-2.0": ["Apache License", "Version 2.0"],
    "BSD-2-Clause": ["BSD 2-Clause", "Redistribution and use"],
    "BSD-3-Clause": ["BSD 3-Clause"],
    "ISC": ["ISC License", "Permission to use, copy, modify"],
    "Unlicense": ["This is free and unencumbered software"],
    "CC0-1.0": ["CC0", "Creative Commons Zero"],
    "0BSD": ["Zero-Clause BSD"],
}


def detect_license(repo_dir: Path) -> str:
    """
    Try to figure out what license a repository uses.

    We scan common license file locations and look for recognizable patterns.
    The approach is deliberately conservative — if we can't confidently identify
    the license, we return "unknown" and the file gets filtered out.

    The detection order matters: we check the most common license files first,
    and within each file we check for SPDX patterns in order. First match wins.

    Returns an SPDX identifier string, or "unknown" if we can't determine it.
    """
    for license_name in LICENSE_FILE_NAMES:
        license_path = repo_dir / license_name
        if not license_path.is_file():
            continue

        try:
            content = license_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        detected = _identify_license_from_content(content)
        if detected != "unknown":
            return detected

    cargo_toml = repo_dir / "Cargo.toml"
    if cargo_toml.is_file():
        try:
            cargo_content = cargo_toml.read_text(encoding="utf-8", errors="replace")
            detected = _extract_license_from_cargo_toml(cargo_content)
            if detected != "unknown":
                return detected
        except OSError:
            pass

    return "unknown"


def _identify_license_from_content(content: str) -> str:
    """
    Match license file content against known SPDX patterns.

    We go through each known license type and check if its signature
    phrases appear in the content. This isn't foolproof — a file could
    contain the phrase "MIT License" in a comment about not being MIT —
    but in practice license files are pretty formulaic and this works well.
    """
    for spdx_id, patterns in SPDX_PATTERNS.items():
        if all(pattern in content for pattern in patterns):
            return spdx_id
    return "unknown"


def _extract_license_from_cargo_toml(content: str) -> str:
    """
    Pull the license field out of Cargo.toml as a fallback.

    Cargo.toml has a `license = "MIT"` or `license = "MIT OR Apache-2.0"` field.
    We do a simple text scan rather than a full TOML parse to avoid adding a
    dependency, and because we only need this one field.
    """
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("license") and "=" in stripped:
            value = stripped.split("=", 1)[1].strip().strip('"').strip("'")

            for part in value.split(" OR "):
                cleaned = part.strip()
                if cleaned in PERMISSIVE_LICENSES:
                    return cleaned

            if value in PERMISSIVE_LICENSES:
                return value

    return "unknown"


def is_permissive_license(license_id: str, allowed: list[str]) -> bool:
    """
    Check if a license identifier is in the allowed list.

    The allowed list comes from config, so different runs can have
    different license policies. The defaults match the spec's §14.
    """
    return license_id in set(allowed)
