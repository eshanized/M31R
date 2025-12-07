# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Release hardening subsystem for M31R.

Provides packaging, checksum verification, manifest generation,
environment validation, security scanning, cleanup, and reproducibility
tooling. This is operations infrastructure — no ML logic lives here.

Per 18_RELEASE_PROCESS.md: every release must be reproducible, tested,
benchmarked, versioned, auditable, and immutable.
"""
