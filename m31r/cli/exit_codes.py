# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
CLI exit codes per 12_CLI_AND_TOOLING_SPEC.md section 6.

These are the only exit codes the system is allowed to use.
Anything else is a spec violation.
"""

SUCCESS: int = 0
USER_ERROR: int = 1
CONFIG_ERROR: int = 2
RUNTIME_ERROR: int = 3
VALIDATION_ERROR: int = 4
