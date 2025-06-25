# ============================================================
# M31R
# CLI and Tooling Specification
# File: 12_CLI_AND_TOOLING_SPEC.md
# Version: 1.0.0
# Status: AUTHORITATIVE
# Owner: Eshan Roy
# Document Order: 12 / 25
# Depends On:
#   01_VISION_PRD.md
#   02_REQUIREMENTS_SPEC.md
#   03_GLOSSARY_AND_DEFINITIONS.md
#   04_SYSTEM_ARCHITECTURE.md
#   05_DATA_ARCHITECTURE.md
#   06_MODEL_ARCHITECTURE.md
#   07_TRAINING_ARCHITECTURE.md
#   08_REASONING_COT_DESIGN.md
#   09_REPOSITORY_STRUCTURE.md
#   10_DEVELOPMENT_WORKFLOW.md
#   11_CONFIGURATION_SPEC.md
# ============================================================


# ============================================================
# 0. PURPOSE
# ============================================================

This document defines the official Command Line Interface (CLI)
and tooling standards for M31R.

This document answers:

"How do users and automation interact with the system?"

The CLI is the only supported interface for:

- pipeline execution
- training
- evaluation
- serving
- maintenance tasks

Direct invocation of internal modules is forbidden.

All operations must be reachable through stable commands.

This document is authoritative for:

- CLI design
- command structure
- argument rules
- behavior contracts
- tooling boundaries


# ============================================================
# 1. DESIGN PHILOSOPHY
# ============================================================

The CLI must be:

- deterministic
- scriptable
- composable
- minimal
- stable
- machine-friendly

The CLI must NOT be:

- interactive only
- stateful
- hidden
- ambiguous

Every command must behave like a build tool.

Commands must:

input → deterministic output


# ============================================================
# 2. CORE PRINCIPLES
# ============================================================

CT-1
CLI is the only execution surface.

CT-2
Commands must be idempotent.

CT-3
All commands must accept --config.

CT-4
All outputs must be logged.

CT-5
Commands must be non-interactive by default.

CT-6
Stable names only.

CT-7
Backward compatibility required.

CT-8
Scripts must be automatable.

CT-9
No hidden side effects.

CT-10
Clear exit codes required.


# ============================================================
# 3. ENTRYPOINT
# ============================================================

Single root command:

m31r

All operations must be subcommands.

No separate executables allowed.

Examples:

m31r crawl
m31r train
m31r eval


# ============================================================
# 4. COMMAND STRUCTURE
# ============================================================

Standard format:

m31r <subcommand> [options]

Example:

m31r train --config configs/train.yaml

Rules:

- lowercase only
- snake_case forbidden for commands
- hyphen allowed


# ============================================================
# 5. GLOBAL OPTIONS
# ============================================================

All commands must support:

--config <path>
--log-level <level>
--dry-run
--seed <int>
--help

Behavior must be consistent.


# ============================================================
# 6. EXIT CODES
# ============================================================

0 = success
1 = user error
2 = config error
3 = runtime error
4 = validation error

No other codes allowed.


# ============================================================
# 7. SUBCOMMAND GROUPS
# ============================================================

Commands are grouped by function:

Group A — Data
Group B — Tokenizer
Group C — Training
Group D — Evaluation
Group E — Runtime
Group F — Utilities


# ============================================================
# 8. GROUP A — DATA COMMANDS
# ============================================================

Commands:

m31r crawl
m31r filter
m31r dataset

Responsibilities:

- acquisition
- filtering
- dataset building

These commands must not touch model code.


# ============================================================
# 9. COMMAND: crawl
# ============================================================

Purpose:

Download raw repositories.

Inputs:

--config dataset.yaml

Outputs:

data/raw/

Behavior:

deterministic cloning only

Must not filter or tokenize.


# ============================================================
# 10. COMMAND: filter
# ============================================================

Purpose:

Apply filtering and cleaning.

Inputs:

raw data

Outputs:

data/filtered/

Behavior:

pure transformation

Must not modify raw data.


# ============================================================
# 11. COMMAND: dataset
# ============================================================

Purpose:

Build versioned dataset.

Inputs:

filtered data

Outputs:

datasets/

Behavior:

generate manifest and partitions


# ============================================================
# 12. GROUP B — TOKENIZER COMMANDS
# ============================================================

Commands:

m31r tokenizer train
m31r tokenizer encode
m31r tokenizer decode


# ============================================================
# 13. COMMAND: tokenizer train
# ============================================================

Purpose:

Train tokenizer.

Outputs:

data/tokenizer/

Must not modify dataset.


# ============================================================
# 14. GROUP C — TRAINING COMMANDS
# ============================================================

Commands:

m31r train
m31r resume
m31r export


# ============================================================
# 15. COMMAND: train
# ============================================================

Purpose:

Train model from scratch.

Inputs:

token shards + config

Outputs:

checkpoints + logs

Must be resumable.


# ============================================================
# 16. COMMAND: resume
# ============================================================

Purpose:

Resume interrupted training.

Inputs:

checkpoint

Behavior:

continue deterministically


# ============================================================
# 17. COMMAND: export
# ============================================================

Purpose:

Create release bundle.

Outputs:

final model + tokenizer + metadata


# ============================================================
# 18. GROUP D — EVALUATION COMMANDS
# ============================================================

Commands:

m31r eval
m31r benchmark


# ============================================================
# 19. COMMAND: eval
# ============================================================

Purpose:

Run evaluation suite.

Outputs:

metrics.json

Must not alter model.


# ============================================================
# 20. GROUP E — RUNTIME COMMANDS
# ============================================================

Commands:

m31r serve
m31r generate


# ============================================================
# 21. COMMAND: serve
# ============================================================

Purpose:

Start local inference server.

Behavior:

offline only

No network dependency required.


# ============================================================
# 22. COMMAND: generate
# ============================================================

Purpose:

Generate tokens from prompt.

Behavior:

single-shot inference


# ============================================================
# 23. GROUP F — UTILITY COMMANDS
# ============================================================

Commands:

m31r info
m31r clean
m31r verify


# ============================================================
# 24. COMMAND: info
# ============================================================

Purpose:

Display environment and config info.


# ============================================================
# 25. COMMAND: clean
# ============================================================

Purpose:

Remove temporary artifacts.

Must not delete checkpoints by default.


# ============================================================
# 26. COMMAND: verify
# ============================================================

Purpose:

Validate dataset or artifacts integrity.

Checks:

hashes
manifests


# ============================================================
# 27. LOGGING RULES
# ============================================================

All commands must:

- log to stdout
- structured logs
- timestamps

Print statements forbidden.


# ============================================================
# 28. OUTPUT RULES
# ============================================================

CLI output must be:

- machine parsable
- minimal
- deterministic

No decorative text.


# ============================================================
# 29. FORBIDDEN PRACTICES
# ============================================================

Not allowed:

- hidden commands
- interactive prompts by default
- modifying multiple stages silently
- training without config
- side effects outside artifacts
- notebooks as entrypoints


# ============================================================
# 30. SUMMARY
# ============================================================

The M31R CLI is:

- the single control plane
- deterministic
- scriptable
- stable
- automation-friendly

Every operation must be accessible through m31r <command>.

No other execution path is supported.

This guarantees reproducibility, maintainability, and enterprise safety.

# END
# ============================================================
