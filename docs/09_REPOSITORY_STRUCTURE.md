# ============================================================
# M31R
# Repository Structure Specification
# File: 09_REPOSITORY_STRUCTURE.md
# Version: 1.0.0
# Status: AUTHORITATIVE
# Owner: Eshan Roy
# Document Order: 09 / 25
# Depends On:
#   01_VISION_PRD.md
#   02_REQUIREMENTS_SPEC.md
#   03_GLOSSARY_AND_DEFINITIONS.md
#   04_SYSTEM_ARCHITECTURE.md
#   05_DATA_ARCHITECTURE.md
#   06_MODEL_ARCHITECTURE.md
#   07_TRAINING_ARCHITECTURE.md
#   08_REASONING_COT_DESIGN.md
# ============================================================


# ============================================================
# 0. PURPOSE
# ============================================================

This document defines the official repository layout for M31R.

This document answers:

"How is the source code organized on disk and what belongs where?"

The repository structure is part of the architecture.

Incorrect structure leads to:

- hidden coupling
- technical debt
- poor maintainability
- onboarding friction
- accidental complexity

Therefore:

The layout defined here is mandatory.


# ============================================================
# 1. DESIGN PRINCIPLES
# ============================================================

RS-1
Structure must reflect architecture.

RS-2
Each folder must have one responsibility.

RS-3
No mixed concerns.

RS-4
Library code and CLI code must be separated.

RS-5
Experiments must not pollute core code.

RS-6
Artifacts must never mix with source.

RS-7
Deterministic rebuild must be possible.

RS-8
LLMs must easily infer responsibilities.

RS-9
Short, explicit names preferred.

RS-10
Flat hierarchy preferred over deep nesting.


# ============================================================
# 2. TOP LEVEL LAYOUT (CANONICAL)
# ============================================================

The repository MUST follow this exact layout:

M31R/
│
├─ m31r/
├─ tools/
├─ configs/
├─ docs/
├─ data/
├─ experiments/
├─ checkpoints/
├─ logs/
├─ scripts/
├─ tests/
├─ pyproject.toml
├─ Makefile
└─ README.md

No additional top-level folders are allowed without justification.


# ============================================================
# 3. LAYERING MODEL
# ============================================================

The repository is divided into 4 layers:

Layer 1 — Core library (m31r/)
Layer 2 — Executables (tools/)
Layer 3 — Configuration (configs/)
Layer 4 — Artifacts (data/, checkpoints/, logs/, experiments/)

Dependencies must only flow downward:

tools → m31r
configs → tools/m31r
artifacts → consumed only

Artifacts must never import code.


# ============================================================
# 4. CORE PACKAGE (m31r/)
# ============================================================

Purpose:

All reusable production logic.

Rules:

- pure Python package
- no CLI parsing
- no side effects
- importable
- testable

This is the only directory allowed to contain business logic.


# ============================================================
# 5. m31r/ SUBDIRECTORIES
# ============================================================

Canonical structure:

m31r/
│
├─ config/
├─ dataset/
├─ tokenizer/
├─ cot/
├─ model/
├─ trainer/
├─ eval/
├─ runtime/
├─ utils/
└─ __init__.py

Each folder corresponds to one subsystem.


# ============================================================
# 6. m31r/config/
# ============================================================

Purpose:

Configuration loading and validation.

Contains:

- schema definitions
- config parser
- validation logic

Must NOT contain:

- training code
- dataset logic


# ============================================================
# 7. m31r/dataset/
# ============================================================

Purpose:

Data acquisition and filtering logic.

Contains:

- crawlers
- filters
- dedupe
- normalization
- manifests

Must NOT contain:

- tokenizer
- training
- model code


# ============================================================
# 8. m31r/tokenizer/
# ============================================================

Purpose:

Tokenizer training and encoding.

Contains:

- tokenizer trainer
- encoder/decoder
- vocab utilities

Must NOT contain:

- neural training


# ============================================================
# 9. m31r/cot/
# ============================================================

Purpose:

Reasoning (Chain-of-Thought) injection logic.

Contains:

- AST heuristics
- reasoning generators
- masking utilities

Must NOT contain:

- training loops


# ============================================================
# 10. m31r/model/
# ============================================================

Purpose:

Neural architecture definitions.

Contains:

- transformer blocks
- embeddings
- forward pass

Must NOT contain:

- dataset
- CLI
- evaluation scripts


# ============================================================
# 11. m31r/trainer/
# ============================================================

Purpose:

Training engine.

Contains:

- train loop
- optimizer setup
- checkpoint logic
- distributed setup

Must NOT contain:

- model architecture definitions


# ============================================================
# 12. m31r/eval/
# ============================================================

Purpose:

Evaluation and metrics.

Contains:

- compile rate checks
- pass@k
- perplexity
- benchmarks

Must NOT contain:

- training code


# ============================================================
# 13. m31r/runtime/
# ============================================================

Purpose:

Inference runtime.

Contains:

- model loading
- token streaming
- generation

Must be lightweight.

Must NOT contain training dependencies.


# ============================================================
# 14. m31r/utils/
# ============================================================

Purpose:

Shared utilities.

Contains:

- logging
- hashing
- IO helpers
- deterministic tools

Must NOT contain business logic.


# ============================================================
# 15. TOOLS/ DIRECTORY
# ============================================================

Purpose:

CLI entrypoints only.

Rules:

- thin wrappers
- no heavy logic
- call library functions only

Examples:

tools/
├─ crawl.py
├─ build_dataset.py
├─ train_tokenizer.py
├─ train_model.py
├─ evaluate.py
└─ serve.py


# ============================================================
# 16. CONFIGS/ DIRECTORY
# ============================================================

Purpose:

All configuration files.

Rules:

- YAML or JSON only
- no logic
- version controlled

Examples:

configs/
├─ model_200m.yaml
├─ train.yaml
├─ tokenizer.yaml
└─ eval.yaml


# ============================================================
# 17. DOCS/ DIRECTORY
# ============================================================

Purpose:

Authoritative documentation set.

Contains:

- ordered specs
- markdown only

Naming must follow:

NN_name.md

Example:

01_vision_prd.md


# ============================================================
# 18. DATA/ DIRECTORY
# ============================================================

Purpose:

Local datasets and intermediate artifacts.

Contains:

- raw/
- filtered/
- datasets/
- shards/
- tokenizer/

Rules:

- not committed to git (except metadata)
- immutable
- large files only


# ============================================================
# 19. EXPERIMENTS/ DIRECTORY
# ============================================================

Purpose:

Training run outputs.

Each run:

experiments/<run_id>/

Contains:

- logs
- metrics
- config snapshot

Never overwritten.


# ============================================================
# 20. CHECKPOINTS/ DIRECTORY
# ============================================================

Purpose:

Saved model states.

Rules:

- large files only
- versioned
- not committed to git


# ============================================================
# 21. LOGS/ DIRECTORY
# ============================================================

Purpose:

System and debug logs.

Rules:

- ephemeral
- may be deleted safely


# ============================================================
# 22. TESTS/ DIRECTORY
# ============================================================

Purpose:

All automated tests.

Rules:

- mirror m31r structure
- unit + integration tests
- deterministic

Test code must not leak into production code.


# ============================================================
# 23. SCRIPTS/ DIRECTORY
# ============================================================

Purpose:

One-off maintenance scripts.

Examples:

- cleanup
- migration
- benchmarking helpers

Must NOT contain core logic.


# ============================================================
# 24. FORBIDDEN PRACTICES
# ============================================================

Not allowed:

- logic in notebooks
- mixing data with source
- business logic in tools/
- hidden folders
- ad-hoc scripts in root
- training inside model/
- experiments inside library


# ============================================================
# 25. DEPENDENCY RULES
# ============================================================

Allowed:

tools → m31r

Forbidden:

m31r → tools
m31r → experiments
m31r → data


# ============================================================
# 26. FILE NAMING CONVENTIONS
# ============================================================

Rules:

- snake_case
- descriptive
- no abbreviations unless standard

Examples:

train_engine.py
not:
trn.py


# ============================================================
# 27. MODULE SIZE RULES
# ============================================================

Guidelines:

- files < 500 lines preferred
- split logically
- avoid monoliths

Improves readability.


# ============================================================
# 28. IMPORT RULES
# ============================================================

Must:

- use absolute imports
- avoid circular dependencies

Circular imports are forbidden.


# ============================================================
# 29. REPOSITORY GUARANTEES
# ============================================================

Structure must ensure:

- fast onboarding
- predictable locations
- easy navigation
- LLM readability
- maintainability


# ============================================================
# 30. SUMMARY
# ============================================================

The M31R repository layout mirrors system architecture:

library → tools → configs → artifacts

Each directory has one responsibility.

No mixing.

No ambiguity.

All contributors must strictly follow this structure.

# END
# ============================================================
