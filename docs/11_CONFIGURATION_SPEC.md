# ============================================================
# M31R
# Configuration Specification
# File: 11_CONFIGURATION_SPEC.md
# Version: 1.0.0
# Status: AUTHORITATIVE
# Owner: Eshan Roy
# Document Order: 11 / 25
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
# ============================================================


# ============================================================
# 0. PURPOSE
# ============================================================

This document defines the configuration system used by M31R.

This document answers:

"How are all system behaviors controlled declaratively and reproducibly?"

Configuration is treated as a first-class artifact.

No behavior must depend on:

- hardcoded constants
- environment hacks
- manual edits
- hidden defaults

All behavior must be configurable and versioned.

This document is authoritative for:

- config structure
- schema rules
- file organization
- validation requirements
- loading behavior


# ============================================================
# 1. CONFIGURATION PHILOSOPHY
# ============================================================

M31R is config-driven.

Principle:

Behavior changes must require config changes, not code changes.

Reasons:

- reproducibility
- traceability
- easier experiments
- fewer bugs
- simpler audits
- LLM readability

Configs define the system.
Code executes the config.


# ============================================================
# 2. CORE PRINCIPLES
# ============================================================

CS-1
All runtime behavior must originate from config.

CS-2
Configs must be explicit.

CS-3
No hidden defaults.

CS-4
All configs must be serializable.

CS-5
All configs must be versioned.

CS-6
Configs must be validated.

CS-7
Configs must be deterministic.

CS-8
No environment-dependent behavior.

CS-9
Same config must produce same result.

CS-10
Human and machine readable formats only.


# ============================================================
# 3. SUPPORTED FORMATS
# ============================================================

Allowed:

- YAML
- JSON

Preferred:

YAML

Forbidden:

- Python scripts
- dynamic evaluation
- environment interpolation
- shell scripts

Reason:

configs must be static and declarative.


# ============================================================
# 4. CONFIG FILE LOCATION
# ============================================================

All configuration files must live under:

configs/

Example:

configs/
├─ dataset.yaml
├─ tokenizer.yaml
├─ model.yaml
├─ train.yaml
├─ eval.yaml
└─ runtime.yaml

No config files outside this directory.


# ============================================================
# 5. CONFIGURATION CATEGORIES
# ============================================================

Config types:

C1 Dataset
C2 Tokenizer
C3 Model
C4 Training
C5 Evaluation
C6 Runtime
C7 Global

Each category must have a dedicated file.


# ============================================================
# 6. GLOBAL CONFIG
# ============================================================

File:

configs/global.yaml

Purpose:

Cross-cutting settings.

Examples:

- project name
- seed
- directories
- logging level

Must not include stage-specific parameters.


# ============================================================
# 7. DATASET CONFIG
# ============================================================

File:

configs/dataset.yaml

Controls:

- sources
- filtering rules
- dedupe thresholds
- limits
- license policy
- shard size

Must not include training parameters.


# ============================================================
# 8. TOKENIZER CONFIG
# ============================================================

File:

configs/tokenizer.yaml

Controls:

- vocab size
- tokenizer type
- training corpus
- normalization rules
- seed

Must not include model or training logic.


# ============================================================
# 9. MODEL CONFIG
# ============================================================

File:

configs/model.yaml

Controls:

- layers
- hidden size
- heads
- context length
- dropout
- architecture flags

Must not include optimizer or dataset options.


# ============================================================
# 10. TRAINING CONFIG
# ============================================================

File:

configs/train.yaml

Controls:

- batch size
- learning rate
- optimizer
- schedule
- checkpoint frequency
- precision
- distributed settings

Must not include architecture definitions.


# ============================================================
# 11. EVALUATION CONFIG
# ============================================================

File:

configs/eval.yaml

Controls:

- benchmark sets
- metrics
- evaluation frequency
- pass@k
- compile checks

Must not include training behavior.


# ============================================================
# 12. RUNTIME CONFIG
# ============================================================

File:

configs/runtime.yaml

Controls:

- device selection
- quantization
- max tokens
- streaming options
- inference limits

Must not include training parameters.


# ============================================================
# 13. CONFIG SCHEMA REQUIREMENTS
# ============================================================

Each config must:

- follow strict schema
- define types
- define required fields
- define defaults explicitly

Validation must fail fast if schema violated.


# ============================================================
# 14. LOADING BEHAVIOR
# ============================================================

Rules:

- load config once at startup
- freeze object
- disallow mutation

Runtime mutation is forbidden.


# ============================================================
# 15. MERGING STRATEGY
# ============================================================

Allowed:

base + override

Example:

train.yaml
train.local.yaml

Rules:

- shallow override only
- deterministic precedence

Complex merging forbidden.


# ============================================================
# 16. ENVIRONMENT VARIABLES
# ============================================================

Environment variables are discouraged.

Allowed only for:

- path overrides
- hardware selection

Never for logic control.


# ============================================================
# 17. SEED MANAGEMENT
# ============================================================

Seed must be:

- explicitly set in global config
- propagated to all subsystems

Random seeds must never be implicit.


# ============================================================
# 18. PATH RULES
# ============================================================

Paths must be:

- relative to project root
- not absolute

Ensures portability.


# ============================================================
# 19. EXPERIMENT SNAPSHOTS
# ============================================================

Every run must:

- copy configs
- store snapshot

Location:

experiments/<run_id>/config_snapshot.yaml

Ensures reproducibility.


# ============================================================
# 20. EXAMPLE STRUCTURE
# ============================================================

Example model.yaml:

model:
  layers: 18
  hidden_size: 1024
  heads: 16
  context_length: 2048
  dropout: 0.1

Clear and explicit only.


# ============================================================
# 21. VALIDATION RULES
# ============================================================

Validation must check:

- missing fields
- unknown fields
- invalid ranges
- type mismatch

Failure must stop execution immediately.


# ============================================================
# 22. IMMUTABILITY
# ============================================================

After load:

Config objects must be immutable.

Mutation indicates bug.


# ============================================================
# 23. DOCUMENTATION REQUIREMENT
# ============================================================

Each field must:

- be documented
- include description
- include default
- include constraints

Undocumented fields forbidden.


# ============================================================
# 24. VERSIONING
# ============================================================

Configs must include:

config_version

Ensures compatibility tracking.


# ============================================================
# 25. NAMING CONVENTIONS
# ============================================================

Rules:

- snake_case
- descriptive names
- avoid abbreviations

Example:

learning_rate
not:
lr


# ============================================================
# 26. TYPE RULES
# ============================================================

Use:

- bool
- int
- float
- string
- list
- dict

Avoid:

- nested dynamic structures

Simple structures preferred.


# ============================================================
# 27. FORBIDDEN PRACTICES
# ============================================================

Not allowed:

- reading config from code comments
- hidden defaults
- hardcoded constants
- dynamic code execution
- environment-based behavior
- editing configs during runtime


# ============================================================
# 28. SECURITY RULES
# ============================================================

Configs must not:

- store secrets
- store credentials
- include API keys

Sensitive values must not exist.


# ============================================================
# 29. TRACEABILITY
# ============================================================

Every artifact must reference:

- config hash
- config snapshot

Without config, artifact is invalid.


# ============================================================
# 30. SUMMARY
# ============================================================

The configuration system makes M31R:

- deterministic
- reproducible
- auditable
- easy to experiment
- easy to automate

All behavior must be declared, never hidden.

Configs are the contract between intent and execution.

All implementations must strictly follow this specification.

# END
# ============================================================
