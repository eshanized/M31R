# ============================================================
# M31R
# System Architecture Specification
# File: 04_SYSTEM_ARCHITECTURE.md
# Version: 1.0.0
# Status: AUTHORITATIVE
# Owner: Eshan Roy
# Document Order: 04 / 25
# Depends On:
#   01_VISION_PRD.md
#   02_REQUIREMENTS_SPEC.md
#   03_GLOSSARY_AND_DEFINITIONS.md
# ============================================================


# ============================================================
# 0. PURPOSE
# ============================================================

This document defines the complete system architecture of M31R.

This document answers:

"How is the system structured end-to-end?"

This document describes:

- components
- boundaries
- responsibilities
- data flow
- control flow
- lifecycle stages
- contracts between subsystems

This document does NOT define:

- low-level code
- hyperparameters
- implementation details

Those appear in later documents.

If an architectural conflict occurs, this document is authoritative.


# ============================================================
# 1. ARCHITECTURAL PRINCIPLES
# ============================================================

AP-1
The system MUST be modular.

AP-2
Each stage MUST be independently executable.

AP-3
Each stage MUST have explicit inputs and outputs.

AP-4
No implicit shared state is allowed.

AP-5
All stages MUST be deterministic.

AP-6
Rebuilding from scratch MUST be possible.

AP-7
Offline execution MUST be supported.

AP-8
External services MUST NOT be required.

AP-9
Artifacts MUST be versioned.

AP-10
The architecture MUST favor simplicity over cleverness.


# ============================================================
# 2. HIGH-LEVEL OVERVIEW
# ============================================================

M31R is implemented as a linear, deterministic pipeline.

Canonical flow:

    Crawl
      ↓
    Filter
      ↓
    Dataset Build
      ↓
    Tokenizer Train
      ↓
    Tokenize + Shard
      ↓
    Model Train
      ↓
    Evaluate
      ↓
    Package
      ↓
    Serve

Each step produces artifacts.

Each step consumes artifacts from the previous step only.

No backward coupling.


# ============================================================
# 3. MAJOR SUBSYSTEMS
# ============================================================

The platform consists of nine major subsystems.

S1  Data Acquisition
S2  Data Filtering
S3  Dataset Builder
S4  Tokenizer System
S5  Tokenization & Sharding
S6  Training Engine
S7  Evaluation Engine
S8  Packaging & Release
S9  Inference Runtime

Each subsystem is isolated and has a defined contract.


# ============================================================
# 4. DATA FLOW MODEL
# ============================================================

The architecture follows strict unidirectional data flow.

Rules:

- No stage may modify upstream artifacts
- No stage may read future artifacts
- All outputs must be immutable
- All outputs must be checksummed

Flow type:

Write-once, read-many.

This guarantees:

- reproducibility
- debuggability
- auditability


# ============================================================
# 5. SUBSYSTEM S1 — DATA ACQUISITION
# ============================================================

Responsibility:

Collect raw Rust repositories.

Inputs:

- repository lists
- registry metadata

Outputs:

- raw source trees
- manifest.json

Responsibilities:

- clone repositories
- download crates
- verify integrity
- log provenance
- store commit hashes

Non-responsibilities:

- cleaning
- deduplication
- tokenization

Isolation rule:

S1 MUST NOT parse code.


# ============================================================
# 6. SUBSYSTEM S2 — DATA FILTERING
# ============================================================

Responsibility:

Remove low-quality and invalid content.

Inputs:

- raw repositories

Outputs:

- filtered source files
- rejection logs

Responsibilities:

- AST validation
- compile validation
- deduplication
- license filtering
- directory exclusion
- normalization

Non-responsibilities:

- tokenization
- reasoning injection

Isolation rule:

S2 MUST NOT modify semantics.


# ============================================================
# 7. SUBSYSTEM S3 — DATASET BUILDER
# ============================================================

Responsibility:

Construct versioned datasets.

Inputs:

- filtered files

Outputs:

- dataset manifests
- dataset versions

Responsibilities:

- deterministic ordering
- file hashing
- splitting into partitions
- metadata storage

Non-responsibilities:

- model training

Isolation rule:

S3 MUST only aggregate.


# ============================================================
# 8. SUBSYSTEM S4 — TOKENIZER SYSTEM
# ============================================================

Responsibility:

Create Rust-specialized tokenizer.

Inputs:

- dataset corpus

Outputs:

- tokenizer.model
- tokenizer.json

Responsibilities:

- train SentencePiece or equivalent
- optimize vocab size
- ensure round-trip encoding

Non-responsibilities:

- model training

Isolation rule:

Tokenizer training is independent of neural training.


# ============================================================
# 9. SUBSYSTEM S5 — TOKENIZATION & SHARDING
# ============================================================

Responsibility:

Convert text into training-ready token streams.

Inputs:

- dataset
- tokenizer

Outputs:

- token shards

Responsibilities:

- encode tokens
- pack sequences
- apply FIM transforms
- inject CoT reasoning
- shard by size

Non-responsibilities:

- neural operations

Isolation rule:

No GPU required.


# ============================================================
# 10. SUBSYSTEM S6 — TRAINING ENGINE
# ============================================================

Responsibility:

Train transformer model.

Inputs:

- token shards
- model config

Outputs:

- checkpoints
- final model
- training logs

Responsibilities:

- forward pass
- backpropagation
- optimizer steps
- checkpointing
- metric logging

Non-responsibilities:

- dataset modification

Isolation rule:

Training MUST be stateless beyond checkpoints.


# ============================================================
# 11. SUBSYSTEM S7 — EVALUATION ENGINE
# ============================================================

Responsibility:

Measure model quality.

Inputs:

- trained model
- benchmark sets

Outputs:

- metrics.json
- reports

Responsibilities:

- compile rate tests
- pass@k evaluation
- perplexity measurement
- regression comparison

Non-responsibilities:

- training

Isolation rule:

Evaluation must not alter model.


# ============================================================
# 12. SUBSYSTEM S8 — PACKAGING & RELEASE
# ============================================================

Responsibility:

Prepare artifacts for consumption.

Inputs:

- trained model
- tokenizer
- configs

Outputs:

- release bundle

Responsibilities:

- version tagging
- checksum creation
- compression
- documentation inclusion

Non-responsibilities:

- retraining

Isolation rule:

Packaging is pure transformation.


# ============================================================
# 13. SUBSYSTEM S9 — INFERENCE RUNTIME
# ============================================================

Responsibility:

Serve model locally.

Inputs:

- release bundle

Outputs:

- generated tokens

Responsibilities:

- load model
- run inference
- support completion
- support FIM
- support streaming

Non-responsibilities:

- training

Isolation rule:

Runtime must operate offline.


# ============================================================
# 14. ARTIFACT MODEL
# ============================================================

All artifacts must:

- be immutable
- be versioned
- include checksums
- include metadata

Artifact types:

- raw data
- filtered data
- tokenizer
- shards
- checkpoints
- final models
- logs
- metrics


# ============================================================
# 15. INTERFACE CONTRACTS
# ============================================================

Each subsystem communicates via files only.

No shared memory.

No direct function calls across boundaries.

Communication mechanism:

Filesystem + manifests.

Benefits:

- reproducibility
- decoupling
- easier debugging


# ============================================================
# 16. EXECUTION MODES
# ============================================================

Supported modes:

- full pipeline
- stage-only execution
- resume execution
- dry run

Each stage must be independently runnable.


# ============================================================
# 17. FAILURE MODEL
# ============================================================

Failures must:

- not corrupt artifacts
- not partially overwrite outputs
- be restartable

Strategy:

- atomic writes
- temp files
- checkpoint safety


# ============================================================
# 18. DETERMINISM MODEL
# ============================================================

Determinism requires:

- fixed seeds
- fixed ordering
- fixed configs
- pinned dependencies

Result:

Same inputs → identical outputs.


# ============================================================
# 19. SCALABILITY MODEL
# ============================================================

Scaling strategy:

Data stages:
    parallel CPU

Training:
    multi-GPU

Inference:
    single GPU or CPU quantized

Architecture must scale horizontally.


# ============================================================
# 20. SECURITY BOUNDARY
# ============================================================

System must:

- not execute untrusted code
- not fetch during inference
- operate offline
- sandbox subprocesses


# ============================================================
# 21. DEPLOYMENT MODEL
# ============================================================

Deployment targets:

- local workstation
- on-prem servers
- air-gapped systems

Cloud-only designs are forbidden.


# ============================================================
# 22. TRACEABILITY
# ============================================================

Every artifact must trace back to:

- dataset version
- config
- git commit
- timestamp

No anonymous artifacts allowed.


# ============================================================
# 23. LIFECYCLE SUMMARY
# ============================================================

Lifecycle:

collect → clean → tokenize → train → evaluate → package → serve

No hidden steps.


# ============================================================
# 24. ARCHITECTURAL ANTI-PATTERNS (FORBIDDEN)
# ============================================================

- monolithic scripts
- hidden global state
- dynamic data mutation
- notebook-only workflows
- cloud dependency
- mixing stages


# ============================================================
# 25. SUMMARY
# ============================================================

M31R uses a strict, modular, deterministic pipeline architecture.

Each subsystem:

- has one responsibility
- communicates through artifacts
- is independently runnable
- is reproducible

This structure guarantees:

- maintainability
- scalability
- reliability
- enterprise readiness

All future design decisions MUST conform to this architecture.

# END
# ============================================================
