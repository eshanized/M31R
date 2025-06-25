# ============================================================
# M31R
# Training Architecture Specification
# File: 07_TRAINING_ARCHITECTURE.md
# Version: 1.0.0
# Status: AUTHORITATIVE
# Owner: Eshan Roy
# Document Order: 07 / 25
# Depends On:
#   01_VISION_PRD.md
#   02_REQUIREMENTS_SPEC.md
#   03_GLOSSARY_AND_DEFINITIONS.md
#   04_SYSTEM_ARCHITECTURE.md
#   05_DATA_ARCHITECTURE.md
#   06_MODEL_ARCHITECTURE.md
# ============================================================


# ============================================================
# 0. PURPOSE
# ============================================================

This document defines the complete training system architecture for M31R.

This document answers:

"How is the model trained reliably, reproducibly, and efficiently?"

This document specifies:

- training pipeline structure
- execution model
- distributed strategy
- checkpointing
- determinism
- logging
- failure handling
- resource usage

This document does NOT define:

- model topology
- dataset rules

Those are defined earlier.

If training behavior conflicts with this document, this document prevails.


# ============================================================
# 1. TRAINING PHILOSOPHY
# ============================================================

Training must be:

- deterministic
- reproducible
- restartable
- resource efficient
- debuggable

Training must NOT be:

- notebook-driven
- ad-hoc
- interactive-only
- stateful
- dependent on external services

Training is treated as an engineering system, not research code.


# ============================================================
# 2. CORE PRINCIPLES
# ============================================================

TA-1
All runs must be reproducible.

TA-2
All configs must be declarative.

TA-3
All artifacts must be versioned.

TA-4
All randomness must be seeded.

TA-5
Training must be resumable.

TA-6
No hidden side effects allowed.

TA-7
Failure must not corrupt state.

TA-8
Training must run offline.

TA-9
Hardware must be optional, not assumed.

TA-10
Simple designs are preferred.


# ============================================================
# 3. TRAINING LIFECYCLE
# ============================================================

Canonical lifecycle:

Load Config
    ↓
Load Token Shards
    ↓
Initialize Model
    ↓
Initialize Optimizer
    ↓
Train Loop
    ↓
Checkpoint
    ↓
Evaluate
    ↓
Export Final Model

Each stage must be independent and restartable.


# ============================================================
# 4. TRAINING ENTRYPOINT
# ============================================================

Training must be launched only through:

CLI command.

Example:

m31r train --config configs/train.yaml

Rules:

- no notebooks
- no hidden scripts
- no manual intervention


# ============================================================
# 5. CONFIGURATION MODEL
# ============================================================

All parameters must come from config files.

Forbidden:

- hardcoded hyperparameters
- magic constants in code

Config categories:

- model
- optimizer
- data
- hardware
- logging

Config must be snapshot into each run.


# ============================================================
# 6. TRAINING MODES
# ============================================================

Supported modes:

Mode A — Single GPU
Mode B — Multi GPU
Mode C — CPU debug
Mode D — Resume

All modes must share identical logic.


# ============================================================
# 7. DATA LOADING ARCHITECTURE
# ============================================================

Requirements:

- streaming shards
- memory efficient
- prefetch enabled
- deterministic ordering with seed

Data must never be fully loaded into memory.

Loader must support:

- sequential
- shuffled
- distributed partitions


# ============================================================
# 8. BATCHING STRATEGY
# ============================================================

Batching rules:

- fixed token count per batch
- dynamic padding avoided
- packed sequences preferred

Goal:

maximize GPU utilization.


# ============================================================
# 9. TRAINING LOOP STRUCTURE
# ============================================================

Each iteration:

1. forward pass
2. compute loss
3. backward pass
4. gradient clipping
5. optimizer step
6. zero gradients
7. log metrics

Order must not change.


# ============================================================
# 10. OPTIMIZER
# ============================================================

Default:

AdamW

Requirements:

- weight decay
- bias correction
- stable defaults

Experimental optimizers are not default.


# ============================================================
# 11. LEARNING RATE SCHEDULE
# ============================================================

Default:

- warmup
- cosine decay

Configurable.

Reason:

stable convergence.


# ============================================================
# 12. GRADIENT ACCUMULATION
# ============================================================

Required to:

- support small GPUs
- simulate larger batches

Behavior:

accumulate steps before optimizer update

Must be deterministic.


# ============================================================
# 13. MIXED PRECISION
# ============================================================

Training precision:

bf16 preferred
fp16 allowed

Benefits:

- speed
- lower memory

Must use:

loss scaling to prevent underflow.


# ============================================================
# 14. DISTRIBUTED TRAINING
# ============================================================

Strategy:

Data Parallel

Preferred methods:

- FSDP
- DeepSpeed
- DDP

Rules:

- identical model replicas
- shard data only
- no model parallel complexity by default


# ============================================================
# 15. CHECKPOINTING
# ============================================================

Checkpoint must contain:

- model weights
- optimizer state
- scheduler state
- global step
- random seed state
- config snapshot

Without all fields, resume is invalid.


# ============================================================
# 16. CHECKPOINT FREQUENCY
# ============================================================

Default:

every N steps or minutes

Must support:

- periodic
- manual
- final

Checkpoint writes must be atomic.


# ============================================================
# 17. RESUME BEHAVIOR
# ============================================================

Resume must:

- restore exact state
- continue deterministically
- not repeat or skip data

Resume must produce identical results as uninterrupted run.


# ============================================================
# 18. LOGGING
# ============================================================

Logs must include:

- step
- loss
- learning rate
- tokens/sec
- GPU memory
- gradient norm

Logs must be structured (JSON or CSV).


# ============================================================
# 19. METRICS STORAGE
# ============================================================

Each run must produce:

metrics.json
config_snapshot.yaml
train.log

Stored under:

experiments/<run_id>/


# ============================================================
# 20. EXPERIMENT TRACKING
# ============================================================

Each run must have:

unique ID

Run ID format:

timestamp + hash

No overwriting allowed.


# ============================================================
# 21. FAILURE HANDLING
# ============================================================

Training must:

- handle OOM gracefully
- handle interruption
- allow resume

Crashes must not corrupt checkpoints.


# ============================================================
# 22. DETERMINISM REQUIREMENTS
# ============================================================

Must fix:

- random seed
- data order
- CUDA determinism flags
- config

Identical inputs must produce identical weights.


# ============================================================
# 23. MEMORY MANAGEMENT
# ============================================================

Must:

- avoid memory leaks
- clear caches
- reuse buffers
- monitor VRAM

Memory spikes must be logged.


# ============================================================
# 24. TRAINING OBJECTIVES INTEGRATION
# ============================================================

Loss components:

- next token
- FIM
- CoT

Weights configurable.

Objectives must be computed in single forward pass.


# ============================================================
# 25. EVALUATION DURING TRAINING
# ============================================================

Periodic evaluation required.

Metrics:

- perplexity
- compile rate
- pass@k

Evaluation must not block training excessively.


# ============================================================
# 26. EXPORT PROCESS
# ============================================================

Final export must include:

- best checkpoint
- tokenizer
- config
- metadata
- checksums

Export must be immutable.


# ============================================================
# 27. SECURITY RULES
# ============================================================

Training must not:

- execute downloaded code
- use internet
- expose secrets

All inputs treated as untrusted.


# ============================================================
# 28. RESOURCE TARGETS
# ============================================================

Target hardware:

1–2 consumer GPUs

System must not require clusters.


# ============================================================
# 29. FORBIDDEN PRACTICES
# ============================================================

Not allowed:

- manual checkpoint edits
- hidden randomness
- notebook-only training
- mutable datasets
- mixing experiments
- external pretrained weights


# ============================================================
# 30. SUMMARY
# ============================================================

The M31R training architecture is:

- deterministic
- restartable
- offline
- resource efficient
- config-driven
- enterprise safe

Training is treated as a reproducible build process.

All implementations must conform to these rules.

# END
# ============================================================
