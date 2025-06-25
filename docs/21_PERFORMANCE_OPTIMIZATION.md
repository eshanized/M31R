# ============================================================
# M31R
# Performance Optimization Specification
# File: 21_PERFORMANCE_OPTIMIZATION.md
# Version: 1.0.0
# Status: AUTHORITATIVE
# Owner: Eshan Roy
# Document Order: 21 / 25
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
#   12_CLI_AND_TOOLING_SPEC.md
#   13_CODING_STANDARDS.md
#   14_EVALUATION_METHODLOGY.md
#   15_TESTING_STRATEGY.md
#   16_BENCHMARK_SUITE.md
#   17_SERVING_ARCHITECTURE.md
#   18_RELEASE_PROCESS.md
#   19_SECURITY_AND_SAFETY.md
#   20_OBSERVABILITY_AND_LOGGING.md
# ============================================================


# ============================================================
# 0. PURPOSE
# ============================================================

This document defines the official performance optimization strategy for M31R.

This document answers:

"How do we make the system fast and memory-efficient without sacrificing correctness or determinism?"

Performance applies to:

- dataset processing
- tokenization
- training
- evaluation
- inference
- CLI tooling

This document specifies:

- targets
- measurement methods
- allowed optimizations
- forbidden optimizations

This document is authoritative for all performance work.


# ============================================================
# 1. PERFORMANCE PHILOSOPHY
# ============================================================

Performance must be:

- measured
- reproducible
- justified

Not:

- speculative
- premature
- clever hacks

Principle:

Correctness first, then speed.

Never trade determinism or clarity for marginal speed.


# ============================================================
# 2. OPTIMIZATION PRIORITIES
# ============================================================

Priority order:

P1  Correctness
P2  Determinism
P3  Simplicity
P4  Memory efficiency
P5  Latency
P6  Throughput

Lower priorities must never compromise higher ones.


# ============================================================
# 3. PERFORMANCE TARGETS
# ============================================================

Training:

>= 20k tokens/sec (single GPU baseline)

Inference:

<= 50 ms/token

Memory:

<= 8GB VRAM

Startup:

<= 5 seconds

Dataset processing:

>= 100k lines/sec CPU

Targets guide decisions.


# ============================================================
# 4. MEASUREMENT FIRST POLICY
# ============================================================

No optimization without:

- baseline measurement
- profiling
- evidence of bottleneck

Guessing is forbidden.

Use metrics from Doc 20.


# ============================================================
# 5. BENCHMARKING TOOLS
# ============================================================

Use:

- built-in timers
- profiler
- memory tracking
- tokens/sec counters

Results must be reproducible.


# ============================================================
# 6. DETERMINISM REQUIREMENT
# ============================================================

Optimizations must not:

- change outputs
- alter randomness
- change seeds
- introduce race conditions

Same inputs must produce identical outputs.


# ============================================================
# 7. DATA PIPELINE OPTIMIZATION
# ============================================================

Goals:

- maximize CPU throughput
- minimize memory usage
- stream data

Strategies:

- streaming I/O
- batching
- multiprocessing
- memory-mapped files


# ============================================================
# 8. DATA PIPELINE RULES
# ============================================================

Must:

- avoid loading full dataset in RAM
- process line-by-line or chunked
- use iterators

Forbidden:

- full dataset in memory
- temporary huge buffers


# ============================================================
# 9. TOKENIZATION OPTIMIZATION
# ============================================================

Goals:

- fast encoding
- minimal allocations

Strategies:

- batch encode
- reuse buffers
- pre-allocated arrays
- compiled tokenizer libraries


# ============================================================
# 10. SHARDING OPTIMIZATION
# ============================================================

Must:

- write sequentially
- avoid small writes
- use large block sizes

Purpose:

reduce disk overhead.


# ============================================================
# 11. TRAINING OPTIMIZATION
# ============================================================

Goals:

- maximize GPU utilization
- minimize idle time

Strategies:

- large effective batch
- gradient accumulation
- prefetch data
- mixed precision


# ============================================================
# 12. MIXED PRECISION
# ============================================================

Allowed:

- bf16
- fp16

Benefits:

- lower memory
- higher throughput

Must:

- preserve stability


# ============================================================
# 13. FLASH ATTENTION
# ============================================================

Recommended:

FlashAttention or equivalent

Benefits:

- lower memory
- faster attention

Allowed as default optimization.


# ============================================================
# 14. CHECKPOINT OPTIMIZATION
# ============================================================

Must:

- save asynchronously if possible
- compress efficiently
- avoid blocking training

Checkpointing must not stall GPU.


# ============================================================
# 15. EVALUATION OPTIMIZATION
# ============================================================

Must:

- parallelize tasks
- cache compilation artifacts
- reuse builds

Avoid recompiling unchanged code.


# ============================================================
# 16. INFERENCE OPTIMIZATION
# ============================================================

Goals:

- low latency
- low memory
- responsive UX

Strategies:

- KV cache reuse
- quantization
- streaming output
- greedy decoding default


# ============================================================
# 17. QUANTIZATION POLICY
# ============================================================

Allowed:

- int8
- int4

Benefits:

- smaller memory
- faster inference

Must:

- preserve accuracy targets


# ============================================================
# 18. MEMORY MANAGEMENT
# ============================================================

Must:

- reuse buffers
- free unused tensors
- clear caches

Avoid:

- memory leaks
- fragmentation


# ============================================================
# 19. CPU/GPU TRANSFERS
# ============================================================

Minimize:

- host-device copies
- blocking syncs

Batch transfers when possible.


# ============================================================
# 20. CONCURRENCY STRATEGY
# ============================================================

Allowed:

- multiprocessing for CPU tasks
- async I/O
- small worker pools

Forbidden:

- complex distributed serving
- nondeterministic thread races


# ============================================================
# 21. CLI PERFORMANCE
# ============================================================

CLI must:

- start quickly
- lazy-load heavy modules
- avoid unnecessary initialization

Fast feedback is essential.


# ============================================================
# 22. DISK I/O OPTIMIZATION
# ============================================================

Must:

- sequential reads/writes
- avoid tiny files
- avoid frequent fsync

Batch operations preferred.


# ============================================================
# 23. CACHING POLICY
# ============================================================

Allowed:

- tokenizer cache
- compilation cache
- dataset manifest cache

Must be:

- deterministic
- invalidated correctly


# ============================================================
# 24. MICRO-OPTIMIZATION RULE
# ============================================================

Do not:

- micro-optimize prematurely
- sacrifice clarity

Only optimize hotspots proven by profiling.


# ============================================================
# 25. CODE CLARITY RULE
# ============================================================

If optimization reduces readability:

must include:

- comments
- justification
- benchmark numbers

Otherwise reject.


# ============================================================
# 26. FORBIDDEN OPTIMIZATIONS
# ============================================================

Not allowed:

- nondeterministic parallelism
- hidden caching
- unsafe memory hacks
- removing checks for speed
- skipping validation
- silent precision loss


# ============================================================
# 27. REGRESSION DETECTION
# ============================================================

Performance regressions must be detected via:

- benchmarks
- metrics comparison
- automated CI checks

Regression > 10% requires investigation.


# ============================================================
# 28. HARDWARE ASSUMPTIONS
# ============================================================

Target hardware:

- consumer GPU
- 8–16GB VRAM
- 8–16 CPU cores

Do not optimize exclusively for clusters.


# ============================================================
# 29. ACCEPTANCE CRITERIA
# ============================================================

Optimization accepted only if:

- measurable improvement
- deterministic
- passes tests
- meets targets
- code remains maintainable


# ============================================================
# 30. SUMMARY
# ============================================================

Performance in M31R is:

- measured
- deterministic
- practical
- hardware-conscious

We optimize:

only what matters
only when proven
without sacrificing correctness

Fast, small, predictable.

That is the goal.

# END
# ============================================================
