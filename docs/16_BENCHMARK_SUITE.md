# ============================================================
# M31R
# Benchmark Suite Specification
# File: 16_BENCHMARK_SUITE.md
# Version: 1.0.0
# Status: AUTHORITATIVE
# Owner: Eshan Roy
# Document Order: 16 / 25
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
#   14_EVALUATION_METHODOLOGY.md
#   15_TESTING_STRATEGY.md
# ============================================================


# ============================================================
# 0. PURPOSE
# ============================================================

This document defines the official benchmark suite for M31R.

This document answers:

"What concrete tasks are used to measure real Rust capability?"

While Document 14 defines evaluation methodology and metrics,
this document defines the actual benchmark datasets and tasks.

In simple terms:

- Doc 14 = how to measure
- Doc 16 = what to measure on

This document is authoritative for:

- benchmark composition
- task types
- dataset structure
- execution format
- versioning rules

Any benchmark outside this specification is unofficial.


# ============================================================
# 1. BENCHMARK PHILOSOPHY
# ============================================================

Benchmarks must represent real developer work.

Not synthetic puzzles.
Not trivia.
Not academic tasks.

The benchmark must answer:

"If a Rust engineer used M31R daily, would it help?"

Therefore tasks must be:

- practical
- compilable
- testable
- deterministic
- representative of production Rust


# ============================================================
# 2. CORE PRINCIPLES
# ============================================================

BS-1
Benchmarks must be Rust-only.

BS-2
Benchmarks must compile.

BS-3
Benchmarks must have automated tests.

BS-4
Benchmarks must be deterministic.

BS-5
Benchmarks must be versioned.

BS-6
Benchmarks must not overlap training data.

BS-7
Benchmarks must be small but meaningful.

BS-8
Benchmarks must reflect real patterns.

BS-9
Benchmarks must be executable offline.

BS-10
Benchmarks must be reproducible.


# ============================================================
# 3. BENCHMARK GOALS
# ============================================================

The suite must measure:

G1  code completion
G2  function synthesis
G3  refactoring
G4  ownership correctness
G5  error handling
G6  type inference
G7  borrow checker reasoning
G8  multi-file reasoning

These map directly to real Rust tasks.


# ============================================================
# 4. BENCHMARK CATEGORIES
# ============================================================

Benchmarks are divided into categories:

C1 Completion
C2 Fill-in-Middle
C3 Function Implementation
C4 Bug Fixing
C5 Refactoring
C6 Type/Ownership Tasks
C7 Multi-File Projects
C8 Standard Library Usage

Each category must exist.


# ============================================================
# 5. CATEGORY C1 — COMPLETION
# ============================================================

Definition:

Predict next tokens or lines.

Examples:

- finish expression
- complete iterator chain
- finish match arms

Purpose:

IDE-style assistance quality.


# ============================================================
# 6. CATEGORY C2 — FILL-IN-MIDDLE
# ============================================================

Definition:

Complete missing middle block.

Format:

prefix + suffix → predict middle

Examples:

- missing function body
- missing logic block

Purpose:

editing and refactoring realism.


# ============================================================
# 7. CATEGORY C3 — FUNCTION IMPLEMENTATION
# ============================================================

Definition:

Implement function from signature and doc.

Example:

fn gcd(a: u64, b: u64) -> u64

Expected:

correct algorithm

Purpose:

algorithmic correctness.


# ============================================================
# 8. CATEGORY C4 — BUG FIXING
# ============================================================

Definition:

Given buggy code, produce fix.

Examples:

- borrow checker errors
- lifetime mistakes
- off-by-one errors

Purpose:

debugging usefulness.


# ============================================================
# 9. CATEGORY C5 — REFACTORING
# ============================================================

Definition:

Transform code while preserving behavior.

Examples:

- replace loops with iterators
- split large function
- add error propagation

Purpose:

structured reasoning ability.


# ============================================================
# 10. CATEGORY C6 — OWNERSHIP TASKS
# ============================================================

Definition:

Tasks that specifically stress borrow checker.

Examples:

- move vs borrow
- lifetimes
- Arc/Mutex usage
- mutable aliasing

Purpose:

Rust-specific reasoning.


# ============================================================
# 11. CATEGORY C7 — MULTI-FILE PROJECTS
# ============================================================

Definition:

Small crate with multiple modules.

Tasks:

- add feature
- implement trait
- modify behavior

Purpose:

cross-file reasoning.


# ============================================================
# 12. CATEGORY C8 — STDLIB USAGE
# ============================================================

Definition:

Use of common Rust patterns.

Examples:

- iterators
- collections
- Result/Option
- async

Purpose:

idiomatic code quality.


# ============================================================
# 13. TASK STRUCTURE
# ============================================================

Each task must contain:

- id
- category
- prompt
- context files
- expected behavior
- compile command
- test command

All machine-readable.


# ============================================================
# 14. DIRECTORY STRUCTURE
# ============================================================

benchmarks/
│
├─ completion/
├─ fim/
├─ functions/
├─ bugs/
├─ refactor/
├─ ownership/
├─ projects/
└─ stdlib/

Each folder contains tasks.


# ============================================================
# 15. TASK FILE FORMAT
# ============================================================

Each task folder:

task_id/
├─ prompt.rs
├─ solution.rs
├─ tests.rs
├─ Cargo.toml
└─ metadata.yaml

Standardized layout only.


# ============================================================
# 16. PROMPT RULES
# ============================================================

Prompt must:

- be incomplete
- specify boundaries
- be minimal

Must not:

- contain solution hints
- include training data fragments


# ============================================================
# 17. SOLUTION RULES
# ============================================================

Solution must:

- compile
- pass tests
- be idiomatic Rust

Used only for evaluation, never for training.


# ============================================================
# 18. TEST RULES
# ============================================================

Tests must:

- be deterministic
- cover behavior
- avoid randomness
- avoid time dependencies

Compile + tests define correctness.


# ============================================================
# 19. METADATA FORMAT
# ============================================================

metadata.yaml must include:

- difficulty
- category
- expected runtime
- tags

Used for filtering and reporting.


# ============================================================
# 20. BENCHMARK SIZE GUIDELINES
# ============================================================

Target counts:

Completion: 200+
FIM: 200+
Functions: 300+
Bug fixes: 150+
Ownership: 150+
Projects: 50+

Balance quality over quantity.


# ============================================================
# 21. DATASET ISOLATION
# ============================================================

Benchmarks must:

- never appear in training data
- be excluded explicitly

Overlap invalidates evaluation.


# ============================================================
# 22. VERSIONING POLICY
# ============================================================

Each benchmark suite has version:

benchmarks_vX

Changing tasks requires new version.

Old versions preserved for comparability.


# ============================================================
# 23. EXECUTION PIPELINE
# ============================================================

Evaluation must:

1. load model
2. iterate tasks
3. generate output
4. write file
5. cargo build
6. cargo test
7. record result

Fully automated.


# ============================================================
# 24. SCORING RULES
# ============================================================

Binary scoring:

pass = 1
fail = 0

No partial credit.

Either works or not.


# ============================================================
# 25. TIME LIMITS
# ============================================================

Each task must:

- compile < 10s
- test < 10s

Prevents slow evaluation.


# ============================================================
# 26. REPORTING FORMAT
# ============================================================

Evaluation output:

metrics.json

Contains:

- compile_rate
- pass@k
- category breakdown
- latency
- memory


# ============================================================
# 27. REGRESSION POLICY
# ============================================================

If new model underperforms baseline:

> 5% drop

Release blocked.

Must investigate.


# ============================================================
# 28. FORBIDDEN PRACTICES
# ============================================================

Not allowed:

- manual judging
- subjective grading
- hidden tests
- non-compilable tasks
- overlapping training data
- random outcomes


# ============================================================
# 29. MAINTENANCE POLICY
# ============================================================

Benchmarks must:

- be reviewed quarterly
- remove obsolete tasks
- add modern patterns

But changes require version bump.


# ============================================================
# 30. SUMMARY
# ============================================================

The benchmark suite defines real Rust work:

- completion
- implementation
- fixing
- refactoring
- ownership reasoning

Success is defined by:

compiles + passes tests

Nothing else matters.

These benchmarks are the objective measure of M31R quality.

# END
# ============================================================
