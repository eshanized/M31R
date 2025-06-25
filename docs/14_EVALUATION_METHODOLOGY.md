# ============================================================
# M31R
# Evaluation Methodology Specification
# File: 14_EVALUATION_METHODOLOGY.md
# Version: 1.0.0
# Status: AUTHORITATIVE
# Owner: Eshan Roy
# Document Order: 14 / 25
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
# ============================================================


# ============================================================
# 0. PURPOSE
# ============================================================

This document defines the official evaluation methodology for M31R.

This document answers:

"How do we objectively measure whether the model is good?"

Evaluation must be:

- deterministic
- reproducible
- automated
- domain-relevant
- resistant to gaming

This document is authoritative for:

- metrics
- benchmarks
- procedures
- reporting
- acceptance criteria

If evaluation is ambiguous, results are meaningless.

If evaluation is not automated, it is invalid.


# ============================================================
# 1. EVALUATION PHILOSOPHY
# ============================================================

M31R is a Rust code model.

Therefore evaluation must measure:

- correctness of code
- compilation success
- functional behavior

NOT:

- English fluency
- BLEU scores
- human-like text
- chat quality

Principle:

Compilation is truth.

If it compiles and passes tests, it works.


# ============================================================
# 2. CORE PRINCIPLES
# ============================================================

EM-1
Metrics must be objective.

EM-2
Metrics must be reproducible.

EM-3
Evaluation must be automated.

EM-4
Benchmarks must be versioned.

EM-5
No manual scoring.

EM-6
No cherry-picking.

EM-7
Same seed → same results.

EM-8
Offline execution only.

EM-9
Fast feedback cycles preferred.

EM-10
Domain-specific metrics dominate generic ones.


# ============================================================
# 3. EVALUATION SCOPE
# ============================================================

Evaluation covers:

- pretraining quality
- code generation
- completion
- reasoning correctness
- runtime performance

Evaluation excludes:

- chat alignment
- personality
- conversational behavior


# ============================================================
# 4. METRIC TIERS
# ============================================================

Metrics are divided into:

Tier 1 — Primary (must meet targets)
Tier 2 — Secondary (optimization signals)
Tier 3 — Diagnostic (debugging only)

Only Tier 1 determines acceptance.


# ============================================================
# 5. PRIMARY METRICS (TIER 1)
# ============================================================

M1  Compile Success Rate
M2  Pass@K
M3  Inference Latency
M4  Memory Usage
M5  Deterministic Reproducibility

These metrics define product viability.


# ============================================================
# 6. METRIC: COMPILE SUCCESS RATE
# ============================================================

Definition:

Percentage of generated outputs that compile without errors.

Procedure:

1. generate code
2. run rustc
3. count successes

Formula:

successes / total

Rationale:

Rust compilation enforces correctness constraints.

This is the most important metric.


# ============================================================
# 7. TARGET: COMPILE SUCCESS
# ============================================================

Minimum target:

>= 70%

Stretch target:

>= 80%

Below 60% is unacceptable.


# ============================================================
# 8. METRIC: PASS@K
# ============================================================

Definition:

Probability at least one of K attempts is correct.

Procedure:

1. generate K samples
2. test each
3. success if any passes

Rationale:

Measures usefulness in IDE scenarios.


# ============================================================
# 9. TARGET: PASS@K
# ============================================================

pass@1 >= 40%
pass@5 >= 60%
pass@10 >= 70%

Targets adjustable by dataset size.


# ============================================================
# 10. METRIC: LATENCY
# ============================================================

Definition:

Time per generated token.

Measurement:

average milliseconds/token

Rationale:

User experience and practicality.


# ============================================================
# 11. TARGET: LATENCY
# ============================================================

<= 50 ms/token

Measured on consumer GPU.

Exceeding 100 ms unacceptable.


# ============================================================
# 12. METRIC: MEMORY USAGE
# ============================================================

Definition:

Peak VRAM during inference.

Target:

<= 8GB

Ensures local usability.


# ============================================================
# 13. METRIC: REPRODUCIBILITY
# ============================================================

Definition:

Identical outputs with identical seeds.

Procedure:

repeat evaluation twice

Outputs must match exactly.


# ============================================================
# 14. SECONDARY METRICS (TIER 2)
# ============================================================

S1  Perplexity
S2  Token efficiency
S3  Throughput
S4  Loss curves
S5  Reasoning overhead

Used for optimization only.


# ============================================================
# 15. METRIC: PERPLEXITY
# ============================================================

Definition:

Language modeling uncertainty.

Use:

training progress indicator only

Not product metric.


# ============================================================
# 16. METRIC: TOKEN EFFICIENCY
# ============================================================

Definition:

Average tokens per Rust line.

Lower is better.

Indicates tokenizer quality.


# ============================================================
# 17. METRIC: THROUGHPUT
# ============================================================

Definition:

tokens/sec during training.

Used to monitor infrastructure efficiency.


# ============================================================
# 18. METRIC: REASONING OVERHEAD
# ============================================================

Definition:

percentage of tokens consumed by CoT.

Target:

<= 20%

Prevents context waste.


# ============================================================
# 19. DIAGNOSTIC METRICS (TIER 3)
# ============================================================

Examples:

- gradient norm
- attention entropy
- loss variance

Used for debugging only.


# ============================================================
# 20. BENCHMARK TYPES
# ============================================================

Benchmarks must include:

B1 function completion
B2 bug fixing
B3 refactoring
B4 type inference
B5 borrow checker sensitive code

These represent real usage.


# ============================================================
# 21. BENCHMARK DATASET RULES
# ============================================================

Benchmarks must:

- be separate from training data
- be versioned
- be immutable
- be deterministic

No overlap with training.


# ============================================================
# 22. BENCHMARK FORMAT
# ============================================================

Each task includes:

- prompt
- expected behavior
- compile command
- tests

Machine executable only.


# ============================================================
# 23. EVALUATION PIPELINE
# ============================================================

Steps:

1. load model
2. load benchmark
3. generate outputs
4. compile
5. run tests
6. record metrics
7. save report

Fully automated.


# ============================================================
# 24. EVALUATION FREQUENCY
# ============================================================

During training:

- periodic lightweight eval

After training:

- full benchmark suite

Before release:

- mandatory full evaluation


# ============================================================
# 25. RESULT STORAGE
# ============================================================

Each run must produce:

metrics.json
report.txt
config_snapshot.yaml

Stored under experiments/<run_id>/eval/


# ============================================================
# 26. COMPARISON POLICY
# ============================================================

Every model must be compared to:

- previous version
- baseline model

Regression > 5% requires investigation.


# ============================================================
# 27. AUTOMATION REQUIREMENT
# ============================================================

Evaluation must run via:

m31r eval

Manual testing is invalid.


# ============================================================
# 28. FORBIDDEN PRACTICES
# ============================================================

Not allowed:

- manual scoring
- cherry-picked examples
- undocumented metrics
- overlapping training/test data
- human-only judgment
- changing benchmarks post-hoc


# ============================================================
# 29. ACCEPTANCE CRITERIA
# ============================================================

Model accepted only if:

- compile rate meets target
- pass@k meets target
- latency within bounds
- memory within bounds
- reproducible

Failure in any metric blocks release.


# ============================================================
# 30. SUMMARY
# ============================================================

Evaluation in M31R is:

- automated
- objective
- compile-centric
- reproducible
- domain-specific

The only question that matters:

"Does the generated Rust code compile and work?"

If yes → model succeeds.

If no → model fails.

All development must optimize for these metrics.

# END
# ============================================================
