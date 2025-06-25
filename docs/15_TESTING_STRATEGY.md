# ============================================================
# M31R
# Testing Strategy Specification
# File: 15_TESTING_STRATEGY.md
# Version: 1.0.0
# Status: AUTHORITATIVE
# Owner: Eshan Roy
# Document Order: 15 / 25
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
# ============================================================


# ============================================================
# 0. PURPOSE
# ============================================================

This document defines the complete testing strategy for M31R.

This document answers:

"How do we ensure the system behaves correctly and safely over time?"

Testing is not optional.

Testing is the primary mechanism that guarantees:

- correctness
- determinism
- reliability
- regression prevention
- long-term maintainability

All changes must be validated through automated tests.

Manual testing is insufficient and not accepted.


# ============================================================
# 1. TESTING PHILOSOPHY
# ============================================================

M31R is infrastructure software.

Infrastructure must not break silently.

Therefore:

- every component must be testable
- every bug must be reproducible
- every regression must be detectable

Principle:

If it is not tested, it is broken by default.


# ============================================================
# 2. CORE PRINCIPLES
# ============================================================

TS-1
All behavior must be testable.

TS-2
All tests must be automated.

TS-3
Tests must be deterministic.

TS-4
Tests must be fast.

TS-5
Tests must isolate concerns.

TS-6
Tests must not depend on internet.

TS-7
Tests must not modify global state.

TS-8
Tests must be reproducible.

TS-9
Tests must run in CI.

TS-10
Failing tests block merges.


# ============================================================
# 3. TEST LAYERS
# ============================================================

Testing is divided into five layers:

Layer 1  Unit Tests
Layer 2  Integration Tests
Layer 3  Pipeline Tests
Layer 4  Regression Tests
Layer 5  Performance Tests

Each layer serves a distinct purpose.


# ============================================================
# 4. TEST EXECUTION POLICY
# ============================================================

All tests must run via:

pytest
or
make test

No manual steps allowed.

Tests must pass locally and in CI.

CI is authoritative.


# ============================================================
# 5. DIRECTORY STRUCTURE
# ============================================================

All tests must live under:

tests/

Structure must mirror source:

m31r/dataset/   → tests/dataset/
m31r/model/     → tests/model/
m31r/trainer/   → tests/trainer/

Ensures discoverability.


# ============================================================
# 6. LAYER 1 — UNIT TESTS
# ============================================================

Definition:

Small tests validating individual functions.

Properties:

- fast (< 100ms each)
- isolated
- no external dependencies

Purpose:

validate logic correctness.


# ============================================================
# 7. UNIT TEST REQUIREMENTS
# ============================================================

Must:

- test edge cases
- test failure modes
- test boundary values

Must not:

- access network
- depend on GPU
- depend on filesystem outside temp


# ============================================================
# 8. EXAMPLES OF UNIT TEST TARGETS
# ============================================================

Examples:

- tokenizer encode/decode
- AST filtering
- hash functions
- config validation
- shard packing logic

Pure logic only.


# ============================================================
# 9. LAYER 2 — INTEGRATION TESTS
# ============================================================

Definition:

Validate cooperation between modules.

Examples:

- dataset → tokenizer
- tokenizer → trainer
- trainer → checkpoint

Purpose:

ensure subsystems connect correctly.


# ============================================================
# 10. INTEGRATION TEST REQUIREMENTS
# ============================================================

Must:

- use small fixtures
- run quickly (< 5 seconds)
- simulate realistic behavior

Must not:

- train full model
- use large datasets


# ============================================================
# 11. LAYER 3 — PIPELINE TESTS
# ============================================================

Definition:

Validate entire pipeline execution.

Examples:

crawl → filter → dataset → tokenize → train (tiny)

Purpose:

verify orchestration correctness.


# ============================================================
# 12. PIPELINE TEST REQUIREMENTS
# ============================================================

Must:

- use toy dataset
- complete quickly
- verify artifacts exist

Must:

- detect broken stage boundaries


# ============================================================
# 13. LAYER 4 — REGRESSION TESTS
# ============================================================

Definition:

Prevent reintroduction of known bugs.

Each bug must:

- have test
- remain permanently

Policy:

No bug fix without regression test.


# ============================================================
# 14. LAYER 5 — PERFORMANCE TESTS
# ============================================================

Definition:

Track speed and memory behavior.

Examples:

- tokens/sec
- latency
- VRAM usage

Purpose:

detect slowdowns.


# ============================================================
# 15. PERFORMANCE TEST POLICY
# ============================================================

Must:

- be separate from unit tests
- run optionally
- produce metrics

Must not:

- block quick development cycles


# ============================================================
# 16. DETERMINISM TESTING
# ============================================================

System must produce identical outputs with same seed.

Tests must:

- run twice
- compare hashes

Mismatch is failure.


# ============================================================
# 17. DATASET TESTING
# ============================================================

Dataset tests must verify:

- filtering correctness
- deduplication accuracy
- license filtering
- manifest integrity

No corrupted data allowed.


# ============================================================
# 18. TOKENIZER TESTING
# ============================================================

Must verify:

- encode/decode roundtrip
- deterministic training
- vocabulary size
- fragmentation metrics


# ============================================================
# 19. MODEL TESTING
# ============================================================

Must verify:

- forward pass shape
- no NaNs
- gradient stability
- parameter count matches config


# ============================================================
# 20. TRAINER TESTING
# ============================================================

Must verify:

- loss decreases
- checkpoint save/load
- resume correctness
- deterministic behavior


# ============================================================
# 21. COT TESTING
# ============================================================

Must verify:

- reasoning injection occurs
- syntax remains valid
- injection deterministic
- masking correct


# ============================================================
# 22. CLI TESTING
# ============================================================

Must verify:

- commands execute
- exit codes correct
- help text exists
- outputs created

CLI is public interface.


# ============================================================
# 23. FIXTURE POLICY
# ============================================================

Fixtures must be:

- small
- local
- deterministic

Large or random fixtures forbidden.


# ============================================================
# 24. MOCKING POLICY
# ============================================================

Allowed:

- filesystem mocks
- small stubs

Forbidden:

- mocking core logic

Tests must reflect reality.


# ============================================================
# 25. TEST SPEED TARGETS
# ============================================================

Unit suite:

< 30 seconds

Full suite:

< 5 minutes

Slow tests discourage usage.


# ============================================================
# 26. COVERAGE TARGETS
# ============================================================

Minimum:

80% overall

Core modules:

> 90%

Coverage used as guidance, not sole metric.


# ============================================================
# 27. CI REQUIREMENTS
# ============================================================

CI must:

- run tests
- fail fast
- report coverage
- block merge on failure

Green CI required.


# ============================================================
# 28. FORBIDDEN PRACTICES
# ============================================================

Not allowed:

- manual tests only
- flaky tests
- network dependency
- time-dependent tests
- skipping failing tests
- silent exceptions


# ============================================================
# 29. TEST FAILURE POLICY
# ============================================================

If test fails:

- fix immediately
- do not ignore
- do not disable

Tests represent system contract.


# ============================================================
# 30. SUMMARY
# ============================================================

Testing in M31R is:

- automated
- deterministic
- layered
- comprehensive
- mandatory

Every component must be covered.

Testing is the safety net that protects long-term reliability.

No code enters production without tests.

# END
# ============================================================
