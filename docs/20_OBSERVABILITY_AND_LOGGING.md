# ============================================================
# M31R
# Observability and Logging Specification
# File: 20_OBSERVABILITY_AND_LOGGING.md
# Version: 1.0.0
# Status: AUTHORITATIVE
# Owner: Eshan Roy
# Document Order: 20 / 25
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
#   16_BENCHMARK_SUITE.md
#   17_SERVING_ARCHITECTURE.md
#   18_RELEASE_PROCESS.md
#   19_SECURITY_AND_SAFETY.md
# ============================================================


# ============================================================
# 0. PURPOSE
# ============================================================

This document defines the observability and logging architecture for M31R.

This document answers:

"How do we understand what the system is doing at runtime and diagnose failures quickly?"

Observability covers:

- logs
- metrics
- traces
- experiment records
- runtime telemetry (local only)

This document is authoritative for:

- what must be logged
- how it must be logged
- formats
- storage
- retention
- privacy constraints

If behavior is not observable, it is not maintainable.


# ============================================================
# 1. OBSERVABILITY PHILOSOPHY
# ============================================================

M31R is:

- deterministic
- offline
- local

Therefore observability must be:

- local-first
- structured
- reproducible
- machine-parseable

Not:

- cloud dashboards
- remote telemetry
- SaaS monitoring

Principle:

Logs are the source of truth.


# ============================================================
# 2. GOALS
# ============================================================

OG-1  Diagnose failures quickly
OG-2  Enable reproducibility
OG-3  Provide experiment traceability
OG-4  Track performance regressions
OG-5  Support automation
OG-6  Avoid leaking sensitive data


# ============================================================
# 3. NON-GOALS
# ============================================================

The system will NOT:

- send data externally
- depend on cloud logging
- use third-party telemetry SDKs
- collect user data automatically


# ============================================================
# 4. CORE PRINCIPLES
# ============================================================

OL-1  Structured logs only
OL-2  Deterministic outputs
OL-3  Minimal noise
OL-4  No secrets
OL-5  Local storage
OL-6  Machine-readable
OL-7  Explicit log levels
OL-8  Immutable experiment logs
OL-9  Low overhead
OL-10 No silent failures


# ============================================================
# 5. OBSERVABILITY COMPONENTS
# ============================================================

M31R observability consists of:

C1 Logging
C2 Metrics
C3 Run metadata
C4 Experiment artifacts
C5 Performance tracking

These together form complete traceability.


# ============================================================
# 6. LOGGING MODEL
# ============================================================

All subsystems must emit logs.

Logging must be:

- structured (JSON or key-value)
- timestamped
- leveled
- consistent format

Human-only text logs are forbidden.


# ============================================================
# 7. LOG LEVELS
# ============================================================

Supported levels:

DEBUG
INFO
WARNING
ERROR
CRITICAL

Usage:

DEBUG   → dev diagnostics
INFO    → normal progress
WARNING → recoverable issues
ERROR   → failures
CRITICAL→ abort


# ============================================================
# 8. LOG FORMAT
# ============================================================

Mandatory fields:

- timestamp
- level
- module
- message
- context (optional structured data)

Example:

{
  "ts": "...",
  "level": "INFO",
  "module": "trainer",
  "msg": "step_complete",
  "step": 1024,
  "loss": 1.23
}


# ============================================================
# 9. LOG DESTINATIONS
# ============================================================

Logs must go to:

- stdout
- file (logs/ or experiments/<run>/)

No external sinks.

Local-only storage.


# ============================================================
# 10. LOG ROTATION
# ============================================================

Must:

- rotate large logs
- avoid unlimited growth

Policies:

- size-based rotation
- timestamped files


# ============================================================
# 11. PRINT STATEMENTS
# ============================================================

Forbidden:

print()

All output must use logger.

Reason:

prints are unstructured and non-parseable.


# ============================================================
# 12. DETERMINISM REQUIREMENT
# ============================================================

Logs must not include:

- random IDs
- nondeterministic ordering

Except timestamps.

Ensures reproducible analysis.


# ============================================================
# 13. TRAINING LOGGING
# ============================================================

Training must log:

- step
- loss
- learning rate
- tokens/sec
- gradient norm
- memory usage
- checkpoint events

Frequency:

every N steps


# ============================================================
# 14. DATA PIPELINE LOGGING
# ============================================================

Dataset stages must log:

- files processed
- files rejected
- duplicates removed
- filter reasons
- shard counts

Purpose:

data auditability.


# ============================================================
# 15. TOKENIZER LOGGING
# ============================================================

Must log:

- vocab size
- coverage
- fragmentation metrics
- training duration


# ============================================================
# 16. EVALUATION LOGGING
# ============================================================

Must log:

- compile rate
- pass@k
- latency
- memory
- per-category scores

Must save metrics.json.


# ============================================================
# 17. SERVING LOGGING
# ============================================================

Must log:

- request count
- latency per request
- tokens generated
- memory usage
- errors

Must not log prompts by default.


# ============================================================
# 18. PRIVACY RULES
# ============================================================

Logs must not include:

- user prompts
- personal data
- file contents
- secrets

Only metadata allowed.

Prompt logging must be opt-in.


# ============================================================
# 19. METRICS MODEL
# ============================================================

Metrics are numeric measurements.

Stored separately from logs.

Format:

metrics.json or CSV.

Machine readable.


# ============================================================
# 20. MANDATORY METRICS
# ============================================================

Must track:

- loss
- compile rate
- pass@k
- tokens/sec
- latency
- memory

These support performance tracking.


# ============================================================
# 21. EXPERIMENT METADATA
# ============================================================

Each run must record:

- config snapshot
- git commit hash
- dataset version
- tokenizer version
- seed
- timestamp

Stored with run artifacts.


# ============================================================
# 22. RUN DIRECTORY STRUCTURE
# ============================================================

experiments/<run_id>/
├─ train.log
├─ metrics.json
├─ config_snapshot.yaml
├─ metadata.json
└─ checkpoints/

Everything required to reproduce run.


# ============================================================
# 23. TRACEABILITY
# ============================================================

Each artifact must trace to:

config → dataset → code commit → seed → metrics

No anonymous runs allowed.


# ============================================================
# 24. PERFORMANCE TRACKING
# ============================================================

Must track:

- training speed trends
- inference latency trends
- memory trends

Allows regression detection.


# ============================================================
# 25. ERROR REPORTING
# ============================================================

Errors must:

- include stack trace
- include context
- not be swallowed

Silent failure forbidden.


# ============================================================
# 26. DEBUG MODE
# ============================================================

Debug mode may:

- increase verbosity
- add extra metrics

Must not change behavior.


# ============================================================
# 27. RETENTION POLICY
# ============================================================

Logs:

may be cleaned

Experiments:

must be preserved until manually deleted

Releases:

permanent


# ============================================================
# 28. FORBIDDEN PRACTICES
# ============================================================

Not allowed:

- unstructured logs
- print debugging
- remote telemetry
- secret logging
- interactive-only metrics
- missing metadata
- deleting failure logs


# ============================================================
# 29. ACCEPTANCE CRITERIA
# ============================================================

System considered observable when:

- every stage logs progress
- failures include context
- runs are reproducible
- metrics saved automatically
- no sensitive data leaked

Otherwise observability is insufficient.


# ============================================================
# 30. SUMMARY
# ============================================================

M31R observability is:

- local
- structured
- deterministic
- privacy-safe
- automation-friendly

Logs + metrics + metadata provide full system insight.

If something goes wrong, the logs must explain why.

Observability is not optional.
It is a core engineering requirement.

# END
# ============================================================
