# ============================================================
# M31R
# Requirements Specification
# File: 02_REQUIREMENTS_SPEC.md
# Version: 1.0.0
# Status: AUTHORITATIVE
# Owner: Eshan Roy
# Document Order: 02 / 25
# Depends On: 01_VISION_PRD.md
# ============================================================


# ============================================================
# 0. DOCUMENT PURPOSE
# ============================================================

This document formally defines all functional and non-functional
requirements for M31R.

This document answers:

"What MUST the system do?"

This document does NOT describe:

- architecture
- implementation
- design
- internal structure

Those are defined later.

Each requirement must be:

- atomic
- testable
- measurable
- deterministic
- unambiguous

If a requirement cannot be tested, it is invalid.


# ============================================================
# 1. REQUIREMENT TERMINOLOGY
# ============================================================

The following keywords are normative:

MUST
    Mandatory. No exceptions.

MUST NOT
    Forbidden.

SHOULD
    Strong recommendation.

MAY
    Optional.

Each requirement is uniquely identified:

Format:
REQ-[CATEGORY]-[NUMBER]

Example:
REQ-DATA-001


# ============================================================
# 2. SYSTEM BOUNDARY
# ============================================================

M31R consists of:

- dataset pipeline
- tokenizer
- model training
- evaluation
- inference runtime
- CLI tooling

M31R excludes:

- hosted SaaS
- external APIs
- proprietary model dependencies
- non-Rust features


# ============================================================
# 3. FUNCTIONAL REQUIREMENTS — DATA COLLECTION
# ============================================================

REQ-DATA-001
System MUST collect Rust repositories from public sources.

REQ-DATA-002
System MUST support cloning repositories via Git.

REQ-DATA-003
System MUST support crates.io package downloads.

REQ-DATA-004
System MUST store raw data locally.

REQ-DATA-005
System MUST maintain deterministic repository lists.

REQ-DATA-006
System MUST allow dataset rebuild from scratch.

REQ-DATA-007
System MUST support incremental crawling.

REQ-DATA-008
System MUST log every fetch event.

REQ-DATA-009
System MUST detect duplicate repositories.

REQ-DATA-010
System MUST handle network failures gracefully.

REQ-DATA-011
System MUST retry failed downloads.

REQ-DATA-012
System MUST verify repository integrity.

REQ-DATA-013
System MUST ignore non-Rust projects.

REQ-DATA-014
System MUST detect Rust via Cargo.toml presence.

REQ-DATA-015
System MUST enforce license filtering.

REQ-DATA-016
System MUST allow whitelist/blacklist configuration.

REQ-DATA-017
System MUST produce reproducible dataset manifests.

REQ-DATA-018
System MUST record commit hashes.

REQ-DATA-019
System MUST avoid binary blobs.

REQ-DATA-020
System MUST exclude generated files.


# ============================================================
# 4. FUNCTIONAL REQUIREMENTS — DATA FILTERING
# ============================================================

REQ-FILTER-001
System MUST parse Rust files using AST.

REQ-FILTER-002
System MUST reject syntactically invalid files.

REQ-FILTER-003
System MUST run rustc --emit=metadata validation.

REQ-FILTER-004
System MUST remove minified or obfuscated code.

REQ-FILTER-005
System MUST deduplicate identical files.

REQ-FILTER-006
System MUST deduplicate near-duplicates.

REQ-FILTER-007
System MUST remove vendor directories.

REQ-FILTER-008
System MUST remove target directories.

REQ-FILTER-009
System MUST remove lock files.

REQ-FILTER-010
System MUST enforce maximum file size.

REQ-FILTER-011
System MUST enforce maximum line count.

REQ-FILTER-012
System MUST normalize formatting.

REQ-FILTER-013
System MUST preserve comments optionally.

REQ-FILTER-014
System MUST produce filtered shards.

REQ-FILTER-015
System MUST version filtered datasets.

REQ-FILTER-016
System MUST track file provenance.

REQ-FILTER-017
System MUST log rejection reasons.

REQ-FILTER-018
System MUST operate offline after initial fetch.

REQ-FILTER-019
System MUST allow deterministic filtering.

REQ-FILTER-020
System MUST support parallel processing.


# ============================================================
# 5. FUNCTIONAL REQUIREMENTS — TOKENIZER
# ============================================================

REQ-TOK-001
System MUST train tokenizer from Rust corpus only.

REQ-TOK-002
System MUST support SentencePiece or equivalent.

REQ-TOK-003
System MUST allow configurable vocab size.

REQ-TOK-004
System MUST generate reproducible tokenizer.

REQ-TOK-005
System MUST store tokenizer artifacts.

REQ-TOK-006
System MUST avoid English-heavy tokens.

REQ-TOK-007
System MUST minimize token fragmentation.

REQ-TOK-008
System MUST support encoding/decoding.

REQ-TOK-009
System MUST validate round-trip accuracy.

REQ-TOK-010
System MUST provide deterministic training seed.


# ============================================================
# 6. FUNCTIONAL REQUIREMENTS — MODEL TRAINING
# ============================================================

REQ-TRAIN-001
System MUST train from random initialization.

REQ-TRAIN-002
System MUST NOT load external pretrained weights.

REQ-TRAIN-003
System MUST support configurable architecture.

REQ-TRAIN-004
System MUST support transformer decoder design.

REQ-TRAIN-005
System MUST support mixed precision.

REQ-TRAIN-006
System MUST support checkpointing.

REQ-TRAIN-007
System MUST resume from checkpoints.

REQ-TRAIN-008
System MUST support gradient accumulation.

REQ-TRAIN-009
System MUST support distributed training.

REQ-TRAIN-010
System MUST log metrics per step.

REQ-TRAIN-011
System MUST log loss values.

REQ-TRAIN-012
System MUST store training configs.

REQ-TRAIN-013
System MUST enforce deterministic seeds.

REQ-TRAIN-014
System MUST support FIM objective.

REQ-TRAIN-015
System MUST support CoT objective.

REQ-TRAIN-016
System MUST export final model.

REQ-TRAIN-017
System MUST validate model integrity.

REQ-TRAIN-018
System MUST prevent NaN divergence.

REQ-TRAIN-019
System MUST support interruption safety.

REQ-TRAIN-020
System MUST produce reproducible runs.


# ============================================================
# 7. FUNCTIONAL REQUIREMENTS — CHAIN OF THOUGHT
# ============================================================

REQ-COT-001
System MUST support reasoning token injection.

REQ-COT-002
System MUST support comment-based reasoning.

REQ-COT-003
System MUST support structured scratchpad blocks.

REQ-COT-004
System MUST allow hidden reasoning mode.

REQ-COT-005
System MUST mask reasoning tokens optionally.

REQ-COT-006
System MUST maintain Rust-only vocabulary.

REQ-COT-007
System MUST auto-generate reasoning heuristically.

REQ-COT-008
System MUST preserve code correctness.

REQ-COT-009
System MUST allow reasoning disable flag.

REQ-COT-010
System MUST track reasoning coverage.


# ============================================================
# 8. FUNCTIONAL REQUIREMENTS — INFERENCE
# ============================================================

REQ-INF-001
System MUST run locally.

REQ-INF-002
System MUST operate offline.

REQ-INF-003
System MUST load models without internet.

REQ-INF-004
System MUST provide CLI interface.

REQ-INF-005
System MUST support completion.

REQ-INF-006
System MUST support fill-in-middle.

REQ-INF-007
System MUST stream tokens.

REQ-INF-008
System MUST limit memory usage.

REQ-INF-009
System MUST allow quantization.

REQ-INF-010
System MUST enforce deterministic outputs with seed.


# ============================================================
# 9. NON-FUNCTIONAL REQUIREMENTS — PERFORMANCE
# ============================================================

REQ-PERF-001
Inference latency <= 50ms/token.

REQ-PERF-002
Model size <= 500M parameters.

REQ-PERF-003
VRAM usage <= 8GB.

REQ-PERF-004
Startup time <= 5 seconds.

REQ-PERF-005
Training throughput must scale linearly with GPUs.

REQ-PERF-006
Tokenizer speed >= 50k tokens/sec.

REQ-PERF-007
Dataset build must support parallelization.

REQ-PERF-008
Checkpoint save <= 30 seconds.

REQ-PERF-009
Resume time <= 60 seconds.

REQ-PERF-010
Memory leaks are forbidden.


# ============================================================
# 10. NON-FUNCTIONAL REQUIREMENTS — RELIABILITY
# ============================================================

REQ-REL-001
All stages MUST be restartable.

REQ-REL-002
Partial failures MUST not corrupt artifacts.

REQ-REL-003
Every stage MUST produce logs.

REQ-REL-004
Checksums MUST verify artifacts.

REQ-REL-005
Training MUST tolerate interruptions.

REQ-REL-006
Crashes MUST be recoverable.

REQ-REL-007
Data corruption MUST be detectable.

REQ-REL-008
Deterministic outputs MUST be reproducible.

REQ-REL-009
System MUST support backup.

REQ-REL-010
System MUST support restore.


# ============================================================
# 11. NON-FUNCTIONAL REQUIREMENTS — SECURITY
# ============================================================

REQ-SEC-001
System MUST avoid proprietary data.

REQ-SEC-002
System MUST filter PII.

REQ-SEC-003
System MUST restrict execution permissions.

REQ-SEC-004
System MUST avoid remote code execution risks.

REQ-SEC-005
Dependencies MUST be pinned.

REQ-SEC-006
Licenses MUST be audited.

REQ-SEC-007
Model artifacts MUST be verifiable.

REQ-SEC-008
No external telemetry by default.

REQ-SEC-009
Offline operation mandatory.

REQ-SEC-010
Supply chain risks minimized.


# ============================================================
# 12. ACCEPTANCE CRITERIA
# ============================================================

System is accepted when:

- all functional requirements pass
- all non-functional targets met
- deterministic rebuild confirmed
- compile success >= target
- local inference verified
- no external dependencies


# ============================================================
# 13. SUMMARY
# ============================================================

This document defines the complete contractual behavior of M31R.

Anything not required here is optional.

Anything required here is mandatory.

All subsequent documents must implement these requirements.

# END
# ============================================================
