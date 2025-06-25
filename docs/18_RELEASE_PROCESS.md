# ============================================================
# M31R
# Release Process Specification
# File: 18_RELEASE_PROCESS.md
# Version: 1.0.0
# Status: AUTHORITATIVE
# Owner: Eshan Roy
# Document Order: 18 / 25
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
# ============================================================


# ============================================================
# 0. PURPOSE
# ============================================================

This document defines the official release process for M31R.

This document answers:

"How does code move from development to a stable, trusted release?"

A release is not:

- a random checkpoint
- an experiment
- a manual zip file

A release is:

- reproducible
- tested
- benchmarked
- versioned
- auditable
- immutable

This document defines the only valid process to produce releases.


# ============================================================
# 1. RELEASE PHILOSOPHY
# ============================================================

Releases must prioritize:

- stability
- determinism
- trustworthiness
- reproducibility

Over:

- speed
- convenience
- experimentation

Principle:

If it cannot be rebuilt from scratch, it is not releasable.


# ============================================================
# 2. CORE PRINCIPLES
# ============================================================

RP-1
Every release must be reproducible.

RP-2
Every release must be tested.

RP-3
Every release must be benchmarked.

RP-4
Every release must be versioned.

RP-5
Every release must be traceable.

RP-6
No manual artifact editing.

RP-7
No hidden changes.

RP-8
Release must be immutable.

RP-9
Automation preferred.

RP-10
Failures block release.


# ============================================================
# 3. RELEASE ARTIFACT DEFINITION
# ============================================================

A valid release must include:

- trained model weights
- tokenizer
- config snapshot
- metadata
- checksums
- documentation reference

Anything missing invalidates the release.


# ============================================================
# 4. RELEASE TYPES
# ============================================================

Supported:

Type A — Model release
Type B — Tooling release
Type C — Patch release
Type D — Experimental (non-stable)

Only Type A is considered production.


# ============================================================
# 5. VERSIONING SCHEME
# ============================================================

Semantic versioning required:

MAJOR.MINOR.PATCH

MAJOR:
breaking changes

MINOR:
new features

PATCH:
bug fixes

Example:

1.2.0


# ============================================================
# 6. PRE-RELEASE CHECKLIST
# ============================================================

Before release:

- tests pass
- benchmarks pass
- metrics meet targets
- config frozen
- dataset version frozen
- tokenizer version frozen
- no TODOs

If any fail → release blocked.


# ============================================================
# 7. TRAINING FREEZE
# ============================================================

Before release:

- stop experiments
- select best checkpoint
- freeze training config

No hyperparameter tweaks allowed post-freeze.


# ============================================================
# 8. DATASET FREEZE
# ============================================================

Dataset version must be:

- immutable
- hashed
- recorded

Releasing with mutable dataset is forbidden.


# ============================================================
# 9. TOKENIZER FREEZE
# ============================================================

Tokenizer must:

- be versioned
- not change post-freeze

Changing tokenizer invalidates compatibility.


# ============================================================
# 10. REPRODUCIBILITY CHECK
# ============================================================

Must verify:

- rebuild model from scratch
- identical hashes produced

If hashes differ → release invalid.


# ============================================================
# 11. TESTING REQUIREMENTS
# ============================================================

Must run:

- unit tests
- integration tests
- pipeline tests
- regression tests

100% pass required.


# ============================================================
# 12. BENCHMARK REQUIREMENTS
# ============================================================

Must run full benchmark suite.

Metrics must meet:

- compile rate target
- pass@k target
- latency target
- memory target

Regression > 5% blocks release.


# ============================================================
# 13. ARTIFACT PACKAGING
# ============================================================

Release directory structure:

release/<version>/
├─ model.safetensors
├─ tokenizer.json
├─ config.yaml
├─ metadata.json
├─ checksum.txt
└─ README.txt

Standardized only.


# ============================================================
# 14. METADATA REQUIREMENTS
# ============================================================

metadata.json must include:

- version
- dataset version
- tokenizer version
- config hash
- git commit hash
- training seed
- date
- metrics summary

Traceability mandatory.


# ============================================================
# 15. CHECKSUM POLICY
# ============================================================

All artifacts must include:

SHA256 checksums

Verification required on load.

Corrupted artifacts rejected.


# ============================================================
# 16. TAGGING POLICY
# ============================================================

Each release must:

- create git tag
- include changelog
- reference commit

Tags immutable.


# ============================================================
# 17. CHANGELOG POLICY
# ============================================================

Must include:

- new features
- bug fixes
- metric changes
- breaking changes

No vague descriptions.


# ============================================================
# 18. AUTOMATION
# ============================================================

Release must be created via:

m31r export

Manual packaging forbidden.

Automation ensures consistency.


# ============================================================
# 19. VALIDATION STEP
# ============================================================

After packaging:

- load model
- run smoke test
- generate sample
- verify compile

Basic sanity required.


# ============================================================
# 20. DISTRIBUTION
# ============================================================

Allowed:

- GitHub releases
- internal storage
- offline transfer

Forbidden:

- auto cloud dependency
- hidden hosting


# ============================================================
# 21. IMMUTABILITY
# ============================================================

Once released:

Artifacts must never change.

If change required:

new version only.


# ============================================================
# 22. PATCH POLICY
# ============================================================

Patch allowed only for:

- critical bug
- packaging mistake

No retraining allowed in patch.


# ============================================================
# 23. ROLLBACK POLICY
# ============================================================

Must support:

- revert to previous version

All previous releases must remain accessible.


# ============================================================
# 24. EXPERIMENT VS RELEASE
# ============================================================

Experiments:

- temporary
- untrusted
- not distributed

Releases:

- stable
- trusted
- versioned

Never confuse the two.


# ============================================================
# 25. SECURITY RULES
# ============================================================

Must:

- verify checksums
- avoid embedded secrets
- avoid remote calls

Artifacts must be self-contained.


# ============================================================
# 26. DOCUMENTATION REQUIREMENT
# ============================================================

Each release must reference:

- docs version
- architecture compatibility

Users must know exact behavior.


# ============================================================
# 27. FORBIDDEN PRACTICES
# ============================================================

Not allowed:

- manual zip files
- editing checkpoints
- post-release modification
- untagged releases
- skipping benchmarks
- skipping tests


# ============================================================
# 28. ACCEPTANCE CRITERIA
# ============================================================

Release accepted only if:

- reproducible
- tested
- benchmarked
- versioned
- packaged
- validated

Otherwise rejected.


# ============================================================
# 29. RELEASE LIFECYCLE SUMMARY
# ============================================================

Flow:

train → evaluate → freeze → test → benchmark → package → tag → publish

No shortcuts.


# ============================================================
# 30. SUMMARY
# ============================================================

M31R releases are:

- deterministic
- reproducible
- tested
- benchmarked
- immutable
- traceable

A release is a verified artifact, not a checkpoint.

Trust is built through discipline.

All releases must strictly follow this process.

# END
# ============================================================
