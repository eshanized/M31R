# ============================================================
# M31R
# Appendix and Reference Specification
# File: 25_APPENDIX_AND_REFERENCE.md
# Version: 1.0.0
# Status: AUTHORITATIVE
# Owner: Eshan Roy
# Document Order: 25 / 25
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
#   20_OBSERVABILITY_AND_LOGGING.md
#   21_PERFORMANCE_OPTIMIZATION.md
#   22_MAINTENANCE_AND_SUPPORT.md
#   23_RISK_MANAGEMENT.md
#   24_SCALABILITY_AND_FUTURE_ROADMAP.md
# ============================================================


# ============================================================
# 0. PURPOSE
# ============================================================

This document serves as the final appendix and reference manual for M31R.

This document answers:

"Where do we define shared terminology, canonical conventions, and quick references that every contributor must align with?"

This file consolidates:

- terminology
- abbreviations
- standard commands
- directory maps
- configuration cheat sheets
- operational checklists
- decision rules
- design constraints

It acts as the single lookup reference for engineers and automation systems.

This document contains no new policy.
It clarifies and centralizes existing ones.


# ============================================================
# 1. HOW TO USE THIS DOCUMENT
# ============================================================

Use this file when you need:

- quick lookup
- standard conventions
- operational commands
- terminology clarification
- onboarding overview

Do not treat this as narrative documentation.

It is a structured reference.


# ============================================================
# 2. TERMINOLOGY
# ============================================================

Artifact
    Any produced file: dataset shard, tokenizer, checkpoint, or release.

Checkpoint
    Intermediate training state.

Release
    Immutable, tested, versioned artifact bundle.

Run
    Single execution of training/evaluation.

Shard
    Chunk of dataset stored independently.

Deterministic
    Same inputs → identical outputs.

Offline-first
    No runtime network dependency.

Local-first
    Runs on a single machine without cloud.


# ============================================================
# 3. ABBREVIATIONS
# ============================================================

CLI   Command Line Interface
CoT   Chain of Thought
FIM   Fill In Middle
KV    Key/Value cache
OOM   Out Of Memory
LTS   Long Term Support
CI    Continuous Integration
PR    Pull Request


# ============================================================
# 4. AUTHORITATIVE DOCUMENT MAP
# ============================================================

01  Vision / PRD
02  Requirements
03  Glossary
04  Architecture
05  Data Architecture
06  Model Architecture
07  Training Architecture
08  CoT Design
09  Repository Structure
10  Development Workflow
11  Configuration
12  CLI & Tooling
13  Coding Standards
14  Evaluation
15  Testing
16  Benchmarks
17  Serving
18  Release
19  Security
20  Observability
21  Performance
22  Maintenance
23  Risk
24  Scalability
25  Appendix (this file)


# ============================================================
# 5. STANDARD PROJECT LAYOUT (CANONICAL)
# ============================================================

m31r/
├─ configs/
├─ data/
├─ datasets/
├─ benchmarks/
├─ experiments/
├─ release/
├─ tests/
├─ m31r/ (source)
└─ docs/

No deviations without justification.


# ============================================================
# 6. CONFIG FILE QUICK REFERENCE
# ============================================================

global.yaml      → shared settings
dataset.yaml     → data pipeline
tokenizer.yaml   → tokenizer
model.yaml       → architecture
train.yaml       → training
eval.yaml        → evaluation
runtime.yaml     → serving

Every run must reference configs.


# ============================================================
# 7. CLI QUICK REFERENCE
# ============================================================

m31r crawl
m31r filter
m31r dataset
m31r tokenizer train
m31r train
m31r resume
m31r eval
m31r benchmark
m31r generate
m31r serve
m31r export
m31r verify
m31r clean
m31r info

All operations must go through CLI.


# ============================================================
# 8. COMMON WORKFLOWS
# ============================================================

Dataset build:

crawl → filter → dataset

Training:

tokenizer → train → eval

Release:

train → benchmark → export

Serving:

serve → generate


# ============================================================
# 9. RUN DIRECTORY TEMPLATE
# ============================================================

experiments/<run_id>/
├─ train.log
├─ metrics.json
├─ config_snapshot.yaml
├─ metadata.json
└─ checkpoints/

This layout is mandatory.


# ============================================================
# 10. RELEASE DIRECTORY TEMPLATE
# ============================================================

release/<version>/
├─ model.safetensors
├─ tokenizer.json
├─ config.yaml
├─ metadata.json
├─ checksum.txt
└─ README.txt

Immutable after creation.


# ============================================================
# 11. DEFAULT TARGET HARDWARE
# ============================================================

CPU: 8–16 cores
RAM: 16–32 GB
GPU: 8–16 GB VRAM

All features must work here.

Do not assume clusters.


# ============================================================
# 12. DEFAULT PERFORMANCE TARGETS
# ============================================================

Training: ≥ 20k tokens/sec
Inference: ≤ 50 ms/token
Memory: ≤ 8GB VRAM
Startup: ≤ 5s

Use these for sanity checks.


# ============================================================
# 13. ACCEPTABLE LICENSES (EXAMPLES)
# ============================================================

MIT
Apache-2.0
BSD variants

Non-permissive licenses rejected.


# ============================================================
# 14. MANDATORY INVARIANTS
# ============================================================

Always true:

- offline capable
- deterministic
- reproducible builds
- immutable releases
- no hidden behavior
- no runtime downloads

Violation = architectural defect.


# ============================================================
# 15. DESIGN DECISION RULES
# ============================================================

When unsure, prefer:

simpler > complex
deterministic > dynamic
local > cloud
explicit > implicit
maintainable > clever
tested > assumed

These rules override preferences.


# ============================================================
# 16. ERROR CLASSIFICATION
# ============================================================

User error
Config error
Validation error
Runtime error
System error

Each must be reported clearly.


# ============================================================
# 17. LOGGING CHEAT SHEET
# ============================================================

Levels:

DEBUG
INFO
WARNING
ERROR
CRITICAL

Never use print().

Always structured logs.


# ============================================================
# 18. TESTING CHEAT SHEET
# ============================================================

Unit → Integration → Pipeline → Regression → Performance

Every bug → regression test.

CI must stay green.


# ============================================================
# 19. SECURITY CHEAT SHEET
# ============================================================

Never:

- execute untrusted code
- embed secrets
- auto-download dependencies
- expose network by default

Always:

- verify hashes
- pin versions
- run offline


# ============================================================
# 20. RELEASE CHECKLIST (SUMMARY)
# ============================================================

Before release:

✓ tests pass
✓ benchmarks pass
✓ configs frozen
✓ reproducible build
✓ artifacts packaged
✓ checksums verified
✓ tag created

Otherwise block.


# ============================================================
# 21. MAINTENANCE CHECKLIST
# ============================================================

Monthly:

✓ tests
✓ benchmarks
✓ dependency audit
✓ log review

Quarterly:

✓ dataset review
✓ benchmark refresh


# ============================================================
# 22. CONTRIBUTOR QUICK RULES
# ============================================================

- write tests first
- no hidden behavior
- small PRs
- document changes
- avoid premature optimization
- follow coding standards
- use config-driven behavior


# ============================================================
# 23. COMMON ANTI-PATTERNS
# ============================================================

Avoid:

- giant scripts
- global mutable state
- interactive-only tools
- manual releases
- dynamic configs
- cloud lock-in
- nondeterministic pipelines


# ============================================================
# 24. GLOSSARY (RUST-SPECIFIC)
# ============================================================

Borrow checker
    Rust ownership enforcement mechanism.

Crate
    Rust package.

Cargo
    Rust build system.

Pass@k
    probability at least one of k generations passes tests.

Compile success rate
    fraction of generations compiling successfully.


# ============================================================
# 25. FUTURE EXTENSION NOTES
# ============================================================

Future additions must:

- follow all invariants
- preserve determinism
- avoid new dependencies
- integrate with CLI
- include tests + docs

Otherwise rejected.


# ============================================================
# 26. FAQ (SHORT)
# ============================================================

Q: Can we add cloud serving?
A: No. Offline-first is invariant.

Q: Can we auto-download models?
A: No. Artifacts must be explicit.

Q: Can we skip tests for speed?
A: No. Tests are mandatory.

Q: Can we modify a release?
A: Never. Create new version.


# ============================================================
# 27. META RULE
# ============================================================

If two solutions exist:

Choose the one easier to:

- explain
- test
- reproduce
- maintain

Complexity compounds risk.


# ============================================================
# 28. DOCUMENT MAINTENANCE RULE
# ============================================================

Whenever behavior changes:

Update documentation first or together.

Docs are part of the product.


# ============================================================
# 29. PROJECT VALUES
# ============================================================

M31R values:

- engineering discipline
- clarity
- reproducibility
- local ownership
- long-term stability

Not:

- hype
- unnecessary scale
- novelty for its own sake


# ============================================================
# 30. FINAL SUMMARY
# ============================================================

This appendix consolidates the entire system into a single reference.

Remember:

M31R succeeds because it is:

- deterministic
- local
- reproducible
- simple
- well-tested
- well-documented

If a decision threatens these properties,
the decision is wrong.

This concludes the complete M31R documentation set.

# END OF DOCUMENTATION
# ============================================================
