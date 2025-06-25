# ============================================================
# M31R
# Development Workflow Specification
# File: 10_DEVELOPMENT_WORKFLOW.md
# Version: 1.0.0
# Status: AUTHORITATIVE
# Owner: Eshan Roy
# Document Order: 10 / 25
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
# ============================================================


# ============================================================
# 0. PURPOSE
# ============================================================

This document defines the official development workflow for M31R.

This document answers:

"How should engineers build, modify, test, and ship changes safely?"

This workflow exists to ensure:

- deterministic builds
- consistent behavior
- minimal regressions
- clean history
- reproducibility
- enterprise-grade reliability

This workflow is mandatory for all contributors.

Ad-hoc development practices are forbidden.


# ============================================================
# 1. DEVELOPMENT PHILOSOPHY
# ============================================================

M31R is infrastructure software, not a research notebook.

Therefore development must be:

- disciplined
- repeatable
- automated
- reviewable
- deterministic

We prioritize:

correctness > speed of hacks

Temporary shortcuts create permanent technical debt.


# ============================================================
# 2. CORE PRINCIPLES
# ============================================================

DW-1
All work must be version controlled.

DW-2
All work must be reproducible.

DW-3
All changes must be reviewed.

DW-4
All behavior must be testable.

DW-5
No hidden state.

DW-6
No local-only hacks.

DW-7
No manual steps.

DW-8
Automation preferred over memory.

DW-9
Configs over hardcoding.

DW-10
Documentation first.


# ============================================================
# 3. CONTRIBUTOR ROLES
# ============================================================

Owner
    Final authority for decisions.

Maintainer
    Reviews and merges changes.

Contributor
    Proposes and implements changes.

Reviewer
    Validates correctness and safety.

All roles must follow the same workflow.


# ============================================================
# 4. ENVIRONMENT SETUP
# ============================================================

Every contributor must use a clean environment.

Requirements:

- Linux or macOS preferred
- Python 3.11+
- CUDA compatible GPU (optional)
- Git
- Make

No system-wide dependencies allowed.

All dependencies must be installed locally or in virtual environment.


# ============================================================
# 5. LOCAL SETUP PROCEDURE
# ============================================================

Canonical steps:

1. clone repository
2. create virtual environment
3. install dependencies
4. verify tests pass

Example:

git clone <repo>
cd M31R
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest

If tests fail, development must stop.


# ============================================================
# 6. DEPENDENCY MANAGEMENT
# ============================================================

Rules:

- all dependencies pinned
- no floating versions
- lock file required

Forbidden:

- implicit installs
- global packages
- manual patching

Reason:

ensures determinism.


# ============================================================
# 7. CODE STYLE RULES
# ============================================================

Must follow:

- PEP8
- type hints mandatory
- small functions
- explicit naming

Tools:

- formatter
- linter
- type checker

Style is enforced automatically.


# ============================================================
# 8. BRANCHING STRATEGY
# ============================================================

Main branches:

main
    stable

dev
    integration

Feature branches:

feature/<name>

Bugfix branches:

fix/<name>

Direct commits to main are forbidden.


# ============================================================
# 9. CHANGE PROCESS
# ============================================================

For every change:

1. create branch
2. implement change
3. add/update tests
4. update docs if needed
5. run checks
6. open pull request
7. review
8. merge

Skipping steps is not allowed.


# ============================================================
# 10. COMMIT RULES
# ============================================================

Commits must be:

- atomic
- descriptive
- small

Commit message format:

type(scope): description

Examples:

feat(dataset): add AST filter
fix(trainer): prevent NaN divergence

Avoid vague messages.


# ============================================================
# 11. TESTING REQUIREMENTS
# ============================================================

Every change must include:

- unit tests
- integration tests (if applicable)

No change without tests is accepted.

Target:

> 90% coverage for core modules.


# ============================================================
# 12. TEST TYPES
# ============================================================

Unit tests
    small, fast, deterministic

Integration tests
    subsystem behavior

End-to-end tests
    full pipeline sanity

Benchmarks
    performance tracking

All tests must be automated.


# ============================================================
# 13. TEST EXECUTION
# ============================================================

Commands:

pytest
make test

Must pass locally before PR.

CI must also pass.

Failing tests block merges.


# ============================================================
# 14. STATIC ANALYSIS
# ============================================================

Required checks:

- linter
- type checker
- formatter

Must run automatically in CI.

No warnings allowed.


# ============================================================
# 15. DOCUMENTATION RULES
# ============================================================

Rules:

- update docs with behavior changes
- maintain numbering
- avoid duplication

If behavior changes but docs are not updated, PR is rejected.


# ============================================================
# 16. CONFIGURATION RULES
# ============================================================

All tunables must be:

- config-based
- not hardcoded

Changing behavior must not require code edits.


# ============================================================
# 17. EXPERIMENT WORKFLOW
# ============================================================

Experiments must:

- use config files
- produce artifacts in experiments/
- not modify library code

Quick hacks inside library are forbidden.


# ============================================================
# 18. REPRODUCIBILITY CHECK
# ============================================================

Every training run must:

- snapshot config
- log seed
- log dataset version
- log commit hash

Without these, results are invalid.


# ============================================================
# 19. LOGGING POLICY
# ============================================================

Logs must be:

- structured
- consistent
- timestamped

Print statements are forbidden.

Use logger only.


# ============================================================
# 20. DEBUGGING POLICY
# ============================================================

Debugging must use:

- logs
- assertions
- tests

Ad-hoc interactive edits are forbidden.

Temporary debug code must not be committed.


# ============================================================
# 21. CODE REVIEW PROCESS
# ============================================================

Every PR must:

- have reviewer
- explain rationale
- pass CI
- be documented

Reviewer checks:

- correctness
- style
- tests
- docs
- architecture compliance


# ============================================================
# 22. MERGE RULES
# ============================================================

Allowed:

- squash merge

Forbidden:

- force push to main
- rewriting public history

History must remain auditable.


# ============================================================
# 23. RELEASE WORKFLOW
# ============================================================

Steps:

1. tag version
2. freeze changes
3. run full tests
4. build artifacts
5. publish release

Manual steps must be scripted.


# ============================================================
# 24. CI REQUIREMENTS
# ============================================================

CI must automatically:

- run tests
- run lint
- run type checks
- verify formatting

Failures block merge.


# ============================================================
# 25. SECURITY PRACTICES
# ============================================================

Must:

- avoid secrets in repo
- pin dependencies
- audit licenses

Must not:

- commit credentials
- fetch unknown code


# ============================================================
# 26. ARTIFACT RULES
# ============================================================

Never commit:

- large datasets
- checkpoints
- logs

Use gitignore.


# ============================================================
# 27. FORBIDDEN PRACTICES
# ============================================================

Not allowed:

- notebooks as source
- hidden configs
- manual steps
- training inside tests
- undocumented hacks
- skipping review


# ============================================================
# 28. ONBOARDING FLOW
# ============================================================

New contributor must:

1. read docs 01–10
2. setup env
3. run tests
4. run small pipeline
5. make small PR

Ensures familiarity.


# ============================================================
# 29. DEVELOPMENT GUARANTEES
# ============================================================

Following this workflow guarantees:

- consistent builds
- predictable behavior
- fewer regressions
- faster onboarding
- enterprise reliability


# ============================================================
# 30. SUMMARY
# ============================================================

M31R development is:

- config-driven
- test-driven
- review-driven
- automation-first
- deterministic

No shortcuts.

Every change must be reproducible, reviewable, and documented.

This is mandatory for maintaining long-term system integrity.

# END
# ============================================================
