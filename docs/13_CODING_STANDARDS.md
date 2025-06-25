# ============================================================
# M31R
# Coding Standards Specification
# File: 13_CODING_STANDARDS.md
# Version: 1.0.0
# Status: AUTHORITATIVE
# Owner: Eshan Roy
# Document Order: 13 / 25
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
# ============================================================


# ============================================================
# 0. PURPOSE
# ============================================================

This document defines the coding standards for the M31R codebase.

This document answers:

"How must source code be written to ensure clarity, reliability,
 maintainability, and determinism?"

This specification is mandatory.

All code must conform to these rules.

Code that violates this specification must not be merged.


# ============================================================
# 1. DESIGN PHILOSOPHY
# ============================================================

M31R is infrastructure software.

Infrastructure code must be:

- predictable
- explicit
- testable
- boring
- easy to reason about

Cleverness is discouraged.

Readability and correctness dominate micro-optimizations.

Principle:

Future maintainers must understand code instantly.


# ============================================================
# 2. LANGUAGE POLICY
# ============================================================

Primary language:

Python 3.11+

Optional:

Rust (performance-critical only)

Forbidden:

- notebooks as production code
- mixed languages without justification

Python is the default for all orchestration logic.


# ============================================================
# 3. CORE PRINCIPLES
# ============================================================

CD-1
Explicit is better than implicit.

CD-2
Small functions over large ones.

CD-3
Pure functions preferred.

CD-4
Side effects minimized.

CD-5
Determinism required.

CD-6
Types everywhere.

CD-7
Tests required.

CD-8
Logging over print.

CD-9
Config over constants.

CD-10
Simplicity over cleverness.


# ============================================================
# 4. STYLE BASELINE
# ============================================================

Must follow:

- PEP8
- PEP484 type hints
- Black formatting
- Ruff or flake8 linting

Style violations block CI.


# ============================================================
# 5. FILE SIZE RULES
# ============================================================

Recommended:

< 500 lines per file

Hard limit:

1000 lines

If exceeded:

split module

Reason:

smaller files are easier to reason about.


# ============================================================
# 6. FUNCTION SIZE RULES
# ============================================================

Recommended:

< 50 lines per function

Hard limit:

100 lines

Long functions must be refactored.

Single responsibility only.


# ============================================================
# 7. NAMING CONVENTIONS
# ============================================================

Use:

snake_case for variables and functions
PascalCase for classes
UPPER_CASE for constants

Names must be descriptive.

Forbidden:

a, b, tmp, data2, foo, bar


# ============================================================
# 8. TYPE HINTING
# ============================================================

All functions must include type hints.

Example:

def train(cfg: TrainConfig) -> None:

Forbidden:

untyped public functions

Reason:

improves correctness and tooling.


# ============================================================
# 9. DOCSTRINGS
# ============================================================

All public functions must include docstrings.

Docstrings must state:

- purpose
- inputs
- outputs
- side effects

Short and precise.

No essays.


# ============================================================
# 10. IMPORT RULES
# ============================================================

Use:

absolute imports only

Forbidden:

relative imports
wildcard imports

Reason:

predictability and clarity.


# ============================================================
# 11. MODULE RESPONSIBILITY
# ============================================================

Each module must:

- have one responsibility
- not mix concerns

Example:

trainer.py must not contain tokenizer logic


# ============================================================
# 12. SIDE EFFECT POLICY
# ============================================================

Functions should:

prefer pure behavior

If side effects required:

must be explicit

Hidden side effects forbidden.


# ============================================================
# 13. GLOBAL STATE
# ============================================================

Global mutable state is forbidden.

Allowed:

constants only

Reason:

global state breaks determinism.


# ============================================================
# 14. CONFIG USAGE
# ============================================================

All behavior must be driven by config objects.

Forbidden:

hardcoded constants inside logic

Example:

bad: lr = 0.0001
good: lr = cfg.learning_rate


# ============================================================
# 15. LOGGING POLICY
# ============================================================

Use structured logger.

Forbidden:

print()

Logging must include:

- level
- timestamp
- context


# ============================================================
# 16. ERROR HANDLING
# ============================================================

Use:

explicit exceptions

Forbidden:

bare except
silent failures

Errors must be actionable.


# ============================================================
# 17. ASSERTIONS
# ============================================================

Use assertions for:

- invariants
- assumptions

Assertions must not replace validation.


# ============================================================
# 18. TESTABILITY
# ============================================================

Code must be:

- easily testable
- dependency injectable
- deterministic

Avoid:

hidden dependencies


# ============================================================
# 19. DEPENDENCY INJECTION
# ============================================================

Prefer:

passing dependencies as parameters

Avoid:

hardcoded singletons


# ============================================================
# 20. DETERMINISM RULES
# ============================================================

Must:

control random seeds
avoid time-dependent logic
avoid non-deterministic ordering

Same inputs → same outputs


# ============================================================
# 21. PERFORMANCE RULES
# ============================================================

Optimize only when:

- measured bottleneck exists

Do not prematurely optimize.

Clarity > micro-optimization.


# ============================================================
# 22. SECURITY PRACTICES
# ============================================================

Must:

validate inputs
sanitize file paths
avoid shell injection

Never execute untrusted content.


# ============================================================
# 23. FILE I/O RULES
# ============================================================

Must:

use atomic writes
handle failures
close resources

Use context managers.


# ============================================================
# 24. CONCURRENCY RULES
# ============================================================

Allowed:

explicit multiprocessing or threading

Forbidden:

implicit shared state

Concurrency must be deterministic.


# ============================================================
# 25. DEPENDENCY POLICY
# ============================================================

Dependencies must be:

- minimal
- maintained
- pinned

Avoid heavy frameworks unless necessary.


# ============================================================
# 26. COMMENTING RULES
# ============================================================

Comments must explain:

WHY

Not:

WHAT (code already shows that)

Avoid redundant comments.


# ============================================================
# 27. LOGIC COMPLEXITY LIMITS
# ============================================================

Cyclomatic complexity should remain low.

If branching becomes complex:

refactor.


# ============================================================
# 28. FORBIDDEN PRACTICES
# ============================================================

Not allowed:

- notebooks as production
- magic numbers
- hidden globals
- copy-paste code
- long monolithic scripts
- interactive prompts
- silent exceptions
- implicit behavior


# ============================================================
# 29. CODE REVIEW CHECKLIST
# ============================================================

Reviewers must verify:

- readability
- types
- tests
- determinism
- config usage
- no globals
- architecture compliance

All must pass before merge.


# ============================================================
# 30. SUMMARY
# ============================================================

M31R code must be:

- simple
- typed
- deterministic
- modular
- testable
- explicit

Infrastructure code must prioritize maintainability over cleverness.

Following these standards ensures long-term stability and scalability.

All contributors must comply without exception.

# END
# ============================================================
