# ============================================================
# M31R
# Vision and Product Requirements Document (PRD)
# File: 01_VISION_PRD.md
# Version: 1.0.0
# Status: AUTHORITATIVE
# Owner: Eshan Roy
# Project Type: Enterprise ML Infrastructure Platform
# Document Order: 01 / 25
# ============================================================


# ============================================================
# 0. DOCUMENT CONTRACT
# ============================================================

This document is the authoritative source of truth for:

- project vision
- scope
- objectives
- constraints
- non-goals
- success criteria
- product boundaries

All future documents MUST derive from this file.

If any later document or code conflicts with this file:
THIS FILE TAKES PRECEDENCE.

This document is intentionally:
- explicit
- deterministic
- unambiguous
- machine-readable
- LLM-parseable
- free of marketing language

This document is NOT:
- design documentation
- architecture documentation
- implementation guidance

Those appear in later documents.


# ============================================================
# 1. PROJECT IDENTIFICATION
# ============================================================

Project Name:
M31R

Expanded Form:
Model 31 Rust

Classification:
Enterprise Internal Language Model Platform

Category:
Specialized Small Language Model (SLM)

Primary Domain:
Rust Programming Language

Ownership:
Fully self-owned
Zero external pretrained weights
Zero vendor dependency for core training


# ============================================================
# 2. EXECUTIVE SUMMARY
# ============================================================

M31R is a specialized Small Language Model platform designed exclusively
for Rust programming language tasks.

The system will:

- collect Rust source code
- clean and filter data deterministically
- train a transformer model from random initialization
- integrate structured reasoning (Chain-of-Thought)
- provide high-quality Rust code generation
- run locally and offline
- operate without external APIs

The system intentionally avoids:

- generic chat behavior
- multi-language capability
- dependence on foundation models
- external inference services

The outcome is:

A compact, fast, controllable, Rust-specialized model that achieves higher
correctness and lower cost than general-purpose LLMs.


# ============================================================
# 3. PROBLEM STATEMENT
# ============================================================

Modern large language models exhibit the following problems for Rust:

1. Excessive parameter count
2. High inference latency
3. High memory usage
4. Weak Rust specialization
5. Poor borrow checker reasoning
6. Frequent compilation failures
7. Vendor lock-in
8. Licensing ambiguity
9. Cloud dependency
10. Inability to run offline

Generic models are trained on mixed corpora:

- English
- JavaScript
- Python
- Web content
- Markdown
- Chat data
- Noise

As a result:

- most parameters are irrelevant to Rust
- tokenizer wastes vocabulary on non-code tokens
- reasoning quality degrades
- hallucinations increase
- cost increases


# ============================================================
# 4. VISION
# ============================================================

Build the best Rust code model in its parameter class.

Not the largest.
Not the most general.

But:

- fastest
- smallest
- most correct
- most reliable
- fully owned

The guiding philosophy:

Specialization beats scale.


# ============================================================
# 5. MISSION
# ============================================================

Provide a fully self-hosted Rust reasoning model that:

- generates compilable code
- understands ownership and lifetimes
- assists in refactoring
- runs on commodity hardware
- is reproducible
- is deterministic
- is legally clean


# ============================================================
# 6. TARGET USERS
# ============================================================

Primary Users:

- Rust developers
- systems engineers
- backend engineers
- security engineers

Secondary Users:

- internal tooling teams
- CI systems
- IDE integrations
- automated code generators

Non-target Users:

- casual chat users
- content creators
- non-technical users


# ============================================================
# 7. CORE PRINCIPLES
# ============================================================

P1 — Ownership
All weights must be trained internally.

P2 — Determinism
Same data + config = same model.

P3 — Reproducibility
Every artifact must be rebuildable.

P4 — Simplicity
Prefer fewer moving parts.

P5 — Specialization
Rust only.

P6 — Transparency
No hidden behavior.

P7 — Offline-first
No mandatory cloud services.

P8 — Cost efficiency
Minimize compute requirements.


# ============================================================
# 8. SCOPE (IN)
# ============================================================

The following ARE in scope:

- Rust source crawling
- data filtering
- dataset versioning
- tokenizer training
- transformer pretraining
- Chain-of-Thought reasoning
- fill-in-middle training
- evaluation metrics
- local inference
- CLI tooling
- IDE integration hooks
- reproducible pipelines


# ============================================================
# 9. SCOPE (OUT)
# ============================================================

The following are explicitly out of scope:

- general chat assistants
- multi-language training
- speech
- vision
- RLHF chat alignment
- SaaS hosting
- proprietary cloud APIs
- billion-parameter scaling
- marketing features
- prompt engineering hacks


# ============================================================
# 10. PRODUCT GOALS
# ============================================================

G1 — High compile success rate
G2 — Low latency inference
G3 — Small memory footprint
G4 — Fully local execution
G5 — Fully reproducible training
G6 — Deterministic dataset pipeline
G7 — Clean licensing


# ============================================================
# 11. NON-GOALS
# ============================================================

NG1 — beating GPT-scale models
NG2 — supporting every language
NG3 — conversational AI
NG4 — human-like personality
NG5 — cloud-first deployment
NG6 — proprietary lock-in


# ============================================================
# 12. SUCCESS METRICS
# ============================================================

Primary Metrics:

- compile success rate >= 70%
- pass@5 >= 60%
- inference latency <= 50ms/token
- VRAM usage <= 8GB
- model size <= 500M parameters

Secondary Metrics:

- perplexity reduction
- memory stability
- deterministic training


# ============================================================
# 13. CONSTRAINTS
# ============================================================

Hardware:

- consumer GPUs
- limited VRAM
- local machines

Legal:

- only permissive licenses

Operational:

- offline capability required

Engineering:

- maintainable codebase
- modular design


# ============================================================
# 14. ASSUMPTIONS
# ============================================================

- sufficient Rust data exists
- small models are adequate
- compile success correlates with usefulness
- specialization improves performance
- internal infrastructure is sufficient


# ============================================================
# 15. RISKS (HIGH LEVEL)
# ============================================================

- insufficient data quality
- underfitting
- reasoning collapse
- training instability
- hardware bottlenecks

Detailed risks are documented later.


# ============================================================
# 16. STAKEHOLDERS
# ============================================================

Owner:
Eshan Roy

Engineering:
Core ML + systems contributors

Consumers:
Internal development teams


# ============================================================
# 17. HIGH LEVEL SYSTEM CONCEPT
# ============================================================

Pipeline:

crawl → filter → tokenize → shard → train → evaluate → serve

Each stage must be isolated.

No implicit coupling.


# ============================================================
# 18. DEFINITIONS
# ============================================================

SLM:
Small Language Model (< 1B params)

CoT:
Chain-of-Thought reasoning tokens

FIM:
Fill-in-Middle objective

Compile Success:
Code builds with rustc without errors


# ============================================================
# 19. ACCEPTANCE CRITERIA
# ============================================================

M31R is considered successful when:

- model trains from scratch
- produces compilable Rust
- runs locally
- requires no external service
- reproducible from source


# ============================================================
# 20. FUTURE EXTENSIBILITY
# ============================================================

Future additions MAY include:

- quantization
- LoRA adapters
- IDE plugins
- benchmarking automation

But MUST NOT violate core principles.


# ============================================================
# 21. SUMMARY STATEMENT
# ============================================================

M31R exists to build the most efficient, correct, and controllable Rust
code language model possible through specialization, ownership, and
deterministic engineering.

All future work must reinforce this objective.

# END OF DOCUMENT
# ============================================================
