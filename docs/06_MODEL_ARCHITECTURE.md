# ============================================================
# M31R
# Model Architecture Specification
# File: 06_MODEL_ARCHITECTURE.md
# Version: 1.0.0
# Status: AUTHORITATIVE
# Owner: Eshan Roy
# Document Order: 06 / 25
# Depends On:
#   01_VISION_PRD.md
#   02_REQUIREMENTS_SPEC.md
#   03_GLOSSARY_AND_DEFINITIONS.md
#   04_SYSTEM_ARCHITECTURE.md
#   05_DATA_ARCHITECTURE.md
# ============================================================


# ============================================================
# 0. PURPOSE
# ============================================================

This document specifies the neural model architecture used by M31R.

This document answers:

"What model is trained, and what are its structural rules?"

This document defines:

- model family
- topology
- parameterization
- constraints
- objectives
- architectural invariants

This document does NOT define:

- training infrastructure
- hardware strategy
- evaluation methods

Those are defined later.

If an implementation deviates from this architecture, it is non-compliant.


# ============================================================
# 1. DESIGN PHILOSOPHY
# ============================================================

The model is intentionally:

- small
- fast
- specialized
- deterministic
- Rust-only

The goal is NOT scale.

The goal is:

maximum Rust correctness per parameter.

Core belief:

Specialization > parameter count.


# ============================================================
# 2. ARCHITECTURAL PRINCIPLES
# ============================================================

MA-1
Use the simplest architecture that works.

MA-2
Prefer proven transformer designs.

MA-3
Avoid experimental research tricks.

MA-4
Optimize for inference speed.

MA-5
Optimize for memory efficiency.

MA-6
Avoid unnecessary parameters.

MA-7
Keep architecture reproducible.

MA-8
Minimize hidden complexity.

MA-9
No dependence on proprietary components.

MA-10
Architecture must be explainable.


# ============================================================
# 3. MODEL FAMILY
# ============================================================

Family:
Decoder-only Transformer

Rationale:

- simplest
- proven for code completion
- supports autoregressive generation
- supports fill-in-middle
- lower complexity than encoder-decoder

Alternative families are explicitly rejected:

- encoder-decoder
- mixture-of-experts
- retrieval-augmented
- recurrent architectures

Reason:

unnecessary complexity.


# ============================================================
# 4. HIGH LEVEL TOPOLOGY
# ============================================================

Structure:

Input Tokens
    ↓
Embedding Layer
    ↓
N × Transformer Blocks
    ↓
Final Norm
    ↓
Linear LM Head
    ↓
Logits

This structure is fixed.


# ============================================================
# 5. EMBEDDING LAYER
# ============================================================

Responsibilities:

- convert tokens to vectors

Requirements:

- shared embedding and LM head weights (weight tying)
- deterministic initialization

Benefits:

- parameter efficiency
- improved generalization


# ============================================================
# 6. POSITIONAL ENCODING
# ============================================================

Method:

RoPE (Rotary Positional Embedding)

Reasons:

- stable long context
- no learned position parameters
- better extrapolation
- lower memory usage

Absolute learned positions are forbidden.


# ============================================================
# 7. TRANSFORMER BLOCK
# ============================================================

Each block contains:

1. RMSNorm
2. Self-Attention
3. Residual
4. RMSNorm
5. FeedForward (SwiGLU)
6. Residual

This order is fixed.

Pre-norm configuration is required.


# ============================================================
# 8. ATTENTION MECHANISM
# ============================================================

Type:

Multi-head self-attention

Requirements:

- causal masking
- FlashAttention implementation when available
- batch-first layout

Benefits:

- faster training
- lower memory
- stable scaling


# ============================================================
# 9. FEEDFORWARD NETWORK
# ============================================================

Activation:

SwiGLU

Structure:

Linear → SwiGLU → Linear

Reason:

Better parameter efficiency and performance compared to ReLU/GELU.


# ============================================================
# 10. NORMALIZATION
# ============================================================

Type:

RMSNorm

Reasons:

- faster than LayerNorm
- lower memory
- simpler computation
- widely adopted in modern LLMs


# ============================================================
# 11. OUTPUT HEAD
# ============================================================

Type:

Linear projection to vocabulary

Weight tied with embedding matrix.

Benefits:

- reduces parameters
- improves learning


# ============================================================
# 12. PARAMETER SIZES (TARGET CONFIGURATIONS)
# ============================================================

Small (local development):

- 60M params

Medium (default production):

- 200M params

Large (upper bound):

- 400–500M params

Models larger than 500M are out of scope.


# ============================================================
# 13. DEFAULT CONFIGURATION (MEDIUM)
# ============================================================

Layers: 18–24
Hidden size: 1024
Heads: 16
Head dim: 64
Context length: 2048
Vocab: 16k–24k
Params: ~200M

This is the recommended baseline.


# ============================================================
# 14. CONTEXT WINDOW
# ============================================================

Minimum:
2048 tokens

Rationale:

- most Rust functions fit
- sufficient for modules
- balances memory

Long context experiments are optional but not required.


# ============================================================
# 15. VOCABULARY DESIGN
# ============================================================

Tokenizer must:

- be Rust-specific
- avoid English-heavy tokens
- include symbols/operators
- minimize fragmentation

Target vocab size:

16k–24k

Large vocabularies are discouraged.


# ============================================================
# 16. OBJECTIVES OVERVIEW
# ============================================================

Training uses multi-objective loss.

Objectives:

1. Next token prediction
2. Fill-in-middle
3. Chain-of-Thought reasoning

Each objective improves a different capability.


# ============================================================
# 17. NEXT TOKEN PREDICTION
# ============================================================

Purpose:

core language modeling

Behavior:

predict next token sequentially

Weight:

primary loss


# ============================================================
# 18. FILL-IN-MIDDLE (FIM)
# ============================================================

Purpose:

code completion and editing

Behavior:

prefix + suffix → predict middle

Benefits:

- IDE integration
- refactoring
- partial completion

Mandatory feature.


# ============================================================
# 19. CHAIN-OF-THOUGHT (CoT)
# ============================================================

Purpose:

improve structured reasoning

Types:

- comment reasoning
- scratchpad reasoning
- hidden reasoning

Constraints:

- Rust-aligned
- concise
- not verbose English


# ============================================================
# 20. LOSS FORMULATION
# ============================================================

Total loss:

L = L_next + α * L_fim + β * L_cot

Where:

α and β are configurable weights.

Default:

α = 0.3
β = 0.2

Exact values tunable.


# ============================================================
# 21. INITIALIZATION
# ============================================================

Weights must:

- use deterministic seeds
- random initialization only

Loading external pretrained weights is forbidden.


# ============================================================
# 22. REGULARIZATION
# ============================================================

Allowed:

- dropout (light)
- weight decay
- gradient clipping

Disallowed:

- heavy stochastic tricks
- unstable experimental methods


# ============================================================
# 23. NUMERICAL PRECISION
# ============================================================

Training:

bf16 or fp16

Inference:

fp16 or int8/int4

Precision must not affect determinism beyond acceptable tolerance.


# ============================================================
# 24. MEMORY TARGETS
# ============================================================

Default inference must run within:

<= 8GB VRAM

Constraint drives architecture choices.


# ============================================================
# 25. LATENCY TARGETS
# ============================================================

Target:

<= 50ms per token

Architecture must favor speed over marginal accuracy gains.


# ============================================================
# 26. EXTENSIBILITY RULES
# ============================================================

Future extensions may include:

- quantization
- LoRA adapters
- distillation

But must not:

- violate determinism
- require external services
- exceed memory targets


# ============================================================
# 27. FORBIDDEN ARCHITECTURES
# ============================================================

Not allowed:

- mixture-of-experts
- retrieval-augmented generation
- reinforcement learning alignment loops
- proprietary backbones
- external hosted models


# ============================================================
# 28. MODEL ARTIFACT FORMAT
# ============================================================

Artifacts must include:

- weights
- config
- tokenizer
- checksum
- version metadata

Single directory bundle.


# ============================================================
# 29. TRACEABILITY
# ============================================================

Each model must trace back to:

- dataset version
- tokenizer version
- config hash
- training seed

No anonymous models allowed.


# ============================================================
# 30. SUMMARY
# ============================================================

M31R uses a compact, decoder-only transformer optimized for Rust code.

Key characteristics:

- 200M class
- RMSNorm
- RoPE
- SwiGLU
- FlashAttention
- FIM + CoT objectives
- trained from scratch

The architecture prioritizes:

speed
simplicity
determinism
specialization

All implementations MUST adhere to this design.

# END
# ============================================================
