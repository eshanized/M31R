# ============================================================
# M31R
# Reasoning (Chain-of-Thought) Design Specification
# File: 08_REASONING_COT_DESIGN.md
# Version: 1.0.0
# Status: AUTHORITATIVE
# Owner: Eshan Roy
# Document Order: 08 / 25
# Depends On:
#   01_VISION_PRD.md
#   02_REQUIREMENTS_SPEC.md
#   03_GLOSSARY_AND_DEFINITIONS.md
#   04_SYSTEM_ARCHITECTURE.md
#   05_DATA_ARCHITECTURE.md
#   06_MODEL_ARCHITECTURE.md
#   07_TRAINING_ARCHITECTURE.md
# ============================================================


# ============================================================
# 0. PURPOSE
# ============================================================

This document defines the complete design of the reasoning system
(Chain-of-Thought, abbreviated CoT) for M31R.

This document answers:

"How does the model learn structured reasoning while staying Rust-only?"

This document specifies:

- reasoning philosophy
- allowed reasoning formats
- injection strategy
- masking strategy
- training behavior
- constraints
- failure modes

This document is authoritative for ALL reasoning behavior.

If reasoning implementation conflicts with this document, this document wins.


# ============================================================
# 1. PROBLEM STATEMENT
# ============================================================

Generic language models often rely on verbose natural language reasoning:

Example:
"Let's think step by step..."

This approach is unsuitable for M31R because:

- adds English tokens
- pollutes tokenizer
- increases sequence length
- reduces Rust specialization
- wastes parameters
- harms compile rate

Rust is not a natural language problem.

Rust reasoning is:

- structural
- symbolic
- compiler-like
- constraint solving

Therefore:

Classic conversational CoT is rejected.


# ============================================================
# 2. DESIGN PHILOSOPHY
# ============================================================

Reasoning must be:

- short
- structured
- code-aligned
- syntax-aware
- deterministic

Reasoning must NOT be:

- verbose
- conversational
- explanatory essays
- natural language heavy

Principle:

Think like a compiler, not a chatbot.


# ============================================================
# 3. OBJECTIVES OF REASONING
# ============================================================

CoT exists to improve:

R1  compile success
R2  type correctness
R3  ownership correctness
R4  borrow checker compliance
R5  multi-step planning
R6  long function synthesis
R7  refactoring reliability

CoT is NOT intended to:

- explain concepts to humans
- produce comments for readability
- simulate conversation


# ============================================================
# 4. REASONING DEFINITION
# ============================================================

Chain-of-Thought (CoT):

Structured intermediate tokens that represent planning or reasoning steps
inserted before or within code during training.

These tokens guide the model to learn problem decomposition.


# ============================================================
# 5. REASONING TYPES
# ============================================================

M31R supports exactly three reasoning types.

RT-1  Comment-based reasoning
RT-2  Scratchpad reasoning
RT-3  Hidden reasoning

No other types are allowed.


# ============================================================
# 6. TYPE 1 — COMMENT-BASED REASONING
# ============================================================

Definition:

Short Rust comments describing planned steps.

Format:

// step description

Example:

// iterate slice
// accumulate sum
// return result

fn sum(xs: &[i32]) -> i32 {
    xs.iter().sum()
}

Characteristics:

- lightweight
- Rust-compatible
- no tokenizer pollution
- human readable
- low overhead

This is the default reasoning mode.


# ============================================================
# 7. TYPE 2 — SCRATCHPAD REASONING
# ============================================================

Definition:

Structured planning block preceding code.

Format:

/* PLAN
1. parse input
2. validate
3. allocate buffer
4. return Result
*/

Example:

/* PLAN
1. open file
2. read bytes
3. deserialize
*/

fn load_config(path: &Path) -> Result<Config, Error> {
    ...
}

Characteristics:

- stronger decomposition
- better for long functions
- multi-step tasks
- moderate token overhead


# ============================================================
# 8. TYPE 3 — HIDDEN REASONING
# ============================================================

Definition:

Internal reasoning tokens not emitted at inference.

Mechanism:

- reasoning tokens present during training
- masked from output
- optionally excluded from loss

Purpose:

- preserve quality gains
- avoid output noise
- maintain clean completions

This is the recommended production mode.


# ============================================================
# 9. REASONING TOKEN RULES
# ============================================================

Rules:

- must be short
- must align with Rust semantics
- must avoid English paragraphs
- must not exceed 2–3 lines per step
- must not alter code behavior

Forbidden:

- storytelling
- explanations
- prose
- markdown
- emojis


# ============================================================
# 10. DOMAIN ALIGNMENT
# ============================================================

Reasoning vocabulary must focus on:

- ownership
- borrowing
- lifetimes
- mutability
- iteration
- allocation
- error handling
- traits
- concurrency safety

Example allowed:

// borrow immutably
// propagate error
// avoid move

Example forbidden:

// this function will elegantly compute the sum


# ============================================================
# 11. REASONING INJECTION STAGE
# ============================================================

Injection occurs during:

Dataset → Tokenization phase

NOT during:

- raw acquisition
- filtering
- inference

Reasoning is a training-time augmentation only.


# ============================================================
# 12. AUTOMATIC GENERATION STRATEGY
# ============================================================

Reasoning must be automatically generated.

Manual annotation is forbidden.

Generation inputs:

- AST
- function names
- control flow
- tests
- type signatures


# ============================================================
# 13. AST-BASED HEURISTICS
# ============================================================

Examples:

If loop detected:
    "iterate items"

If Result returned:
    "propagate error"

If Vec created:
    "allocate vector"

If match:
    "branch on enum"

These heuristics must be deterministic.


# ============================================================
# 14. FUNCTION NAME HEURISTICS
# ============================================================

Examples:

parse_* → "parse input"
load_*  → "read resource"
save_*  → "write resource"
calc_*  → "compute value"

Used only when safe.


# ============================================================
# 15. TEST-DRIVEN HEURISTICS
# ============================================================

Tests provide behavior hints.

Example:

test_negative_input

→ "handle negative values"

Improves semantic correctness.


# ============================================================
# 16. INJECTION RATIO
# ============================================================

Not all samples require reasoning.

Default:

30–50% of samples include reasoning.

Reasons:

- prevent overfitting
- avoid token bloat
- preserve raw code distribution


# ============================================================
# 17. SEQUENCE LAYOUT
# ============================================================

Layouts allowed:

[reasoning][code]
[prefix][reasoning][suffix]
[hidden reasoning][code]

Layouts must remain consistent.


# ============================================================
# 18. TOKEN BUDGET
# ============================================================

Reasoning must not exceed:

20% of total sequence length

Prevents context waste.


# ============================================================
# 19. LOSS INTEGRATION
# ============================================================

Options:

Mode A:
    include reasoning in loss

Mode B:
    reduced weight

Mode C:
    masked (hidden)

Default:

reduced weight


# ============================================================
# 20. MASKING RULES
# ============================================================

Hidden reasoning:

- excluded from output
- optionally excluded from gradient

Mask tokens must be clearly identifiable.


# ============================================================
# 21. INFERENCE POLICY
# ============================================================

Inference must:

- not emit reasoning by default

Optional debug flag may show reasoning.

Default output:

pure Rust only.


# ============================================================
# 22. METRICS FOR REASONING
# ============================================================

Track:

- compile rate improvement
- pass@k improvement
- sequence overhead
- latency impact

If no measurable benefit, adjust.


# ============================================================
# 23. FAILURE MODES
# ============================================================

Common issues:

- overlong reasoning
- English pollution
- syntax breakage
- token inflation

These must trigger filtering.


# ============================================================
# 24. SECURITY RULES
# ============================================================

Reasoning must never:

- execute code
- leak secrets
- include external data

Pure text only.


# ============================================================
# 25. DETERMINISM
# ============================================================

Given same:

- dataset
- config
- seed

Injection must produce identical reasoning.


# ============================================================
# 26. STORAGE FORMAT
# ============================================================

Reasoning is embedded in text samples.

No separate files.

Shards contain final injected sequences only.


# ============================================================
# 27. FORBIDDEN PRACTICES
# ============================================================

Not allowed:

- natural language essays
- interactive prompting
- external LLM generation
- manual labeling
- chat-style reasoning
- runtime reasoning generation


# ============================================================
# 28. EXTENSIBILITY
# ============================================================

Future enhancements may include:

- improved AST heuristics
- static analysis signals
- type inference hints

Must remain deterministic.


# ============================================================
# 29. RATIONALE SUMMARY
# ============================================================

Reasoning helps models plan.

But verbose language harms specialization.

Therefore:

Use short, structured, Rust-native reasoning only.


# ============================================================
# 30. SUMMARY
# ============================================================

M31R reasoning system:

- structured
- deterministic
- Rust-aligned
- lightweight
- optionally hidden

Goal:

Improve correctness without sacrificing specialization or speed.

All reasoning implementations MUST follow this specification.

# END
# ============================================================
