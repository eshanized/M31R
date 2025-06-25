# ============================================================
# M31R
# Glossary and Definitions
# File: 03_GLOSSARY_AND_DEFINITIONS.md
# Version: 1.0.0
# Status: AUTHORITATIVE
# Owner: Eshan Roy
# Document Order: 03 / 25
# Depends On: 01_VISION_PRD.md, 02_REQUIREMENTS_SPEC.md
# ============================================================


# ============================================================
# 0. DOCUMENT PURPOSE
# ============================================================

This document defines the canonical terminology for the M31R project.

The goal is to eliminate ambiguity.

All future documents, source code, comments, and tooling MUST use
the terminology defined here exactly as written.

This document exists primarily for:

- engineers
- reviewers
- maintainers
- automation systems
- large language models

Every term must have:

- one meaning only
- no synonyms
- no overloaded semantics

If a term is not defined here, it MUST NOT be used in formal documents.


# ============================================================
# 1. RULES FOR TERMINOLOGY USAGE
# ============================================================

R1
Terms defined in this file are canonical.

R2
Do not invent alternative names.

R3
Avoid synonyms.

R4
Avoid marketing language.

R5
Use precise technical language.

R6
Prefer short, unambiguous phrases.

R7
Each term MUST map to exactly one concept.

R8
Each concept MUST have exactly one term.


# ============================================================
# 2. PROJECT LEVEL TERMS
# ============================================================

M31R
    The complete system including dataset, tokenizer, model,
    training infrastructure, evaluation, and inference runtime.

Platform
    The full lifecycle system used to build and operate M31R.

Artifact
    Any produced file including:
    models, tokenizer, shards, logs, metrics, configs.

Pipeline
    A sequence of deterministic processing stages.

Stage
    A single step in the pipeline with clear inputs and outputs.

Build
    The process of producing artifacts from raw inputs.

Run
    A single execution of a pipeline or stage.

Experiment
    A controlled training run with defined configuration and metrics.

Release
    A versioned set of stable artifacts ready for consumption.


# ============================================================
# 3. DATA TERMS
# ============================================================

Raw Data
    Unprocessed repositories or source files directly fetched
    from external sources.

Filtered Data
    Data that passed syntax and quality filters.

Dataset
    A versioned collection of filtered files used for training.

Shard
    A contiguous chunk of tokenized data stored for streaming.

Corpus
    The complete set of text/code used for tokenizer and training.

Sample
    A single training instance.

Sequence
    A contiguous token window fed to the model.

Token Stream
    Ordered sequence of integer tokens.

Manifest
    Deterministic file describing dataset contents and hashes.

Provenance
    Metadata describing origin of a file.

Deduplication
    Removal of identical or near-identical content.

Normalization
    Consistent formatting and encoding cleanup.

Filtering
    Removal of low-quality or invalid content.

License Filtering
    Exclusion of non-permissive licensed code.


# ============================================================
# 4. RUST-SPECIFIC TERMS
# ============================================================

Rust Source
    A file with .rs extension containing valid Rust code.

Crate
    A Rust package with Cargo.toml.

Workspace
    A collection of related crates.

Compile Success
    rustc completes without errors.

Compile Failure
    rustc produces errors.

Borrow Checker
    Rust compiler component enforcing ownership rules.

Lifetime
    Rust construct describing reference validity duration.

Ownership
    Memory safety model of Rust.

Trait
    Rust abstraction defining shared behavior.

Macro
    Compile-time code generation mechanism.

Generated Code
    Code created automatically by tools or build scripts.

Vendor Directory
    Third-party dependency folder not authored by project owner.

Target Directory
    Build artifacts directory produced by Cargo.


# ============================================================
# 5. TOKENIZER TERMS
# ============================================================

Tokenizer
    Component converting text to tokens and back.

Vocabulary
    Set of tokens known to tokenizer.

Token
    Smallest unit processed by model.

Special Token
    Reserved control token with predefined behavior.

Subword
    Partial token used by BPE/Unigram methods.

Encoding
    Text → tokens conversion.

Decoding
    Tokens → text conversion.

Fragmentation
    Excessive splitting of logical tokens.

Coverage
    Percentage of corpus represented efficiently.

Round Trip
    Encoding then decoding produces identical text.


# ============================================================
# 6. MODEL TERMS
# ============================================================

Model
    Neural network trained to predict tokens.

Small Language Model (SLM)
    Model with fewer than 1 billion parameters.

Parameter
    Trainable weight inside neural network.

Checkpoint
    Saved model state during training.

Layer
    Single transformer block.

Head
    Attention mechanism inside transformer.

Hidden Size
    Embedding dimensionality.

Context Length
    Maximum tokens processed simultaneously.

Inference
    Using trained model to generate tokens.

Pretraining
    Training from random initialization.

Fine-Tuning
    Additional training on specialized tasks.

Random Initialization
    Starting weights without pretrained knowledge.


# ============================================================
# 7. TRAINING TERMS
# ============================================================

Training Step
    Single optimizer update.

Epoch
    Full pass over dataset.

Batch
    Group of sequences processed together.

Gradient
    Derivative used for weight updates.

Optimizer
    Algorithm adjusting weights.

Loss
    Numeric error metric minimized during training.

Learning Rate
    Step size of optimization.

Mixed Precision
    Using reduced precision floats for speed.

Gradient Accumulation
    Combining gradients across steps.

Distributed Training
    Training across multiple devices.

Determinism
    Identical outputs given identical inputs and seeds.

Seed
    Random number initialization value.

Divergence
    Unstable training causing exploding loss.


# ============================================================
# 8. REASONING (CoT) TERMS
# ============================================================

Chain-of-Thought (CoT)
    Structured intermediate reasoning tokens.

Reasoning Token
    Token representing planning or thought step.

Comment Reasoning
    Reasoning expressed as Rust comments.

Scratchpad
    Structured planning block.

Hidden CoT
    Internal reasoning not emitted to output.

Masking
    Excluding tokens from loss or output.

Injection
    Adding reasoning content to samples.

Reasoning Coverage
    Percentage of samples containing reasoning.

Structured Reasoning
    Code-aligned reasoning rather than natural language.


# ============================================================
# 9. OBJECTIVE TERMS
# ============================================================

Next Token Prediction
    Predict next token in sequence.

Fill-in-the-Middle (FIM)
    Predict missing middle section.

Span Corruption
    Replace spans with masks and reconstruct.

Multi-Objective Training
    Combining multiple loss targets.

Compile-Aware Objective
    Training strategy emphasizing compilable output.

Loss Weight
    Relative importance of loss component.


# ============================================================
# 10. EVALUATION TERMS
# ============================================================

Metric
    Quantitative measurement.

Compile Rate
    Percentage of compilable outputs.

Pass@K
    Probability at least one of K attempts is correct.

Perplexity
    Measure of predictive uncertainty.

Benchmark
    Standardized evaluation set.

Regression
    Performance decrease compared to baseline.

Baseline
    Reference model for comparison.

Validation Set
    Data used for evaluation during training.

Test Set
    Data reserved for final evaluation only.


# ============================================================
# 11. INFRASTRUCTURE TERMS
# ============================================================

Node
    Single machine participating in training.

Device
    GPU or CPU used for computation.

VRAM
    GPU memory.

Throughput
    Tokens processed per second.

Latency
    Time per token generated.

Quantization
    Reducing numeric precision for efficiency.

Artifact Store
    Location where artifacts are saved.

Run Log
    File capturing execution details.

Checksum
    Hash used for integrity verification.

Snapshot
    Point-in-time state capture.


# ============================================================
# 12. CLI TERMS
# ============================================================

Command
    Executable entry point.

Flag
    Optional argument modifying behavior.

Subcommand
    Command nested under main command.

Exit Code
    Numeric success/failure indicator.

Dry Run
    Execution without side effects.


# ============================================================
# 13. GOVERNANCE TERMS
# ============================================================

Owner
    Responsible decision maker.

Contributor
    Individual submitting changes.

Reviewer
    Individual approving changes.

Approval
    Formal acceptance.

Breaking Change
    Modification invalidating compatibility.

Version
    Numeric identifier for releases.

Semantic Versioning
    Major.Minor.Patch scheme.


# ============================================================
# 14. FORBIDDEN TERMS
# ============================================================

The following terms are disallowed in formal docs:

- AI magic
- smart
- intelligent assistant
- human-like
- chatbot
- GPT-like
- general AI

Reason:
They are ambiguous and non-technical.


# ============================================================
# 15. CANONICAL ABBREVIATIONS
# ============================================================

SLM  -> Small Language Model
CoT  -> Chain-of-Thought
FIM  -> Fill-in-the-Middle
AST  -> Abstract Syntax Tree
CLI  -> Command Line Interface
PRD  -> Product Requirements Document
VRAM -> Video RAM


# ============================================================
# 16. SUMMARY
# ============================================================

This glossary standardizes vocabulary across the entire M31R platform.

All future documentation MUST use these definitions exactly.

Ambiguous terminology is not allowed.

Consistency is mandatory for:

- engineering clarity
- maintainability
- automation
- LLM comprehension

# END
# ============================================================
