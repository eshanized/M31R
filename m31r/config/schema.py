# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Type-safe configuration schemas for M31R.

Every config category from 11_CONFIGURATION_SPEC.md gets its own frozen pydantic
model. Frozen means once you create it, you cannot mutate it — this is intentional.
Config mutation at runtime is a bug per spec.

The models use pydantic v2's ConfigDict with:
  - frozen=True: immutability after construction
  - extra="forbid": unknown fields cause immediate failure
  - validate_default=True: even defaults get type-checked

If you're adding a new config section for a future phase, create a new model here,
add it as an Optional field to M31RConfig, and add the corresponding YAML file to
configs/.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class DirectoryConfig(BaseModel):
    """Paths to the standard project directories, all relative to project root."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    data: str = Field(default="data", description="Root directory for datasets and raw data")
    checkpoints: str = Field(default="checkpoints", description="Saved model states")
    logs: str = Field(default="logs", description="System and debug logs")
    experiments: str = Field(default="experiments", description="Training run outputs")
    configs: str = Field(default="configs", description="Configuration files directory")


class GlobalConfig(BaseModel):
    """
    Cross-cutting settings that apply to the entire system.
    Maps to configs/global.yaml per spec section 6.

    This is the first config loaded and it controls things like
    reproducibility (seed), observability (log_level), and project identity.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    config_version: str = Field(
        description="Schema version for compatibility tracking, e.g. '1.0.0'"
    )
    project_name: str = Field(
        default="m31r", description="Human-readable project identifier"
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Global random seed propagated to all subsystems",
    )
    log_level: str = Field(
        default="INFO",
        description="One of DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    log_file: Optional[str] = Field(
        default=None,
        description="Optional path for file-based log output, relative to project root",
    )
    directories: DirectoryConfig = Field(default_factory=DirectoryConfig)


class SourceConfig(BaseModel):
    """
    A single data source — one Git repository to crawl.

    Each source is pinned to a specific commit so re-running the pipeline always
    fetches the exact same snapshot. The license field is what we expect to find;
    it gets verified during the filter stage.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    name: str = Field(description="Short identifier for this source, used in directory names")
    url: str = Field(description="Git clone URL for the repository")
    commit: str = Field(description="Exact commit hash to pin this source to")
    license: str = Field(
        default="unknown",
        description="Expected SPDX license identifier, verified during filtering",
    )


class FilterConfig(BaseModel):
    """
    All the knobs that control what gets kept and what gets thrown away
    during the filter stage. These come straight from the data architecture spec.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    max_file_size_bytes: int = Field(
        default=5_242_880,
        ge=1,
        description="Files bigger than this get dropped (default 5 MB per spec §11)",
    )
    max_lines: int = Field(
        default=5000,
        ge=1,
        description="Files with more lines than this get dropped (per spec §11)",
    )
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".rs"],
        description="Only files with these extensions survive filtering",
    )
    excluded_directories: list[str] = Field(
        default_factory=lambda: [
            "target", "vendor", "node_modules", "build", "dist",
            "generated", ".git",
        ],
        description="Any file inside these directories gets dropped (per spec §9)",
    )
    allowed_licenses: list[str] = Field(
        default_factory=lambda: [
            "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause",
            "ISC", "Unlicense", "CC0-1.0",
        ],
        description="Only repos with these SPDX identifiers are kept (per spec §14)",
    )
    enable_deduplication: bool = Field(
        default=True,
        description="Whether to run exact hash-based deduplication",
    )


class ShardConfig(BaseModel):
    """Controls how filtered files get packed into fixed-size shards."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    shard_size_bytes: int = Field(
        default=268_435_456,
        ge=1,
        description="Target size per shard in bytes (default 256 MB)",
    )


class DatasetConfig(BaseModel):
    """
    Everything the data pipeline needs — sources to crawl, filtering rules,
    sharding parameters, and where to put the output. Maps to configs/dataset.yaml.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    config_version: str = Field(description="Schema version")
    sources: list[SourceConfig] = Field(
        default_factory=list,
        description="Repositories to crawl — empty list means nothing to crawl",
    )
    filter: FilterConfig = Field(default_factory=FilterConfig)
    shard: ShardConfig = Field(default_factory=ShardConfig)
    raw_directory: str = Field(
        default="data/raw",
        description="Where crawled repos land, relative to project root",
    )
    filtered_directory: str = Field(
        default="data/filtered",
        description="Where filtered files go, relative to project root",
    )
    dataset_directory: str = Field(
        default="data/datasets",
        description="Where versioned dataset shards end up, relative to project root",
    )


class NormalizationConfig(BaseModel):
    """How raw text gets cleaned up before tokenization."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    nfkc: bool = Field(
        default=True,
        description="Apply Unicode NFKC normalization to stabilize character variants",
    )
    lowercase: bool = Field(
        default=False,
        description="Convert text to lowercase before tokenization (usually off for code)",
    )


class TokenizerConfig(BaseModel):
    """
    Everything the tokenizer subsystem needs — training parameters, vocabulary
    settings, normalization rules, and I/O paths. Maps to configs/tokenizer.yaml.

    The vocab_size target of 16384 comes from the model architecture spec (§15),
    which recommends 16k–24k for a Rust-specific tokenizer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    config_version: str = Field(description="Schema version for compatibility tracking")
    vocab_size: int = Field(
        default=16384,
        ge=256,
        le=65536,
        description="Target vocabulary size — 16k is the recommended baseline for Rust code",
    )
    tokenizer_type: str = Field(
        default="bpe",
        description="Algorithm for vocabulary construction: 'bpe' or 'unigram'",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for deterministic training — same seed = same vocab",
    )
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    special_tokens: list[str] = Field(
        default_factory=lambda: ["<pad>", "<unk>", "<bos>", "<eos>"],
        description="Tokens reserved for model control signals, added before training",
    )
    min_frequency: int = Field(
        default=2,
        ge=1,
        description="Minimum number of times a token must appear to enter the vocabulary",
    )
    max_token_length: int = Field(
        default=64,
        ge=1,
        description="Longest single token allowed in characters — keeps vocab sensible",
    )
    pre_tokenizer_type: str = Field(
        default="byte_level",
        description="How text gets split before BPE/Unigram: 'byte_level' or 'whitespace'",
    )
    dataset_directory: str = Field(
        default="data/datasets",
        description="Where to find dataset shards for training, relative to project root",
    )
    output_directory: str = Field(
        default="data/tokenizer",
        description="Where the finished tokenizer bundle gets written, relative to project root",
    )


class ModelConfig(BaseModel):
    """
    Neural architecture definitions. Maps to configs/model.yaml.
    Per 06_MODEL_ARCHITECTURE.md §13, default is the Medium config (~200M params).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    config_version: str = Field(description="Schema version")
    n_layers: int = Field(
        default=18,
        ge=1,
        le=96,
        description="Number of transformer blocks (spec default: 18-24)",
    )
    hidden_size: int = Field(
        default=1024,
        ge=64,
        description="Model hidden dimension (spec default: 1024)",
    )
    n_heads: int = Field(
        default=16,
        ge=1,
        description="Number of attention heads (spec default: 16)",
    )
    head_dim: int = Field(
        default=64,
        ge=1,
        description="Dimension per attention head (spec default: 64)",
    )
    context_length: int = Field(
        default=2048,
        ge=128,
        description="Maximum sequence length (spec minimum: 2048)",
    )
    dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Dropout probability for attention and FFN (light per spec §22)",
    )
    norm_eps: float = Field(
        default=1e-6,
        gt=0.0,
        description="Epsilon for RMSNorm numerical stability",
    )
    rope_theta: float = Field(
        default=10000.0,
        gt=0.0,
        description="Base frequency for Rotary Positional Embedding",
    )
    init_std: float = Field(
        default=0.02,
        gt=0.0,
        description="Standard deviation for weight initialization",
    )
    vocab_size: int = Field(
        default=16384,
        ge=256,
        le=65536,
        description="Vocabulary size (must match tokenizer, default 16k per spec §15)",
    )


class TrainConfig(BaseModel):
    """
    Training hyperparameters and schedule. Maps to configs/train.yaml.
    Per 07_TRAINING_ARCHITECTURE.md.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    config_version: str = Field(description="Schema version")
    batch_size: int = Field(
        default=8,
        ge=1,
        description="Micro-batch size (number of sequences per step)",
    )
    gradient_accumulation_steps: int = Field(
        default=4,
        ge=1,
        description="Number of micro-batches before optimizer step (simulates larger batch)",
    )
    max_steps: int = Field(
        default=100000,
        ge=1,
        description="Total number of optimizer steps for the training run",
    )
    learning_rate: float = Field(
        default=3e-4,
        gt=0.0,
        description="Peak learning rate for AdamW optimizer",
    )
    min_learning_rate: float = Field(
        default=1e-5,
        ge=0.0,
        description="Minimum learning rate at the end of cosine decay",
    )
    weight_decay: float = Field(
        default=0.1,
        ge=0.0,
        description="L2 weight decay for AdamW (per spec §10)",
    )
    beta1: float = Field(
        default=0.9,
        ge=0.0,
        lt=1.0,
        description="AdamW beta1 (first moment decay)",
    )
    beta2: float = Field(
        default=0.95,
        ge=0.0,
        lt=1.0,
        description="AdamW beta2 (second moment decay)",
    )
    grad_clip: float = Field(
        default=1.0,
        gt=0.0,
        description="Maximum gradient norm for gradient clipping (per spec §22)",
    )
    warmup_steps: int = Field(
        default=1000,
        ge=0,
        description="Number of linear warmup steps before cosine decay (per spec §11)",
    )
    precision: str = Field(
        default="bf16",
        description="Training precision: 'bf16', 'fp16', or 'fp32' (per spec §13)",
    )
    checkpoint_interval: int = Field(
        default=1000,
        ge=1,
        description="Save checkpoint every N optimizer steps (per spec §16)",
    )
    log_interval: int = Field(
        default=10,
        ge=1,
        description="Log training metrics every N steps",
    )
    dataset_directory: str = Field(
        default="data/datasets",
        description="Path to dataset shards directory, relative to project root",
    )
    tokenizer_directory: str = Field(
        default="data/tokenizer",
        description="Path to tokenizer bundle directory, relative to project root",
    )


class EvalConfig(BaseModel):
    """
    Everything the evaluation harness needs to run benchmarks against a model.

    This controls where to find benchmark tasks, how many attempts to give
    the model (pass@k), how long to let compilation and tests run before
    giving up, and where results land.

    Maps to configs/eval.yaml.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    config_version: str = Field(description="Schema version")
    benchmark_directory: str = Field(
        default="benchmarks",
        description="Where benchmark task folders live, relative to project root",
    )
    k_values: list[int] = Field(
        default_factory=lambda: [1, 5, 10],
        description="Values of K for pass@k evaluation — each task gets this many attempts",
    )
    compile_timeout_seconds: int = Field(
        default=10,
        ge=1,
        le=300,
        description="Max seconds to wait for cargo build before declaring failure",
    )
    test_timeout_seconds: int = Field(
        default=10,
        ge=1,
        le=300,
        description="Max seconds to wait for cargo test before declaring failure",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Seed for deterministic generation — same seed means same outputs",
    )
    max_workers: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Number of parallel task evaluations (kept at 1 for determinism by default)",
    )
    output_directory: str = Field(
        default="experiments",
        description="Where evaluation results get written, relative to project root",
    )


class RuntimeConfig(BaseModel):
    """Inference runtime settings. Maps to configs/runtime.yaml."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    config_version: str = Field(description="Schema version")


class M31RConfig(BaseModel):
    """
    Top-level config container. Each CLI command loads the relevant section.

    In practice, a single YAML file might contain just `global:` for foundation
    work, or `global:` + `train:` for a training run. Sections not present in
    the YAML stay None and that's fine — commands validate they have what they need.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    global_config: GlobalConfig = Field(alias="global")
    dataset: Optional[DatasetConfig] = Field(default=None)
    tokenizer: Optional[TokenizerConfig] = Field(default=None)
    model: Optional[ModelConfig] = Field(default=None)
    train: Optional[TrainConfig] = Field(default=None)
    eval: Optional[EvalConfig] = Field(default=None)
    runtime: Optional[RuntimeConfig] = Field(default=None)
