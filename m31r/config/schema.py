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


class DatasetConfig(BaseModel):
    """Controls data acquisition, filtering, deduplication. Maps to configs/dataset.yaml."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    config_version: str = Field(description="Schema version")


class TokenizerConfig(BaseModel):
    """Controls tokenizer training and encoding. Maps to configs/tokenizer.yaml."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    config_version: str = Field(description="Schema version")


class ModelConfig(BaseModel):
    """Neural architecture definitions. Maps to configs/model.yaml."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    config_version: str = Field(description="Schema version")


class TrainConfig(BaseModel):
    """Training hyperparameters and schedule. Maps to configs/train.yaml."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    config_version: str = Field(description="Schema version")


class EvalConfig(BaseModel):
    """Evaluation benchmarks and metrics. Maps to configs/eval.yaml."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    config_version: str = Field(description="Schema version")


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
