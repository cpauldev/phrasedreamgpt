from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import torch
import torch.nn as nn

SECTION_TITLE_OVERRIDES = {
    "dreamphrasegpt": "DreamPhraseGPT",
}


def fail(message: str, hint: str | None = None) -> NoReturn:
    if hint:
        raise SystemExit(f"{message}\nHow to fix it: {hint}")
    raise SystemExit(message)


def ensure_utf8_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def format_section_title(title: str) -> str:
    normalized = title.strip()
    override = SECTION_TITLE_OVERRIDES.get(normalized.lower())
    if override is not None:
        return override
    if normalized and normalized == normalized.lower():
        return normalized.title()
    return normalized


def print_section(title: str) -> None:
    print(f"\n--- {format_section_title(title)} ---")


@dataclass(frozen=True)
class Dataset:
    data: torch.Tensor
    id_to_char: list[str]
    bos_id: int
    vocab_size: int


@dataclass(frozen=True)
class PrecisionSettings:
    amp_dtype: torch.dtype | None
    use_amp: bool


@dataclass(frozen=True)
class RuntimeSettings:
    requested_device: str
    resolved_device: str
    requested_dtype: str
    amp_requested: bool | None
    amp_enabled: bool
    amp_dtype: str | None
    compile_requested: bool | None
    compile_enabled: bool


@dataclass(frozen=True)
class TrainingResult:
    model: nn.Module
    elapsed: float
    total_tokens: int
    tok_s: float
    steps_s: float
    final_loss: float
    completed_steps: int
    optimizer_state: dict
    scaler_state: dict | None
    runtime: RuntimeSettings


@dataclass(frozen=True)
class ArtifactPaths:
    directory: Path
    resume: Path
    model: Path
    js_bundle: Path


ARTIFACT_SCHEMA_HINT = "Use a file created by this script, or retrain and save a new one."
RESUME_ARTIFACT_HINT = (
    "Resume from a saved model that still has its resume companion data, "
    "or load the model from the interactive models menu for generation only."
)
MODEL_MLP_TYPE = "swiglu"
JS_BUNDLE_MAGIC = b"PDBGONNX"
JS_BUNDLE_FORMAT = "dreamphrasegpt-onnx-bundle"
JS_BUNDLE_VERSION = 1


def artifact_subject(label: str, artifact_path: Path | None = None) -> str:
    if artifact_path is None:
        return label
    return f"{label} ({artifact_path})"


def swiglu_hidden_dim(n_embd: int) -> int:
    # Match the parameter budget of a classic 4x FFN while using three projections.
    return max(1, (8 * n_embd) // 3)


def require_positive_integer_field(
    value: object,
    *,
    field_name: str,
    label: str,
    hint: str,
    artifact_path: Path | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(
            (
                f"{artifact_subject(label, artifact_path)} field '{field_name}' "
                "must be a positive integer."
            ),
            hint,
        )
    if value <= 0:
        fail(
            (
                f"{artifact_subject(label, artifact_path)} field '{field_name}' "
                "must be greater than 0."
            ),
            hint,
        )
    return value


def require_literal_string_field(
    value: object,
    *,
    field_name: str,
    expected: str,
    label: str,
    hint: str,
    artifact_path: Path | None = None,
) -> str:
    if not isinstance(value, str) or value != expected:
        fail(
            f"{artifact_subject(label, artifact_path)} field '{field_name}' must be '{expected}'.",
            hint,
        )
    return value


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_embd: int
    n_head: int
    mlp_type: str
    mlp_hidden_dim: int

    @classmethod
    def from_dimensions(
        cls,
        *,
        vocab_size: int,
        block_size: int,
        n_layer: int,
        n_embd: int,
        n_head: int,
    ) -> ModelConfig:
        config = cls(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=n_layer,
            n_embd=n_embd,
            n_head=n_head,
            mlp_type=MODEL_MLP_TYPE,
            mlp_hidden_dim=swiglu_hidden_dim(n_embd),
        )
        config.validate(
            label="Model config",
            hint="Choose values such as n_embd 128 and n_head 4.",
        )
        return config

    @classmethod
    def from_mapping(
        cls,
        mapping: dict,
        *,
        label: str = "Artifact model_config",
        hint: str | None = None,
        artifact_path: Path | None = None,
    ) -> ModelConfig:
        if hint is None:
            hint = ARTIFACT_SCHEMA_HINT
        config = cls(
            vocab_size=require_positive_integer_field(
                mapping.get("vocab_size"),
                field_name="vocab_size",
                label=label,
                hint=hint,
                artifact_path=artifact_path,
            ),
            block_size=require_positive_integer_field(
                mapping.get("block_size"),
                field_name="block_size",
                label=label,
                hint=hint,
                artifact_path=artifact_path,
            ),
            n_layer=require_positive_integer_field(
                mapping.get("n_layer"),
                field_name="n_layer",
                label=label,
                hint=hint,
                artifact_path=artifact_path,
            ),
            n_embd=require_positive_integer_field(
                mapping.get("n_embd"),
                field_name="n_embd",
                label=label,
                hint=hint,
                artifact_path=artifact_path,
            ),
            n_head=require_positive_integer_field(
                mapping.get("n_head"),
                field_name="n_head",
                label=label,
                hint=hint,
                artifact_path=artifact_path,
            ),
            mlp_type=require_literal_string_field(
                mapping.get("mlp_type"),
                field_name="mlp_type",
                expected=MODEL_MLP_TYPE,
                label=label,
                hint=hint,
                artifact_path=artifact_path,
            ),
            mlp_hidden_dim=require_positive_integer_field(
                mapping.get("mlp_hidden_dim"),
                field_name="mlp_hidden_dim",
                label=label,
                hint=hint,
                artifact_path=artifact_path,
            ),
        )
        config.validate(label=label, hint=hint, artifact_path=artifact_path)
        return config

    def validate(
        self,
        *,
        label: str,
        hint: str,
        artifact_path: Path | None = None,
    ) -> None:
        if self.n_embd % self.n_head != 0:
            fail(
                (
                    f"{artifact_subject(label, artifact_path)} fields 'n_embd' and "
                    "'n_head' must divide evenly."
                ),
                hint,
            )
        expected_hidden_dim = swiglu_hidden_dim(self.n_embd)
        if self.mlp_hidden_dim != expected_hidden_dim:
            fail(
                (
                    f"{artifact_subject(label, artifact_path)} field "
                    f"'mlp_hidden_dim' must be {expected_hidden_dim}."
                ),
                hint,
            )

    def to_artifact_dict(self) -> dict[str, int | str]:
        return {
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "n_layer": self.n_layer,
            "n_embd": self.n_embd,
            "n_head": self.n_head,
            "mlp_type": self.mlp_type,
            "mlp_hidden_dim": self.mlp_hidden_dim,
        }


@dataclass(frozen=True)
class TrainingConfig:
    dataset_path: str | None
    seed: int
    steps: int
    batch_size: int
    model: ModelConfig
    learning_rate: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    requested_device: str
    requested_dtype: str
    amp_requested: bool | None
    compile_requested: bool | None
    print_every: int

    def validate(self) -> None:
        if self.steps <= 0:
            fail("--steps must be greater than 0.", "Pass a positive integer to --steps.")
        if self.batch_size <= 0:
            fail("--batch-size must be greater than 0.", "Pass a positive integer to --batch-size.")
        if self.print_every <= 0:
            fail(
                "--print-every must be greater than 0.",
                "Pass a positive integer to --print-every.",
            )
        if self.learning_rate <= 0:
            fail("--learning-rate must be greater than 0.", "Use a positive value such as 3e-4.")
        if self.eps <= 0:
            fail("--eps must be greater than 0.", "Use a positive value such as 1e-8.")
        self.model.validate(
            label="Training model_config",
            hint="Choose values such as n_embd 128 and n_head 4.",
        )


@dataclass(frozen=True)
class GenerationConfig:
    num_samples: int
    temperature: float
    requested_block_size: int

    def validate(self) -> None:
        if self.num_samples < 0:
            fail("--samples cannot be negative.", "Use 0 or a positive integer for --samples.")
        if self.temperature <= 0:
            fail("--temperature must be greater than 0.", "Use a small positive value such as 0.8.")
        if self.requested_block_size <= 0:
            fail("--block-size must be greater than 0.", "Pass a positive integer to --block-size.")


@dataclass(frozen=True)
class ArtifactRuntimePolicy:
    load_device: torch.device
    target_device: torch.device
    include_training_state: bool

    @classmethod
    def for_inference(cls, target_device: torch.device) -> ArtifactRuntimePolicy:
        return cls(
            load_device=torch.device("cpu"),
            target_device=target_device,
            include_training_state=False,
        )

    @classmethod
    def for_resume(cls, target_device: torch.device) -> ArtifactRuntimePolicy:
        return cls(
            load_device=torch.device("cpu"),
            target_device=target_device,
            include_training_state=True,
        )
