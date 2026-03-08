from __future__ import annotations

import io
import json
import os
import shutil
import struct
import sys
import tempfile
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from .config import (
    ARTIFACT_SCHEMA_HINT,
    JS_BUNDLE_FORMAT,
    JS_BUNDLE_MAGIC,
    JS_BUNDLE_VERSION,
    RESUME_ARTIFACT_HINT,
    ArtifactPaths,
    ArtifactRuntimePolicy,
    Dataset,
    ModelConfig,
    RuntimeSettings,
    TrainingConfig,
    artifact_subject,
    ensure_utf8_stdio,
    fail,
    print_section,
)
from .runtime import build_model, capture_rng_state, unwrap_model


@dataclass(frozen=True)
class ArtifactBundle:
    artifact_path: Path
    raw_artifact: dict
    model_config: ModelConfig
    dataset: Dataset
    state_dict: dict
    artifact_type: str
    training_metadata: dict
    resume_state: dict
    optimizer_state: dict
    scaler_state: dict | None
    rng_state: dict


def export_onnx_quietly(*args, **kwargs) -> None:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            torch.onnx.export(*args, **kwargs)
    except Exception:
        stdout_text = stdout_buffer.getvalue().strip()
        stderr_text = stderr_buffer.getvalue().strip()
        if stdout_text:
            print(stdout_text)
        if stderr_text:
            print(stderr_text, file=sys.stderr)
        raise


ARTIFACT_EXTENSIONS = frozenset({".pt", ".pth"})
ARTIFACT_SUFFIXES = (
    (".resume.pt", "resume", ".pt"),
    (".resume.pth", "resume", ".pth"),
    (".checkpoint.pt", "resume", ".pt"),
    (".checkpoint.pth", "resume", ".pth"),
    (".model.pt", "model", ".pt"),
    (".model.pth", "model", ".pth"),
)
MODEL_ARTIFACT_KEYS = frozenset({"model_config", "tokenizer", "state_dict"})
RESUME_ARTIFACT_KEYS = frozenset(
    {"training_config", "dataset_data", "optimizer_state", "resume_state", "rng_state"}
)
MODEL_CONFIG_KEYS = frozenset(
    {"vocab_size", "block_size", "n_layer", "n_embd", "n_head", "mlp_type", "mlp_hidden_dim"}
)
TOKENIZER_KEYS = frozenset({"id_to_char", "bos_id", "vocab_size"})
EXACT_RESUME_ARTIFACT_KEYS = frozenset(
    {"dataset_data", "optimizer_state", "resume_state", "rng_state"}
)
RESUME_TRAINING_CONFIG_KEYS = frozenset(
    {"batch_size", "learning_rate", "beta1", "beta2", "eps", "weight_decay"}
)
ARTIFACT_TYPE_ALIASES = {
    "checkpoint": "resume",
    "training": "resume",
    "resume": "resume",
    "model": "model",
    "inference": "model",
}


@dataclass(frozen=True)
class ArtifactSpec:
    path: Path
    base_path: Path
    extension: str
    explicit_type: str | None

    @property
    def has_tensor_extension(self) -> bool:
        return self.path.suffix in ARTIFACT_EXTENSIONS

    @property
    def artifact_directory(self) -> Path:
        if self.explicit_type is None and not self.has_tensor_extension:
            return self.path
        return self.path.parent

    @property
    def artifact_stem(self) -> str:
        if self.explicit_type is None and not self.has_tensor_extension:
            return self.path.name
        return self.base_path.name

    @property
    def paired_paths(self) -> ArtifactPaths:
        return build_artifact_paths(self.artifact_directory, self.artifact_stem, self.extension)

    def save_paths(self) -> ArtifactPaths:
        paired_paths = self.paired_paths
        if self.explicit_type == "resume":
            return ArtifactPaths(
                directory=paired_paths.directory,
                resume=self.path,
                model=paired_paths.model,
                js_bundle=paired_paths.js_bundle,
            )
        if self.explicit_type == "model":
            return ArtifactPaths(
                directory=paired_paths.directory,
                resume=paired_paths.resume,
                model=self.path,
                js_bundle=paired_paths.js_bundle,
            )
        if self.has_tensor_extension:
            return ArtifactPaths(
                directory=self.path.parent,
                resume=self.path,
                model=paired_paths.model,
                js_bundle=paired_paths.js_bundle,
            )
        return paired_paths


def build_artifact_paths(run_dir: Path, stem: str, extension: str = ".pt") -> ArtifactPaths:
    return ArtifactPaths(
        directory=run_dir,
        resume=run_dir / f"{stem}.resume{extension}",
        model=run_dir / f"{stem}.model{extension}",
        js_bundle=run_dir / f"{stem}.model",
    )


def build_staged_artifact_paths(final_paths: ArtifactPaths, staging_dir: Path) -> ArtifactPaths:
    return ArtifactPaths(
        directory=staging_dir,
        resume=staging_dir / final_paths.resume.name,
        model=staging_dir / final_paths.model.name,
        js_bundle=staging_dir / final_paths.js_bundle.name,
    )


def legacy_resume_path(final_paths: ArtifactPaths) -> Path:
    return final_paths.resume.with_name(
        final_paths.resume.name.replace(".resume", ".checkpoint", 1)
    )


def allowed_artifact_names(final_paths: ArtifactPaths) -> set[str]:
    return {
        final_paths.model.name,
        final_paths.resume.name,
        legacy_resume_path(final_paths).name,
        final_paths.js_bundle.name,
    }


def create_staging_directory(final_dir: Path) -> Path:
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.staging-", dir=final_dir.parent))


def ensure_artifact_directory_safe(final_paths: ArtifactPaths) -> None:
    final_dir = final_paths.directory
    if not final_dir.exists():
        return

    if not final_dir.is_dir():
        fail(
            f"Save destination exists but is not a directory: {final_dir}",
            "Choose a different save path.",
        )

    unexpected_entries = [
        entry.name
        for entry in final_dir.iterdir()
        if entry.name not in allowed_artifact_names(final_paths)
    ]
    if unexpected_entries:
        fail(
            f"Refusing to overwrite non-artifact files in {final_dir}.",
            ("Move or remove these files first: " + ", ".join(sorted(unexpected_entries))),
        )


def commit_staged_directory(staging_dir: Path, final_dir: Path) -> None:
    backup_dir: Path | None = None
    final_dir = final_dir.resolve()
    staging_dir = staging_dir.resolve()

    try:
        if final_dir.exists():
            backup_dir = final_dir.parent / f".{final_dir.name}.backup-{time.time_ns()}"
            final_dir.rename(backup_dir)

        staging_dir.rename(final_dir)
    except Exception as exc:
        if backup_dir is not None and backup_dir.exists() and not final_dir.exists():
            try:
                backup_dir.rename(final_dir)
            except OSError:
                pass
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        fail(
            f"Failed to save artifact set into {final_dir}.",
            f"Retry the save. Original error: {exc}",
        )

    if backup_dir is not None and backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)


def describe_artifact_path(path: Path) -> ArtifactSpec:
    name = path.name
    for suffix, artifact_type, extension in ARTIFACT_SUFFIXES:
        if name.endswith(suffix):
            base_name = name[: -len(suffix)]
            return ArtifactSpec(
                path=path,
                base_path=path.with_name(base_name),
                extension=extension,
                explicit_type=artifact_type,
            )

    if path.suffix in ARTIFACT_EXTENSIONS:
        return ArtifactSpec(
            path=path,
            base_path=path.with_suffix(""),
            extension=path.suffix,
            explicit_type=None,
        )

    return ArtifactSpec(
        path=path,
        base_path=path,
        extension=".pt",
        explicit_type=None,
    )


def resume_companion_candidates(path: Path) -> list[Path]:
    artifact_spec = describe_artifact_path(path)
    return [
        artifact_spec.artifact_directory
        / f"{artifact_spec.artifact_stem}.resume{artifact_spec.extension}",
        artifact_spec.artifact_directory
        / f"{artifact_spec.artifact_stem}.checkpoint{artifact_spec.extension}",
    ]


def find_existing_resume_companion(path: Path) -> Path | None:
    for candidate in resume_companion_candidates(path):
        if candidate.exists():
            return candidate
    return None


def model_companion_path(path: Path) -> Path:
    artifact_spec = describe_artifact_path(path)
    return (
        artifact_spec.artifact_directory
        / f"{artifact_spec.artifact_stem}.model{artifact_spec.extension}"
    )


def related_artifact_paths(path: Path) -> list[Path]:
    artifact_spec = describe_artifact_path(path)
    related_paths: list[Path] = []

    if artifact_spec.explicit_type == "model":
        related_paths.append(path)
        resume_path = find_existing_resume_companion(path)
        if resume_path is not None:
            related_paths.append(resume_path)
        js_bundle_path = artifact_spec.paired_paths.js_bundle
        if js_bundle_path.exists():
            related_paths.append(js_bundle_path)
        return related_paths

    related_paths.append(path)
    model_path = model_companion_path(path)
    if model_path.exists():
        related_paths.append(model_path)
    js_bundle_path = artifact_spec.paired_paths.js_bundle
    if js_bundle_path.exists():
        related_paths.append(js_bundle_path)
    return related_paths


def should_list_artifact_path(path: Path) -> bool:
    artifact_spec = describe_artifact_path(path)
    if artifact_spec.explicit_type == "model":
        return True
    if artifact_spec.explicit_type == "resume":
        return not model_companion_path(path).exists()
    return path.suffix in ARTIFACT_EXTENSIONS


def list_artifact_files(models_dir: Path) -> list[Path]:
    if not models_dir.exists():
        return []
    return sorted(
        (
            path
            for path in models_dir.rglob("*")
            if path.is_file()
            and path.suffix in ARTIFACT_EXTENSIONS
            and not any(part.startswith(".") for part in path.relative_to(models_dir).parts)
            and should_list_artifact_path(path)
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def format_file_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024.0
    return f"{int(num_bytes)} B"


def format_display_path(path: Path, root_dir: Path) -> str:
    try:
        return str(path.relative_to(root_dir))
    except ValueError:
        return str(path)


def is_standard_run_directory_name(directory_name: str, artifact_stem: str) -> bool:
    if directory_name == artifact_stem:
        return True
    if not directory_name.startswith(f"{artifact_stem}_"):
        return False
    suffix = directory_name.removeprefix(f"{artifact_stem}_")
    return suffix.isdigit() and int(suffix) >= 2


def format_artifact_display_name(path: Path, root_dir: Path) -> str:
    artifact_spec = describe_artifact_path(path)
    expected_model_path = artifact_spec.paired_paths.model

    try:
        relative_directory = artifact_spec.artifact_directory.relative_to(root_dir)
    except ValueError:
        return format_display_path(path, root_dir)

    if (
        artifact_spec.explicit_type == "model"
        and path == expected_model_path
        and is_standard_run_directory_name(
            artifact_spec.artifact_directory.name,
            artifact_spec.artifact_stem,
        )
    ):
        return str(relative_directory)

    return format_display_path(path, root_dir)


def next_available_artifact_directory(parent_dir: Path, base_name: str) -> Path:
    candidate = parent_dir / base_name
    if not candidate.exists():
        return candidate

    suffix = 2
    while True:
        candidate = parent_dir / f"{base_name}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def default_artifact_directory(models_dir: Path, input_stem: str) -> Path:
    return next_available_artifact_directory(models_dir, input_stem)


def resolve_output_path_arg(path_arg: str, default_dir: Path) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    if path.parent == Path("."):
        return default_dir / path
    return path


def resolve_save_paths(
    save_model_arg: str | None, models_dir: Path, input_stem: str = "dreamphrasegpt"
) -> ArtifactPaths | None:
    if save_model_arg is None:
        return None
    if save_model_arg == "auto":
        return build_artifact_paths(default_artifact_directory(models_dir, input_stem), input_stem)

    output_path = resolve_output_path_arg(save_model_arg, models_dir)
    artifact_spec = describe_artifact_path(output_path)

    if output_path.parent == models_dir:
        run_dir = next_available_artifact_directory(models_dir, artifact_spec.base_path.name)
        return build_artifact_paths(run_dir, artifact_spec.base_path.name)

    if artifact_spec.explicit_type is None and not artifact_spec.has_tensor_extension:
        return build_artifact_paths(output_path, output_path.name)

    return artifact_spec.save_paths()


def resolve_resume_save_paths(
    save_model_arg: str | None,
    source_artifact_path: Path,
    models_dir: Path,
    input_stem: str = "dreamphrasegpt",
) -> ArtifactPaths | None:
    if save_model_arg is None:
        return None
    if save_model_arg == "auto":
        return describe_artifact_path(source_artifact_path).save_paths()

    resolved_paths = resolve_save_paths(save_model_arg, models_dir, input_stem)
    if resolved_paths is None:
        fail("Internal error: resume save paths were not resolved.", "Retry the command.")
    return resolved_paths


def require_artifact_mapping(
    value: object,
    *,
    label: str,
    hint: str,
    artifact_path: Path | None = None,
) -> dict:
    if not isinstance(value, dict):
        fail(f"{artifact_subject(label, artifact_path)} must be a mapping.", hint)
    return value


def require_artifact_keys(
    mapping: dict,
    required_keys: frozenset[str],
    *,
    label: str,
    hint: str,
    artifact_path: Path | None = None,
) -> None:
    missing_keys = sorted(required_keys - set(mapping))
    if missing_keys:
        fail(
            (
                f"{artifact_subject(label, artifact_path)} is missing required "
                f"fields: {', '.join(missing_keys)}"
            ),
            hint,
        )


def load_raw_artifact_file(artifact_path: Path, device: torch.device) -> dict:
    if not artifact_path.exists():
        fail(
            f"Artifact file not found: {artifact_path}",
            (
                "Pass an existing saved artifact path from models/, or use the "
                "interactive models menu to browse saved files."
            ),
        )

    try:
        artifact = torch.load(artifact_path, map_location=device, weights_only=False)
    except Exception as exc:
        fail(
            f"Failed to load artifact file: {artifact_path}",
            f"Make sure the file is a valid artifact created by this script. Original error: {exc}",
        )

    return require_artifact_mapping(
        artifact,
        label="Artifact file",
        hint=ARTIFACT_SCHEMA_HINT,
        artifact_path=artifact_path,
    )


def parse_artifact_model_config(
    artifact: dict, *, artifact_path: Path | None = None
) -> ModelConfig:
    model_config_mapping = require_artifact_mapping(
        artifact["model_config"],
        label="Artifact model_config",
        hint=ARTIFACT_SCHEMA_HINT,
        artifact_path=artifact_path,
    )
    require_artifact_keys(
        model_config_mapping,
        MODEL_CONFIG_KEYS,
        label="Artifact model_config",
        hint=ARTIFACT_SCHEMA_HINT,
        artifact_path=artifact_path,
    )
    return ModelConfig.from_mapping(model_config_mapping, artifact_path=artifact_path)


def validate_model_artifact(
    artifact: dict, *, artifact_path: Path | None = None, hint: str = ARTIFACT_SCHEMA_HINT
) -> None:
    require_artifact_keys(
        artifact,
        MODEL_ARTIFACT_KEYS,
        label="Artifact file",
        hint=hint,
        artifact_path=artifact_path,
    )
    parse_artifact_model_config(artifact, artifact_path=artifact_path)
    require_artifact_keys(
        require_artifact_mapping(
            artifact["tokenizer"],
            label="Artifact tokenizer",
            hint=hint,
            artifact_path=artifact_path,
        ),
        TOKENIZER_KEYS,
        label="Artifact tokenizer",
        hint=hint,
        artifact_path=artifact_path,
    )
    require_artifact_mapping(
        artifact["state_dict"],
        label="Artifact state_dict",
        hint=hint,
        artifact_path=artifact_path,
    )


def validate_resume_artifact(
    artifact: dict, *, artifact_path: Path | None = None, hint: str = RESUME_ARTIFACT_HINT
) -> None:
    require_artifact_keys(
        artifact,
        RESUME_ARTIFACT_KEYS,
        label="Resume data",
        hint=hint,
        artifact_path=artifact_path,
    )
    require_artifact_mapping(
        artifact.get("resume_state"),
        label="Resume state",
        hint=hint,
        artifact_path=artifact_path,
    )
    resolve_resume_training_metadata(artifact, artifact_path or Path("<resume>"))


def merge_model_and_resume_artifacts(
    model_artifact: dict,
    resume_artifact: dict,
    *,
    model_path: Path | None = None,
    resume_path: Path | None = None,
) -> dict:
    validate_model_artifact(model_artifact, artifact_path=model_path)
    validate_resume_artifact(resume_artifact, artifact_path=resume_path)
    merged_artifact = dict(model_artifact)
    for key in (
        "training_config",
        "dataset_data",
        "optimizer_state",
        "scaler_state",
        "resume_state",
        "rng_state",
    ):
        if key in resume_artifact:
            merged_artifact[key] = resume_artifact[key]
    if merged_artifact.get("created_at") is None and resume_artifact.get("created_at") is not None:
        merged_artifact["created_at"] = resume_artifact["created_at"]
    if resume_artifact.get("source_model"):
        merged_artifact["source_model"] = resume_artifact["source_model"]
    if resume_path is not None:
        merged_artifact["resume_data_path"] = str(resume_path)
    return merged_artifact


def artifact_supports_exact_resume(artifact: dict) -> bool:
    return EXACT_RESUME_ARTIFACT_KEYS.issubset(artifact)


def describe_artifact_type(artifact: dict) -> str:
    checkpoint_type = ARTIFACT_TYPE_ALIASES.get(artifact.get("checkpoint_type"))
    if checkpoint_type is not None:
        return checkpoint_type
    if artifact_supports_exact_resume(artifact):
        return "resume"
    if MODEL_ARTIFACT_KEYS.issubset(artifact):
        return "model"
    return "unknown"


def resolve_resume_training_metadata(artifact: dict, artifact_path: Path) -> dict:
    training_config = require_artifact_mapping(
        artifact.get("training_config"),
        label="Resume training_config",
        hint=RESUME_ARTIFACT_HINT,
        artifact_path=artifact_path,
    )
    require_artifact_keys(
        training_config,
        RESUME_TRAINING_CONFIG_KEYS,
        label="Resume training_config",
        hint=RESUME_ARTIFACT_HINT,
        artifact_path=artifact_path,
    )
    return training_config


def require_exact_resume_artifact(artifact: dict, artifact_path: Path) -> None:
    require_artifact_keys(
        artifact,
        EXACT_RESUME_ARTIFACT_KEYS,
        label="Resume data",
        hint=RESUME_ARTIFACT_HINT,
        artifact_path=artifact_path,
    )
    require_artifact_mapping(
        artifact.get("resume_state"),
        label="Resume state",
        hint=RESUME_ARTIFACT_HINT,
        artifact_path=artifact_path,
    )
    resolve_resume_training_metadata(artifact, artifact_path)


def dataset_from_artifact(artifact: dict) -> Dataset:
    tokenizer = artifact["tokenizer"]
    dataset_data = artifact.get("dataset_data")
    if dataset_data is None:
        dataset_data = torch.empty(0, dtype=torch.long)
    return Dataset(
        data=dataset_data.to(dtype=torch.long, device="cpu"),
        id_to_char=list(tokenizer["id_to_char"]),
        bos_id=int(tokenizer["bos_id"]),
        vocab_size=int(tokenizer["vocab_size"]),
    )


def build_artifact_bundle(artifact_path: Path, artifact: dict) -> ArtifactBundle:
    return ArtifactBundle(
        artifact_path=artifact_path,
        raw_artifact=artifact,
        model_config=parse_artifact_model_config(artifact, artifact_path=artifact_path),
        dataset=dataset_from_artifact(artifact),
        state_dict=require_artifact_mapping(
            artifact["state_dict"],
            label="Artifact state_dict",
            hint=ARTIFACT_SCHEMA_HINT,
            artifact_path=artifact_path,
        ),
        artifact_type=describe_artifact_type(artifact),
        training_metadata=artifact.get("training_config", {}),
        resume_state=artifact.get("resume_state", {}),
        optimizer_state=artifact.get("optimizer_state", {}),
        scaler_state=artifact.get("scaler_state"),
        rng_state=artifact.get("rng_state", {}),
    )


def _load_inference_artifact(artifact_path: Path, device: torch.device) -> dict:
    artifact = load_raw_artifact_file(artifact_path, device)
    has_model_payload = MODEL_ARTIFACT_KEYS.issubset(artifact)
    has_resume_payload = RESUME_ARTIFACT_KEYS.issubset(artifact)

    if has_model_payload:
        validate_model_artifact(artifact, artifact_path=artifact_path)
        return artifact

    if has_resume_payload:
        model_path = model_companion_path(artifact_path)
        model_artifact = load_raw_artifact_file(model_path, device)
        validate_model_artifact(model_artifact, artifact_path=model_path)
        return model_artifact

    fail(
        f"Artifact file is missing required model fields: {artifact_path}",
        ARTIFACT_SCHEMA_HINT,
    )


def _load_resumable_artifact(artifact_path: Path, device: torch.device) -> dict:
    artifact = load_raw_artifact_file(artifact_path, device)
    artifact_type = ARTIFACT_TYPE_ALIASES.get(artifact.get("checkpoint_type"))

    has_model_payload = MODEL_ARTIFACT_KEYS.issubset(artifact)
    has_resume_payload = RESUME_ARTIFACT_KEYS.issubset(artifact)

    if artifact_type == "resume" and not has_model_payload:
        model_path = model_companion_path(artifact_path)
        model_artifact = load_raw_artifact_file(model_path, device)
        return merge_model_and_resume_artifacts(
            model_artifact,
            artifact,
            model_path=model_path,
            resume_path=artifact_path,
        )

    if has_model_payload and has_resume_payload:
        validate_model_artifact(artifact, artifact_path=artifact_path)
        validate_resume_artifact(artifact, artifact_path=artifact_path)
        return artifact

    if has_model_payload:
        validate_model_artifact(artifact, artifact_path=artifact_path)
        resume_path = find_existing_resume_companion(artifact_path)
        if resume_path is None or resume_path.resolve() == artifact_path.resolve():
            return artifact
        try:
            resume_artifact = load_raw_artifact_file(resume_path, device)
            return merge_model_and_resume_artifacts(
                artifact,
                resume_artifact,
                model_path=artifact_path,
                resume_path=resume_path,
            )
        except SystemExit:
            return artifact

    if has_resume_payload:
        model_path = model_companion_path(artifact_path)
        model_artifact = load_raw_artifact_file(model_path, device)
        return merge_model_and_resume_artifacts(
            model_artifact,
            artifact,
            model_path=model_path,
            resume_path=artifact_path,
        )

    fail(
        f"Artifact file is missing required model fields: {artifact_path}",
        ARTIFACT_SCHEMA_HINT,
    )


def load_artifact_bundle(artifact_path: Path, policy: ArtifactRuntimePolicy) -> ArtifactBundle:
    if policy.include_training_state:
        artifact = _load_resumable_artifact(artifact_path, policy.load_device)
    else:
        artifact = _load_inference_artifact(artifact_path, policy.load_device)
    return build_artifact_bundle(artifact_path, artifact)


def artifact_path_supports_resume(path: Path) -> bool:
    artifact_spec = describe_artifact_path(path)
    if artifact_spec.explicit_type == "model":
        return find_existing_resume_companion(path) is not None

    try:
        bundle = load_artifact_bundle(path, ArtifactRuntimePolicy.for_resume(torch.device("cpu")))
    except SystemExit:
        return False
    return artifact_supports_exact_resume(bundle.raw_artifact)


def infer_artifact_type_from_path(path: Path) -> str:
    artifact_spec = describe_artifact_path(path)
    if artifact_spec.explicit_type == "model":
        return "resumable" if artifact_path_supports_resume(path) else "model"
    if artifact_spec.explicit_type == "resume":
        return "resume"

    try:
        bundle = load_artifact_bundle(path, ArtifactRuntimePolicy.for_resume(torch.device("cpu")))
    except SystemExit:
        return "unknown"
    if artifact_path_supports_resume(path) and bundle.artifact_type == "model":
        return "resumable"
    return bundle.artifact_type


def print_available_artifacts(models_dir: Path) -> None:
    artifacts = list_artifact_files(models_dir)
    if not artifacts:
        print(f"no saved model files found in {models_dir}")
        return

    rows = []
    for path in artifacts:
        stat = path.stat()
        rows.append(
            (
                format_artifact_display_name(path, models_dir),
                infer_artifact_type_from_path(path),
                datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                format_file_size(stat.st_size),
            )
        )

    index_w = len(str(len(rows)))
    name_w = max(len(r[0]) for r in rows + [("run", "", "", "")])
    type_w = max(len(r[1]) for r in rows + [("", "type", "", "")])
    size_w = max(len(r[3]) for r in rows + [("", "", "", "size")])

    pad = " " * (index_w + 2)
    print(f"{pad}{'run':{name_w}}  {'type':{type_w}}  {'modified':{19}}  {'size':>{size_w}}")
    print(f"{pad}{'-' * name_w}  {'-' * type_w}  {'-' * 19}  {'-' * size_w}")
    for index, (name, artifact_type, modified_at, size) in enumerate(rows, start=1):
        print(
            f"{index:{index_w}d}. {name:{name_w}}  "
            f"{artifact_type:{type_w}}  {modified_at}  {size:>{size_w}}"
        )


def build_training_checkpoint(
    model: nn.Module,
    optimizer_state: dict,
    scaler_state: dict | None,
    dataset: Dataset,
    training_config: TrainingConfig,
    runtime: RuntimeSettings,
    completed_steps: int,
    total_tokens: int,
    final_loss: float,
) -> dict:
    raw_model = unwrap_model(model)
    return {
        "format_version": 1,
        "checkpoint_type": "resume",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_config": raw_model.config.to_artifact_dict(),
        "tokenizer": {
            "id_to_char": dataset.id_to_char,
            "bos_id": dataset.bos_id,
            "vocab_size": dataset.vocab_size,
        },
        "dataset_data": dataset.data.cpu(),
        "training_config": {
            "input_path": training_config.dataset_path,
            "seed": training_config.seed,
            "steps": completed_steps,
            "last_run_steps": training_config.steps,
            "batch_size": training_config.batch_size,
            "learning_rate": training_config.learning_rate,
            "beta1": training_config.beta1,
            "beta2": training_config.beta2,
            "eps": training_config.eps,
            "weight_decay": training_config.weight_decay,
            "device": training_config.requested_device,
            "dtype": training_config.requested_dtype,
            "amp": training_config.amp_requested,
            "compile": training_config.compile_requested,
            "requested_device": runtime.requested_device,
            "resolved_device": runtime.resolved_device,
            "requested_dtype": runtime.requested_dtype,
            "amp_requested": runtime.amp_requested,
            "amp_enabled": runtime.amp_enabled,
            "amp_dtype": runtime.amp_dtype,
            "compile_requested": runtime.compile_requested,
            "compile_enabled": runtime.compile_enabled,
        },
        "resume_state": {
            "completed_steps": completed_steps,
            "total_tokens": total_tokens,
            "final_loss": final_loss,
        },
        "state_dict": raw_model.state_dict(),
        "optimizer_state": optimizer_state,
        "scaler_state": scaler_state,
        "rng_state": capture_rng_state(),
    }


def save_artifact_file(artifact: dict, artifact_path: Path) -> Path:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, artifact_path)
    return artifact_path


def build_model_artifact(source_checkpoint: dict, source_path: Path | None = None) -> dict:
    model_artifact = {
        "format_version": source_checkpoint.get("format_version", 1),
        "checkpoint_type": "model",
        "created_at": source_checkpoint.get(
            "created_at", datetime.now().isoformat(timespec="seconds")
        ),
        "model_config": source_checkpoint["model_config"],
        "tokenizer": source_checkpoint["tokenizer"],
        "state_dict": source_checkpoint["state_dict"],
    }
    if source_path is not None:
        model_artifact["source_resume"] = str(source_path)
    return model_artifact


def build_resume_artifact(source_checkpoint: dict, source_path: Path | None = None) -> dict:
    resume_artifact = {
        "format_version": source_checkpoint.get("format_version", 1),
        "checkpoint_type": "resume",
        "created_at": source_checkpoint.get(
            "created_at", datetime.now().isoformat(timespec="seconds")
        ),
        "training_config": source_checkpoint["training_config"],
        "dataset_data": source_checkpoint["dataset_data"],
        "optimizer_state": source_checkpoint["optimizer_state"],
        "scaler_state": source_checkpoint.get("scaler_state"),
        "resume_state": source_checkpoint["resume_state"],
        "rng_state": source_checkpoint["rng_state"],
    }
    if source_path is not None:
        resume_artifact["source_model"] = str(source_path)
    return resume_artifact


def build_js_bundle_bytes(
    *, onnx_bytes: bytes, tokenizer: dict[str, object], source_path: Path
) -> bytes:
    header = {
        "format": JS_BUNDLE_FORMAT,
        "version": JS_BUNDLE_VERSION,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "source_artifact": str(source_path),
        "tokenizer": tokenizer,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return JS_BUNDLE_MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + onnx_bytes


def export_js_model_bundle_from_model(
    model: nn.Module,
    dataset: Dataset,
    output_path: Path,
    *,
    source_path: Path,
    model_config: ModelConfig,
) -> Path:
    ensure_utf8_stdio()
    raw_model = build_model(
        model_config,
        unwrap_model(model).state_dict(),
        torch.device("cpu"),
    )
    raw_model.eval()

    seq_len = torch.export.Dim("seq_len", min=1, max=model_config.block_size)
    dummy = torch.zeros(1, 1, dtype=torch.long)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as handle:
        temp_onnx_path = Path(handle.name)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"`isinstance\(treespec, LeafSpec\)` is deprecated",
                category=FutureWarning,
            )
            export_onnx_quietly(
                raw_model,
                dummy,
                str(temp_onnx_path),
                input_names=["idx"],
                output_names=["logits"],
                external_data=False,
                dynamic_shapes={"idx": {1: seq_len}},
                opset_version=18,
            )

        bundle_bytes = build_js_bundle_bytes(
            onnx_bytes=temp_onnx_path.read_bytes(),
            tokenizer={
                "id_to_char": list(dataset.id_to_char),
                "bos_id": int(dataset.bos_id),
                "vocab_size": int(dataset.vocab_size),
                "block_size": model_config.block_size,
            },
            source_path=source_path,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(bundle_bytes)
    finally:
        temp_onnx_path.unlink(missing_ok=True)

    return output_path


def export_js_model_bundle(
    source_path: Path, output_path: Path, *, source_artifact_path: Path | None = None
) -> Path:
    bundle = load_artifact_bundle(
        source_path,
        ArtifactRuntimePolicy.for_inference(torch.device("cpu")),
    )
    model = build_model(bundle.model_config, bundle.state_dict, torch.device("cpu"))
    return export_js_model_bundle_from_model(
        model,
        bundle.dataset,
        output_path,
        source_path=source_artifact_path or source_path,
        model_config=bundle.model_config,
    )


def save_artifact_set(
    model: nn.Module,
    optimizer_state: dict,
    scaler_state: dict | None,
    dataset: Dataset,
    training_config: TrainingConfig,
    runtime: RuntimeSettings,
    completed_steps: int,
    total_tokens: int,
    final_loss: float,
    artifact_paths: ArtifactPaths,
) -> ArtifactPaths:
    checkpoint = build_training_checkpoint(
        model,
        optimizer_state,
        scaler_state,
        dataset,
        training_config,
        runtime,
        completed_steps,
        total_tokens,
        final_loss,
    )
    ensure_artifact_directory_safe(artifact_paths)
    staging_dir = create_staging_directory(artifact_paths.directory)
    staged_paths = build_staged_artifact_paths(artifact_paths, staging_dir)

    try:
        save_artifact_file(
            build_model_artifact(checkpoint, source_path=artifact_paths.resume),
            staged_paths.model,
        )
        save_artifact_file(
            build_resume_artifact(checkpoint, source_path=artifact_paths.model),
            staged_paths.resume,
        )
        export_js_model_bundle_from_model(
            model,
            dataset,
            staged_paths.js_bundle,
            source_path=artifact_paths.model,
            model_config=unwrap_model(model).config,
        )
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    commit_staged_directory(staging_dir, artifact_paths.directory)
    return artifact_paths


def resolve_resume_training_config(
    user_training_config: TrainingConfig, bundle: ArtifactBundle
) -> TrainingConfig:
    training_metadata = resolve_resume_training_metadata(bundle.raw_artifact, bundle.artifact_path)
    return TrainingConfig(
        dataset_path=training_metadata.get("input_path", user_training_config.dataset_path),
        seed=user_training_config.seed,
        steps=user_training_config.steps,
        batch_size=int(training_metadata["batch_size"]),
        model=bundle.model_config,
        learning_rate=float(training_metadata["learning_rate"]),
        beta1=float(training_metadata["beta1"]),
        beta2=float(training_metadata["beta2"]),
        eps=float(training_metadata["eps"]),
        weight_decay=float(training_metadata["weight_decay"]),
        requested_device=user_training_config.requested_device,
        requested_dtype=training_metadata.get(
            "requested_dtype",
            training_metadata.get("dtype", user_training_config.requested_dtype),
        ),
        amp_requested=training_metadata.get(
            "amp_requested", training_metadata.get("amp", user_training_config.amp_requested)
        ),
        compile_requested=training_metadata.get(
            "compile_requested",
            training_metadata.get("compile", user_training_config.compile_requested),
        ),
        print_every=user_training_config.print_every,
    )


def _print_row(label: str, value: object, width: int) -> None:
    print(f"{label:{width}}  {value}")


def _print_artifact_file_section(
    artifact_path: Path, bundle: ArtifactBundle, stat: os.stat_result
) -> None:
    print_section("inspect")
    width = len("modified")
    _print_row("path", artifact_path, width)
    _print_row("size", format_file_size(stat.st_size), width)
    _print_row(
        "modified",
        datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        width,
    )
    if bundle.raw_artifact.get("created_at"):
        _print_row("created", bundle.raw_artifact["created_at"], width)
    if bundle.raw_artifact.get("exported_at"):
        _print_row("exported", bundle.raw_artifact["exported_at"], width)
    _print_row("type", bundle.artifact_type, width)
    _print_row("version", bundle.raw_artifact.get("format_version", "unknown"), width)
    if (
        bundle.raw_artifact.get("source_resume")
        and Path(str(bundle.raw_artifact["source_resume"])).exists()
    ):
        _print_row("source", bundle.raw_artifact["source_resume"], width)
    if bundle.raw_artifact.get("resume_data_path"):
        _print_row("resume data", bundle.raw_artifact["resume_data_path"], width)


def _print_artifact_model_section(bundle: ArtifactBundle) -> None:
    print_section("model")
    width = len("mlp hidden")
    param_count = sum(t.numel() for t in bundle.state_dict.values())
    _print_row("params", f"{param_count:,}", width)
    _print_row("vocab", bundle.model_config.vocab_size, width)
    _print_row("block", bundle.model_config.block_size, width)
    _print_row("layers", bundle.model_config.n_layer, width)
    _print_row("embd", bundle.model_config.n_embd, width)
    _print_row("heads", bundle.model_config.n_head, width)
    _print_row("mlp type", bundle.model_config.mlp_type, width)
    _print_row("mlp hidden", bundle.model_config.mlp_hidden_dim, width)


def _print_artifact_tokenizer_section(bundle: ArtifactBundle) -> None:
    print_section("tokenizer")
    width = len("bos id")
    tokenizer = bundle.raw_artifact["tokenizer"]
    _print_row("vocab", f"{tokenizer['vocab_size']} chars", width)
    _print_row("bos id", tokenizer["bos_id"], width)


def _print_artifact_training_sections(bundle: ArtifactBundle) -> None:
    training_config = bundle.training_metadata
    if not training_config:
        return

    print_section("training")
    width = len("weight decay")
    if training_config.get("input_path"):
        _print_row("dataset", training_config["input_path"], width)
    total = training_config.get("steps")
    last = training_config.get("last_run_steps", total)
    steps_str = (
        f"{total:,} total  ({last:,} last run)"
        if isinstance(total, int) and isinstance(last, int)
        else f"{total}  ({last} last run)"
    )
    _print_row("steps", steps_str, width)
    _print_row("seed", training_config.get("seed", "unknown"), width)
    _print_row("batch", training_config.get("batch_size", "unknown"), width)
    _print_row("lr", training_config.get("learning_rate", "unknown"), width)
    _print_row("beta1", training_config.get("beta1", "unknown"), width)
    _print_row("beta2", training_config.get("beta2", "unknown"), width)
    _print_row("eps", training_config.get("eps", "unknown"), width)
    _print_row("weight decay", training_config.get("weight_decay", "unknown"), width)
    dataset_data = bundle.raw_artifact.get("dataset_data")
    if dataset_data is not None and hasattr(dataset_data, "numel"):
        _print_row("stored tokens", f"{dataset_data.numel():,}", width)

    print_section("runtime")
    width = len("compile")
    req_device = training_config.get("requested_device", training_config.get("device", "unknown"))
    eff_device = training_config.get("resolved_device")
    device_str = eff_device or req_device
    if eff_device is not None and eff_device != req_device:
        device_str = f"{eff_device}  ({req_device})"
    _print_row("device", device_str, width)

    req_amp = training_config.get("amp_requested", training_config.get("amp", "unknown"))
    amp_enabled = training_config.get("amp_enabled")
    amp_dtype = training_config.get("amp_dtype")
    if amp_enabled is not None:
        amp_str = "on" if amp_enabled else "off"
        if amp_dtype:
            amp_str += f"  {amp_dtype}"
        amp_str += f"  (requested {req_amp})"
    else:
        amp_str = str(req_amp)
    _print_row("amp", amp_str, width)

    req_compile = training_config.get(
        "compile_requested", training_config.get("compile", "unknown")
    )
    compile_enabled = training_config.get("compile_enabled")
    if compile_enabled is not None:
        compile_str = f"{'on' if compile_enabled else 'off'}  (requested {req_compile})"
    else:
        compile_str = str(req_compile)
    _print_row("compile", compile_str, width)


def _print_artifact_resume_section(bundle: ArtifactBundle) -> None:
    if not bundle.resume_state:
        return

    print_section("resume")
    width = len("final loss")
    completed = bundle.resume_state.get("completed_steps")
    _print_row(
        "completed",
        f"{completed:,} steps" if isinstance(completed, int) else "unknown",
        width,
    )
    tokens = bundle.resume_state.get("total_tokens")
    _print_row("tokens", f"{tokens:,}" if isinstance(tokens, int) else "unknown", width)
    final_loss = bundle.resume_state.get("final_loss")
    _print_row(
        "final loss",
        f"{final_loss:.4f}" if isinstance(final_loss, float) else "unknown",
        width,
    )


def _print_artifact_summary_section(bundle: ArtifactBundle) -> None:
    print_section("artifact")
    width = len("resumable")
    _print_row(
        "resumable",
        "yes" if artifact_supports_exact_resume(bundle.raw_artifact) else "no",
        width,
    )
    _print_row("tensors", len(bundle.state_dict), width)


def print_artifact_details(artifact_path: Path) -> None:
    bundle = load_artifact_bundle(
        artifact_path, ArtifactRuntimePolicy.for_resume(torch.device("cpu"))
    )
    stat = artifact_path.stat()
    _print_artifact_file_section(artifact_path, bundle, stat)
    _print_artifact_model_section(bundle)
    _print_artifact_tokenizer_section(bundle)
    _print_artifact_training_sections(bundle)
    _print_artifact_resume_section(bundle)
    _print_artifact_summary_section(bundle)


def delete_artifact_file(artifact_path: Path) -> None:
    deleted_paths: list[Path] = []
    for path in related_artifact_paths(artifact_path):
        if not path.exists():
            continue
        try:
            path.unlink()
        except OSError as exc:
            fail(
                f"Failed to delete artifact: {path}",
                f"Close any program using the file and try again. Original error: {exc}",
            )
        deleted_paths.append(path)

    if not deleted_paths:
        print(f"artifact already deleted: {artifact_path}")
        return

    print("deleted artifacts:")
    for path in deleted_paths:
        print(f"  {path}")
