"""
PhraseDreamGPT: train a character-level GPT model from newline-delimited text.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import random
import shutil
import struct
import sys
import tempfile
import time
import warnings
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NoReturn

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required.\n"
        "Install it using the official selector:\n"
        "  https://pytorch.org/get-started/locally/\n"
        "Choose the wheel that matches your OS, Python version, and accelerator."
    ) from exc


if not hasattr(F, "scaled_dot_product_attention"):
    raise SystemExit(
        "This script requires a PyTorch build that provides "
        "`torch.nn.functional.scaled_dot_product_attention`.\n"
        "Upgrade PyTorch and try again."
    )


def fail(message: str, hint: str | None = None) -> NoReturn:
    if hint:
        raise SystemExit(f"{message}\nHow to fix it: {hint}")
    raise SystemExit(message)


def ensure_utf8_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


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
JS_BUNDLE_FORMAT = "phrasedreamgpt-onnx-bundle"
JS_BUNDLE_VERSION = 1


def artifact_subject(label: str, artifact_path: Path | None = None) -> str:
    if artifact_path is None:
        return label
    return f"{label} ({artifact_path})"


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
            (
                "The previous saved run was left untouched. "
                f"Retry the save after fixing the underlying error. Original error: {exc}"
            ),
        )

    if backup_dir is not None and backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)


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
    if not isinstance(value, str):
        fail(
            f"{artifact_subject(label, artifact_path)} field '{field_name}' must be '{expected}'.",
            hint,
        )
    if value != expected:
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
        return cls(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=n_layer,
            n_embd=n_embd,
            n_head=n_head,
            mlp_type=MODEL_MLP_TYPE,
            mlp_hidden_dim=swiglu_hidden_dim(n_embd),
        )

    @classmethod
    def from_training_args(cls, args: argparse.Namespace, vocab_size: int) -> ModelConfig:
        return cls.from_dimensions(
            vocab_size=vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_embd=args.n_embd,
            n_head=args.n_head,
        )

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

    def validate(self, *, label: str, hint: str, artifact_path: Path | None = None) -> None:
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PhraseDreamGPT on character data.")
    parser.add_argument(
        "--dataset",
        default=None,
        metavar="PATH",
        help=(
            "Path to a .txt dataset file. Bare file names resolve inside datasets/. "
            "When omitted, auto-selects from the datasets/ directory."
        ),
    )
    parser.add_argument("--steps", type=int, default=3000, help="Training steps.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--block-size", type=int, default=32, help="Context length.")
    parser.add_argument("--n-layer", type=int, default=4, help="Transformer depth.")
    parser.add_argument("--n-embd", type=int, default=128, help="Embedding width.")
    parser.add_argument("--n-head", type=int, default=4, help="Attention heads.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Base learning rate.")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1.")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2.")
    parser.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16"],
        help="Autocast dtype on CUDA.",
    )
    parser.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        default=None,
        help="Force-enable automatic mixed precision on CUDA (strict if unavailable).",
    )
    parser.add_argument(
        "--no-amp", dest="amp", action="store_false", help="Disable automatic mixed precision."
    )
    parser.add_argument(
        "--compile",
        dest="compile",
        action="store_true",
        default=None,
        help="Force-enable torch.compile (strict; errors if unavailable).",
    )
    parser.add_argument(
        "--no-compile", dest="compile", action="store_false", help="Disable torch.compile."
    )
    parser.add_argument("--print-every", type=int, default=50, help="Progress print interval.")
    parser.add_argument(
        "--samples", type=int, default=20, help="How many samples to generate after training."
    )
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument(
        "--output",
        dest="save",
        default="auto",
        metavar="PATH",
        help=(
            "Override the automatic save location. Bare names resolve inside models/. "
            "Use --no-save to disable writing artifacts."
        ),
    )
    parser.add_argument(
        "--no-save",
        dest="save",
        action="store_const",
        const=None,
        help="Disable automatic saving after training or resume.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run short benchmark on CPU and the detected accelerator (CUDA or MPS).",
    )
    parser.add_argument(
        "--compare-steps", type=int, default=400, help="Steps per device in compare mode."
    )
    parser.add_argument(
        "--no-generate", action="store_true", help="Skip post-training text generation."
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help=(
            "Open the interactive model manager for loading, resuming, "
            "inspecting, or deleting saved artifacts."
        ),
    )
    return parser


def parse_args() -> argparse.Namespace:
    args = build_arg_parser().parse_args()
    validate_args(args)
    return args


def require_positive(value: int, flag: str) -> None:
    if value <= 0:
        fail(f"{flag} must be greater than 0.", f"Pass a positive integer to {flag}.")


def require_non_negative(value: int, flag: str) -> None:
    if value < 0:
        fail(f"{flag} cannot be negative.", f"Use 0 or a positive integer for {flag}.")


def validate_args(args: argparse.Namespace) -> None:
    require_positive(args.steps, "--steps")
    require_positive(args.batch_size, "--batch-size")
    require_positive(args.block_size, "--block-size")
    require_positive(args.n_layer, "--n-layer")
    require_positive(args.n_embd, "--n-embd")
    require_positive(args.n_head, "--n-head")
    require_positive(args.print_every, "--print-every")
    require_positive(args.compare_steps, "--compare-steps")
    require_non_negative(args.samples, "--samples")

    if args.n_embd % args.n_head != 0:
        fail(
            "--n-embd must be divisible by --n-head.",
            "Choose values such as --n-embd 128 --n-head 4.",
        )

    if args.temperature <= 0:
        fail(
            "--temperature must be greater than 0.",
            "Use a small positive value such as 0.8 or 0.5.",
        )

    artifact_operations = [("--models", args.models)]
    active_artifact_operations = [name for name, active in artifact_operations if active]

    if len(active_artifact_operations) > 1:
        fail(
            "Only one artifact operation can be used at a time.",
            f"Choose one of: {', '.join(active_artifact_operations)}.",
        )

    if args.compare and active_artifact_operations:
        fail(
            "--compare cannot be combined with artifact management commands.",
            "Run compare mode by itself, or run the artifact operation separately.",
        )

    explicit_save_path = args.save not in {None, "auto"}

    if args.compare and explicit_save_path:
        fail(
            "--compare cannot be combined with --output.",
            "Run compare mode by itself, or remove --output.",
        )

    if explicit_save_path and active_artifact_operations:
        fail(
            "--output cannot be combined with artifact management commands.",
            "Train a model to save it, or run the artifact operation separately.",
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def has_mps() -> bool:
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            fail(
                "--device cuda was requested but CUDA is not available.",
                (
                    "Install a CUDA-enabled PyTorch build, verify "
                    "`torch.cuda.is_available()`, or rerun with --device cpu."
                ),
            )
        return torch.device("cuda")

    if device_arg == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_built():
            fail(
                "--device mps was requested but this PyTorch build does not include MPS support.",
                "Install a PyTorch build with MPS support or rerun with --device cpu.",
            )
        if not torch.backends.mps.is_available():
            fail(
                "--device mps was requested but MPS is unavailable on this machine.",
                "Use an Apple Silicon machine with MPS enabled, or rerun with --device cpu.",
            )
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if has_mps():
        return torch.device("mps")
    return torch.device("cpu")


def load_dataset(path: str) -> Dataset:
    input_path = Path(path)
    if not input_path.exists():
        fail(
            f"Dataset file not found: {input_path}",
            "Pass a valid path with --dataset PATH.",
        )

    try:
        with input_path.open("r", encoding="utf-8") as handle:
            docs = [line.strip() for line in handle if line.strip()]
    except UnicodeDecodeError as exc:
        fail(
            f"Failed to decode input file as UTF-8: {input_path}",
            f"Save the dataset as UTF-8 text and try again. Original error: {exc}",
        )
    except OSError as exc:
        fail(
            f"Failed to read input file: {input_path}",
            f"Check that the file exists and is readable. Original error: {exc}",
        )

    if not docs:
        fail(
            f"Input file has no non-empty lines: {input_path}",
            "Add at least one non-empty line to the dataset file.",
        )

    random.shuffle(docs)

    id_to_char = sorted(set("".join(docs)))
    char_to_id = {ch: i for i, ch in enumerate(id_to_char)}
    bos_id = len(id_to_char)
    vocab_size = bos_id + 1

    stream = [bos_id]
    for doc in docs:
        stream.extend(char_to_id[ch] for ch in doc)
        stream.append(bos_id)

    data = torch.tensor(stream, dtype=torch.long)
    print_section("dataset")
    print(f"file    {Path(path).name}")
    print(f"docs    {len(docs):,}")
    print(f"vocab   {vocab_size} chars")
    print(f"tokens  {data.numel():,}")
    return Dataset(data=data, id_to_char=id_to_char, bos_id=bos_id, vocab_size=vocab_size)


def ensure_dataset_supports_block_size(dataset: Dataset, block_size: int) -> None:
    minimum_tokens = block_size + 2
    if dataset.data.numel() < minimum_tokens:
        fail(
            f"The dataset is too small for --block-size {block_size}.",
            (
                "Reduce --block-size or provide more input data. This setup needs "
                f"at least {minimum_tokens} tokens after tokenization."
            ),
        )


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


def should_list_artifact_path(path: Path) -> bool:
    artifact_spec = describe_artifact_path(path)
    if artifact_spec.explicit_type == "model":
        return True
    if artifact_spec.explicit_type == "resume":
        return not model_companion_path(path).exists()
    return path.suffix in ARTIFACT_EXTENSIONS


def artifact_path_supports_resume(path: Path) -> bool:
    artifact_spec = describe_artifact_path(path)
    if artifact_spec.explicit_type == "model":
        return find_existing_resume_companion(path) is not None

    try:
        artifact = load_artifact_file(path, torch.device("cpu"))
    except SystemExit:
        return False
    return artifact_supports_exact_resume(artifact)


def infer_artifact_type_from_path(path: Path) -> str:
    artifact_spec = describe_artifact_path(path)
    if artifact_spec.explicit_type == "model":
        return "resumable" if artifact_path_supports_resume(path) else "model"
    if artifact_spec.explicit_type == "resume":
        return "resume"

    try:
        artifact = load_artifact_file(path, torch.device("cpu"))
    except SystemExit:
        return "unknown"
    if artifact_path_supports_resume(path) and describe_artifact_type(artifact) == "model":
        return "resumable"
    return describe_artifact_type(artifact)


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


def resolve_save_paths(
    save_model_arg: str | None, models_dir: Path, input_stem: str = "phrasedreamgpt"
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
    input_stem: str = "phrasedreamgpt",
) -> ArtifactPaths | None:
    if save_model_arg is None:
        return None
    if save_model_arg == "auto":
        return describe_artifact_path(source_artifact_path).save_paths()

    resolved_paths = resolve_save_paths(save_model_arg, models_dir, input_stem)
    if resolved_paths is None:
        fail("Internal error: resume save paths were not resolved.", "Retry the command.")
    return resolved_paths


def list_input_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix == ".txt")


DATASETS_DIR = Path("datasets")
MODELS_DIR = Path("models")


def resolve_existing_path_arg(path_arg: str, default_dir: Path) -> Path:
    path = Path(path_arg)
    if path.is_absolute() or path.exists():
        return path
    if path.parent == Path("."):
        return default_dir / path
    return path


def format_display_path(path: Path, root_dir: Path) -> str:
    try:
        return str(path.relative_to(root_dir))
    except ValueError:
        return str(path)


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
        and artifact_spec.artifact_directory.name == artifact_spec.artifact_stem
    ):
        return str(relative_directory)

    return format_display_path(path, root_dir)


def resolve_output_path_arg(path_arg: str, default_dir: Path) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    if path.parent == Path("."):
        return default_dir / path
    return path


def select_dataset(dataset_arg: str | None) -> Path:
    if dataset_arg is not None:
        path = resolve_existing_path_arg(dataset_arg, DATASETS_DIR)
        if not path.exists():
            fail(
                f"Dataset file not found: {path}",
                (
                    "Pass a valid path with --dataset PATH. "
                    "Bare file names are resolved inside datasets/."
                ),
            )
        return path

    candidates = list_input_files(DATASETS_DIR)

    if not candidates:
        fail(
            f"No .txt dataset files found in {DATASETS_DIR}.",
            (
                "Add a .txt file with one sample per line to datasets/, "
                "or pass --dataset PATH to specify a file explicitly."
            ),
        )

    if len(candidates) == 1:
        print(f"dataset  {candidates[0].name}")
        return candidates[0]

    if not sys.stdin.isatty():
        fail(
            f"Multiple .txt files found in {DATASETS_DIR} but stdin is not a terminal.",
            "Pass --dataset PATH to specify a dataset file explicitly.",
        )

    print_section("dataset")
    for index, path in enumerate(candidates, start=1):
        print(f"{index}  {path.name}")
    print()

    while True:
        selection = prompt_user("select: ").lower()
        if selection in {"q", "quit", "exit"}:
            raise SystemExit(0)
        if not selection.isdigit():
            print("enter a valid number")
            continue
        index = int(selection)
        if not 1 <= index <= len(candidates):
            print(f"selection must be between 1 and {len(candidates)}")
            continue
        return candidates[index - 1]


def unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def prompt_user(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return "q"


def prompt_with_default(label: str, default: str, choices: str = "") -> str:
    hint = f"  ({choices})" if choices else ""
    response = prompt_user(f"{label}{hint}  [{default}]: ")
    return response.strip() if response.strip() else default


def prompt_positive_int(label: str, default: int, choices: str = "") -> int:
    while True:
        value = prompt_with_default(label, str(default), choices)
        try:
            n = int(value)
            if n > 0:
                return n
        except ValueError:
            pass
        print(f"{label.strip()} must be a positive integer")


def prompt_non_negative_int(label: str, default: int) -> int:
    while True:
        value = prompt_with_default(label, str(default))
        try:
            n = int(value)
            if n >= 0:
                return n
        except ValueError:
            pass
        print(f"{label.strip()} must be 0 or a positive integer")


def prompt_positive_float(label: str, default: float) -> float:
    while True:
        value = prompt_with_default(label, str(default))
        try:
            n = float(value)
            if n > 0:
                return n
        except ValueError:
            pass
        print(f"{label.strip()} must be a positive number")


def prompt_bool(label: str, default: bool = False) -> bool:
    default_str = "y" if default else "n"
    while True:
        value = prompt_with_default(label, default_str, "y/n").lower()
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print(f"{label.strip()}: enter y or n")


def dtype_name(dtype: torch.dtype | None) -> str | None:
    if dtype is None:
        return None
    return str(dtype).replace("torch.", "")


def is_compiled_model(model: nn.Module) -> bool:
    return hasattr(model, "_orig_mod")


def capture_rng_state() -> dict:
    state = {
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "get_rng_state")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        state["mps_rng_state"] = torch.mps.get_rng_state()
    return state


def as_cpu_bytetensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def restore_rng_state(rng_state: dict) -> None:
    if "python_random_state" in rng_state:
        random.setstate(rng_state["python_random_state"])
    if "torch_rng_state" in rng_state:
        torch.set_rng_state(as_cpu_bytetensor(rng_state["torch_rng_state"]))

    cuda_rng_state_all = rng_state.get("cuda_rng_state_all")
    if cuda_rng_state_all is not None:
        if not torch.cuda.is_available():
            fail(
                "This checkpoint requires CUDA RNG state restoration, but CUDA is not available.",
                (
                    "Resume on a CUDA-enabled machine or load the model from "
                    "the interactive models menu for generation only."
                ),
            )
        torch.cuda.set_rng_state_all([as_cpu_bytetensor(state) for state in cuda_rng_state_all])

    mps_rng_state = rng_state.get("mps_rng_state")
    if mps_rng_state is not None:
        if not (
            hasattr(torch, "mps")
            and hasattr(torch.mps, "set_rng_state")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            fail(
                "This checkpoint requires MPS RNG state restoration, but MPS is not available.",
                (
                    "Resume on a machine with MPS support or load the model "
                    "from the interactive models menu for generation only."
                ),
            )
        torch.mps.set_rng_state(as_cpu_bytetensor(mps_rng_state))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(channels, dim=2)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.c_proj(y)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, n_embd: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config.n_embd, config.n_head)
        self.norm2 = RMSNorm(config.n_embd)
        self.feed_forward = SwiGLUFeedForward(config.n_embd, config.mlp_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.08)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block_size {self.block_size}")

        positions = torch.arange(0, seq_len, device=idx.device)
        x = self.wte(idx) + self.wpe(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.lm_head(x)


def build_training_checkpoint(
    model: nn.Module,
    optimizer_state: dict,
    scaler_state: dict | None,
    dataset: Dataset,
    args: argparse.Namespace,
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
            "input_path": args.dataset,
            "seed": args.seed,
            "steps": completed_steps,
            "last_run_steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "eps": args.eps,
            "weight_decay": args.weight_decay,
            "device": args.device,
            "dtype": args.dtype,
            "amp": args.amp,
            "compile": args.compile,
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


def save_artifact_set(
    model: nn.Module,
    optimizer_state: dict,
    scaler_state: dict | None,
    dataset: Dataset,
    args: argparse.Namespace,
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
        args,
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
        export_js_model_bundle(
            staged_paths.model,
            staged_paths.js_bundle,
            source_artifact_path=artifact_paths.model,
        )
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    commit_staged_directory(staging_dir, artifact_paths.directory)
    return artifact_paths


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
    resolve_resume_training_config(artifact, artifact_path or Path("<resume>"))


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
    return ModelConfig.from_mapping(
        model_config_mapping,
        artifact_path=artifact_path,
    )


def load_artifact_file(artifact_path: Path, device: torch.device) -> dict:
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
            if MODEL_ARTIFACT_KEYS.issubset(resume_artifact) and RESUME_ARTIFACT_KEYS.issubset(
                resume_artifact
            ):
                validate_resume_artifact(resume_artifact, artifact_path=resume_path)
                return merge_model_and_resume_artifacts(
                    artifact,
                    resume_artifact,
                    model_path=artifact_path,
                    resume_path=resume_path,
                )
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


def resolve_resume_training_config(artifact: dict, artifact_path: Path) -> dict:
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
    resolve_resume_training_config(artifact, artifact_path)


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


def build_model_from_artifact(artifact: dict, device: torch.device) -> GPT:
    model = GPT(parse_artifact_model_config(artifact)).to(device)

    try:
        model.load_state_dict(artifact["state_dict"])
    except Exception as exc:
        fail(
            "Checkpoint weights could not be loaded into the model.",
            f"The checkpoint may be incompatible or corrupted. Original error: {exc}",
        )

    return model


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


def export_js_model_bundle(
    source_path: Path, output_path: Path, *, source_artifact_path: Path | None = None
) -> Path:
    ensure_utf8_stdio()
    artifact = load_artifact_file(source_path, torch.device("cpu"))
    model = build_model_from_artifact(artifact, torch.device("cpu"))
    model.eval()

    seq_len = torch.export.Dim(
        "seq_len",
        min=1,
        max=parse_artifact_model_config(artifact, artifact_path=source_path).block_size,
    )
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
                model,
                dummy,
                str(temp_onnx_path),
                input_names=["idx"],
                output_names=["logits"],
                external_data=False,
                dynamic_shapes={"idx": {1: seq_len}},
                opset_version=18,
            )

        tokenizer = artifact["tokenizer"]
        bundle_bytes = build_js_bundle_bytes(
            onnx_bytes=temp_onnx_path.read_bytes(),
            tokenizer={
                "id_to_char": list(tokenizer["id_to_char"]),
                "bos_id": int(tokenizer["bos_id"]),
                "vocab_size": int(tokenizer["vocab_size"]),
                "block_size": parse_artifact_model_config(
                    artifact, artifact_path=source_path
                ).block_size,
            },
            source_path=source_artifact_path or source_path,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(bundle_bytes)
    finally:
        temp_onnx_path.unlink(missing_ok=True)

    return output_path


def resolve_resume_args(
    user_args: argparse.Namespace, artifact: dict, artifact_path: Path
) -> argparse.Namespace:
    training_config = resolve_resume_training_config(artifact, artifact_path)
    model_config = parse_artifact_model_config(artifact, artifact_path=artifact_path)
    resumed_args = argparse.Namespace(**vars(user_args))

    resumed_args.block_size = model_config.block_size
    resumed_args.n_layer = model_config.n_layer
    resumed_args.n_embd = model_config.n_embd
    resumed_args.n_head = model_config.n_head

    resumed_args.batch_size = int(training_config["batch_size"])
    resumed_args.learning_rate = float(training_config["learning_rate"])
    resumed_args.beta1 = float(training_config["beta1"])
    resumed_args.beta2 = float(training_config["beta2"])
    resumed_args.eps = float(training_config["eps"])
    resumed_args.weight_decay = float(training_config["weight_decay"])
    resumed_args.dtype = training_config.get(
        "requested_dtype", training_config.get("dtype", user_args.dtype)
    )
    resumed_args.amp = training_config.get(
        "amp_requested", training_config.get("amp", user_args.amp)
    )
    resumed_args.compile = training_config.get(
        "compile_requested", training_config.get("compile", user_args.compile)
    )
    return resumed_args


def verify_resume_runtime(artifact: dict, runtime: RuntimeSettings) -> None:
    training_config = artifact.get("training_config", {})
    expected_device = training_config.get("resolved_device")
    expected_amp_enabled = training_config.get("amp_enabled")
    expected_amp_dtype = training_config.get("amp_dtype")
    expected_compile_enabled = training_config.get("compile_enabled")

    if expected_device is not None and runtime.resolved_device != expected_device:
        fail(
            (
                "Exact resume requires the same execution backend that was used "
                "when the checkpoint was created."
            ),
            (
                f"Resume on {expected_device} or load the artifact from the "
                "models menu for generation "
                f"only. Current device resolved to {runtime.resolved_device}."
            ),
        )

    if expected_amp_enabled is not None and runtime.amp_enabled != expected_amp_enabled:
        fail(
            (
                "Exact resume requires the same AMP setting that was active when "
                "the checkpoint was created."
            ),
            (
                "Resume with the original environment/settings, or start a new "
                "training run instead of exact resume."
            ),
        )

    if expected_amp_dtype is not None and runtime.amp_dtype != expected_amp_dtype:
        fail(
            (
                "Exact resume requires the same AMP dtype that was active when "
                "the checkpoint was created."
            ),
            (
                "Resume with the original environment/settings. Expected "
                f"{expected_amp_dtype}, got {runtime.amp_dtype}."
            ),
        )

    if expected_compile_enabled is not None and runtime.compile_enabled != expected_compile_enabled:
        fail(
            (
                "Exact resume requires the same compile setting that was active "
                "when the checkpoint was created."
            ),
            (
                "Resume with the original environment/settings, or start a new "
                "training run instead of exact resume."
            ),
        )


def print_artifact_details(artifact_path: Path) -> None:
    artifact = load_artifact_file(artifact_path, torch.device("cpu"))
    stat = artifact_path.stat()
    model_config = parse_artifact_model_config(artifact, artifact_path=artifact_path)
    tokenizer = artifact["tokenizer"]
    training_config = artifact.get("training_config", {})
    resume_state = artifact.get("resume_state", {})
    param_count = sum(t.numel() for t in artifact["state_dict"].values())

    def row(label: str, value: object, w: int) -> None:
        print(f"{label:{w}}  {value}")

    # File
    print_section("inspect")
    w = len("modified")
    row("path", artifact_path, w)
    row("size", format_file_size(stat.st_size), w)
    row("modified", datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"), w)
    if artifact.get("created_at"):
        row("created", artifact["created_at"], w)
    if artifact.get("exported_at"):
        row("exported", artifact["exported_at"], w)
    row("type", describe_artifact_type(artifact), w)
    row("version", artifact.get("format_version", "unknown"), w)
    if artifact.get("source_resume") and Path(str(artifact["source_resume"])).exists():
        row("source", artifact["source_resume"], w)
    if artifact.get("resume_data_path"):
        row("resume data", artifact["resume_data_path"], w)

    # Model
    print_section("model")
    w = len("mlp hidden")
    row("params", f"{param_count:,}", w)
    row("vocab", model_config.vocab_size, w)
    row("block", model_config.block_size, w)
    row("layers", model_config.n_layer, w)
    row("embd", model_config.n_embd, w)
    row("heads", model_config.n_head, w)
    row("mlp type", model_config.mlp_type, w)
    row("mlp hidden", model_config.mlp_hidden_dim, w)

    # Tokenizer
    print_section("tokenizer")
    w = len("bos id")
    row("vocab", f"{tokenizer['vocab_size']} chars", w)
    row("bos id", tokenizer["bos_id"], w)

    # Training
    if training_config:
        print_section("training")
        w = len("weight decay")
        if training_config.get("input_path"):
            row("dataset", training_config["input_path"], w)
        total = training_config.get("steps")
        last = training_config.get("last_run_steps", total)
        steps_str = (
            f"{total:,} total  ({last:,} last run)"
            if isinstance(total, int) and isinstance(last, int)
            else f"{total}  ({last} last run)"
        )
        row("steps", steps_str, w)
        row("seed", training_config.get("seed", "unknown"), w)
        row("batch", training_config.get("batch_size", "unknown"), w)
        row("lr", training_config.get("learning_rate", "unknown"), w)
        row("beta1", training_config.get("beta1", "unknown"), w)
        row("beta2", training_config.get("beta2", "unknown"), w)
        row("eps", training_config.get("eps", "unknown"), w)
        row("weight decay", training_config.get("weight_decay", "unknown"), w)
        dataset_data = artifact.get("dataset_data")
        if dataset_data is not None and hasattr(dataset_data, "numel"):
            row("stored tokens", f"{dataset_data.numel():,}", w)

        # Runtime
        print_section("runtime")
        w = len("compile")
        req_device = training_config.get(
            "requested_device", training_config.get("device", "unknown")
        )
        eff_device = training_config.get("resolved_device")
        if eff_device is None or eff_device == req_device:
            device_str = eff_device or req_device
        else:
            device_str = f"{eff_device}  ({req_device})"
        row("device", device_str, w)

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
        row("amp", amp_str, w)

        req_compile = training_config.get(
            "compile_requested", training_config.get("compile", "unknown")
        )
        compile_enabled = training_config.get("compile_enabled")
        if compile_enabled is not None:
            compile_str = f"{'on' if compile_enabled else 'off'}  (requested {req_compile})"
        else:
            compile_str = str(req_compile)
        row("compile", compile_str, w)

    # Resume state
    if resume_state:
        print_section("resume")
        w = len("final loss")
        completed = resume_state.get("completed_steps")
        row("completed", f"{completed:,} steps" if isinstance(completed, int) else "unknown", w)
        tokens = resume_state.get("total_tokens")
        row("tokens", f"{tokens:,}" if isinstance(tokens, int) else "unknown", w)
        final_loss = resume_state.get("final_loss")
        row("final loss", f"{final_loss:.4f}" if isinstance(final_loss, float) else "unknown", w)

    # Summary
    print_section("artifact")
    w = len("resumable")
    row("resumable", "yes" if artifact_supports_exact_resume(artifact) else "no", w)
    row("tensors", len(artifact["state_dict"]), w)


class BatchProvider:
    def __init__(self, data: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
        self.data = data.to(device)
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.offsets = torch.arange(block_size, device=device)

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = self.data.numel() - self.block_size - 1
        indices = torch.randint(0, max_start, (self.batch_size,), device=self.device)
        positions = indices[:, None] + self.offsets[None, :]
        x = self.data[positions]
        y = self.data[positions + 1]
        return x, y


def resolve_amp_settings(
    amp_pref: bool | None, dtype_arg: str, device: torch.device
) -> PrecisionSettings:
    if dtype_arg != "auto" and device.type != "cuda":
        fail(
            (
                f"--dtype {dtype_arg} was requested but non-auto dtype selection "
                "requires --device cuda."
            ),
            "Use --device cuda for fp16/bf16 selection, or switch back to --dtype auto.",
        )

    if amp_pref is True and device.type != "cuda":
        fail(
            "--amp was requested but AMP is only supported with --device cuda in this script.",
            "Use --device cuda or rerun with --no-amp.",
        )

    if device.type != "cuda":
        return PrecisionSettings(amp_dtype=None, use_amp=False)

    if dtype_arg == "fp32":
        if amp_pref is True:
            fail(
                "--amp with --dtype fp32 is invalid.",
                "Use --dtype fp16, --dtype bf16, or --dtype auto when AMP is enabled.",
            )
        return PrecisionSettings(amp_dtype=None, use_amp=False)

    if dtype_arg == "bf16":
        if not torch.cuda.is_bf16_supported():
            fail(
                "--dtype bf16 was requested but this GPU/PyTorch build does not support bfloat16.",
                "Use --dtype fp16, --dtype auto, or rerun with --no-amp.",
            )
        amp_dtype = torch.bfloat16
    elif dtype_arg == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if amp_pref is False:
        return PrecisionSettings(amp_dtype=amp_dtype, use_amp=False)

    return PrecisionSettings(amp_dtype=amp_dtype, use_amp=True)


def make_optimizer(
    model: nn.Module, args: argparse.Namespace, device: torch.device
) -> torch.optim.Optimizer:
    kwargs = {
        "lr": args.learning_rate,
        "betas": (args.beta1, args.beta2),
        "eps": args.eps,
        "weight_decay": args.weight_decay,
    }
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
        except TypeError:
            pass
    return torch.optim.AdamW(model.parameters(), **kwargs)


def has_triton() -> bool:
    return importlib.util.find_spec("triton") is not None


def triton_install_hint() -> str:
    if sys.platform == "win32":
        return (
            "Install triton-windows (it provides the `triton` module on Windows), "
            "or rerun with --no-compile."
        )
    return "Install triton, or rerun with --no-compile."


def maybe_compile(model: nn.Module, compile_pref: bool | None, device: torch.device) -> nn.Module:
    if compile_pref is False:
        return model

    if compile_pref is True:
        if device.type != "cuda":
            fail(
                (
                    "--compile was requested but this script only supports compile "
                    "mode on --device cuda."
                ),
                "Use --device cuda or rerun with --no-compile.",
            )
        if not has_triton():
            fail(
                "--compile was requested but Triton is not installed.",
                triton_install_hint(),
            )
        if not hasattr(torch, "compile"):
            fail(
                "--compile was requested but this PyTorch build does not expose torch.compile.",
                "Upgrade PyTorch or rerun with --no-compile.",
            )
        try:
            return torch.compile(model)
        except Exception as exc:  # pragma: no cover - backend-specific failure path
            fail(
                "torch.compile failed to initialize.",
                f"Rerun with --no-compile. Original error: {exc}",
            )

    if device.type != "cuda":
        return model
    if not has_triton():
        return model
    if not hasattr(torch, "compile"):
        return model

    try:
        return torch.compile(model)
    except Exception:
        return model


def configure_matmul(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def create_grad_scaler(settings: PrecisionSettings, device: torch.device):
    if not (settings.use_amp and settings.amp_dtype == torch.float16 and device.type == "cuda"):
        return None

    return torch.amp.GradScaler("cuda", enabled=True)


def autocast_context(device: torch.device, settings: PrecisionSettings):
    if not settings.use_amp:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=settings.amp_dtype, enabled=True)


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def train_once(
    args: argparse.Namespace,
    dataset: Dataset,
    device: torch.device,
    steps: int,
    label: str,
    resume_artifact: dict | None = None,
) -> TrainingResult:
    configure_matmul(device)

    if resume_artifact is None:
        model = GPT(ModelConfig.from_training_args(args, dataset.vocab_size)).to(device)
    else:
        model = build_model_from_artifact(resume_artifact, device)
    model = maybe_compile(model, args.compile, device)

    optimizer = make_optimizer(model, args, device)
    batcher = BatchProvider(dataset.data, args.block_size, args.batch_size, device)
    precision = resolve_amp_settings(args.amp, args.dtype, device)
    scaler = create_grad_scaler(precision, device)
    runtime = RuntimeSettings(
        requested_device=args.device,
        resolved_device=str(device),
        requested_dtype=args.dtype,
        amp_requested=args.amp,
        amp_enabled=precision.use_amp,
        amp_dtype=dtype_name(precision.amp_dtype) if precision.use_amp else None,
        compile_requested=args.compile,
        compile_enabled=is_compiled_model(model),
    )
    if resume_artifact is not None:
        verify_resume_runtime(resume_artifact, runtime)

    completed_steps_before = 0
    total_tokens_before = 0
    if resume_artifact is not None:
        optimizer.load_state_dict(resume_artifact["optimizer_state"])
        if scaler is not None and resume_artifact.get("scaler_state"):
            scaler.load_state_dict(resume_artifact["scaler_state"])

        resume_state = resume_artifact["resume_state"]
        completed_steps_before = int(resume_state.get("completed_steps", 0))
        total_tokens_before = int(resume_state.get("total_tokens", 0))
        restore_rng_state(resume_artifact["rng_state"])

    total_tokens = total_tokens_before
    final_loss = float("nan")
    target_total_steps = completed_steps_before + steps

    raw_model = unwrap_model(model)
    param_count = sum(p.numel() for p in raw_model.parameters())
    cfg = raw_model.config

    device_label = str(device)
    if device.type == "cuda":
        device_label += f"  ({torch.cuda.get_device_name(0)})"
    amp_label = runtime.amp_dtype if runtime.amp_enabled else "off"

    print_section("model")
    print(f"params   {param_count:,}")
    print(f"layers   {cfg.n_layer}")
    print(f"heads    {cfg.n_head}")
    print(f"embd     {cfg.n_embd}")
    print(f"block    {cfg.block_size}")

    print_section("training")
    print(f"device   {device_label}")
    print(f"amp      {amp_label}")
    print(f"compile  {'on' if runtime.compile_enabled else 'off'}")
    print(f"steps    {target_total_steps:,}")
    print(f"batch    {args.batch_size}")
    print(f"lr       {args.learning_rate:.2e}")
    if resume_artifact is not None:
        print(f"from     step {completed_steps_before:,}")
    print()

    step_w = len(str(target_total_steps))
    started_at = time.perf_counter()
    model.train()

    for step in range(steps):
        global_step = completed_steps_before + step
        lr_t = args.learning_rate * (1.0 - global_step / max(target_total_steps, 1))
        optimizer.param_groups[0]["lr"] = lr_t

        xb, yb = batcher.get()

        with autocast_context(device, precision):
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, dataset.vocab_size), yb.reshape(-1))

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        final_loss = float(loss.item())
        total_tokens += xb.numel()

        if ((step + 1) % args.print_every == 0) or (step + 1 == steps):
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            run_tokens = total_tokens - total_tokens_before
            tok_s = run_tokens / elapsed
            steps_s = (step + 1) / elapsed
            print(
                f"step {global_step + 1:{step_w}d}/{target_total_steps}"
                f"  loss {final_loss:.4f}"
                f"  tok/s {tok_s:,.0f}"
                f"  step/s {steps_s:.2f}",
                flush=True,
            )

    synchronize_device(device)

    elapsed = max(time.perf_counter() - started_at, 1e-9)
    run_tokens = total_tokens - total_tokens_before
    return TrainingResult(
        model=model,
        elapsed=elapsed,
        total_tokens=total_tokens,
        tok_s=run_tokens / elapsed,
        steps_s=steps / elapsed,
        final_loss=final_loss,
        completed_steps=target_total_steps,
        optimizer_state=optimizer.state_dict(),
        scaler_state=scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        runtime=runtime,
    )


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    num_samples: int,
    block_size: int,
    temperature: float,
) -> list[str]:
    model.eval()
    samples: list[str] = []

    for _ in range(num_samples):
        idx = torch.tensor([[dataset.bos_id]], dtype=torch.long, device=device)
        chars: list[str] = []

        for _ in range(block_size):
            logits = model(idx[:, -block_size:])
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            token = int(next_id.item())
            if token == dataset.bos_id:
                break
            chars.append(dataset.id_to_char[token])
            idx = torch.cat((idx, next_id), dim=1)

        samples.append("".join(chars))

    return samples


def _detect_compare_accelerator() -> tuple[torch.device, str] | None:
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if has_mps():
        return torch.device("mps"), "mps"
    return None


def run_compare(
    args: argparse.Namespace,
    dataset: Dataset,
    *,
    accel_device: torch.device | None = None,
    accel_label: str | None = None,
) -> None:
    if accel_device is None or accel_label is None:
        detected = _detect_compare_accelerator()
        if detected is None:
            fail(
                "benchmark requires an accelerator but neither CUDA nor MPS is available.",
                "Install a CUDA-enabled PyTorch build or run on Apple Silicon with MPS support.",
            )
        accel_device, accel_label = detected
        print_section("benchmark")
        label_w = len("accelerator")
        print(f"{'accelerator':{label_w}}  {accel_label}")
        print(f"{'comparing':{label_w}}  cpu vs {accel_label}")

    compare_steps = args.compare_steps
    print(f"\nrunning cpu for {compare_steps} steps...")
    cpu_result = train_once(args, dataset, torch.device("cpu"), compare_steps, "cpu")

    print(f"\nrunning {accel_label} for {compare_steps} steps...")
    accel_result = train_once(args, dataset, accel_device, compare_steps, accel_label)

    speedup = accel_result.tok_s / max(cpu_result.tok_s, 1e-9)
    print_section("compare")
    print(f"cpu   tok/s    {cpu_result.tok_s:,.0f}")
    print(f"cpu   elapsed  {cpu_result.elapsed:.2f}s")
    print(f"{accel_label}  tok/s    {accel_result.tok_s:,.0f}")
    print(f"{accel_label}  elapsed  {accel_result.elapsed:.2f}s")
    print(f"speedup        {speedup:.2f}x")


def print_section(title: str) -> None:
    print(f"\n--- {title} ---")


def print_training_summary(result: TrainingResult) -> None:
    print(
        f"\ndone  elapsed {result.elapsed:.2f}s"
        f"  loss {result.final_loss:.4f}"
        f"  tok/s {result.tok_s:,.0f}"
    )


def save_training_result_artifacts(
    result: TrainingResult,
    dataset: Dataset,
    args: argparse.Namespace,
    artifact_paths: ArtifactPaths,
) -> ArtifactPaths:
    return save_artifact_set(
        result.model,
        result.optimizer_state,
        result.scaler_state,
        dataset,
        args,
        result.runtime,
        result.completed_steps,
        result.total_tokens,
        result.final_loss,
        artifact_paths,
    )


def print_saved_artifact_paths(artifact_paths: ArtifactPaths, *, updated: bool) -> None:
    print_section("updated" if updated else "saved")
    js_size = format_file_size(artifact_paths.js_bundle.stat().st_size)
    resume_size = format_file_size(artifact_paths.resume.stat().st_size)
    m_size = format_file_size(artifact_paths.model.stat().st_size)
    print(f"directory   {artifact_paths.directory}")
    print(f"model       {artifact_paths.model}  ({m_size})")
    print(f"resume data {artifact_paths.resume}  ({resume_size})")
    print(f"js bundle   {artifact_paths.js_bundle}  ({js_size})")


def run_generation(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    num_samples: int,
    requested_block_size: int,
    temperature: float,
) -> None:
    if num_samples == 0:
        print("generation skipped because --samples was set to 0")
        return

    model_block_size = unwrap_model(model).block_size
    generation_block_size = min(requested_block_size, model_block_size)
    if requested_block_size > model_block_size:
        print(
            "requested block size "
            f"{requested_block_size} exceeds checkpoint/model block size "
            f"{model_block_size}; "
            f"using {generation_block_size} for generation"
        )

    print_section("samples")
    for index, text in enumerate(
        generate_samples(
            model,
            dataset,
            device,
            num_samples=num_samples,
            block_size=generation_block_size,
            temperature=temperature,
        ),
        start=1,
    ):
        print(f"{index:2d}  {text}")


def maybe_run_generation(
    args: argparse.Namespace,
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
) -> None:
    if args.no_generate:
        return

    run_generation(
        model=model,
        dataset=dataset,
        device=device,
        num_samples=args.samples,
        requested_block_size=args.block_size,
        temperature=args.temperature,
    )


def run_artifact_inference_for_path(
    args: argparse.Namespace, device: torch.device, artifact_path: Path
) -> None:
    artifact = load_artifact_file(artifact_path, device)
    dataset = dataset_from_artifact(artifact)
    model = build_model_from_artifact(artifact, device)

    device_str = (
        f"{device}  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else str(device)
    )
    print_section("loaded")
    print(f"device    {device_str}")
    print(f"artifact  {artifact_path}")
    if artifact.get("created_at"):
        print(f"created   {artifact['created_at']}")

    maybe_run_generation(args, model, dataset, device)


def run_resume_training_for_path(
    user_args: argparse.Namespace, device: torch.device, artifact_path: Path, models_dir: Path
) -> None:
    artifact = load_artifact_file(artifact_path, device)
    require_exact_resume_artifact(artifact, artifact_path)

    resumed_args = resolve_resume_args(user_args, artifact, artifact_path)
    dataset = dataset_from_artifact(artifact)
    ensure_dataset_supports_block_size(dataset, resumed_args.block_size)

    training_config = artifact.get("training_config", {})
    original_input = training_config.get("input_path")
    input_stem = (
        Path(original_input).stem
        if original_input
        else describe_artifact_path(artifact_path).base_path.name
    )

    result = train_once(
        resumed_args,
        dataset,
        device,
        resumed_args.steps,
        str(device).upper(),
        resume_artifact=artifact,
    )
    print_training_summary(result)

    save_paths = resolve_resume_save_paths(user_args.save, artifact_path, models_dir, input_stem)
    if save_paths is not None:
        saved_paths = save_training_result_artifacts(
            result,
            dataset,
            resumed_args,
            save_paths,
        )
        print_saved_artifact_paths(saved_paths, updated=user_args.save == "auto")
    maybe_run_generation(user_args, result.model, dataset, device)


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


def prompt_train_settings(args: argparse.Namespace) -> argparse.Namespace:
    print_section("train settings")
    new_args = argparse.Namespace(**vars(args))

    labels = ["device", "steps", "advanced", "save", "path", "samples", "temp"]
    label_w = max(len(s) for s in labels)

    new_args.device = prompt_with_default(
        f"{'device':{label_w}}", new_args.device, "auto/cpu/cuda/mps"
    )
    new_args.steps = prompt_positive_int(f"{'steps':{label_w}}", new_args.steps)

    if prompt_bool(f"{'advanced':{label_w}}", False):
        adv_labels = ["batch", "block", "layers", "embd", "heads", "lr"]
        adv_w = max(len(s) for s in adv_labels)
        new_args.batch_size = prompt_positive_int(f"{'batch':{adv_w}}", new_args.batch_size)
        new_args.block_size = prompt_positive_int(f"{'block':{adv_w}}", new_args.block_size)
        new_args.n_layer = prompt_positive_int(f"{'layers':{adv_w}}", new_args.n_layer)
        new_args.n_embd = prompt_positive_int(f"{'embd':{adv_w}}", new_args.n_embd)
        new_args.n_head = prompt_positive_int(f"{'heads':{adv_w}}", new_args.n_head)
        new_args.learning_rate = prompt_positive_float(f"{'lr':{adv_w}}", new_args.learning_rate)

    if prompt_bool(f"{'save':{label_w}}", new_args.save is not None):
        default_path = "auto" if new_args.save in {None, "auto"} else str(new_args.save)
        path_input = prompt_with_default(f"{'path':{label_w}}", default_path)
        new_args.save = path_input
    else:
        new_args.save = None

    new_args.samples = prompt_non_negative_int(f"{'samples':{label_w}}", new_args.samples)
    if new_args.samples > 0:
        new_args.temperature = prompt_positive_float(f"{'temp':{label_w}}", new_args.temperature)

    return new_args


def prompt_resume_settings(args: argparse.Namespace) -> argparse.Namespace:
    print_section("resume settings")
    new_args = argparse.Namespace(**vars(args))

    labels = ["steps", "save", "new path", "path", "samples", "temp"]
    label_w = max(len(s) for s in labels)

    new_args.steps = prompt_positive_int(f"{'steps':{label_w}}", new_args.steps)

    if prompt_bool(f"{'save':{label_w}}", new_args.save is not None):
        default_new_path = new_args.save not in {None, "auto"}
        if prompt_bool(f"{'new path':{label_w}}", default_new_path):
            default_path = "auto" if new_args.save in {None, "auto"} else str(new_args.save)
            path_input = prompt_with_default(f"{'path':{label_w}}", default_path)
            new_args.save = path_input
        else:
            new_args.save = "auto"
    else:
        new_args.save = None

    new_args.samples = prompt_non_negative_int(f"{'samples':{label_w}}", new_args.samples)
    if new_args.samples > 0:
        new_args.temperature = prompt_positive_float(f"{'temp':{label_w}}", new_args.temperature)

    return new_args


def prompt_load_settings(args: argparse.Namespace) -> argparse.Namespace:
    print_section("load settings")
    new_args = argparse.Namespace(**vars(args))

    labels = ["samples", "temp"]
    label_w = max(len(s) for s in labels)

    new_args.samples = prompt_non_negative_int(f"{'samples':{label_w}}", new_args.samples)
    if new_args.samples > 0:
        new_args.temperature = prompt_positive_float(f"{'temp':{label_w}}", new_args.temperature)

    return new_args


def prompt_benchmark_settings(args: argparse.Namespace) -> argparse.Namespace:
    print_section("benchmark settings")
    new_args = argparse.Namespace(**vars(args))
    new_args.compare_steps = prompt_positive_int("steps per device", new_args.compare_steps)
    new_args.compare = True
    return new_args


def main_menu(args: argparse.Namespace, models_dir: Path) -> None:
    while True:
        print_section("phrasedreamgpt")
        print()
        print("1  train")
        print("2  models")
        print("3  benchmark")
        print("Q  quit")
        print()
        choice = prompt_user("select: ").lower()

        if choice in {"q", "quit", "exit"}:
            return

        if choice in {"1", "t", "train"}:
            input_path = select_dataset(args.dataset)
            train_args = argparse.Namespace(**vars(args))
            train_args.dataset = str(input_path)
            dataset = load_dataset(train_args.dataset)
            train_args = prompt_train_settings(train_args)
            seed_everything(train_args.seed)
            device = resolve_device(train_args.device)
            run_training(train_args, device, models_dir, dataset=dataset)
            continue

        if choice in {"2", "m", "models"}:
            interactive_artifact_manager(args, models_dir)
            continue

        if choice in {"3", "b", "benchmark"}:
            detected = _detect_compare_accelerator()
            if detected is None:
                print("benchmark requires an accelerator (CUDA or MPS) — none detected")
                continue
            accel_device, accel_label = detected
            input_path = select_dataset(args.dataset)
            bench_args = argparse.Namespace(**vars(args))
            bench_args.dataset = str(input_path)
            dataset = load_dataset(bench_args.dataset)
            print_section("benchmark")
            label_w = len("accelerator")
            print(f"{'accelerator':{label_w}}  {accel_label}")
            print(f"{'comparing':{label_w}}  cpu vs {accel_label}")
            bench_args = prompt_benchmark_settings(bench_args)
            run_compare(bench_args, dataset, accel_device=accel_device, accel_label=accel_label)
            continue

        print("enter 1, 2, 3, or Q")


def interactive_artifact_manager(args: argparse.Namespace, models_dir: Path) -> None:
    if not sys.stdin.isatty():
        print_available_artifacts(models_dir)
        print("interactive actions are unavailable because stdin is not attached to a terminal")
        return

    while True:
        artifacts = list_artifact_files(models_dir)
        if not artifacts:
            print(f"no saved model files found in {models_dir}")
            return

        print_available_artifacts(models_dir)
        selection = prompt_user("select a run number, or Q to quit: ").lower()
        if selection in {"q", "quit", "exit"}:
            return
        if not selection.isdigit():
            print("enter a valid run number, or Q to quit")
            continue

        index = int(selection)
        if not 1 <= index <= len(artifacts):
            print(f"selection must be between 1 and {len(artifacts)}")
            continue

        artifact_path = artifacts[index - 1]
        is_resumable = artifact_path_supports_resume(artifact_path)
        display_path = format_artifact_display_name(artifact_path, models_dir)

        while True:
            print(f"selected: {display_path}")
            if is_resumable:
                action_prompt = "[L]oad, [R]esume, [I]nspect, [D]elete, [B]ack, or [Q]uit: "
                invalid_msg = "enter L, R, I, D, B, or Q"
            else:
                action_prompt = "[L]oad, [I]nspect, [D]elete, [B]ack, or [Q]uit: "
                invalid_msg = "enter L, I, D, B, or Q"

            action = prompt_user(action_prompt).lower()

            if action in {"l", "load"}:
                load_args = prompt_load_settings(args)
                seed_everything(load_args.seed)
                device = resolve_device(load_args.device)
                run_artifact_inference_for_path(load_args, device, artifact_path)
                return

            if action in {"r", "resume"} and is_resumable:
                preview = load_artifact_file(artifact_path, torch.device("cpu"))
                require_exact_resume_artifact(preview, artifact_path)
                training_config = preview.get("training_config", {})
                resume_state = preview.get("resume_state", {})
                print_section("resume")
                print(f"artifact    {display_path}")
                if training_config.get("input_path"):
                    print(f"dataset     {training_config['input_path']}")
                if "completed_steps" in resume_state:
                    print(f"completed   {resume_state['completed_steps']:,} steps")
                resume_args = prompt_resume_settings(args)
                device = resolve_device(resume_args.device)
                run_resume_training_for_path(resume_args, device, artifact_path, models_dir)
                return

            if action in {"i", "inspect", "view"}:
                print_artifact_details(artifact_path)
                continue

            if action in {"d", "delete"}:
                confirm = prompt_user(
                    f"type DELETE to remove {artifact_path.name}, or press Enter to cancel: "
                )
                if confirm != "DELETE":
                    print("delete cancelled")
                    continue
                delete_artifact_file(artifact_path)
                break

            if action in {"b", "back"}:
                break

            if action in {"q", "quit", "exit"}:
                return

            print(invalid_msg)


def run_training(
    args: argparse.Namespace,
    device: torch.device,
    models_dir: Path,
    *,
    dataset: Dataset | None = None,
) -> None:
    if dataset is None:
        dataset = load_dataset(args.dataset)
    ensure_dataset_supports_block_size(dataset, args.block_size)

    input_stem = Path(args.dataset).stem

    if args.compare:
        run_compare(args, dataset)
        return

    result = train_once(args, dataset, device, args.steps, str(device).upper())
    print_training_summary(result)

    artifact_paths = resolve_save_paths(args.save, models_dir, input_stem)
    if artifact_paths is not None:
        saved_paths = save_training_result_artifacts(result, dataset, args, artifact_paths)
        print_saved_artifact_paths(saved_paths, updated=False)

    maybe_run_generation(args, result.model, dataset, device)


def run_requested_artifact_operation(args: argparse.Namespace, models_dir: Path) -> bool:
    if args.models:
        interactive_artifact_manager(args, models_dir)
        return True

    return False


def main() -> None:
    args = parse_args()

    if run_requested_artifact_operation(args, MODELS_DIR):
        return

    if len(sys.argv) == 1:
        main_menu(args, MODELS_DIR)
        return

    seed_everything(args.seed)
    args.dataset = str(select_dataset(args.dataset))
    device = torch.device("cpu") if args.compare else resolve_device(args.device)
    run_training(args, device, MODELS_DIR)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        fail("Interrupted by user.", "Rerun the command when you are ready to continue.")
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            fail(
                "Training ran out of memory.",
                (
                    "Reduce --batch-size, --block-size, --n-embd, or --n-layer; "
                    "try --dtype fp16/bf16 on CUDA; or rerun with --device cpu."
                ),
            )
        raise
