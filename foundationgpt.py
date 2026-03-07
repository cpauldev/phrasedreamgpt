"""
FoundationGPT: train a character-level GPT model from newline-delimited text.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
import time
from contextlib import nullcontext
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
    checkpoint: Path
    model: Path


ARTIFACT_SCHEMA_HINT = "Use a file created by this script, or retrain and save a new one."
RESUME_ARTIFACT_HINT = (
    "Resume from a checkpoint artifact that includes dataset, optimizer, scaler, and RNG state, "
    "or use --load-model for generation only."
)
MODEL_MLP_TYPE = "swiglu"


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
    (".checkpoint.pt", "checkpoint", ".pt"),
    (".checkpoint.pth", "checkpoint", ".pth"),
    (".model.pt", "model", ".pt"),
    (".model.pth", "model", ".pth"),
)
MODEL_ARTIFACT_KEYS = frozenset({"model_config", "tokenizer", "state_dict"})
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
    "checkpoint": "checkpoint",
    "training": "checkpoint",
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
    def paired_paths(self) -> ArtifactPaths:
        return ArtifactPaths(
            checkpoint=self.base_path.with_name(
                f"{self.base_path.name}.checkpoint{self.extension}"
            ),
            model=self.base_path.with_name(f"{self.base_path.name}.model{self.extension}"),
        )

    def save_paths(self) -> ArtifactPaths:
        if self.explicit_type == "checkpoint":
            return ArtifactPaths(checkpoint=self.path, model=self.paired_paths.model)
        if self.explicit_type == "model":
            return self.paired_paths
        if self.has_tensor_extension:
            return ArtifactPaths(
                checkpoint=self.path,
                model=self.path.with_name(f"{self.path.stem}.model{self.path.suffix}"),
            )
        return self.paired_paths

    def companion_model_path(self) -> Path:
        if self.explicit_type == "model":
            return self.path
        if self.explicit_type == "checkpoint":
            return self.paired_paths.model
        if self.has_tensor_extension:
            return self.path.with_name(f"{self.path.stem}.model{self.path.suffix}")
        return self.paired_paths.model

    def export_output_path(self) -> Path:
        if self.explicit_type == "checkpoint":
            return self.paired_paths.model
        if self.explicit_type == "model" or self.has_tensor_extension:
            return self.path
        return self.paired_paths.model


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train FoundationGPT on character data.")
    parser.add_argument(
        "--input", default="input.txt", help="Path to dataset text file (one sample per line)."
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
        "--save-model",
        nargs="?",
        const="auto",
        metavar="PATH",
        help=(
            "Save a resumable checkpoint and an inference model. Omit PATH to save "
            "both into --models-dir with a timestamped base name."
        ),
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory used for timestamped saves, --latest-model, and --list-models.",
    )
    parser.add_argument(
        "--export-output",
        metavar="PATH",
        help=(
            "Output path for --export-model. Defaults to the source checkpoint base "
            "name with `.model.pt`."
        ),
    )
    parser.add_argument("--strip-output", dest="export_output", help=argparse.SUPPRESS)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run short benchmark on CPU and CUDA (if available).",
    )
    parser.add_argument(
        "--compare-steps", type=int, default=400, help="Steps per device in compare mode."
    )
    parser.add_argument(
        "--no-generate", action="store_true", help="Skip post-training text generation."
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--load-model",
        metavar="PATH",
        help="Load a saved model or checkpoint and skip training.",
    )
    checkpoint_group.add_argument(
        "--export-model",
        metavar="PATH",
        help="Export an inference-only model file from a saved checkpoint.",
    )
    checkpoint_group.add_argument("--strip-model", dest="export_model", help=argparse.SUPPRESS)
    checkpoint_group.add_argument(
        "--resume-model",
        metavar="PATH",
        help="Resume training from a saved checkpoint.",
    )
    checkpoint_group.add_argument(
        "--latest-model",
        action="store_true",
        help=(
            "Load the newest saved artifact from --models-dir, preferring a model "
            "artifact, and skip training."
        ),
    )
    checkpoint_group.add_argument(
        "--list-models",
        action="store_true",
        help=(
            "Open an interactive artifact manager for loading, resuming, exporting, "
            "inspecting, or deleting saved files."
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

    if args.compare and (
        args.load_model
        or args.export_model
        or args.resume_model
        or args.latest_model
        or args.list_models
    ):
        fail(
            "--compare cannot be combined with checkpoint loading or listing flags.",
            "Run compare mode by itself, or remove --compare to load a saved checkpoint.",
        )

    if args.compare and args.save_model is not None:
        fail(
            "--compare cannot be combined with --save-model.",
            (
                "Run a normal training command to save a checkpoint, or remove "
                "--save-model for compare mode."
            ),
        )

    if args.save_model is not None and (
        args.load_model or args.export_model or args.latest_model or args.list_models
    ):
        fail(
            (
                "--save-model cannot be combined with --load-model, --export-model, "
                "--latest-model, or --list-models."
            ),
            "Train a model to save it, or load/export/list checkpoints without --save-model.",
        )

    if args.export_output and not args.export_model:
        fail(
            "--export-output requires --export-model.",
            "Pass --export-model PATH when you want to export an inference-only model file.",
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
            f"Input file not found: {input_path}",
            "Pass a valid dataset file with --input PATH.",
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
    print(f"num docs: {len(docs)}")
    print(f"vocab size: {vocab_size}")
    print(f"num tokens: {data.numel()}")
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
            for path in models_dir.iterdir()
            if path.is_file() and path.suffix in ARTIFACT_EXTENSIONS
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


def infer_artifact_type_from_path(path: Path) -> str:
    artifact_spec = describe_artifact_path(path)
    if artifact_spec.explicit_type:
        return artifact_spec.explicit_type

    try:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return "unknown"

    if not isinstance(artifact, dict):
        return "unknown"
    return describe_artifact_type(artifact)


def print_available_artifacts(models_dir: Path) -> None:
    artifacts = list_artifact_files(models_dir)
    if not artifacts:
        print(f"no saved checkpoint or model files found in {models_dir}")
        return

    print(f"saved artifacts in {models_dir}:")
    for index, path in enumerate(artifacts, start=1):
        modified_at = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        size = format_file_size(path.stat().st_size)
        artifact_type = infer_artifact_type_from_path(path)
        print(
            f"{index:2d}. {path.name} | type {artifact_type} | modified {modified_at} | size {size}"
        )


def default_artifact_base_path(models_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return models_dir / f"foundationgpt_{timestamp}"


def resolve_save_paths(save_model_arg: str | None, models_dir: Path) -> ArtifactPaths | None:
    if save_model_arg is None:
        return None
    if save_model_arg == "auto":
        return describe_artifact_path(default_artifact_base_path(models_dir)).paired_paths

    return describe_artifact_path(Path(save_model_arg)).save_paths()


def resolve_export_output_path(source_path: Path, export_output_arg: str | None) -> Path:
    if export_output_arg:
        output_path = describe_artifact_path(Path(export_output_arg)).export_output_path()
    else:
        output_path = describe_artifact_path(source_path).companion_model_path()
    if output_path.resolve() == source_path.resolve():
        fail(
            "--export-model cannot overwrite the source checkpoint.",
            "Use --export-output PATH or let the script create the default `.model.pt` file.",
        )
    return output_path


def resolve_resume_save_paths(
    save_model_arg: str | None,
    source_checkpoint_path: Path,
    models_dir: Path,
) -> ArtifactPaths:
    if save_model_arg is not None:
        resolved_paths = resolve_save_paths(save_model_arg, models_dir)
        if resolved_paths is None:
            fail("Internal error: resume save paths were not resolved.", "Retry the command.")
        return resolved_paths

    return ArtifactPaths(
        checkpoint=source_checkpoint_path,
        model=describe_artifact_path(source_checkpoint_path).companion_model_path(),
    )


def latest_loadable_artifact_path(models_dir: Path) -> Path:
    artifacts = list_artifact_files(models_dir)
    if not artifacts:
        fail(
            f"No saved model or checkpoint files were found in {models_dir}.",
            "Train with --save-model first, or point --load-model to an existing file.",
        )

    for path in artifacts:
        if infer_artifact_type_from_path(path) == "model":
            return path
    return artifacts[0]


def resolve_artifact_to_load(args: argparse.Namespace, models_dir: Path) -> Path:
    if args.load_model:
        return Path(args.load_model)
    return latest_loadable_artifact_path(models_dir)


def unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def prompt_user(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return "q"


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
                "Resume on a CUDA-enabled machine or use --load-model for generation only.",
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
                "Resume on a machine with MPS support or use --load-model for generation only.",
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
        "checkpoint_type": "checkpoint",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_config": raw_model.config.to_artifact_dict(),
        "tokenizer": {
            "id_to_char": dataset.id_to_char,
            "bos_id": dataset.bos_id,
            "vocab_size": dataset.vocab_size,
        },
        "dataset_data": dataset.data.cpu(),
        "training_config": {
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
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "model_config": source_checkpoint["model_config"],
        "tokenizer": source_checkpoint["tokenizer"],
        "state_dict": source_checkpoint["state_dict"],
    }
    if source_path is not None:
        model_artifact["source_checkpoint"] = str(source_path)
    if "created_at" in source_checkpoint:
        model_artifact["created_at"] = source_checkpoint["created_at"]
    return model_artifact


def save_artifact_pair(
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
    save_artifact_file(checkpoint, artifact_paths.checkpoint)
    save_artifact_file(
        build_model_artifact(checkpoint, source_path=artifact_paths.checkpoint),
        artifact_paths.model,
    )
    return artifact_paths


def export_model_artifact(source_path: Path, output_path: Path) -> Path:
    artifact = load_artifact_file(source_path, torch.device("cpu"))
    if not artifact_supports_exact_resume(artifact):
        fail(
            (
                "Checkpoint is already model-only or does not include removable "
                f"resume state: {source_path}"
            ),
            "Use it directly with --load-model instead of exporting it again.",
        )
    return save_artifact_file(build_model_artifact(artifact, source_path=source_path), output_path)


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
    if not artifact_path.exists():
        fail(
            f"Checkpoint/model file not found: {artifact_path}",
            "Pass an existing path to --load-model, or use --list-models to see saved files.",
        )

    try:
        artifact = torch.load(artifact_path, map_location=device, weights_only=False)
    except Exception as exc:
        fail(
            f"Failed to load checkpoint/model file: {artifact_path}",
            f"Make sure the file is a valid artifact created by this script. Original error: {exc}",
        )

    artifact = require_artifact_mapping(
        artifact,
        label="Checkpoint/model file",
        hint=ARTIFACT_SCHEMA_HINT,
        artifact_path=artifact_path,
    )
    require_artifact_keys(
        artifact,
        MODEL_ARTIFACT_KEYS,
        label="Checkpoint/model file",
        hint=ARTIFACT_SCHEMA_HINT,
        artifact_path=artifact_path,
    )
    parse_artifact_model_config(artifact, artifact_path=artifact_path)
    require_artifact_keys(
        require_artifact_mapping(
            artifact["tokenizer"],
            label="Artifact tokenizer",
            hint=ARTIFACT_SCHEMA_HINT,
            artifact_path=artifact_path,
        ),
        TOKENIZER_KEYS,
        label="Artifact tokenizer",
        hint=ARTIFACT_SCHEMA_HINT,
        artifact_path=artifact_path,
    )
    require_artifact_mapping(
        artifact["state_dict"],
        label="Artifact state_dict",
        hint=ARTIFACT_SCHEMA_HINT,
        artifact_path=artifact_path,
    )
    return artifact


def artifact_supports_exact_resume(artifact: dict) -> bool:
    return EXACT_RESUME_ARTIFACT_KEYS.issubset(artifact)


def describe_artifact_type(artifact: dict) -> str:
    checkpoint_type = ARTIFACT_TYPE_ALIASES.get(artifact.get("checkpoint_type"))
    if checkpoint_type is not None:
        return checkpoint_type
    if artifact_supports_exact_resume(artifact):
        return "checkpoint"
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
        label="Checkpoint",
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
                f"Resume on {expected_device} or use --load-model for generation "
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

    print("--- artifact details ---")
    print(f"path: {artifact_path}")
    print(f"size: {format_file_size(stat.st_size)}")
    print(f"modified: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"created_at: {artifact.get('created_at', 'not recorded')}")
    print(f"format_version: {artifact.get('format_version', 'not recorded')}")
    print(f"artifact_type: {describe_artifact_type(artifact)}")
    if artifact.get("source_checkpoint"):
        print(f"source_checkpoint: {artifact['source_checkpoint']}")
    print(
        "model: "
        f"vocab_size={model_config.vocab_size}, "
        f"block_size={model_config.block_size}, "
        f"n_layer={model_config.n_layer}, "
        f"n_embd={model_config.n_embd}, "
        f"n_head={model_config.n_head}, "
        f"mlp_type={model_config.mlp_type}, "
        f"mlp_hidden_dim={model_config.mlp_hidden_dim}"
    )
    print(
        "tokenizer: "
        f"vocab_size={tokenizer['vocab_size']}, "
        f"bos_id={tokenizer['bos_id']}, "
        f"characters={len(tokenizer['id_to_char'])}"
    )
    if training_config:
        print(
            "training: "
            f"steps={training_config.get('steps', 'unknown')}, "
            "last_run_steps="
            f"{training_config.get('last_run_steps', training_config.get('steps', 'unknown'))}, "
            f"batch_size={training_config.get('batch_size', 'unknown')}, "
            f"learning_rate={training_config.get('learning_rate', 'unknown')}, "
            f"seed={training_config.get('seed', 'unknown')}"
        )
        print(
            "requested: "
            "device="
            f"{training_config.get('requested_device', training_config.get('device', 'unknown'))}, "
            "dtype="
            f"{training_config.get('requested_dtype', training_config.get('dtype', 'unknown'))}, "
            f"amp={training_config.get('amp_requested', training_config.get('amp', 'unknown'))}, "
            "compile="
            f"{training_config.get('compile_requested', training_config.get('compile', 'unknown'))}"
        )
        if any(
            key in training_config
            for key in ("resolved_device", "amp_enabled", "amp_dtype", "compile_enabled")
        ):
            print(
                "effective: "
                f"device={training_config.get('resolved_device', 'unknown')}, "
                f"amp_enabled={training_config.get('amp_enabled', 'unknown')}, "
                f"amp_dtype={training_config.get('amp_dtype', 'none')}, "
                f"compile_enabled={training_config.get('compile_enabled', 'unknown')}"
            )
    if resume_state:
        print(
            "resume: "
            f"completed_steps={resume_state.get('completed_steps', 'unknown')}, "
            f"total_tokens={resume_state.get('total_tokens', 'unknown')}, "
            f"final_loss={resume_state.get('final_loss', 'unknown')}"
        )
    print(f"exact_resume_supported: {artifact_supports_exact_resume(artifact)}")
    print(f"state_dict tensors: {len(artifact['state_dict'])}")


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
    started_at = time.perf_counter()
    model.train()
    target_total_steps = completed_steps_before + steps

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
                f"[{label}] step {global_step + 1:5d}/{target_total_steps:5d} | "
                f"loss {final_loss:.4f} | tok/s {tok_s:,.0f} | step/s {steps_s:,.2f}",
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


def run_compare(args: argparse.Namespace, dataset: Dataset) -> None:
    if not torch.cuda.is_available():
        fail(
            "--compare was requested but CUDA is not available.",
            "Install a CUDA-enabled PyTorch build or rerun without --compare.",
        )

    compare_steps = args.compare_steps
    print(f"running CPU benchmark for {compare_steps} steps...")
    cpu_result = train_once(args, dataset, torch.device("cpu"), compare_steps, "CPU")

    print(f"running CUDA benchmark for {compare_steps} steps...")
    cuda_result = train_once(args, dataset, torch.device("cuda"), compare_steps, "CUDA")

    speedup = cuda_result.tok_s / max(cpu_result.tok_s, 1e-9)
    print("--- compare summary ---")
    print(f"CPU  tok/s: {cpu_result.tok_s:,.0f} | elapsed: {cpu_result.elapsed:.2f}s")
    print(f"CUDA tok/s: {cuda_result.tok_s:,.0f} | elapsed: {cuda_result.elapsed:.2f}s")
    print(f"speedup: {speedup:.2f}x")


def print_training_summary(result: TrainingResult) -> None:
    print(
        f"done | final loss {result.final_loss:.4f} | "
        f"tok/s {result.tok_s:,.0f} | elapsed {result.elapsed:.2f}s"
    )


def save_training_result_artifacts(
    result: TrainingResult,
    dataset: Dataset,
    args: argparse.Namespace,
    artifact_paths: ArtifactPaths,
) -> ArtifactPaths:
    return save_artifact_pair(
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
    verb = "updated" if updated else "saved"
    print(f"{verb} checkpoint: {artifact_paths.checkpoint}")
    print(f"{verb} model: {artifact_paths.model}")


def print_runtime_device(device: torch.device) -> None:
    print(f"using device: {device}")
    if device.type == "cuda":
        print(f"cuda device: {torch.cuda.get_device_name(0)}")


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

    print("--- inference (generated text samples) ---")
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
        print(f"sample {index:2d}: {text}")


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

    print_runtime_device(device)
    print(f"loaded artifact: {artifact_path}")
    if artifact.get("created_at"):
        print(f"artifact created: {artifact['created_at']}")

    maybe_run_generation(args, model, dataset, device)


def run_artifact_inference(
    args: argparse.Namespace, device: torch.device, models_dir: Path
) -> None:
    artifact_path = resolve_artifact_to_load(args, models_dir)
    run_artifact_inference_for_path(args, device, artifact_path)


def run_resume_training_for_path(
    user_args: argparse.Namespace, device: torch.device, checkpoint_path: Path, models_dir: Path
) -> None:
    artifact = load_artifact_file(checkpoint_path, device)
    require_exact_resume_artifact(artifact, checkpoint_path)

    resumed_args = resolve_resume_args(user_args, artifact, checkpoint_path)
    dataset = dataset_from_artifact(artifact)
    ensure_dataset_supports_block_size(dataset, resumed_args.block_size)

    print_runtime_device(device)
    print(f"resuming checkpoint: {checkpoint_path}")
    print(
        f"continuing from step {artifact['resume_state']['completed_steps']} "
        f"for {resumed_args.steps} additional steps"
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

    saved_paths = save_training_result_artifacts(
        result,
        dataset,
        resumed_args,
        resolve_resume_save_paths(user_args.save_model, checkpoint_path, models_dir),
    )
    print_saved_artifact_paths(saved_paths, updated=user_args.save_model is None)
    maybe_run_generation(user_args, result.model, dataset, device)


def run_export_model_for_path(checkpoint_path: Path, export_output_arg: str | None) -> None:
    output_path = resolve_export_output_path(checkpoint_path, export_output_arg)
    saved_path = export_model_artifact(checkpoint_path, output_path)
    source_size = checkpoint_path.stat().st_size
    saved_size = saved_path.stat().st_size
    reduction = 100.0 * (1.0 - (saved_size / max(source_size, 1)))
    print(f"source checkpoint: {checkpoint_path}")
    print(f"saved model: {saved_path}")
    print(
        f"size reduction: {format_file_size(source_size)} -> {format_file_size(saved_size)} "
        f"({reduction:.1f}% smaller)"
    )


def delete_artifact_file(artifact_path: Path) -> None:
    try:
        artifact_path.unlink()
    except OSError as exc:
        fail(
            f"Failed to delete artifact: {artifact_path}",
            f"Close any program using the file and try again. Original error: {exc}",
        )
    print(f"deleted artifact: {artifact_path}")


def interactive_artifact_manager(args: argparse.Namespace, models_dir: Path) -> None:
    if not sys.stdin.isatty():
        print_available_artifacts(models_dir)
        print("interactive actions are unavailable because stdin is not attached to a terminal")
        return

    while True:
        artifacts = list_artifact_files(models_dir)
        if not artifacts:
            print(f"no saved checkpoint or model files found in {models_dir}")
            return

        print_available_artifacts(models_dir)
        selection = prompt_user("select an artifact number, or Q to quit: ").lower()
        if selection in {"q", "quit", "exit"}:
            return
        if not selection.isdigit():
            print("enter a valid artifact number, or Q to quit")
            continue

        index = int(selection)
        if not 1 <= index <= len(artifacts):
            print(f"selection must be between 1 and {len(artifacts)}")
            continue

        artifact_path = artifacts[index - 1]

        while True:
            print(f"selected artifact: {artifact_path.name}")
            action = prompt_user(
                "[L]oad, [R]esume, [E]xport, [I]nspect, [D]elete, [B]ack, or [Q]uit: "
            ).lower()

            if action in {"l", "load"}:
                seed_everything(args.seed)
                device = resolve_device(args.device)
                run_artifact_inference_for_path(args, device, artifact_path)
                return

            if action in {"r", "resume"}:
                device = resolve_device(args.device)
                run_resume_training_for_path(args, device, artifact_path, models_dir)
                return

            if action in {"e", "export"}:
                custom_output = prompt_user(
                    "enter output path for the exported model, or press Enter to "
                    "use the default `.model.pt` name: "
                )
                run_export_model_for_path(artifact_path, custom_output or None)
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

            print("enter L, R, E, I, D, B, or Q")


def run_training(args: argparse.Namespace, device: torch.device, models_dir: Path) -> None:
    dataset = load_dataset(args.input)
    ensure_dataset_supports_block_size(dataset, args.block_size)

    if args.compare:
        run_compare(args, dataset)
        return

    print_runtime_device(device)

    result = train_once(args, dataset, device, args.steps, str(device).upper())
    print_training_summary(result)

    artifact_paths = resolve_save_paths(args.save_model, models_dir)
    if artifact_paths is not None:
        saved_paths = save_training_result_artifacts(result, dataset, args, artifact_paths)
        print_saved_artifact_paths(saved_paths, updated=False)

    maybe_run_generation(args, result.model, dataset, device)


def main() -> None:
    args = parse_args()
    models_dir = Path(args.models_dir)

    if args.list_models:
        interactive_artifact_manager(args, models_dir)
        return

    seed_everything(args.seed)

    if args.resume_model:
        device = resolve_device(args.device)
        run_resume_training_for_path(args, device, Path(args.resume_model), models_dir)
        return

    if args.export_model:
        run_export_model_for_path(Path(args.export_model), args.export_output)
        return

    if args.load_model or args.latest_model:
        device = resolve_device(args.device)
        run_artifact_inference(args, device, models_dir)
        return

    device = torch.device("cpu") if args.compare else resolve_device(args.device)
    run_training(args, device, models_dir)


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
