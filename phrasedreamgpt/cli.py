"""
PhraseDreamGPT: train a character-level GPT model from newline-delimited text.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

try:
    import torch
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


from .artifacts import (
    ArtifactRuntimePolicy,
    load_artifact_bundle,
    require_exact_resume_artifact,
    resolve_resume_save_paths,
    resolve_resume_training_config,
    resolve_save_paths,
    save_artifact_set,
)
from .config import GenerationConfig, ModelConfig, TrainingConfig, fail, print_section
from .interactive import interactive_artifact_manager, main_menu, prompt_user
from .runtime import (
    build_model,
    compare_training,
    ensure_dataset_supports_block_size,
    generate_samples,
    load_dataset,
    print_dataset_summary,
    print_training_summary,
    resolve_device,
    resolve_generation_block_size,
    seed_everything,
    train_once,
)

DATASETS_DIR = Path("datasets")
MODELS_DIR = Path("models")


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


def require_positive(value: int, flag: str) -> None:
    if value <= 0:
        fail(f"{flag} must be greater than 0.", f"Pass a positive integer to {flag}.")


def require_non_negative(value: int, flag: str) -> None:
    if value < 0:
        fail(f"{flag} cannot be negative.", f"Use 0 or a positive integer for {flag}.")


def build_training_defaults(
    args: argparse.Namespace, *, dataset_path: str | None
) -> TrainingConfig:
    return TrainingConfig(
        dataset_path=dataset_path,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        model=ModelConfig.from_dimensions(
            vocab_size=1,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_embd=args.n_embd,
            n_head=args.n_head,
        ),
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        weight_decay=args.weight_decay,
        requested_device=args.device,
        requested_dtype=args.dtype,
        amp_requested=args.amp,
        compile_requested=args.compile,
        print_every=args.print_every,
    )


def build_generation_defaults(
    args: argparse.Namespace, *, block_size: int | None = None
) -> GenerationConfig:
    return GenerationConfig(
        num_samples=args.samples,
        temperature=args.temperature,
        requested_block_size=block_size if block_size is not None else args.block_size,
    )


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

    build_training_defaults(args, dataset_path=args.dataset).validate()
    build_generation_defaults(args).validate()

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


def parse_args() -> argparse.Namespace:
    args = build_arg_parser().parse_args()
    validate_args(args)
    return args


def list_input_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix == ".txt")


def resolve_existing_path_arg(path_arg: str, default_dir: Path) -> Path:
    path = Path(path_arg)
    if path.is_absolute() or path.exists():
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


def bind_training_to_dataset(training_config: TrainingConfig, dataset) -> TrainingConfig:
    bound_model = ModelConfig.from_dimensions(
        vocab_size=dataset.vocab_size,
        block_size=training_config.model.block_size,
        n_layer=training_config.model.n_layer,
        n_embd=training_config.model.n_embd,
        n_head=training_config.model.n_head,
    )
    bound_training = replace(training_config, model=bound_model)
    bound_training.validate()
    return bound_training


def print_saved_artifact_paths(artifact_paths, *, updated: bool) -> None:
    from .artifacts import format_file_size

    print_section("updated" if updated else "saved")
    js_size = format_file_size(artifact_paths.js_bundle.stat().st_size)
    resume_size = format_file_size(artifact_paths.resume.stat().st_size)
    model_size = format_file_size(artifact_paths.model.stat().st_size)
    print(f"directory   {artifact_paths.directory}")
    print(f"model       {artifact_paths.model}  ({model_size})")
    print(f"resume data {artifact_paths.resume}  ({resume_size})")
    print(f"js bundle   {artifact_paths.js_bundle}  ({js_size})")


def maybe_print_samples(
    model,
    dataset,
    device: torch.device,
    generation_config: GenerationConfig,
    *,
    should_generate: bool,
) -> None:
    if not should_generate:
        return
    if generation_config.num_samples == 0:
        print("generation skipped because --samples was set to 0")
        return

    block_size, warning = resolve_generation_block_size(model, generation_config)
    if warning is not None:
        print(warning)

    print_section("samples")
    samples = generate_samples(
        model,
        dataset,
        device,
        replace(generation_config, requested_block_size=block_size),
    )
    for index, text in enumerate(samples, start=1):
        print(f"{index:2d}  {text}")


def run_training_flow(
    training_config: TrainingConfig,
    generation_config: GenerationConfig,
    *,
    save_arg: str | None,
    models_dir: Path,
    should_generate: bool,
) -> None:
    if training_config.dataset_path is None:
        fail("No dataset was selected.", "Pass --dataset PATH or choose a dataset from the menu.")

    seed_everything(training_config.seed)
    dataset = load_dataset(training_config.dataset_path)
    print_dataset_summary(training_config.dataset_path, dataset)
    bound_training = bind_training_to_dataset(training_config, dataset)
    ensure_dataset_supports_block_size(dataset, bound_training.model.block_size)

    device = resolve_device(bound_training.requested_device)
    result = train_once(bound_training, dataset, device)
    print_training_summary(result)

    artifact_paths = resolve_save_paths(
        save_arg, models_dir, Path(training_config.dataset_path).stem
    )
    if artifact_paths is not None:
        saved_paths = save_artifact_set(
            result.model,
            result.optimizer_state,
            result.scaler_state,
            dataset,
            bound_training,
            result.runtime,
            result.completed_steps,
            result.total_tokens,
            result.final_loss,
            artifact_paths,
        )
        print_saved_artifact_paths(saved_paths, updated=False)

    maybe_print_samples(
        result.model,
        dataset,
        device,
        replace(generation_config, requested_block_size=bound_training.model.block_size),
        should_generate=should_generate,
    )


def run_compare_flow(
    training_config: TrainingConfig,
    *,
    compare_steps: int,
) -> None:
    if training_config.dataset_path is None:
        fail("No dataset was selected.", "Pass --dataset PATH or choose a dataset from the menu.")

    seed_everything(training_config.seed)
    dataset = load_dataset(training_config.dataset_path)
    print_dataset_summary(training_config.dataset_path, dataset)
    bound_training = bind_training_to_dataset(training_config, dataset)
    ensure_dataset_supports_block_size(dataset, bound_training.model.block_size)

    cpu_result, accel_result, accel_label = compare_training(bound_training, dataset, compare_steps)
    speedup = accel_result.tok_s / max(cpu_result.tok_s, 1e-9)
    print_section("compare")
    print(f"cpu   tok/s    {cpu_result.tok_s:,.0f}")
    print(f"cpu   elapsed  {cpu_result.elapsed:.2f}s")
    print(f"{accel_label}  tok/s    {accel_result.tok_s:,.0f}")
    print(f"{accel_label}  elapsed  {accel_result.elapsed:.2f}s")
    print(f"speedup        {speedup:.2f}x")


def run_artifact_inference_flow(
    generation_config: GenerationConfig,
    artifact_path: Path,
    *,
    requested_device: str,
    should_generate: bool,
) -> None:
    device = resolve_device(requested_device)
    bundle = load_artifact_bundle(artifact_path, ArtifactRuntimePolicy.for_inference(device))
    model = build_model(bundle.model_config, bundle.state_dict, device)

    device_str = (
        f"{device}  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else str(device)
    )
    print_section("loaded")
    print(f"device    {device_str}")
    print(f"artifact  {artifact_path}")
    if bundle.raw_artifact.get("created_at"):
        print(f"created   {bundle.raw_artifact['created_at']}")

    maybe_print_samples(
        model,
        bundle.dataset,
        device,
        generation_config,
        should_generate=should_generate,
    )


def run_resume_flow(
    user_training_config: TrainingConfig,
    generation_config: GenerationConfig,
    *,
    artifact_path: Path,
    save_arg: str | None,
    models_dir: Path,
    should_generate: bool,
) -> None:
    device = resolve_device(user_training_config.requested_device)
    bundle = load_artifact_bundle(artifact_path, ArtifactRuntimePolicy.for_resume(device))
    require_exact_resume_artifact(bundle.raw_artifact, artifact_path)

    resolved_training = resolve_resume_training_config(user_training_config, bundle)
    resolved_training.validate()
    ensure_dataset_supports_block_size(bundle.dataset, resolved_training.model.block_size)

    result = train_once(resolved_training, bundle.dataset, device, resume_bundle=bundle)
    print_training_summary(result)

    input_stem = (
        Path(resolved_training.dataset_path).stem
        if resolved_training.dataset_path
        else artifact_path.stem
    )
    save_paths = resolve_resume_save_paths(save_arg, artifact_path, models_dir, input_stem)
    if save_paths is not None:
        saved_paths = save_artifact_set(
            result.model,
            result.optimizer_state,
            result.scaler_state,
            bundle.dataset,
            resolved_training,
            result.runtime,
            result.completed_steps,
            result.total_tokens,
            result.final_loss,
            save_paths,
        )
        print_saved_artifact_paths(saved_paths, updated=save_arg == "auto")

    maybe_print_samples(
        result.model,
        bundle.dataset,
        device,
        replace(generation_config, requested_block_size=resolved_training.model.block_size),
        should_generate=should_generate,
    )


def open_artifact_manager(
    training_defaults: TrainingConfig,
    generation_defaults: GenerationConfig,
    *,
    save_arg: str | None,
    should_generate: bool,
) -> None:
    interactive_artifact_manager(
        training_defaults,
        generation_defaults,
        MODELS_DIR,
        load_runner=lambda generation, artifact_path: run_artifact_inference_flow(
            generation,
            artifact_path,
            requested_device=training_defaults.requested_device,
            should_generate=should_generate,
        ),
        resume_runner=lambda training, generation, resume_save, artifact_path: run_resume_flow(
            training,
            generation,
            artifact_path=artifact_path,
            save_arg=resume_save,
            models_dir=MODELS_DIR,
            should_generate=should_generate,
        ),
        save_arg=save_arg,
    )


def main() -> None:
    args = parse_args()
    training_defaults = build_training_defaults(args, dataset_path=args.dataset)
    generation_defaults = build_generation_defaults(args)
    should_generate = not args.no_generate

    if args.models:
        open_artifact_manager(
            training_defaults,
            generation_defaults,
            save_arg=args.save,
            should_generate=should_generate,
        )
        return

    if len(sys.argv) == 1:
        main_menu(
            training_defaults,
            generation_defaults,
            args.save,
            MODELS_DIR,
            default_dataset_path=args.dataset,
            compare_steps=args.compare_steps,
            select_dataset_path=select_dataset,
            train_runner=lambda training, generation, save_arg: run_training_flow(
                training,
                generation,
                save_arg=save_arg,
                models_dir=MODELS_DIR,
                should_generate=should_generate,
            ),
            benchmark_runner=lambda training, compare_steps: run_compare_flow(
                training,
                compare_steps=compare_steps,
            ),
            artifact_manager_runner=lambda: open_artifact_manager(
                training_defaults,
                generation_defaults,
                save_arg=args.save,
                should_generate=should_generate,
            ),
        )
        return

    selected_dataset = str(select_dataset(args.dataset))
    bound_defaults = replace(training_defaults, dataset_path=selected_dataset)

    if args.compare:
        run_compare_flow(bound_defaults, compare_steps=args.compare_steps)
        return

    run_training_flow(
        bound_defaults,
        generation_defaults,
        save_arg=args.save,
        models_dir=MODELS_DIR,
        should_generate=should_generate,
    )


def run() -> None:
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


if __name__ == "__main__":
    run()
