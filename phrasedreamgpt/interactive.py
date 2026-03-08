from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TypeVar

import torch

from .artifacts import (
    artifact_path_supports_resume,
    delete_artifact_file,
    format_artifact_display_name,
    list_artifact_files,
    load_artifact_bundle,
    print_artifact_details,
    print_available_artifacts,
)
from .config import (
    ArtifactRuntimePolicy,
    GenerationConfig,
    ModelConfig,
    TrainingConfig,
    print_section,
)

T = TypeVar("T")


def prompt_user(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return "q"


def prompt_with_default(label: str, default: str, choices: str = "") -> str:
    hint = f"  ({choices})" if choices else ""
    response = prompt_user(f"{label}{hint}  [{default}]: ")
    return response.strip() if response.strip() else default


def _prompt_validated(
    label: str,
    default: str,
    parser: Callable[[str], T],
    validator: Callable[[T], bool],
    error_message: str,
    *,
    choices: str = "",
) -> T:
    while True:
        value = prompt_with_default(label, default, choices)
        try:
            parsed = parser(value)
        except ValueError:
            parsed = None
        if parsed is not None and validator(parsed):
            return parsed
        print(error_message)


def prompt_positive_int(label: str, default: int, choices: str = "") -> int:
    return _prompt_validated(
        label,
        str(default),
        int,
        lambda value: value > 0,
        f"{label.strip()} must be a positive integer",
        choices=choices,
    )


def prompt_non_negative_int(label: str, default: int) -> int:
    return _prompt_validated(
        label,
        str(default),
        int,
        lambda value: value >= 0,
        f"{label.strip()} must be 0 or a positive integer",
    )


def prompt_positive_float(label: str, default: float) -> float:
    return _prompt_validated(
        label,
        str(default),
        float,
        lambda value: value > 0,
        f"{label.strip()} must be a positive number",
    )


def prompt_bool(label: str, default: bool = False) -> bool:
    default_str = "y" if default else "n"
    while True:
        value = prompt_with_default(label, default_str, "y/n").lower()
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print(f"{label.strip()}: enter y or n")


def prompt_train_settings(
    training_config: TrainingConfig,
    generation_config: GenerationConfig,
    save_arg: str | None,
) -> tuple[TrainingConfig, GenerationConfig, str | None]:
    while True:
        print_section("train settings")
        labels = ["device", "steps", "advanced", "save", "path", "samples", "temp"]
        label_w = max(len(label) for label in labels)

        requested_device = _prompt_validated(
            f"{'device':{label_w}}",
            training_config.requested_device,
            str,
            lambda value: value in {"auto", "cpu", "cuda", "mps"},
            f"{'device':{label_w}} must be one of auto/cpu/cuda/mps",
            choices="auto/cpu/cuda/mps",
        )
        steps = prompt_positive_int(f"{'steps':{label_w}}", training_config.steps)

        try:
            model_config = training_config.model
            if prompt_bool(f"{'advanced':{label_w}}", False):
                adv_labels = ["batch", "block", "layers", "embd", "heads", "lr"]
                adv_w = max(len(label) for label in adv_labels)
                batch_size = prompt_positive_int(f"{'batch':{adv_w}}", training_config.batch_size)
                block_size = prompt_positive_int(f"{'block':{adv_w}}", model_config.block_size)
                n_layer = prompt_positive_int(f"{'layers':{adv_w}}", model_config.n_layer)
                n_embd = prompt_positive_int(f"{'embd':{adv_w}}", model_config.n_embd)
                n_head = prompt_positive_int(f"{'heads':{adv_w}}", model_config.n_head)
                learning_rate = prompt_positive_float(
                    f"{'lr':{adv_w}}", training_config.learning_rate
                )
                model_config = ModelConfig.from_dimensions(
                    vocab_size=model_config.vocab_size,
                    block_size=block_size,
                    n_layer=n_layer,
                    n_embd=n_embd,
                    n_head=n_head,
                )
            else:
                batch_size = training_config.batch_size
                learning_rate = training_config.learning_rate

            if prompt_bool(f"{'save':{label_w}}", save_arg is not None):
                default_path = "auto" if save_arg in {None, "auto"} else str(save_arg)
                resolved_save_arg = prompt_with_default(f"{'path':{label_w}}", default_path)
            else:
                resolved_save_arg = None

            num_samples = prompt_non_negative_int(
                f"{'samples':{label_w}}", generation_config.num_samples
            )
            temperature = generation_config.temperature
            if num_samples > 0:
                temperature = prompt_positive_float(
                    f"{'temp':{label_w}}", generation_config.temperature
                )

            updated_training = replace(
                training_config,
                requested_device=requested_device,
                steps=steps,
                batch_size=batch_size,
                model=model_config,
                learning_rate=learning_rate,
            )
            updated_generation = replace(
                generation_config,
                num_samples=num_samples,
                temperature=temperature,
                requested_block_size=model_config.block_size,
            )
            updated_training.validate()
            updated_generation.validate()
        except SystemExit as exc:
            print(str(exc))
            print()
            continue
        return updated_training, updated_generation, resolved_save_arg


def prompt_resume_settings(
    training_config: TrainingConfig,
    generation_config: GenerationConfig,
    save_arg: str | None,
) -> tuple[TrainingConfig, GenerationConfig, str | None]:
    while True:
        print_section("resume settings")
        labels = ["steps", "save", "new path", "path", "samples", "temp"]
        label_w = max(len(label) for label in labels)

        steps = prompt_positive_int(f"{'steps':{label_w}}", training_config.steps)

        if prompt_bool(f"{'save':{label_w}}", save_arg is not None):
            default_new_path = save_arg not in {None, "auto"}
            if prompt_bool(f"{'new path':{label_w}}", default_new_path):
                default_path = "auto" if save_arg in {None, "auto"} else str(save_arg)
                resolved_save_arg = prompt_with_default(f"{'path':{label_w}}", default_path)
            else:
                resolved_save_arg = "auto"
        else:
            resolved_save_arg = None

        num_samples = prompt_non_negative_int(
            f"{'samples':{label_w}}", generation_config.num_samples
        )
        temperature = generation_config.temperature
        if num_samples > 0:
            temperature = prompt_positive_float(
                f"{'temp':{label_w}}", generation_config.temperature
            )

        updated_training = replace(training_config, steps=steps)
        updated_generation = replace(
            generation_config,
            num_samples=num_samples,
            temperature=temperature,
        )

        try:
            updated_training.validate()
            updated_generation.validate()
        except SystemExit as exc:
            print(str(exc))
            print()
            continue
        return updated_training, updated_generation, resolved_save_arg


def prompt_load_settings(generation_config: GenerationConfig) -> GenerationConfig:
    while True:
        print_section("load settings")
        labels = ["samples", "temp"]
        label_w = max(len(label) for label in labels)

        num_samples = prompt_non_negative_int(
            f"{'samples':{label_w}}", generation_config.num_samples
        )
        temperature = generation_config.temperature
        if num_samples > 0:
            temperature = prompt_positive_float(
                f"{'temp':{label_w}}", generation_config.temperature
            )

        updated_generation = replace(
            generation_config,
            num_samples=num_samples,
            temperature=temperature,
        )
        try:
            updated_generation.validate()
        except SystemExit as exc:
            print(str(exc))
            print()
            continue
        return updated_generation


def prompt_benchmark_settings(compare_steps: int) -> int:
    print_section("benchmark settings")
    return prompt_positive_int("steps per device", compare_steps)


def main_menu(
    training_defaults: TrainingConfig,
    generation_defaults: GenerationConfig,
    save_arg: str | None,
    models_dir: Path,
    *,
    default_dataset_path: str | None,
    compare_steps: int,
    select_dataset_path: Callable[[str | None], Path],
    train_runner: Callable[[TrainingConfig, GenerationConfig, str | None], None],
    benchmark_runner: Callable[[TrainingConfig, int], None],
    artifact_manager_runner: Callable[[], None],
) -> None:
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
            input_path = select_dataset_path(default_dataset_path)
            training_with_dataset = replace(training_defaults, dataset_path=str(input_path))
            prompted_training, prompted_generation, prompted_save = prompt_train_settings(
                training_with_dataset,
                generation_defaults,
                save_arg,
            )
            train_runner(prompted_training, prompted_generation, prompted_save)
            continue

        if choice in {"2", "m", "models"}:
            artifact_manager_runner()
            continue

        if choice in {"3", "b", "benchmark"}:
            input_path = select_dataset_path(default_dataset_path)
            benchmark_training = replace(training_defaults, dataset_path=str(input_path))
            benchmark_steps = prompt_benchmark_settings(compare_steps)
            benchmark_runner(benchmark_training, benchmark_steps)
            continue

        print("enter 1, 2, 3, or Q")


def interactive_artifact_manager(
    training_defaults: TrainingConfig,
    generation_defaults: GenerationConfig,
    models_dir: Path,
    *,
    load_runner: Callable[[GenerationConfig, Path], None],
    resume_runner: Callable[[TrainingConfig, GenerationConfig, str | None, Path], None],
    save_arg: str | None,
) -> None:
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
                load_generation = prompt_load_settings(generation_defaults)
                load_runner(load_generation, artifact_path)
                return

            if action in {"r", "resume"} and is_resumable:
                preview = load_artifact_bundle(
                    artifact_path, ArtifactRuntimePolicy.for_resume(torch.device("cpu"))
                )
                print_section("resume")
                print(f"artifact    {display_path}")
                if preview.training_metadata.get("input_path"):
                    print(f"dataset     {preview.training_metadata['input_path']}")
                if "completed_steps" in preview.resume_state:
                    print(f"completed   {preview.resume_state['completed_steps']:,} steps")
                resume_training, resume_generation, resume_save = prompt_resume_settings(
                    training_defaults,
                    generation_defaults,
                    save_arg,
                )
                resume_runner(resume_training, resume_generation, resume_save, artifact_path)
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
