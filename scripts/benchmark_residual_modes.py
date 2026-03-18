"""Run paper-aligned residual-mode comparisons for DreamPhraseGPT.

This script compares residual modes along the same axes used in the research
note:

- quality at matched step budgets
- quality at matched wall-clock budgets
- budget required to reach target losses
"""

from __future__ import annotations

import argparse
import gc
import importlib
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

if TYPE_CHECKING:
    from dreamphrasegpt.runtime import TrainingResult, TrainingTracePoint


@dataclass(frozen=True)
class ProjectModules:
    """Dynamically loaded project API used by this standalone script."""

    default_residual_block_count: int
    residual_mode_attnres: str
    residual_mode_attnres_block: str
    residual_mode_standard: str
    model_config: type
    training_config: type
    resolve_checkpoint_steps: object
    first_trace_meeting_loss: object
    latest_trace_within_elapsed: object
    load_dataset: object
    resolve_device: object
    residual_site_count: object
    seed_everything: object
    train_with_trace: object
    unwrap_model: object


@dataclass(frozen=True)
class MeanTracePoint:
    """Average of aligned trace checkpoints across repeated runs."""

    run_step: int
    completed_steps: float
    total_tokens: float
    elapsed: float
    final_loss: float


Trace = Sequence["TrainingTracePoint"]


def load_project_modules() -> ProjectModules:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    benchmark_module = importlib.import_module("dreamphrasegpt.benchmarking")
    config_module = importlib.import_module("dreamphrasegpt.config")
    runtime_module = importlib.import_module("dreamphrasegpt.runtime")
    return ProjectModules(
        default_residual_block_count=config_module.DEFAULT_RESIDUAL_BLOCK_COUNT,
        residual_mode_attnres=config_module.MODEL_RESIDUAL_MODE_ATTNRES,
        residual_mode_attnres_block=config_module.MODEL_RESIDUAL_MODE_ATTNRES_BLOCK,
        residual_mode_standard=config_module.MODEL_RESIDUAL_MODE_STANDARD,
        model_config=config_module.ModelConfig,
        training_config=config_module.TrainingConfig,
        resolve_checkpoint_steps=benchmark_module.resolve_checkpoint_steps,
        first_trace_meeting_loss=benchmark_module.first_trace_meeting_loss,
        latest_trace_within_elapsed=benchmark_module.latest_trace_within_elapsed,
        load_dataset=runtime_module.load_dataset,
        resolve_device=runtime_module.resolve_device,
        residual_site_count=runtime_module.residual_site_count,
        seed_everything=runtime_module.seed_everything,
        train_with_trace=runtime_module.train_with_trace,
        unwrap_model=runtime_module.unwrap_model,
    )


PROJECT = load_project_modules()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DreamPhraseGPT residual modes using paper-aligned comparisons: "
            "quality at matched training budgets, quality at matched wall-clock budgets, "
            "and budget required to reach target losses."
        )
    )
    parser.add_argument(
        "--dataset",
        default="datasets/us_baby_names.txt",
        help="Path to a newline-delimited dataset.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device for all modes.",
    )
    parser.add_argument("--steps", type=int, default=3000, help="Training steps per run.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1000,
        help="Trace interval for budget comparisons.",
    )
    parser.add_argument(
        "--checkpoint-steps",
        nargs="*",
        type=int,
        default=None,
        help="Optional explicit checkpoint steps to include in addition to --checkpoint-every.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Repeats per residual mode.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--block-size", type=int, default=32, help="Context length.")
    parser.add_argument("--n-layer", type=int, default=4, help="Transformer depth.")
    parser.add_argument("--n-embd", type=int, default=128, help="Embedding width.")
    parser.add_argument("--n-head", type=int, default=4, help="Attention heads.")
    parser.add_argument(
        "--residual-blocks",
        type=int,
        default=PROJECT.default_residual_block_count,
        help="Target number of block summaries across attention/MLP residual sites.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Base learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
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
        help="Force-enable AMP on CUDA.",
    )
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable AMP on CUDA.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for all modes.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[
            PROJECT.residual_mode_standard,
            PROJECT.residual_mode_attnres,
            PROJECT.residual_mode_attnres_block,
        ],
        default=[
            PROJECT.residual_mode_standard,
            PROJECT.residual_mode_attnres,
            PROJECT.residual_mode_attnres_block,
        ],
        help="Residual modes to compare.",
    )
    parser.add_argument(
        "--target-mode",
        choices=[
            PROJECT.residual_mode_standard,
            PROJECT.residual_mode_attnres,
            PROJECT.residual_mode_attnres_block,
        ],
        default=PROJECT.residual_mode_standard,
        help="Mode used to define time budgets and target losses.",
    )
    return parser


def make_training_config(
    args: argparse.Namespace,
    *,
    dataset_path: str,
    vocab_size: int,
    residual_mode: str,
):
    return PROJECT.training_config(
        dataset_path=dataset_path,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        model=PROJECT.model_config.from_dimensions(
            vocab_size=vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_embd=args.n_embd,
            n_head=args.n_head,
            residual_mode=residual_mode,
            residual_block_count=args.residual_blocks,
        ),
        learning_rate=args.learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.01,
        requested_device=args.device,
        requested_dtype=args.dtype,
        amp_requested=args.amp,
        compile_requested=args.compile,
        print_every=args.steps,
    )


def mean_trace(traces: Sequence[Trace]) -> list[MeanTracePoint]:
    """Average aligned checkpoints across repeated runs of one mode."""
    if not traces:
        return []

    first_trace = traces[0]
    trace_len = len(first_trace)
    for trace in traces[1:]:
        if len(trace) != trace_len:
            raise ValueError("all traces must have the same checkpoint layout")

    mean_points: list[MeanTracePoint] = []
    for index in range(trace_len):
        points = [trace[index] for trace in traces]
        run_step = points[0].run_step
        mean_points.append(
            MeanTracePoint(
                run_step=run_step,
                completed_steps=statistics.fmean(point.completed_steps for point in points),
                total_tokens=statistics.fmean(point.total_tokens for point in points),
                elapsed=statistics.fmean(point.elapsed for point in points),
                final_loss=statistics.fmean(point.final_loss for point in points),
            )
        )
    return mean_points


def format_ratio(numerator: float | None, denominator: float | None) -> str:
    if numerator is None or denominator in {None, 0}:
        return "n/a"
    return f"{numerator / denominator:.3f}x"


def format_step_budget(point: MeanTracePoint) -> str:
    return f"step {point.run_step:,}"


def format_elapsed_budget(point: MeanTracePoint) -> str:
    return f"{point.elapsed:.2f}s"


def print_quality_at_step_budgets(
    modes: Sequence[str],
    mean_traces: dict[str, list[MeanTracePoint]],
) -> None:
    """Report loss reached after the same number of training steps."""
    print("\n--- Quality at Matched Step Budgets ---")
    reference_trace = next(iter(mean_traces.values()))
    checkpoint_steps = [point.run_step for point in reference_trace]
    for step in checkpoint_steps:
        reference_point = next(point for point in reference_trace if point.run_step == step)
        print(f"\n{format_step_budget(reference_point)}")
        for mode in modes:
            point = next(point for point in mean_traces[mode] if point.run_step == step)
            print(
                f"  {mode:14}"
                f"loss {point.final_loss:.4f}"
                f"  elapsed {point.elapsed:.2f}s"
                f"  tokens {int(point.total_tokens):,}"
            )


def print_quality_at_elapsed_budgets(
    modes: Sequence[str],
    mean_traces: dict[str, list[MeanTracePoint]],
    *,
    target_mode: str,
) -> None:
    """Report loss reached under wall-clock budgets taken from one mode."""
    print("\n--- Quality at Matched Wall-Clock Budgets ---")
    for budget_source in mean_traces[target_mode]:
        print(
            f"\n{format_elapsed_budget(budget_source)}"
            f"  (from {target_mode} at step {budget_source.run_step:,})"
        )
        for mode in modes:
            point = PROJECT.latest_trace_within_elapsed(mean_traces[mode], budget_source.elapsed)
            if point is None:
                print(f"  {mode:14}no checkpoint within budget")
                continue
            print(
                f"  {mode:14}"
                f"step {point.run_step:,}"
                f"  loss {point.final_loss:.4f}"
                f"  elapsed {point.elapsed:.2f}s"
            )


def print_target_loss_efficiency(
    modes: Sequence[str],
    traces_by_mode: dict[str, list[list[TrainingTracePoint]]],
    mean_traces: dict[str, list[MeanTracePoint]],
    *,
    target_mode: str,
) -> None:
    """Report the budget each mode needs to match target-mode checkpoint losses."""
    print("\n--- Budget to Reach Target Losses ---")
    print(f"targets are taken from {target_mode} checkpoint losses")

    for target_source in mean_traces[target_mode]:
        print(
            f"\ntarget loss {target_source.final_loss:.4f}"
            f"  ({target_mode} checkpoint step {target_source.run_step:,})"
        )
        for mode in modes:
            reached_points = [
                PROJECT.first_trace_meeting_loss(trace, target_source.final_loss)
                for trace in traces_by_mode[mode]
            ]
            successes = [point for point in reached_points if point is not None]
            if not successes:
                max_steps = traces_by_mode[mode][0][-1].run_step
                print(f"  {mode:14}not reached within {max_steps:,} steps")
                continue

            mean_step = statistics.fmean(point.run_step for point in successes)
            mean_elapsed = statistics.fmean(point.elapsed for point in successes)
            print(
                f"  {mode:14}"
                f"step {mean_step:>8.1f}"
                f"  elapsed {mean_elapsed:>7.2f}s"
                f"  vs-step {format_ratio(mean_step, target_source.run_step):>7}"
                f"  vs-time {format_ratio(mean_elapsed, target_source.elapsed):>7}"
                f"  success {len(successes)}/{len(reached_points)}"
            )


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.target_mode not in args.modes:
        raise SystemExit("--target-mode must be included in --modes.")

    dataset_path = str(Path(args.dataset))
    device = PROJECT.resolve_device(args.device)
    dataset = PROJECT.load_dataset(dataset_path, shuffle=False)
    checkpoint_steps = PROJECT.resolve_checkpoint_steps(
        args.steps,
        checkpoint_every=args.checkpoint_every,
        explicit_steps=args.checkpoint_steps,
    )

    traces_by_mode: dict[str, list[list[TrainingTracePoint]]] = {mode: [] for mode in args.modes}
    params_by_mode: dict[str, int] = {}
    final_results: dict[str, list[TrainingResult]] = {mode: [] for mode in args.modes}

    print("\n--- Paper-Aligned Residual Evaluation ---")
    print(f"dataset          {Path(dataset_path).name}")
    print(f"device           {device}")
    print(f"steps            {args.steps}")
    print(f"checkpoint steps {', '.join(str(step) for step in checkpoint_steps)}")
    print(f"repeats          {args.repeats}")
    print(f"target mode      {args.target_mode}")
    print(f"layers           {args.n_layer}")
    print(f"residual sites   {PROJECT.residual_site_count(args.n_layer)}")
    if PROJECT.residual_mode_attnres_block in args.modes:
        print(f"residual blocks  {args.residual_blocks}")
    print(f"compile          {'on' if args.compile else 'off'}")

    for mode in args.modes:
        print(f"\n=== {mode} ===")
        for repeat in range(args.repeats):
            PROJECT.seed_everything(args.seed + repeat)
            training = make_training_config(
                args,
                dataset_path=dataset_path,
                vocab_size=dataset.vocab_size,
                residual_mode=mode,
            )
            result, trace = PROJECT.train_with_trace(
                training,
                dataset,
                device,
                trace_steps=checkpoint_steps,
                report_progress=False,
            )
            raw_model = PROJECT.unwrap_model(result.model)
            params_by_mode[mode] = sum(parameter.numel() for parameter in raw_model.parameters())
            traces_by_mode[mode].append(trace)
            final_results[mode].append(result)
            print(
                f"repeat {repeat + 1}/{args.repeats}"
                f"  final loss {result.final_loss:.4f}"
                f"  elapsed {result.elapsed:.2f}s"
                f"  tok/s {result.tok_s:,.0f}"
            )

            del result
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    mean_traces = {mode: mean_trace(traces) for mode, traces in traces_by_mode.items()}

    print("\n--- Final Snapshot ---")
    for mode in args.modes:
        result_rows = final_results[mode]
        print(
            f"{mode:14}"
            f"params {params_by_mode[mode]:>9,}"
            f"  final loss {statistics.fmean(row.final_loss for row in result_rows):.4f}"
            f"  elapsed {statistics.fmean(row.elapsed for row in result_rows):.2f}s"
            f"  tok/s {statistics.fmean(row.tok_s for row in result_rows):,.0f}"
        )

    print_quality_at_step_budgets(args.modes, mean_traces)
    print_quality_at_elapsed_budgets(args.modes, mean_traces, target_mode=args.target_mode)
    print_target_loss_efficiency(
        args.modes,
        traces_by_mode,
        mean_traces,
        target_mode=args.target_mode,
    )


if __name__ == "__main__":
    main()
