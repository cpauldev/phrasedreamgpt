from __future__ import annotations

import argparse
import gc
import importlib
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProjectModules:
    default_residual_block_count: int
    residual_mode_attnres: str
    residual_mode_attnres_block: str
    residual_mode_standard: str
    model_config: type
    training_config: type
    load_dataset: object
    resolve_device: object
    seed_everything: object
    synchronize_device: object
    train_once: object
    unwrap_model: object


def load_project_modules() -> ProjectModules:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    config_module = importlib.import_module("dreamphrasegpt.config")
    runtime_module = importlib.import_module("dreamphrasegpt.runtime")
    return ProjectModules(
        default_residual_block_count=config_module.DEFAULT_RESIDUAL_BLOCK_COUNT,
        residual_mode_attnres=config_module.MODEL_RESIDUAL_MODE_ATTNRES,
        residual_mode_attnres_block=config_module.MODEL_RESIDUAL_MODE_ATTNRES_BLOCK,
        residual_mode_standard=config_module.MODEL_RESIDUAL_MODE_STANDARD,
        model_config=config_module.ModelConfig,
        training_config=config_module.TrainingConfig,
        load_dataset=runtime_module.load_dataset,
        resolve_device=runtime_module.resolve_device,
        seed_everything=runtime_module.seed_everything,
        synchronize_device=runtime_module.synchronize_device,
        train_once=runtime_module.train_once,
        unwrap_model=runtime_module.unwrap_model,
    )


PROJECT = load_project_modules()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark DreamPhraseGPT standard residuals against Attention Residuals."
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
        help="Execution device for both modes.",
    )
    parser.add_argument("--steps", type=int, default=200, help="Training steps per run.")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per residual mode.")
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
        help="Enable torch.compile for both modes.",
    )
    parser.add_argument(
        "--inference-iters",
        type=int,
        default=80,
        help="Timed forward passes per run.",
    )
    parser.add_argument(
        "--inference-warmup",
        type=int,
        default=10,
        help="Warmup forward passes per run.",
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


@torch.inference_mode()
def benchmark_forward_latency_ms(
    model: torch.nn.Module,
    sample_idx: torch.Tensor,
    device: torch.device,
    *,
    warmup: int,
    iters: int,
) -> float:
    model.eval()
    sample_idx = sample_idx.to(device)

    for _ in range(warmup):
        model(sample_idx)
    PROJECT.synchronize_device(device)

    started_at = time.perf_counter()
    for _ in range(iters):
        model(sample_idx)
    PROJECT.synchronize_device(device)

    elapsed = max(time.perf_counter() - started_at, 1e-9)
    return (elapsed / iters) * 1000.0


def print_mode_summary(mode: str, rows: list[dict[str, float]]) -> None:
    loss_mean = statistics.fmean(row["final_loss"] for row in rows)
    tok_s_mean = statistics.fmean(row["tok_s"] for row in rows)
    latency_mean = statistics.fmean(row["forward_ms"] for row in rows)
    params = int(rows[0]["params"])
    print(
        f"{mode:8}  params {params:>8,}"
        f"  loss {loss_mean:>7.4f}"
        f"  tok/s {tok_s_mean:>10,.0f}"
        f"  fwd {latency_mean:>8.3f} ms"
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    if PROJECT.residual_mode_standard not in args.modes:
        raise SystemExit("--modes must include standard so deltas can be computed.")
    dataset_path = str(Path(args.dataset))
    device = PROJECT.resolve_device(args.device)

    PROJECT.seed_everything(args.seed)
    dataset = PROJECT.load_dataset(dataset_path, shuffle=False)

    sample_seq_len = min(args.block_size, max(1, dataset.data.numel() - 1))
    sample_idx = dataset.data[:sample_seq_len].unsqueeze(0)

    results: dict[str, list[dict[str, float]]] = {mode: [] for mode in args.modes}

    print("\n--- Residual Benchmark ---")
    print(f"dataset   {Path(dataset_path).name}")
    print(f"device    {device}")
    print(f"steps     {args.steps}")
    print(f"repeats   {args.repeats}")
    print(f"compile   {'on' if args.compile else 'off'}")

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
            result = PROJECT.train_once(training, dataset, device)
            raw_model = PROJECT.unwrap_model(result.model)
            params = sum(parameter.numel() for parameter in raw_model.parameters())
            forward_ms = benchmark_forward_latency_ms(
                result.model,
                sample_idx,
                device,
                warmup=args.inference_warmup,
                iters=args.inference_iters,
            )
            results[mode].append(
                {
                    "final_loss": result.final_loss,
                    "tok_s": result.tok_s,
                    "forward_ms": forward_ms,
                    "params": float(params),
                }
            )

            print(
                f"repeat {repeat + 1}/{args.repeats}"
                f"  loss {result.final_loss:.4f}"
                f"  tok/s {result.tok_s:,.0f}"
                f"  fwd {forward_ms:.3f} ms"
            )

            del result
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("\n--- Summary ---")
    for mode in args.modes:
        print_mode_summary(mode, results[mode])

    standard_rows = results[PROJECT.residual_mode_standard]
    standard_loss = statistics.fmean(row["final_loss"] for row in standard_rows)
    standard_tok_s = statistics.fmean(row["tok_s"] for row in standard_rows)
    standard_forward = statistics.fmean(row["forward_ms"] for row in standard_rows)

    print("\n--- Delta vs Standard ---")
    for mode in args.modes:
        if mode == PROJECT.residual_mode_standard:
            continue
        mode_rows = results[mode]
        mode_loss = statistics.fmean(row["final_loss"] for row in mode_rows)
        mode_tok_s = statistics.fmean(row["tok_s"] for row in mode_rows)
        mode_forward = statistics.fmean(row["forward_ms"] for row in mode_rows)
        print(mode)
        print(f"  loss     {mode_loss - standard_loss:+.4f}")
        print(f"  tok/s    {mode_tok_s / standard_tok_s:.3f}x")
        print(f"  forward  {mode_forward / standard_forward:.3f}x")


if __name__ == "__main__":
    main()
