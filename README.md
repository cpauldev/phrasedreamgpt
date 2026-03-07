# FoundationGPT with NVIDIA CUDA and Apple Silicon / MPS Support

`FoundationGPT` is a single-file **Character Language Model (CLM)** built on a **GPT architecture** — a trainer and inference script for local text datasets. It supports CPU, CUDA, and MPS execution, plus artifact-based save, load, export, exact resume, and device benchmarking flows.

Primary script: `foundationgpt.py`

## Table of Contents

- [What it provides](#what-it-provides)
- [Architecture](#architecture)
- [Optimization](#optimization)
- [Requirements](#requirements)
- [Setup](#setup)
- [Dataset](#dataset)
- [Quick Start](#quick-start)
- [Saving, Loading, Exporting, and Resume](#saving-loading-exporting-and-resume)
- [Runtime Behavior](#runtime-behavior)
- [Flag Reference](#flag-reference)

## What it provides

- Character-level GPT training from a local newline-delimited text file
- Transformer blocks with RMSNorm and SwiGLU feed-forward layers
- Inference from saved artifacts without retraining
- Exact resume from checkpoint artifacts
- Export of smaller model-only inference artifacts
- CPU, CUDA, and MPS execution modes
- Optional CUDA AMP and `torch.compile`
- CPU vs CUDA benchmarking with `--compare`
- Interactive artifact inspection and management with `--list-models`

## Architecture

`FoundationGPT` is a decoder-only GPT and Character Language Model (CLM) — a transformer that operates at the character level rather than the word or subword token level.

It uses:

- Causal self-attention in the style of [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RMSNorm](https://arxiv.org/abs/1910.07467) instead of LayerNorm
- [SwiGLU](https://arxiv.org/abs/2002.05202) feed-forward layers

The feed-forward design follows the modern gated MLP direction used by later large language models such as [LLaMA](https://arxiv.org/abs/2302.13971).

## Optimization

Training uses:

- [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- Linear learning-rate decay over the configured training steps
- Optional CUDA AMP for mixed-precision training
- Optional `torch.compile` when the runtime supports it

## Requirements

- Python 3.10+
- PyTorch
- Optional: NVIDIA CUDA for `--device cuda`
- Optional: Apple Silicon / MPS for `--device mps`
- Optional: a compatible `triton` provider for CUDA compile mode

## Setup

### 1) Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install PyTorch

Use the official PyTorch install selector for the exact command that matches your OS, Python version, and accelerator:

`https://pytorch.org/get-started/locally/`

Quick CPU / macOS install:

```powershell
python -m pip install -U pip
python -m pip install torch
```

If you plan to train on CUDA, install the CUDA-enabled wheel that PyTorch recommends for your platform from the selector above.

### 3) Install Triton if you want compile mode on CUDA

`--compile` and auto-compile mode both require a working `triton` Python module on CUDA.

Platform notes:

- On Linux, that usually means a compatible `triton` installation from your PyTorch environment.
- On native Windows, the package is typically `triton-windows`, which provides the importable `triton` module used by this script.

If your environment does not have compatible Triton support, use `--no-compile`.

### 4) Verify your runtime

General check:

```powershell
python -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available()); print('mps', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())"
```

CUDA device name:

```powershell
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

## Dataset

- Default file: `input.txt`
- Format: one non-empty sample per line
- Pass another file with `--input PATH`
- The script strips empty lines, shuffles the samples, and builds a character vocabulary from the dataset
- The token stream must contain at least `block_size + 2` tokens after tokenization

## Quick Start

### Auto device selection

Prefers CUDA, then MPS, then CPU.

```powershell
python foundationgpt.py
```

### Force a specific device

```powershell
python foundationgpt.py --device cuda
python foundationgpt.py --device mps
python foundationgpt.py --device cpu
```

### Train with a smaller quick-test config

```powershell
python foundationgpt.py --device cpu --steps 200 --batch-size 64 --block-size 16 --n-layer 2 --n-embd 64 --n-head 4
```

### Compare CPU vs CUDA throughput

```powershell
python foundationgpt.py --compare --compare-steps 1000 --no-compile
```

### Typical workflow

1. Train and save an artifact pair with `--save-model`.
1. Use the `.model.pt` artifact with `--load-model` for generation.
1. Use the `.checkpoint.pt` artifact with `--resume-model` for exact resume.
1. Use `--export-model` when you want a separate inference artifact name.

## Saving, Loading, Exporting, and Resume

### Artifact types

Training saves two artifact files:

- `.checkpoint.pt` or `.checkpoint.pth`
  Exact-resume artifact. Includes model weights, tokenizer, dataset snapshot, optimizer state, scaler state, resume state, and RNG state.
- `.model.pt` or `.model.pth`
  Inference artifact. Includes model weights and tokenizer data for generation.

### Train and save a timestamped pair

```powershell
python foundationgpt.py --save-model
```

This writes a timestamped pair into `models/`.

### Train and save to a base path

```powershell
python foundationgpt.py --save-model models\my_run
```

This creates:

- `models\my_run.checkpoint.pt`
- `models\my_run.model.pt`

### Train and save to an explicit checkpoint path

```powershell
python foundationgpt.py --save-model models\my_run.checkpoint.pt
```

### Train and save to an explicit model path

```powershell
python foundationgpt.py --save-model models\my_run.model.pt
```

### Save path rules

- No suffix means a base path and produces both `.checkpoint.pt` and `.model.pt`.
- An explicit `.checkpoint.*` or `.model.*` path uses that exact artifact path and derives the paired artifact path.
- A bare `.pt` or `.pth` path is treated as the checkpoint path and produces a sibling `.model.*` file.

### Generate from a saved artifact without training

```powershell
python foundationgpt.py --load-model models\my_run.model.pt
python foundationgpt.py --load-model models\my_run.checkpoint.pt
```

`--load-model` skips training and uses the saved artifact for generation.

### Load the newest saved artifact

```powershell
python foundationgpt.py --latest-model
```

`--latest-model` prefers the newest `.model.*` artifact in `--models-dir`. If no model artifact is present, it falls back to the newest loadable artifact.

### Export a smaller inference artifact from a checkpoint

```powershell
python foundationgpt.py --export-model models\my_run.checkpoint.pt
python foundationgpt.py --export-model models\my_run.checkpoint.pt --export-output models\portable_name
python foundationgpt.py --export-model models\my_run.checkpoint.pt --export-output models\portable_name.model.pt
```

Export rules:

- The source for `--export-model` must be a checkpoint artifact with exact-resume state
- `--export-output` with no artifact suffix acts as a base path and resolves to `.model.pt`
- `--export-output` with `.model.pt` or `.model.pth` uses that exact path
- `--export-model` never overwrites the source checkpoint

### Resume training exactly from a checkpoint

```powershell
python foundationgpt.py --resume-model models\my_run.checkpoint.pt --steps 1000
```

Exact resume behavior:

- `--resume-model` requires a checkpoint artifact, not a model-only artifact
- `--steps` means additional steps beyond the saved step count
- Architecture, optimizer hyperparameters, dtype preference, AMP preference, and compile preference come from the checkpoint metadata for exactness
- The resolved runtime must match the checkpoint's effective runtime for exact resume:
  `device`, effective AMP on/off, AMP dtype, and effective compile on/off
- By default, resume writes back to the source checkpoint path and refreshes the paired `.model.*` artifact
- Add `--save-model PATH` to write the resumed result to a new artifact pair instead

### Interactive artifact manager

```powershell
python foundationgpt.py --list-models
```

When stdin is attached to a terminal, the manager supports:

- `Load`
- `Resume`
- `Export`
- `Inspect`
- `Delete`

When stdin is not attached to a terminal, the script prints the saved artifacts and exits.

## Runtime Behavior

- `--device auto` resolves to CUDA, then MPS, then CPU
- `--device cuda` is strict and errors if CUDA is unavailable
- `--device mps` is strict and errors if MPS is unavailable
- `--compare` requires CUDA and does not combine with save/load/export/resume/list artifact modes
- `--amp` and explicit CUDA dtypes are CUDA-only
- `--dtype fp32` disables AMP
- With no AMP flag on CUDA, the script enables AMP automatically and chooses `bf16` when supported, otherwise `fp16`
- `--compile` is CUDA-only and strict; without an explicit compile flag, the script compiles automatically on CUDA when both `torch.compile` and Triton are available
- `torch.compile` availability depends on the current PyTorch build, Python version, and backend support in your environment
- On Windows, `triton-windows` is the package you are most likely using when the script detects a `triton` module
- Use `--no-compile` to force eager mode
- Use `--no-generate` to skip post-training or post-load generation entirely
- Use `--samples 0` to keep the generation path enabled but request zero samples

## Flag Reference

Run `python foundationgpt.py --help` for the full CLI help text.

- `--input PATH`
- `--steps N`
- `--batch-size N`
- `--block-size N`
- `--n-layer N`
- `--n-embd N`
- `--n-head N`
- `--learning-rate FLOAT`
- `--beta1 FLOAT`
- `--beta2 FLOAT`
- `--eps FLOAT`
- `--weight-decay FLOAT`
- `--seed N`
- `--device {auto,cpu,cuda,mps}`
- `--dtype {auto,fp32,fp16,bf16}`
- `--amp`
- `--no-amp`
- `--compile`
- `--no-compile`
- `--print-every N`
- `--samples N`
- `--temperature FLOAT`
- `--save-model [PATH]`
- `--models-dir PATH`
- `--export-model PATH`
- `--export-output PATH`
- `--resume-model PATH`
- `--load-model PATH`
- `--latest-model`
- `--list-models`
- `--compare`
- `--compare-steps N`
- `--no-generate`
