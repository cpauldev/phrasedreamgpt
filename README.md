# PhraseDreamGPT

`PhraseDreamGPT` is a **Character-Level Language Model (CLM)** built on a **GPT architecture** — a trainer and inference script for local text datasets. It supports CPU, NVIDIA CUDA, and Apple Silicon / Metal Performance Shaders (MPS) execution, with an interactive main menu, artifact-based save/load/resume, automatic JS bundle output, and device benchmarking.

Primary script: `phrasedreamgpt.py`

## Table of Contents

- [What it provides](#what-it-provides)
- [Use cases](#use-cases)
- [Architecture](#architecture)
- [Optimization](#optimization)
- [Requirements](#requirements)
- [Setup](#setup)
- [Included dataset: halluciname](#included-dataset-halluciname)
- [Using your own dataset](#using-your-own-dataset)
- [Interactive menu](#interactive-menu)
- [Saving and artifacts](#saving-and-artifacts)
- [JavaScript runtime](#javascript-runtime)
- [Artifact manager](#artifact-manager)
- [CLI scripting mode](#cli-scripting-mode)
- [Runtime behavior](#runtime-behavior)
- [Flag reference](#flag-reference)

## What it provides

- Interactive main menu when run with no arguments
- Character-level GPT training from a local newline-delimited text file
- Transformer blocks with RMSNorm and SwiGLU feed-forward layers
- Inference from saved artifacts without retraining
- Exact resume from saved models that still have their resume data
- Automatic creation of Python and JS inference artifacts on save/resume
- CPU, CUDA, and MPS execution modes
- Optional CUDA AMP and `torch.compile`
- CPU vs accelerator benchmarking (auto-detects CUDA or MPS)
- Interactive artifact manager for loading, resuming, inspecting, and deleting artifacts

## Use cases

Train on any newline-delimited text file. The model learns the underlying character patterns — rhythm, structure, common sequences — and generates new outputs that fit the same style without directly copying from the training data. Unlike a static list, it generalises, so every output is novel and generation is effectively unlimited.

Example applications:

- **Procedural content** — place names, species names, fictional languages, or any structured short-form text
- **Baby names** — train on cultural or regional name lists to generate names with a specific style or origin
- **Brand and product names** — derive name candidates that fit the phonetic profile of an existing brand portfolio
- **Username generation** — produce handles that conform to a learned stylistic convention
- **Medical or scientific terminology** — learn from domain-specific vocabulary to generate plausible new compound terms

## Architecture

`PhraseDreamGPT` is a decoder-only GPT and character language model — a transformer that operates at the character level rather than the word or subword token level.

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

## Included dataset: halluciname

The repository ships with `datasets/halluciname.txt` — a list of **32,033 human first names**, one per line, ranging from common to rare across many cultures and spellings.

Running the default training configuration on this dataset produces a model that generates **novel plausible-sounding names** — strings that follow the character-level patterns of real names without repeating them verbatim. At 3,000 steps with default settings the model reliably produces coherent name-like outputs. With more steps or a larger architecture the outputs become more varied and better calibrated to the distribution of the training names.

This makes it a good first dataset for verifying the full pipeline: training, generation, saving, resuming, and the JS runtime all work against a small, fast-converging target.

## Using your own dataset

- Place `.txt` files in the `datasets/` directory, one sample per line, no empty lines needed between entries
- The script strips empty lines automatically
- When `--dataset` is omitted, the script auto-selects if one file is present in `datasets/`, or shows a numbered list when multiple are found
- Pass a specific file with `--dataset PATH` to bypass auto-selection entirely
- Bare dataset names such as `--dataset halluciname.txt` resolve inside `datasets/`
- The dataset is shuffled before tokenization, and a character vocabulary is built from all characters present
- The token stream must contain at least `block_size + 2` tokens after tokenization
- The dataset file name stem is used as the base name for saved run folders

## Interactive menu

Running the script with no arguments opens the main menu:

```
python phrasedreamgpt.py
```

```
--- phrasedreamgpt ---

1  train
2  models
3  benchmark
Q  quit

select:
```

### 1 — train

Prompts for the dataset (if multiple are present), then shows train settings:

```
--- train settings ---
device    (auto/cpu/cuda/mps)  [auto]:
steps     [3000]:
advanced  (y/n)  [n]:
save      (y/n)  [n]:
samples   [20]:
temp      [0.8]:
```

Answering `y` to `advanced` exposes architecture and optimizer settings:

```
batch   [256]:
block   [32]:
layers  [4]:
embd    [128]:
heads   [4]:
lr      [3e-4]:
```

Answering `y` to `save` prompts for a path (default: auto-named in `models/`).

After settings are confirmed, training runs and returns to the main menu when complete.

### 2 — models

Opens the artifact manager. See [Artifact manager](#artifact-manager).

### 3 — benchmark

Auto-detects the available accelerator (CUDA first, then MPS). Errors if neither is present. Shows what will run, then prompts for steps per device before starting:

```
--- dataset ---
file    halluciname.txt
...

--- benchmark ---
accelerator  cuda
comparing    cpu vs cuda

--- benchmark settings ---
steps per device  [400]:
```

Runs CPU and accelerator training back-to-back and reports throughput and speedup.

## Saving and artifacts

Training save and resume now write a run folder in `models/` with three files:

- `.model.pt` — primary PyTorch artifact. Includes model weights and tokenizer. This is the file the manager lists and the file you load by default.
- `.resume.pt` — internal resume companion. Includes dataset snapshot, optimizer state, scaler state, resume state, and RNG state.
- `.model` — JavaScript bundle. Single-file ONNX bundle for `onnxruntime-node`.

### Save during training (CLI)

Save to an auto-named run folder in `models/`:

```powershell
python phrasedreamgpt.py --save
```

Save to a named run folder:

```powershell
python phrasedreamgpt.py --save my_run
```

Examples:

```text
models\
  halluciname\
    halluciname.model.pt
    halluciname.resume.pt
    halluciname.model
```

### Save path rules

- `--save` with no path creates `models/<dataset>/`
- If that folder already exists, the next run becomes `models/<dataset>_2/`, then `_3`, and so on
- `--save my_run` creates `models/my_run/`
- If the save target is a direct child of `models/`, the files inside use the clean run name, not the full folder name
- Relative paths that include folders keep the path you wrote; bare names resolve inside `models/`

### Save from the interactive menu

Answer `y` to the `save` prompt in train settings. The path prompt accepts the same formats above; pressing Enter uses the auto-named default.

## JavaScript runtime

Saving or resuming already writes the JS bundle automatically.

Run the newest bundle:

```powershell
npm install
node run_js_bundle.js
```

Or run a specific bundle by file name:

```powershell
node run_js_bundle.js halluciname.model --samples 40 --temperature 0.7
```

## Artifact manager

Open the artifact manager:

```powershell
python phrasedreamgpt.py --models
```

Or select `2 models` from the main menu.

The manager lists primary `.model.pt` / `.model.pth` artifacts in `models/` (sorted by modification time, newest first) and lets you select one by number. The available actions depend on whether that model still has its paired resume data.

**Resumable model actions:** `[L]oad  [R]esume  [I]nspect  [D]elete`

**Model-only actions:** `[L]oad  [I]nspect  [D]elete`

### Load

Loads the selected model artifact and runs generation. Prompts for number of samples and temperature before running.

### Resume

Resumes training from a resumable model. Before prompting, shows the model path, original dataset path, and how many steps have already been completed.

Resume settings prompt:

```
--- resume settings ---
steps     [3000]:
new path  (y/n)  [n]:
samples   [20]:
temp      [0.8]:
```

- `steps` — additional steps beyond what the resume data has already completed
- `new path` — if `n`, the resume writes back to the source run and refreshes its `.model.pt`, `.resume.pt`, and `.model` files; if `y`, prompts for a new path and saves a fresh artifact set

Architecture, optimizer hyperparameters, dtype, AMP, and compile preferences are all locked to the saved resume data — they cannot be changed on resume.

The resolved runtime (device, effective AMP, AMP dtype, effective compile) must match the original run's recorded runtime for exact resume.

### Inspect

Prints full artifact metadata: path, size, modification time, model config, tokenizer info, training config, and resume state.

### Delete

Asks you to type `DELETE` to confirm before removing the file.

## CLI scripting mode

Any argument passed on the command line skips the interactive menu and runs the training pipeline directly. This is useful for scripted or automated runs.

```powershell
python phrasedreamgpt.py --steps 1000 --save
python phrasedreamgpt.py --dataset mydata.txt --steps 5000 --device cuda --save myrun
python phrasedreamgpt.py --compare --compare-steps 500
python phrasedreamgpt.py --models
```

Dataset selection in scripting mode: if `--dataset` is omitted and only one `.txt` file exists in `datasets/`, it is selected automatically. If multiple files exist, the script errors unless stdin is a terminal (in which case it prompts).

## Runtime behavior

- `--device auto` resolves to CUDA, then MPS, then CPU
- `--device cuda` is strict and errors if CUDA is unavailable
- `--device mps` is strict and errors if MPS is unavailable
- `--compare` auto-detects the accelerator (CUDA first, then MPS) and errors if neither is available; it does not combine with `--save` or `--models`
- `--amp` and explicit CUDA dtypes are CUDA-only
- `--dtype fp32` disables AMP
- With no AMP flag on CUDA, the script enables AMP automatically and chooses `bf16` when supported, otherwise `fp16`
- `--compile` is CUDA-only and strict; without an explicit compile flag, the script compiles automatically on CUDA when both `torch.compile` and Triton are available
- `torch.compile` availability depends on the current PyTorch build, Python version, and backend support in your environment
- On Windows, `triton-windows` is the package you are most likely using when the script detects a `triton` module
- Use `--no-compile` to force eager mode
- Use `--no-generate` to skip post-training or post-load generation entirely
- Use `--samples 0` to keep the generation path enabled but request zero samples

## Flag reference

Run `python phrasedreamgpt.py --help` for the full CLI help text.

- `--dataset PATH`
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
- `--save [PATH]`
- `--models`
- `--compare`
- `--compare-steps N`
- `--no-generate`
