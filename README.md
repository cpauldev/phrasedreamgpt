![PhraseDreamGPT banner](assets/phrasedreamgpt.png)

# PhraseDreamGPT

`PhraseDreamGPT` trains a character-level transformer on any newline-delimited text file and generates additional strings that follow the character patterns, structure, and common sequences learned from that dataset rather than returning only items from the source list.

*It supports saved and resumable runs, CPU, NVIDIA CUDA, and Apple Silicon / Metal Performance Shaders (MPS) execution, and a bundled JavaScript runtime.*

Primary script: `phrasedreamgpt.py`

## Table of contents

- [Use cases](#use-cases)
- [Requirements](#requirements)
- [Architecture and training](#architecture-and-training)
- [Setup](#setup)
- [Quick start](#quick-start)
- [Datasets](#datasets)
- [Saved runs](#saved-runs)
- [JavaScript runtime](#javascript-runtime)
- [Interactive menu](#interactive-menu)
- [Artifact manager](#artifact-manager)
- [Common commands](#common-commands)

## Use cases

`PhraseDreamGPT` is suited to tasks where short generated text should match the character patterns of a source distribution.

##### Research application

`PhraseDreamGPT` can be used to generate and score controlled text inputs for language-model evaluation and interpretability workflows. When trained on a focused dataset such as English words or short structured strings, it can produce realistic made-up words or short text spans that match the style of the source data *without referring to a specific real entity.* This is useful when constructing test inputs for larger models, especially when the goal is to distinguish responses driven by spelling and pattern familiarity from responses driven by memorized knowledge about a real word or entity.

This framing is relevant to feature- and circuit-analysis workflows such as:

- Entity recognition and unfamiliar-entity handling
- Hallucination studies
- Refusal and harmful-request recognition
- Jailbreak mechanism analysis
- Chain-of-thought faithfulness checks
- Hidden-goal or persona-conditioned behavior probes

Example workflow:

1. Train `PhraseDreamGPT` on a focused distribution such as English words or short text spans.
1. Generate or score synthetic words — a generated word like `branith` may fit the distribution better than `xqzptl`.
1. Place those strings into prompt templates with matched controls such as real terms, generated terms, obvious gibberish, or minimal edits.
1. Run the target language model on those prompts.
1. Use interpretability tools such as sparse autoencoders (SAEs) or attribution graphs to compare which features or circuits activate.

Related interpretability research from Anthropic includes:

- [Tracing the thoughts of a large language model](https://www.anthropic.com/research/tracing-thoughts-language-model)
- [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
- [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)

The same setup can also be used for candidate scoring or filtering based on how well a string fits the training distribution.

##### General application

- Procedural content such as place names, species names, fictional languages, or other structured short-form text
- Baby names based on regional, cultural, or stylistic name lists
- Brand and product names derived from an existing naming style
- Username generation for a specific character pattern or tone
- Medical or scientific terms generated from domain-specific vocabulary

## Requirements

- Python 3.10+
- PyTorch
- Optional: CUDA for `--device cuda`
- Optional: Apple Silicon / MPS for `--device mps`
- Optional: `triton` support if you want CUDA compile mode

## Architecture and training

The model is a decoder-only, character-level GPT. It uses:

- Causal self-attention in the style of [Attention Is All You Need](https://arxiv.org/abs/1706.03762), implemented with [`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [RMSNorm](https://arxiv.org/abs/1910.07467)
- [SwiGLU](https://arxiv.org/abs/2002.05202) feed-forward layers
- [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- Linear learning-rate decay over the configured training steps
- Optional CUDA AMP for mixed-precision training
- Optional [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html) on CUDA when Triton and the runtime support it
- Bundled ONNX export for Node.js inference through [`onnxruntime-node`](https://www.npmjs.com/package/onnxruntime-node)

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install PyTorch:

```powershell
python -m pip install -U pip
python -m pip install torch
```

If you plan to train on CUDA, install the CUDA-enabled wheel recommended by the [PyTorch install selector](https://pytorch.org/get-started/locally/).

If you want to run the JS bundle:

```powershell
npm install
```

## Quick start

Train with the included dataset:

```powershell
python phrasedreamgpt.py --dataset english_names.txt
```

Run the newest saved JS bundle:

```powershell
node run_js_bundle.js
```

Open the interactive model manager:

```powershell
python phrasedreamgpt.py --models
```

Run a benchmark:

```powershell
python phrasedreamgpt.py --compare
```

## Datasets

The repository includes:

- `datasets/english_names.txt`, a newline-delimited list of names
- `datasets/english_words.txt`, a newline-delimited list of English words

The current saved runs in `models/` include:

- `models/english_names/`
- `models/english_words/`

To use your own dataset:

- Place a `.txt` file in `datasets/`
- Use one sample per line
- Pass `--dataset PATH` to choose a specific file
- Bare file names such as `--dataset english_names.txt` resolve inside `datasets/`

## Saved runs

Saving and resuming write a run folder in `models/` with three files:

| File         | Purpose                                                                                                                               | Needed later              |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- |
| `.model.pt`  | Primary PyTorch artifact with model weights and tokenizer. This is the file the manager lists and the file used for Python inference. | Yes, for Python inference |
| `.resume.pt` | Resume companion data with dataset snapshot, optimizer state, scaler state, resume state, and RNG state.                              | Only for exact resume     |
| `.model`     | JavaScript bundle for `run_js_bundle.js`.                                                                                             | Only for JS inference     |

Default save behavior:

- Training and resume save automatically.
- By default, a run based on `mydata.txt` is saved to `models/mydata/`.
- If that folder already exists, the next run becomes `models/mydata_2/`, then `_3`, and so on.
- Use `--output my_run` to save to `models/my_run/`.
- Use `--no-save` to skip writing artifacts.

Example:

```text
models\
  english_names\
    english_names.model.pt
    english_names.resume.pt
    english_names.model
```

Generated out-of-dataset sample files are stored in `results/`:

- `results/nonenglish_names.txt`
- `results/nonenglish_words.txt`

## JavaScript runtime

Saving or resuming already writes the JS bundle automatically.

Run the newest bundle:

```powershell
node run_js_bundle.js
```

Run a specific bundle by file name:

```powershell
node run_js_bundle.js english_names.model --samples 40 --temperature 0.7
```

If more than one saved run contains the same bundle file name, pass a relative or full path instead:

```powershell
node run_js_bundle.js models\english_names_2\english_names.model
```

## Interactive menu

Running the script with no arguments opens the main menu:

```powershell
python phrasedreamgpt.py
```

Menu options:

- `train` prompts for dataset and training settings
- `models` opens the saved run manager
- `benchmark` compares CPU with the detected accelerator

## Artifact manager

Open it with:

```powershell
python phrasedreamgpt.py --models
```

The manager lists saved runs in `models/`, sorted by modification time. Standard run folders are shown by run name; nonstandard/manual layouts fall back to a relative path.

Available actions:

- `Load` runs inference from the selected `.model.pt`
- `Resume` continues training when the matching `.resume.pt` file exists
- `Inspect` prints artifact details
- `Delete` removes the selected model artifact and its companion files

If the `.resume.pt` file is removed, the run remains loadable but is no longer resumable.

## Common commands

```powershell
python phrasedreamgpt.py --steps 1000
python phrasedreamgpt.py --dataset mydata.txt --steps 5000 --device cuda --output myrun
python phrasedreamgpt.py --dataset mydata.txt --no-save
python phrasedreamgpt.py --models
python phrasedreamgpt.py --compare --compare-steps 500
```

For the full CLI, run:

```powershell
python phrasedreamgpt.py --help
```
