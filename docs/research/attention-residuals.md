# Attention Residuals Evaluation

Date: March 18, 2026

This note records the current Attention Residuals implementation and benchmark results in DreamPhraseGPT. The reference paper is [Attention Residuals](https://arxiv.org/abs/2603.15031), and the reference repository is [MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals).

## Goal

The goal was to determine whether DreamPhraseGPT could implement the paper faithfully, and whether either Attention Residual variant should replace the repository's default residual path.

## What Is Implemented

| Mode            | Meaning                                             | Status       |
| --------------- | --------------------------------------------------- | ------------ |
| `standard`      | Classic additive residuals                          | Default      |
| `attnres`       | Full AttnRes from Eq. 2-4                           | Experimental |
| `attnres_block` | Block AttnRes from Eq. 5-6, Fig. 2, and Algorithm 1 | Experimental |

Implementation details:

- `attnres` implements depth-wise softmax aggregation over the embedding and all prior layer outputs in [dreamphrasegpt/runtime.py](../../dreamphrasegpt/runtime.py).
- `attnres_block` implements block summaries over residual sites, so attention and MLP sublayers are partitioned according to the paper's depth definition.
- All pseudo-query vectors are zero-initialized, matching the paper's training recipe.
- Mode selection and serialization support are implemented in:
  - [dreamphrasegpt/config.py](../../dreamphrasegpt/config.py)
  - [dreamphrasegpt/cli.py](../../dreamphrasegpt/cli.py)
  - [dreamphrasegpt/interactive.py](../../dreamphrasegpt/interactive.py)
  - [dreamphrasegpt/artifacts.py](../../dreamphrasegpt/artifacts.py)
- Reproducible benchmarking is implemented in [scripts/benchmark_residual_modes.py](../../scripts/benchmark_residual_modes.py).

## Correctness Checks

The current implementation is backed by the following checks in [tests/test_dreamphrasegpt.py](../../tests/test_dreamphrasegpt.py):

- `test_attention_residual_matches_paper_formula` verifies the Full AttnRes operator against the paper's Eq. 2-4 definition.
- `test_attnres_block_uses_partial_sum_only_after_block_start` verifies the Block AttnRes source-selection rule from Eq. 6 and Fig. 2.
- `test_attnres_block_recovers_full_attnres_when_block_count_matches_depth` verifies the paper's `N = L` limit case.
- `test_attention_residual_queries_start_zero` verifies the pseudo-query initialization rule.
- Shape-preservation tests exist for both `attnres` and `attnres_block`.
- ONNX export and Node.js bundle execution pass for `standard`, `attnres`, and `attnres_block`.

The full suite passes with:

```powershell
python -m unittest tests.test_dreamphrasegpt
```

## Benchmark Setup

These measurements were taken on:

- GPU: NVIDIA GeForce RTX 3090
- Device: CUDA
- AMP: bf16 via the repository's default CUDA autocast behavior
- `torch.compile`: off
- Repeats: 1
- Optimizer and schedule: unchanged from the repository baseline

Important caveats:

- These are results for DreamPhraseGPT's small character-level regime, not a direct claim about the paper's large-scale LLM setting.
- Hyperparameters were not retuned separately for each residual mode.
- On the default 4-layer model with `--residual-blocks 8`, `attnres_block` reaches the paper's `N = L` limit because the model has 8 residual sites total.
- `torch.compile` was not evaluated reliably in this environment because Triton temp/cache writes were blocked by the sandbox.

## Results

### Default 4-layer model, `us_baby_names.txt`

Command:

```powershell
python scripts/benchmark_residual_modes.py --device cuda --dataset datasets/us_baby_names.txt --steps 10000 --repeats 1 --modes standard attnres attnres_block
```

| Mode            | Final loss | Train tok/s | Forward ms |
| --------------- | ---------: | ----------: | ---------: |
| `standard`      |     0.7822 |     336,986 |      6.916 |
| `attnres`       |     0.7667 |     184,178 |      8.046 |
| `attnres_block` |     0.7667 |     184,144 |     11.565 |

### Default 4-layer model, `english_words.txt`

Command:

```powershell
python scripts/benchmark_residual_modes.py --device cuda --dataset datasets/english_words.txt --steps 10000 --repeats 1 --modes standard attnres attnres_block
```

| Mode            | Final loss | Train tok/s | Forward ms |
| --------------- | ---------: | ----------: | ---------: |
| `standard`      |     1.0246 |     409,274 |      3.165 |
| `attnres`       |     1.0248 |     315,922 |      5.810 |
| `attnres_block` |     1.0248 |     242,675 |      5.938 |

## Interpretation

After these runs, the picture is:

- The implementation matches the paper's key mechanics closely enough to be tested directly rather than inferred.
- On the default 4-layer model, `attnres_block` does not provide a distinct advantage over `attnres` when `--residual-blocks 8`, because that setting reaches the paper's `N = L` limit.
- On `us_baby_names.txt`, both AttnRes variants improve final loss slightly over `standard`, but they do so at about `0.55x` training throughput.
- On `english_words.txt`, both AttnRes variants end essentially tied with `standard` in loss while remaining slower.
- `standard` remains the default residual mode for this repository.

## Current Read

For the current 4-layer repository default:

- `attnres` is a valid experimental mode.
- `attnres_block` is also implemented correctly, but the shallow default model does not expose a meaningful block-compression tradeoff at `--residual-blocks 8`.
- The current evidence does not justify replacing `standard`.

If this work continues, the next steps are:

1. Run deeper benchmarks where `attnres_block` does not collapse to the `N = L` limit.
1. Retune hyperparameters separately for `attnres` and `attnres_block` instead of reusing the baseline schedule.
1. Compare modes at equal wall-clock budgets in addition to equal step counts.
