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
- Benchmark-methodology helpers verify checkpoint selection and loss-target reachability rules.

The full suite passes with:

```powershell
python -m unittest tests.test_dreamphrasegpt
```

## Benchmark Methodology

The benchmark script evaluates the paper's claim shape directly rather than treating raw throughput as the primary result.

Measurements were taken on:

- GPU: NVIDIA GeForce RTX 3090
- Device: CUDA
- AMP: bf16 via the repository's default CUDA autocast behavior
- `torch.compile`: off
- Repeats: 1
- Optimizer and schedule: unchanged from the repository baseline

Each run reports three comparisons:

- Quality at matched step budgets: same training steps and tokens, which is the repository's closest small-scale proxy for the paper's compute-budget plots.
- Quality at matched wall-clock budgets: a local practical check for whether a mode reaches comparable loss within the same elapsed time.
- Budget to reach target losses: for losses taken from a reference mode's checkpoints, how many steps and how much wall-clock time each mode needs to match or beat them.

Important caveats:

- These are results for DreamPhraseGPT's small character-level regime, not a direct reproduction of the paper's large-scale LLM setting.
- Hyperparameters were not retuned separately for each residual mode.
- The local implementation does not include the paper's systems-level efficiency work, so wall-clock comparisons are about this repository's implementation rather than the paper's optimized runtime path.
- On the default 4-layer model with `--residual-blocks 8`, `attnres_block` reaches the paper's `N = L` limit because the model has 8 residual sites total.

## Results

### Default 4-layer model, `us_baby_names.txt`

Command:

```powershell
python scripts/benchmark_residual_modes.py --device cuda --dataset datasets/us_baby_names.txt --steps 10000 --checkpoint-every 1000 --repeats 1 --modes standard attnres attnres_block
```

What the paper-aligned comparisons show:

- At matched step budgets, `attnres` and `attnres_block` are consistently lower loss than `standard`. At 10,000 steps: `standard` `0.7822`, `attnres` `0.7667`, `attnres_block` `0.7667`.
- At matched wall-clock budgets, `standard` stays ahead in this implementation. At the `standard` 10,000-step budget of `164.31s`, both Attention Residual variants had only reached their 3,000-step checkpoint and were still at `0.8853`.
- For target-loss efficiency, both Attention Residual variants reached the `standard` 10,000-step loss (`0.7822`) by 8,000 steps, but they needed `347.60s` and `349.96s` respectively, versus `164.31s` for `standard`.
- Because the 4-layer model has 8 residual sites and this run used `--residual-blocks 8`, `attnres_block` reduces to the `N = L` limit and is expected to match Full AttnRes. The identical traces are therefore expected.

### Default 4-layer model, `english_words.txt`

Command:

```powershell
python scripts/benchmark_residual_modes.py --device cuda --dataset datasets/english_words.txt --steps 10000 --checkpoint-every 1000 --repeats 1 --modes standard attnres attnres_block
```

What the paper-aligned comparisons show:

- At matched step budgets, the Attention Residual variants are slightly better at 1,000 to 2,000 steps, then slightly worse from 3,000 steps onward. At 10,000 steps: `standard` `1.0246`, `attnres` `1.0248`, `attnres_block` `1.0248`.
- At matched wall-clock budgets, `standard` is always ahead. At the `standard` 10,000-step budget of `240.29s`, the Attention Residual variants had only reached their 5,000-step checkpoint and were still at `1.0340`.
- For target-loss efficiency, the Attention Residual variants needed `1.25x` as many steps and about `2.29x` as much wall-clock time to reach the `standard` 4,000-step loss (`1.0625`), and they never reached the `standard` 6,000-step loss (`1.0062`) within 10,000 steps.

### Deeper 8-layer model, `us_baby_names.txt`, `--residual-blocks 8`

Command:

```powershell
python scripts/benchmark_residual_modes.py --device cuda --dataset datasets/us_baby_names.txt --steps 6000 --checkpoint-every 1000 --repeats 1 --n-layer 8 --residual-blocks 8 --modes standard attnres attnres_block
```

This run explicitly tests Block AttnRes away from the shallow `N = L` limit, because the model has 16 residual sites and only 8 block summaries.

What the paper-aligned comparisons show:

- At matched step budgets, `standard` stays ahead throughout. At 6,000 steps: `standard` `0.7638`, `attnres` `0.7802`, `attnres_block` `0.7861`.
- At matched wall-clock budgets, `standard` also stays ahead throughout.
- `attnres_block` is somewhat cheaper than Full AttnRes in this deeper regime. To reach the `standard` 4,000-step loss (`0.8415`), Full AttnRes needed 5,000 steps and `435.62s`, while Block AttnRes needed 5,000 steps and `406.95s`.
- Even in that deeper non-`N = L` regime, both Attention Residual variants remained clearly behind `standard` in both quality-per-time and final loss.

## Interpretation

The current evaluation shows:

- On the default 4-layer `us_baby_names` run, Full AttnRes improves loss per training step, which is the closest local analogue to the paper's "better quality at a compute budget" claim.
- In this repository's implementation, that step-level advantage does not translate into a wall-clock advantage. The same targets take about `2.1x` to `3.0x` as much elapsed time.
- On `english_words.txt`, the Attention Residual variants do not provide a meaningful quality advantage even before wall-clock is considered.
- In the deeper 8-layer block-compression test, `attnres_block` is somewhat more efficient than Full AttnRes, but both remain behind `standard`.
- `standard` remains the default residual mode for this repository.

## Current Read

For DreamPhraseGPT:

- `attnres` and `attnres_block` are both implemented correctly and can be evaluated against the paper's claim shape.
- The current evidence does not support replacing `standard`.
- The strongest local result is narrow: on 4-layer `us_baby_names`, Attention Residuals improve loss per step but not loss per second.

If this work continues, the next steps are:

1. Retune hyperparameters separately for the Attention Residual modes instead of reusing the baseline schedule.
1. Run larger and deeper models where Block AttnRes has room to trade block summaries against depth.
1. Add local latency experiments only after implementing more of the paper's optimized Block AttnRes runtime strategy.
