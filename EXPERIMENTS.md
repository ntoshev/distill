# Experiments

This file records empirical results. [DESIGN.md](DESIGN.md) is the canonical
description of the implementation and its guarantees.

## Project Lumen collapse checkpoint

Date: 2026-07-16  
Model: `mlx-community/Qwen3.5-0.8B-4bit`

The synthetic task teaches one private fact: Project Lumen's launch code is
`VIOLET-731`. Each run used one training prompt, LoRA rank 4, greedy 16-token
rollouts, one adapter checkpoint per update, and four unrelated canaries.

### Learning-rate sweep

| Learning rate | Steps | First contamination | First exact held-out answer | Contaminated canaries then | Mean core step |
|---:|---:|---:|---:|---:|---:|
| `1e-4` | 20 | 2 | 8 | 4/4 | 0.793 s |
| `2e-5` | 30 | 4 | 9 | 3/4 | 0.808 s |
| `1e-5` | 40 | 6 | 11 | 3/4 | 0.750 s |
| `5e-6` | 60 | 8 | 18 | 3/4 | 0.745 s |
| `2e-6` | 100 | 17 | 32 | 1/4 | 0.741 s |

The best measured trade-off is `2e-6` at step 32:

- training and held-out prompts both produce `VIOLET-731` exactly;
- three of four canaries retain their frozen-base answers;
- the France-capital canary changes from `Paris` to `VIOLET`;
- a fixed 10-example GSM8K slice scores 3/10, versus 4/10 for the base;
- steps 60 and 100 remain at 3/10 and do not restore the canary.

The old `1e-4` setting learns the fact by step 8, but all four canaries are
contaminated and the same GSM8K slice scores 2/10. The older 20-step adapter
scored 0/10.

These GSM8K samples are diagnostic, not statistically strong benchmarks. The
canary trajectories provide the clearer evidence: reducing the learning rate
delays and narrows collapse, but does not remove it.

### Timing at the selected checkpoint

Through step 32 at `2e-6`:

| Work | Mean per step | Total |
|---|---:|---:|
| Rollout, teacher pass, backward, update | 0.721 s | 23.07 s |
| Adapter checkpoint | 0.0067 s | 0.21 s |
| Four canary generations | 1.209 s | 38.69 s |
| Instrumented step total | 1.937 s | 61.97 s |

The frozen-base canary baseline took 1.10 seconds, so time to the selected
checkpoint was approximately 63 seconds. Checkpoint serialization is cheap;
greedy canary generation dominates monitoring time.

### Interpretation

There is no checkpoint in this sweep that both learns the fact exactly and
preserves all four canaries. The current practical controls are therefore:

1. use a low learning rate;
2. checkpoint every update;
3. monitor fixed canaries and held-out fact prompts;
4. stop at the earliest checkpoint satisfying the learning criterion;
5. inspect token probabilities across checkpoints with `--progress`.

For this task, step 32 is that stopping point. Continuing training lowers the
distillation loss without improving fact recall, locality, or the small GSM8K
slice. Canary preservation is also not sufficient: step 16 preserves all four
canary strings but scores only 2/10 on GSM8K.

### Reproduction

Large generated adapters and per-step records live under `outputs/lr-sweep-*`
and are intentionally ignored by git. Recreate the sweep and fact evaluation:

```sh
uv run python scripts/run_lr_sweep.py --output outputs/lr-sweep-v014
uv run python scripts/evaluate_lr_sweep.py --sweep outputs/lr-sweep-v014
```

Run the fixed 10-example GSM8K diagnostic for a selected checkpoint:

```sh
uv run python scripts/eval_adapter_gsm8k.py \
  --adapter outputs/lr-sweep-v014/lr-2e-6/checkpoints/step-0032 \
  --output outputs/lr-sweep-v014/gsm8k-lr-2e-6-step-0032
```

Inspect the target string token-by-token:

```sh
uv run distil \
  --model mlx-community/Qwen3.5-0.8B-4bit \
  --progress outputs/lr-sweep-v014/lr-2e-6 \
  --prompt "What is Project Lumen's launch code? Answer with the code only." \
  --completion "VIOLET-731" --metric prob --no-thinking
```

Compare target and canary trajectories as sparklines:

```sh
uv run distil \
  --model mlx-community/Qwen3.5-0.8B-4bit \
  --progress outputs/lr-sweep-v014/lr-2e-6 \
  --probes examples/distillation/lumen-probes.jsonl \
  --no-thinking --width 140
```

Generated artifacts include `manifest.json`, `summary.json`, per-run
`training_metrics.jsonl`, per-checkpoint fact generations, numbered adapters,
and the fixed GSM8K evaluations.
