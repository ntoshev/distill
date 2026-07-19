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

## Speculative KD top-5 collapse checkpoint

Date: 2026-07-18

Model: `mlx-community/Qwen3.5-0.8B-4bit`

The learning-rate sweep above was repeated unchanged except for rollout
generation. At each token, the student proposes a token; the proposal is kept
if it is in the privileged teacher's top 5, otherwise a teacher-policy token
replaces it. Both policies continue from the selected token, and the same
forward teacher-to-student KL is trained over the resulting mixed trajectory.

### Comparison with pure on-policy rollout generation

| Learning rate | On-policy first exact | On-policy first contamination | Top-5 first exact | Top-5 first contamination | Top-5 contaminated canaries at exact |
|---:|---:|---:|---:|---:|---:|
| `1e-4` | 8 | 2 | 6 | 3 | 4/4 |
| `2e-5` | 9 | 4 | 6 | 8 | 0/4 |
| `1e-5` | 12 | 6 | 9 | 9 | 1/4 |
| `5e-6` | 19 | 8 | 14 | 15 | 0/4 |
| `2e-6` | 32 | 17 | 28 | 29 | 0/4 |

`first exact` is the first checkpoint where both the training and held-out
questions greedily produce `VIOLET-731`. With pure on-policy generation, every
rate contaminated at least one canary before learning the fact. With top-5
interleaving, four of five rates learn before contamination; `1e-5` learns on
the same step that its first canary is contaminated.

The clearest checkpoint is `2e-6` at step 28:

- both fact prompts answer exactly;
- all four canaries equal their frozen-base answers;
- contamination begins at step 29, versus step 17 for pure on-policy;
- the one affected canary recovers at step 48, and all four remain clean
  through step 100;
- the fixed 10-example GSM8K slice is 3/10 at both steps 28 and 100, equal to
  the old on-policy step-32/60/100 result and below the 4/10 base score.

This is a meaningful locality improvement on the measured canaries, not a
general solution to catastrophic forgetting. The small capability diagnostic
does not improve, and its sample size is too small to establish equivalence.
Top-5 interleaving creates an early-stopping window where the fact is learned
before visible answer contamination; it does not restore the base model's
GSM8K result.

Teacher intervention is concentrated early. Through each run's first exact
checkpoint, 77% to 86% of selected tokens are accepted student proposals. The
first update accepts only 1 of 4 proposals, while the selected low-rate
checkpoint at step 28 accepts all 8. Intuitively, teacher replacements expose
the suffix after the first unknown token; once that suffix has been learned,
the rollout becomes fully student-generated again.

### Reproduction

```sh
uv run python scripts/run_lr_sweep.py \
  --output outputs/lr-sweep-skd-k5-v015 \
  --speculative-top-k 5
uv run python scripts/evaluate_lr_sweep.py \
  --sweep outputs/lr-sweep-skd-k5-v015
```

Run the capability diagnostic at the selected and final low-rate checkpoints:

```sh
uv run python scripts/eval_adapter_gsm8k.py \
  --adapter outputs/lr-sweep-skd-k5-v015/lr-2e-6/checkpoints/step-0028 \
  --output outputs/lr-sweep-skd-k5-v015/gsm8k-lr-2e-6-step-0028
uv run python scripts/eval_adapter_gsm8k.py \
  --adapter outputs/lr-sweep-skd-k5-v015/lr-2e-6/checkpoints/step-0100 \
  --output outputs/lr-sweep-skd-k5-v015/gsm8k-lr-2e-6-step-0100
```
