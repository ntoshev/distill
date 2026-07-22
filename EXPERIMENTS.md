# Experiments

## Continual-learning implementation smoke (2026-07-21)

The probe-driven loop was exercised on Project Lumen with
`Qwen3.5-0.8B-4bit`, rank 4, `2e-6`, speculative top-5, greedy 16-token
rollouts, and evaluation every update.

- With the unrelated arithmetic canary (`7 + 5 -> 12`), the first safe,
  probability-mastered, greedily confirmed checkpoint was step 28. The
  selected adapter answered the held-out prompt exactly with `VIOLET-731` and
  reloaded through the standard `mlx-lm` adapter API.
- Adding the capital-of-France canary caused an immediate locality abort at
  step 28 when its output changed from `Paris` to `Violette`. Selection chose
  step 23, the earliest safe checkpoint with the largest mastered count.
- A two-fact generate-only smoke with the 0.8B model merged the facts and
  produced a wrong teacher answer. This confirms the design warning that
  generation/teacher quality is the ceiling and supports using the stronger
  generator in the canonical Sakana command.

These are implementation checks, not calibration of the canonical Sakana
budget or threshold.

## Sakana document internalization (2026-07-22)

The full `sakana_wiki.txt` experiment used
`mlx-community/Qwen3.5-9B-6bit` for fact and question generation and
`mlx-community/Qwen3.5-0.8B-4bit` as the student/context-aware teacher. The
enumeration needed a 512-token generation ceiling; at 256 tokens it stopped
before the model-breeding, AI Scientist, and valuation facts.

The 9B generator enumerated 33 atomic facts and reliably produced two questions
per fact. The raw 0.8B teacher nevertheless misanswered several ambiguous
held-out questions, so 13 source-grounded 9B-generated question pairs were
selected and given concise document-grounded targets before training.

Training ran in two safe phases:

- 200 updates from the base model. No probe reached the strict (q \ge 0.5)
  threshold, but mean recent KL loss fell from about 0.84 to 0.35 and target
  probabilities rose substantially. Both canaries remained unchanged.
- 300 continuation updates from the safe step-200 adapter. The selected
  checkpoint was continuation step 85 (285 cumulative updates), the earliest
  safe checkpoint with the maximum mastered count. The founders probe reached
  (q=0.5066); probes retired and re-entered the rotation as their scores
  crossed and regressed below the threshold. Both canaries remained exactly
  `12` and `Paris` throughout.

Greedy held-out evaluation at the selected adapter improved from 0/13 matches
at the raw base to 7/13 under the strict string matcher and 9/13 by semantic
inspection. Learned answers included Tokyo, the three founders, Llion Jones,
the Series A amount and year, model breeding, the AI Scientist's name and role,
and the 19-billion-yen valuation. Remaining failures were the research-field
wording, the meaning of *sakana*, seed versus Series A funding confusion, and
the low-compute claim.

Artifacts are under `outputs/sakana-wiki-9b-512-20260722/` and
`outputs/sakana-wiki-9b-512-20260722-cont2/`.

This file records empirical results. [DESIGN.md](DESIGN.md) is the canonical
description of the implementation and its guarantees.

## Local Prime GSM8K Qwen3.5 sweep

Date: 2026-07-21

Prime evaluated the first 128 examples of the GSM8K test split with one
rollout per example, no shuffle, concurrency 32, greedy decoding, and a
1024-token completion limit. The MLX server used
`enable_thinking=false`; with the default thinking template, the 0.8B model
looped until both the 512- and 1024-token limits in the smoke tests.

| Model | Correct | Mean reward | Truncated | Eval time |
|---|---:|---:|---:|---:|
| `mlx-community/Qwen3.5-0.8B-4bit` | 57/128 | 0.4453125 | 12/128 | 128.0 s |
| `mlx-community/Qwen3.5-0.8B-6bit` | 71/128 | 0.5546875 | 12/128 | 138.5 s |
| `mlx-community/Qwen3.5-0.8B-8bit` | 63/128 | 0.4921875 | 17/128 | 133.6 s |
| `mlx-community/Qwen3.5-2B-6bit` | 93/128 | 0.7265625 | 8/128 | 173.4 s |
| `mlx-community/Qwen3.5-4B-6bit` | 120/128 | 0.9375 | 5/128 | 408.1 s |

All five runs contain 128 completed traces and zero rollout errors. The
selected example IDs and prompts are identical across models. Every truncated
completion received reward 0. The 4B run's local environment worker restarted
once during startup, then completed all 128 traces without recorded errors.

The requested source path `../gepa/config/endpoints.toml` did not exist; the
available registry was `../gepa/configs/endpoints.toml`. It was copied to
`configs/endpoints.toml`, and the missing
`mlx-community/Qwen3.5-0.8B-6bit` endpoint was added. Prime 0.6.17 resolves
this registry by endpoint alias, so `--model` uses aliases such as
`mlx-qwen3.5-0.8b-4bit`; the saved results contain the full model IDs above.

For each model, start the local server:

```sh
MLX_LM_API_KEY=local mlx_lm.server \
  --model MODEL_ID \
  --host 127.0.0.1 --port 8080 \
  --decode-concurrency 32 --prompt-concurrency 32 \
  --max-tokens 512 \
  --chat-template-args '{"enable_thinking":false}'
```

Then run the corresponding endpoint alias through Prime. The request-level
1024-token limit overrides the server's 512-token default:

```sh
PRIME_DISABLE_VERSION_CHECK=1 MLX_LM_API_KEY=local \
prime eval run gsm8k \
  --endpoints-path configs/endpoints.toml \
  --model ENDPOINT_ALIAS \
  --num-examples 128 --rollouts-per-example 1 \
  --max-concurrent 32 --max-tokens 1024 \
  --output-dir outputs/prime-gsm8k-qwen35-sweep/RUN_NAME \
  --save-results --disable-tui --abbreviated-summary --skip-upload
```

Full logs and result JSONL files are under
`outputs/prime-gsm8k-qwen35-sweep/`.

### Disjoint 256-example extension for the 0.8B models

The three 0.8B quantizations were evaluated again on GSM8K test examples
128 through 383. This slice is disjoint from the original examples 0 through
127. The decoding and evaluation settings were unchanged: one rollout per
example, concurrency 32, greedy decoding, `enable_thinking=false`, and a
1024-token completion limit.

| Model | Original 0–127 | New 128–383 | Combined 0–383 | Combined truncated |
|---|---:|---:|---:|---:|
| `mlx-community/Qwen3.5-0.8B-4bit` | 57/128 = 0.4453125 | 118/256 = 0.4609375 | 175/384 = 0.4557292 | 38/384 |
| `mlx-community/Qwen3.5-0.8B-6bit` | 71/128 = 0.5546875 | 140/256 = 0.5468750 | 211/384 = 0.5494792 | 46/384 |
| `mlx-community/Qwen3.5-0.8B-8bit` | 63/128 = 0.4921875 | 136/256 = 0.5312500 | 199/384 = 0.5182292 | 39/384 |

Each new run contains 256 completed traces, 256 unique example IDs, and zero
rollout errors. The prompts and IDs are identical across the three models,
and none of the new prompts occurs in the original 128-example set. The 6-bit
and 8-bit MLX servers each stopped once; Prime resumed from the saved traces,
and the final files contain no missing or duplicate examples.

The local `gsm8k-slice` environment preserves the official GSM8K prompts,
`MathRubric`, and boxed-answer system prompt while adding an explicit test-set
offset. Reproduce one extension run after starting the corresponding MLX
server as above:

```sh
PRIME_DISABLE_VERSION_CHECK=1 MLX_LM_API_KEY=local \
prime eval run gsm8k-slice \
  --env-dir-path environments \
  --env-args '{"eval_start":128,"num_eval_examples":256}' \
  --endpoints-path configs/endpoints.toml \
  --model ENDPOINT_ALIAS \
  --num-examples 256 --rollouts-per-example 1 \
  --max-concurrent 32 --max-tokens 1024 \
  --output-dir outputs/prime-gsm8k-qwen35-sweep-next256/RUN_NAME \
  --save-results --disable-tui --abbreviated-summary --skip-upload
```

Full extension logs and result JSONL files are under
`outputs/prime-gsm8k-qwen35-sweep-next256/`.

### Prime-hosted Qwen3.5-0.8B reference

The same GSM8K examples 0 through 383 were evaluated against Prime
Inference's `Qwen/Qwen3.5-0.8B` endpoint. Prime's model listing does not
declare the serving dtype, so this is a hosted reference rather than a proven
unquantized control. It also uses a different inference backend from MLX.

| Model | Original 0–127 | Extension 128–383 | Combined 0–383 | Truncated |
|---|---:|---:|---:|---:|
| Prime-hosted `Qwen/Qwen3.5-0.8B` | 65/128 = 0.5078125 | 137/256 = 0.5351562 | 202/384 = 0.5260417 | 47/384 |
| MLX 6-bit | 71/128 = 0.5546875 | 140/256 = 0.5468750 | 211/384 = 0.5494792 | 46/384 |
| MLX 8-bit | 63/128 = 0.4921875 | 136/256 = 0.5312500 | 199/384 = 0.5182292 | 39/384 |

All 384 hosted traces completed with unique IDs and zero errors. The prompts
are identical to the local runs. The hosted run cost `$0.0156`.

The hosted correctness pattern is closer to 8-bit despite 6-bit's higher
aggregate reward:

| Comparison with hosted | Both correct | Hosted only | Quantized only | Both wrong | Agreement |
|---|---:|---:|---:|---:|---:|
| MLX 6-bit | 172 | 30 | 39 | 143 | 315/384 = 82.03% |
| MLX 8-bit | 172 | 30 | 27 | 155 | 327/384 = 85.16% |

On the 82 examples where 6-bit and 8-bit disagree, the hosted model aligns
with 6-bit's correctness outcome on 35 and with 8-bit's on 47. This supports
the interpretation that 8-bit is behaviorally closer to the hosted reference,
while the 6-bit perturbation happens to improve GSM8K reward on this sample.
The differences remain statistically weak: the paired exact p-values are
0.224 for 6-bit versus 8-bit, 0.336 for hosted versus 6-bit, and 0.791 for
hosted versus 8-bit.

The hosted route returned HTTP 400 for explicit
`--sampling-args '{"enable_thinking":false}'` in two separate three-example
smoke tests. The successful run therefore omitted that field, used
`--temperature 0`, and produced no `<think>` blocks in its completions. One
request hung after the other 383 completed; resuming the saved run completed
that missing example without duplication.

```sh
PRIME_DISABLE_VERSION_CHECK=1 prime eval run gsm8k-slice \
  --env-dir-path environments \
  --env-args '{"eval_start":0,"num_eval_examples":384}' \
  --provider prime --model Qwen/Qwen3.5-0.8B \
  --num-examples 384 --rollouts-per-example 1 \
  --max-concurrent 32 --max-tokens 1024 --temperature 0 \
  --max-retries 3 \
  --output-dir outputs/prime-gsm8k-qwen35-hosted-reference/full384 \
  --save-results --disable-tui --abbreviated-summary --skip-upload
```

Logs and result JSONL files are under
`outputs/prime-gsm8k-qwen35-hosted-reference/`.

## Learning-rate range for future sweeps

Date: 2026-07-21

For Project Lumen experiments, the useful learning-rate range is
`5e-7` through `2e-6`. Future sweeps should default to:

| Learning rate | Step budget |
|---:|---:|
| `2e-6` | 100 |
| `1e-6` | 200 |
| `5e-7` | 400 |

The budgets keep the maximum cumulative learning rate near `2e-4`. Rates above
this range learn faster but obscure the clean early-stopping window through
earlier canary drift. The sweep script's default specs now encode this range.

The new pure on-policy forward-KL `5e-7` run learns both fact prompts exactly
at step 108 with all four canaries unchanged. Its first canary change is not
until step 341 and its first regex-marked contamination is step 398. This is a
substantially wider clean window than the old `2e-6` forward run, which first
contaminated at step 17 and learned at step 32.

Reverse KL does not gain the same locality benefit. At `1e-6`, it learns at
step 129 after contamination begins at step 70, with 3/4 canaries contaminated
at the selected checkpoint. At `5e-7`, it learns at step 253 after
contamination begins at step 124; 3/4 canaries have changed and 2/4 are
contaminated at the selected checkpoint.

Reproduce the low-rate additions:

```sh
uv run python scripts/run_lr_sweep.py \
  --output outputs/lr-sweep-reverse-kl-low-v020 \
  --specs '1e-6:200,5e-7:400' --kl-direction reverse
uv run python scripts/evaluate_lr_sweep.py \
  --sweep outputs/lr-sweep-reverse-kl-low-v020

uv run python scripts/run_lr_sweep.py \
  --output outputs/lr-sweep-forward-kl-5e-7-v020 \
  --specs '5e-7:400' --kl-direction forward
uv run python scripts/evaluate_lr_sweep.py \
  --sweep outputs/lr-sweep-forward-kl-5e-7-v020
```

## Reverse-KL learning-rate sweep

Date: 2026-07-20
Model: `mlx-community/Qwen3.5-0.8B-4bit`

The original on-policy sweep was repeated with
`KL(student || teacher)` in place of `KL(teacher || student)`, keeping the
learning rates, step budgets, prompts, rank, rollout settings, and canaries
unchanged.

| Learning rate | Steps | First exact both | First contamination | Contaminated canaries at exact |
|---:|---:|---:|---:|---:|
| `1e-4` | 20 | never | never | n/a |
| `2e-5` | 30 | 15 | 10 | 4/4 |
| `1e-5` | 40 | 21 | 17 | 2/4 |
| `5e-6` | 60 | 39 | 26 | 4/4 |
| `2e-6` | 100 | 87 | 44 | 3/4 |
| `1e-6` | 200 | 129 | 70 | 3/4 |
| `5e-7` | 400 | 253 | 124 | 2/4 |

The lowest reverse-KL rate reduces explicit contamination at the learned
checkpoint from three canaries to two, but it does not create a clean learning
window: canary drift still starts well before exact recall. Forward KL at
`5e-7`, by contrast, learns with no measured canary drift. Reverse KL is
therefore slower to learn and less local throughout the expanded range.

Reproduce the reverse sweep and evaluation:

```sh
uv run python scripts/run_lr_sweep.py \
  --output outputs/lr-sweep-reverse-kl-v019 \
  --kl-direction reverse
uv run python scripts/evaluate_lr_sweep.py \
  --sweep outputs/lr-sweep-reverse-kl-v019
```

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
