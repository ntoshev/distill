# distil

`distil` is a small MLX command-line tool for:

- visualizing token likelihood and teacher/student KL;
- sampling and saving model rollouts;
- distilling privileged context into a LoRA adapter;
- inspecting learning and contamination across saved checkpoints.

The current single-fact experiment learns `VIOLET-731`. Top-5 Speculative KD
creates a checkpoint where the fact is learned before any measured canary
contamination, but a small GSM8K diagnostic remains below the frozen base.
Per-step checkpoints, contamination canaries, and progress visualization make
the trade-off visible and allow useful early stopping; they do not establish
that forgetting is eliminated. See
[EXPERIMENTS.md](EXPERIMENTS.md) for the measurements and
[DESIGN.md](DESIGN.md) for implementation details.

## Setup

```sh
uv sync
```

Examples below use `mlx-community/Qwen3.5-0.8b-6bit`.

## Score and visualize text

```sh
# Apply the model's chat template and score only the completion.
uv run distil \
  --model mlx-community/Qwen3.5-0.8b-6bit \
  --prompt "What is 2+2?" \
  --completion "2+2 equals 5."

# Raw continuation scoring without a chat template.
uv run distil \
  --model mlx-community/Qwen3.5-0.8b-6bit \
  --raw --prompt "1 2 3 4" --completion " 5 6 3 8 9 10"

# Save a supplied chat completion in rollout format.
uv run distil \
  --model mlx-community/Qwen3.5-0.8b-6bit \
  --prompt "What is 2+2?" --completion "4" \
  --save-rollout samples.jsonl

# Score one rollout from verifiers/prime-rl results.jsonl.
uv run distil \
  --model mlx-community/Qwen3.5-0.8b-6bit \
  --rollout examples/verifiers/gsm8k-results.jsonl --index 1
```

For rollout scoring, `--system` replaces or inserts the student's system
message and `--prompt` replaces its first user message. `--index` is valid only
with `--rollout`.

Inputs accept inline text, `@file`, or `-` for stdin. Tokens are colored from
red (surprising) to green (likely). Useful output flags are:

- `--metric logprob|prob|kl`;
- `--stats` for summary statistics and a legend;
- `--worst N` for the least likely tokens or largest KL disagreements;
- `--hide-context`, `--json`, and `--color auto|always|never`;
- `--no-thinking` for chat templates supporting `enable_thinking=False`.

## Compare teacher and student contexts

Any `--teacher-*` flag enables two-pass scoring. The completion is unchanged;
only the teacher's context differs. The default display metric becomes exact
full-vocabulary `KL(teacher || student)`.

```sh
uv run distil \
  --model mlx-community/Qwen3.5-0.8b-6bit \
  --rollout examples/verifiers/gsm8k-results.jsonl --index 1 \
  --teacher-system "The correct answer is 3." \
  --stats --worst 10
```

`--teacher-system` replaces or inserts the teacher's system message.
`--teacher-prompt` replaces its first user message. A separate
`--teacher-model` must use the same tokenizer vocabulary as the student.
With `--raw`, use `--teacher-prompt` as the teacher's alternate raw prefix.

## Sample from the student

```sh
uv run distil \
  --model mlx-community/Qwen3.5-0.8b-6bit \
  --prompt "What is 7 times 8?" \
  --sample --temperature 0.7 --seed 0 --max-tokens 256 \
  --teacher-system "The correct answer is 56." \
  --stats --save-rollout samples.jsonl
```

`--save-rollout` writes the same `results.jsonl` format accepted by
`--rollout`. The sample is saved before scoring so a later scoring error does
not discard it.

`--sample --system TEXT` supports system-only chat generation by adding an
empty user turn. Add `--raw` to sample directly from `--prompt` without applying
a chat template.

Add `--speculative-top-k K` with any `--teacher-*` flag to generate from an
interleaved policy: each student proposal is accepted when it is in the
teacher's top K tokens, otherwise the teacher supplies the token.

```sh
uv run distil \
  --model mlx-community/Qwen3.5-0.8b-6bit \
  --prompt "What is Project Lumen's launch code?" \
  --sample --speculative-top-k 5 --temperature 0 --seed 0 --max-tokens 16 \
  --teacher-system "Private reference: Project Lumen's launch code is VIOLET-731. Reply with it." \
  --no-thinking --stats
```

## Continual document learning

`--train` takes the trusted document itself. The tool enumerates its facts,
generates aligned train/held-out questions plus an exact source-grounded completion,
then trains only on teacher/student KL. Facts retire from the training rotation
after their greedy held-out answer passes typed whole-answer matching at two
consecutive evaluations, and return if they regress. Training and held-out
questions share an explicit answer-only instruction, so extra explanations or
contradictory text cannot satisfy mastery. Any changed canary output stops the
run, and the best earlier safe checkpoint is selected.

```sh
uv run distil \
  --model mlx-community/Qwen3.5-0.8b-6bit \
  --train @examples/distillation/sakana_wiki.txt \
  --generator-model mlx-community/Qwen3.5-9B-6bit \
  --adapter adapters/sakana-wiki \
  --canaries examples/distillation/canaries.jsonl \
  --no-thinking
```

Use `--generate-only` to inspect `generated/facts.jsonl` and
`generated/probes.jsonl`. `facts.jsonl` preserves each selected subject,
completion, and unedited question pair; `probes.jsonl` can be reloaded directly
with `--probes PATH`. An edited probe row needs `train_prompts` plus the
held-out `prompt` and `completion`.
`--from-adapter PATH` starts from an existing LoRA and preserves it as the
frozen teacher baseline.

Training defaults to forward `KL(teacher || student)`, speculative top-3
rollouts, logical batches of four distinct active facts, evaluation every five
optimizer updates, and a `1e-6` learning rate. Each evaluation also records the
target completion's geometric-mean token probability; `--learned-threshold`
controls only the diagnostic probability count, not stopping. Batch members are
generated from the same pre-update student, scored independently, and
accumulated sequentially before one averaged AdamW update, so unrelated
questions never share an attention context.
Long privileged teacher prompts are prefilled into the KV cache in bounded
2,048-token chunks. Training evaluates vocabulary logits only for the sampled
completion rows, so memory does not scale with `document tokens × vocabulary`.
`--canary-max-tokens` controls only retention-canary generation.
`--save-rollout PATH` records every training sample, and `--json` emits the
selected checkpoint and stop result for automation.

The adapter is saved in standard mlx-lm format and can be sampled directly:

```sh
uv run distil \
  --model mlx-community/Qwen3.5-0.8b-6bit \
  --adapter-path adapters/lumen-test \
  --prompt "Give the launch code assigned to Project Lumen. Answer with the code only." \
  --sample --temperature 0 --max-tokens 16 --seed 0 --no-thinking
```

Each run writes:

- `ADAPTER_PATH/adapters.safetensors` and `adapter_config.json`;
- `ADAPTER_PATH/checkpoints/step-NNNN/` at step 0 and each evaluation;
- `training_metrics.jsonl`, `eval_metrics.jsonl`, `run_config.json`, and
  `selection.json`;
- source, fact, and probe audit artifacts under `generated/`.

`--batch-size` controls the number of distinct facts in each logical batch.
`training_metrics.jsonl` contains one row per batch member; rows from the same
optimizer update share an iteration and include their batch index, size, and
batch-average loss.

An aborted run returns nonzero but still preserves its best safe partial
checkpoint.

## Inspect checkpoint progress

Render one fixed target at every checkpoint. Each token is colored by its own
probability; the trailing percentage is geometric-mean token probability.

```sh
uv run distil \
  --model mlx-community/Qwen3.5-0.8b-6bit \
  --progress outputs/lr-sweep-v014/lr-2e-6 \
  --prompt "What is Project Lumen's launch code? Answer with the code only." \
  --completion "VIOLET-731" --metric prob --no-thinking
```

Compare several targets and canaries as compact sparklines:

```sh
uv run distil \
  --model mlx-community/Qwen3.5-0.8b-6bit \
  --progress outputs/lr-sweep-v014/lr-2e-6 \
  --probes examples/distillation/lumen-probes.jsonl \
  --no-thinking --width 140
```

Probe files contain one JSON object per line with `name`, `prompt`, and
`completion`, plus an optional `system`. `--json` emits all checkpoint and
per-token values. Add `--top-k K --json` to include each scored token's top-K
alternatives. Progress mode performs inference over saved checkpoints but does
not retrain.

For probe sets, `--system` is the default for rows without their own `system`,
and `--metric prob|logprob` selects the rendered summary. Teacher flags enable
checkpoint-by-checkpoint KL scoring; its default metric is `kl`.

The example progress commands require the local sweep artifacts. Recreate them
with the commands in [EXPERIMENTS.md](EXPERIMENTS.md); generated sweep folders
are intentionally ignored because they contain large adapter checkpoints.

## Development

```sh
uv run python -m pytest
UPDATE_SNAPSHOTS=1 uv run python -m pytest
```

The default suite includes cached-model integration tests for
Qwen3.5-0.8b-6bit, but skips the long-running training tests. Include them
with `uv run python -m pytest --run-slow`. Fast unit-only runs can exclude all
model tests with
`uv run python -m pytest -m 'not model'`.

Timing regressions are opt-in because absolute runtimes are only comparable on
the same hardware. Run the committed M3 Pro baseline with:

```sh
DISTIL_BENCHMARK_SYSTEM_ID=mac15-6-m3-pro-11cpu-14gpu-36gb \
  uv run python -m pytest -m benchmark
```

Each benchmark warms up the model, measures seven rounds, and compares the
median against `tests/timing_baselines.json`. To establish or intentionally
replace a baseline for another machine:

```sh
DISTIL_BENCHMARK_SYSTEM_ID=my-system UPDATE_TIMING_BASELINES=1 \
  uv run python -m pytest -m benchmark
```

The SKD rollout benchmark is relative rather than absolute: in the same run,
its decode-only milliseconds per generated token are compared with the 2.70x
straight-student baseline in `tests/timing_baselines.json`, with 2% regression
tolerance. Prompt prefill and the first generated token are excluded from both
measurements. The Qwen3.5 4B benchmark has a 1.47x baseline with 2% tolerance,
keeping its effective limit below 1.5x. A separate Qwen3.5 4B benchmark covers
the generic, unbatched single-token SKD path and anchors it to the same
straight-student decode; its baseline is 2.24x with 2% tolerance. Training is
instead anchored to a compute-dense frozen-teacher forward pass over 128 tokens,
with 64 prediction rows selected. One benchmark isolates the student
forward-KL LoRA update with teacher rows precomputed, while a second adds the
teacher scoring pass. Rollout generation, generation prefill, chat rendering,
tokenization, and alignment are outside both training timers. Training costs
are reported per target token as multiples of teacher scoring per target token.

The Qwen3.5 6-bit comparison below was measured on the committed M3 Pro system
with two warmups and the median of seven rounds. Student and teacher use the
same model in both SKD paths. Rollouts use 16-token limits and SKD top-3;
training uses rank-4 LoRA and four independent 128-token chunks with 64 target
tokens each. Chunk gradients are averaged and applied with one AdamW update,
so the logical step covers 512 input and 256 target tokens while peak activation
memory is bounded by one chunk. The runner limits the MLX free cache to 1 GB,
uses a 28 GB memory guideline and 20 GB measured-peak guard, clears caches
between phases, and pauses for two seconds between phases and models.

| Model (6-bit) | Student rollout (ms/token) | SKD generic | SKD optimized | Teacher scoring (ms/token) | LoRA update | Training step | Peak MLX memory (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 6.76 | 2.67x | 1.69x | 1.25 | 14.64x | 15.64x | 4.55 |
| Qwen3.5-2B | 14.08 | 2.40x | 1.47x | 2.53 | 8.22x | 9.16x | 5.56 |
| Qwen3.5-4B | 29.64 | 2.24x | 1.35x | 5.09 | 6.71x | 7.71x | 10.84 |
| Qwen3.5-9B | 52.62 | 2.25x | 1.24x | 8.97 | 4.59x | 5.96x | 14.59 |

Legend:

- **Student rollout** is straight base-model generation, reported as
  decode-only milliseconds per generated token. Prompt prefill and the first
  generated token are excluded.
- **SKD generic** uses the same model and tokenizer for student and privileged
  teacher, with unbatched one-token proposal/verification
  (`max_block_size=1`, `batch_shared_model=False`). The timer includes student
  proposal plus teacher verification and repair, but excludes prompt prefill
  and the first generated token.
- **SKD optimized** measures the same dual-context rollout with shared-model
  batching and adaptive proposal blocks up to eight tokens
  (`max_block_size=8`, `batch_shared_model=True`). Its timer has the same
  decode-only boundaries as the generic path.
- **Teacher scoring** runs a frozen, adapter-disabled sequence forward on each
  independent 128-token chunk, selecting the logits that predict its final 64
  tokens. Each vector is long enough to use dense full-sequence matrix kernels
  instead of the one-token decode path. Only one chunk's logits remain live at
  a time. The result is reported in milliseconds per selected target token.
- **LoRA update** computes teacher rows one chunk at a time outside its reported
  timer. Inside the timer, each chunk performs the student sequence forward,
  forward KL over 64 target rows, and backward pass. LoRA gradients are
  accumulated across all four chunks, averaged, and applied with one rank-4
  AdamW update. Its multiple is relative to **Teacher scoring**, not decode.
- **Training step** includes one **Teacher scoring** pass followed by the full
  **LoRA update**. Rollout generation, generation prefill, chat rendering,
  tokenization, and alignment remain outside the timer. Its multiple is also
  relative to **Teacher scoring**.
- **Peak MLX memory** is the largest MLX allocator usage observed during that
  model row, including its model and measured operations. The runner checks the
  configured guard after every decode phase, scoring chunk, backward chunk, and
  optimizer update, aborting before further allocation-heavy work if exceeded.
- **SKD generic** and **SKD optimized** multiples use **Student rollout** as
  their denominator. **LoRA update** and **Training step** multiples use
  **Teacher scoring** as their denominator.

Regenerate the table with:

```sh
uv run python scripts/benchmark_qwen35_matrix.py
```
