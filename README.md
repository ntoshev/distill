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

Examples below use `mlx-community/Qwen3.5-0.8B-4bit`.

## Score and visualize text

```sh
# Apply the model's chat template and score only the completion.
uv run distil \
  --model mlx-community/Qwen3.5-0.8B-4bit \
  --prompt "What is 2+2?" \
  --completion "2+2 equals 5."

# Raw continuation scoring without a chat template.
uv run distil \
  --model mlx-community/Qwen3.5-0.8B-4bit \
  --raw --prompt "1 2 3 4" --completion " 5 6 3 8 9 10"

# Score one rollout from verifiers/prime-rl results.jsonl.
uv run distil \
  --model mlx-community/Qwen3.5-0.8B-4bit \
  --rollout examples/verifiers/gsm8k-results.jsonl --index 1
```

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
  --model mlx-community/Qwen3.5-0.8B-4bit \
  --rollout examples/verifiers/gsm8k-results.jsonl --index 1 \
  --teacher-system "The correct answer is 3." \
  --stats --worst 10
```

`--teacher-system` replaces or inserts the teacher's system message.
`--teacher-prompt` replaces its first user message. A separate
`--teacher-model` must use the same tokenizer vocabulary as the student.

## Sample from the student

```sh
uv run distil \
  --model mlx-community/Qwen3.5-0.8B-4bit \
  --prompt "What is 7 times 8?" \
  --sample --temperature 0.7 --seed 0 --max-tokens 256 \
  --teacher-system "The correct answer is 56." \
  --stats --save-rollout samples.jsonl
```

`--save-rollout` writes the same `results.jsonl` format accepted by
`--rollout`. The sample is saved before scoring so a later scoring error does
not discard it.

## Continual document learning

`--train` takes the trusted document itself. The tool enumerates its facts,
generates train/held-out questions, asks the context-aware frozen teacher for
held-out answers, then trains only on teacher/student KL. Facts retire from the
training rotation when mastered and return if they regress. Any changed canary
output stops the run, and the best earlier safe checkpoint is selected.

```sh
uv run distil \
  --model mlx-community/Qwen3.5-0.8B-4bit \
  --train @examples/distillation/sakana_wiki.txt \
  --generator-model mlx-community/Qwen3.5-2B-6bit \
  --adapter adapters/sakana-wiki \
  --canaries examples/distillation/canaries.jsonl \
  --no-thinking
```

Use `--generate-only` to inspect `generated/facts.jsonl` and
`generated/probes.jsonl`. To use an edited dataset, pass `--probes PATH`; each
row needs `train_prompts` plus the held-out `prompt` and `completion`.
`--from-adapter PATH` starts from an existing LoRA and preserves it as the
frozen teacher baseline.

Training defaults to forward `KL(teacher || student)`, speculative top-5
rollouts, evaluation every five successful updates, a target-token geometric
mean probability threshold of 0.5, and a `1e-6` learning rate.

The adapter is saved in standard mlx-lm format and can be sampled directly:

```sh
uv run distil \
  --model mlx-community/Qwen3.5-0.8B-4bit \
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

An aborted run returns nonzero but still preserves its best safe partial
checkpoint.

## Inspect checkpoint progress

Render one fixed target at every checkpoint. Each token is colored by its own
probability; the trailing percentage is geometric-mean token probability.

```sh
uv run distil \
  --model mlx-community/Qwen3.5-0.8B-4bit \
  --progress outputs/lr-sweep-v014/lr-2e-6 \
  --prompt "What is Project Lumen's launch code? Answer with the code only." \
  --completion "VIOLET-731" --metric prob --no-thinking
```

Compare several targets and canaries as compact sparklines:

```sh
uv run distil \
  --model mlx-community/Qwen3.5-0.8B-4bit \
  --progress outputs/lr-sweep-v014/lr-2e-6 \
  --probes examples/distillation/lumen-probes.jsonl \
  --no-thinking --width 140
```

Probe files contain one JSON object per line with `name`, `prompt`, and
`completion`, plus an optional `system`. `--json` emits all checkpoint and
per-token values. Progress mode performs inference over saved checkpoints but
does not retrain.

The example progress commands require the local sweep artifacts. Recreate them
with the commands in [EXPERIMENTS.md](EXPERIMENTS.md); generated sweep folders
are intentionally ignored because they contain large adapter checkpoints.

## Development

```sh
uv run pytest
UPDATE_SNAPSHOTS=1 uv run pytest
```

The full suite includes cached-model integration tests for Qwen3.5-0.8B-4bit.
Fast unit-only runs can exclude them with `uv run pytest -m 'not model'`.
