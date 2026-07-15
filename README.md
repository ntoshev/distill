# distil

Visualize the per-token likelihood of a text/rollout according to an LLM, using a
single forward pass with [mlx-lm](https://github.com/ml-explore/mlx-lm).

Tokens are colored on a red→orange→yellow→green gradient by their log-probability
under the model: green = the model finds the token likely, red = surprising. The
palette lives in `src/distil/render.py` as hex color literals so editors can show
inline swatches. The far tail below logprob \(-7.5\), roughly probability
\(e^{-7.5} \approx 0.06\%\), is bold and underlined so near-impossible tokens
remain visually distinct.

## Usage

```sh
# score a completion given a prompt (model chat template applied)
uv run distil --model mlx-community/Qwen3.5-0.8B-4bit \
    --prompt "What is 2+2?" --completion "2+2 equals 4."

# optional system prompt; @file reads inputs from disk, "-" reads stdin
uv run distil --model mlx-community/Qwen3.5-0.8B-4bit \
    --system @system.txt --prompt @prompt.txt --completion -

# raw mode: no chat template, completion scored as continuation of prompt
uv run distil --model mlx-community/Qwen3.5-0.8B-4bit \
    --raw --prompt "1 2 3 4 5" --completion " 6 7 8 9 10"

# score a verifiers / prime-rl rollout (results.jsonl from `prime eval run`)
uv run distil --model mlx-community/Qwen3.5-0.8B-4bit \
    --rollout examples/verifiers/gsm8k-results.jsonl --index 1

# score a generated multi-turn Qwen3.5-9B-6bit Wordle rollout
uv run distil --model mlx-community/Qwen3.5-9B-6bit \
    --rollout examples/verifiers/wordle-qwen3.5-9b-6bit-results.jsonl --index 0
```

The example rollouts are copied from or generated with real verifiers evals:
GSM8K for a single-turn task and Wordle for multi-turn environments. For your
own eval runs, point `--rollout` at the generated `results.jsonl`; `...` is
accepted as a recursive wildcard path segment when it matches exactly one file,
e.g. `outputs/evals/.../results.jsonl`.

By default the output is just the text, gradient-colored (context in dim grey).
Useful flags: `--stats` (color legend + mean logprob/perplexity), `--worst N`
(table of most surprising tokens with top-k alternatives), `--metric logprob|prob|kl`
(color scale), `--json` (machine-readable output), `--hide-context` (omit the
uncolored prompt/context), `--no-thinking` (pass `enable_thinking=False` to
Qwen-style chat templates), `--color always|never|auto`.

## Teacher/student KL

Any `--teacher-*` flag switches to two-pass scoring: the same completion is
scored under two contexts — the student sees the plain conversation, the
teacher (typically the *same weights*, loaded once) sees privileged
conditioning — and each token is colored by its exact full-vocabulary
`KL(teacher ‖ student)` in nats: green = the two distributions agree, red =
strong disagreement. `--worst N` becomes a table of the largest disagreements
showing what each model preferred.

```sh
# teacher = same weights + a system prompt that reveals the answer
uv run distil --model mlx-community/Qwen3.5-0.8B-4bit     --rollout examples/verifiers/gsm8k-results.jsonl --index 1     --teacher-system "You are grading a solution. The correct answer is 3."     --stats --worst 10

# different teacher weights (must share the student's tokenizer)
uv run distil --model mlx-community/Qwen3.5-0.8B-8bit     --teacher-model mlx-community/Qwen3.5-9B-6bit  --system "Respond with one number only, the final answer"   --prompt "What is 189+67?" --sample --no-thinking --worst 3 --temperature 0
```

`--teacher-system` replaces the system message (or inserts one) for the teacher
pass only; `--teacher-prompt` replaces the first user message. With a teacher,
`--metric` defaults to `kl`; pass `--metric logprob` to color by the student's
likelihood instead.

## Sampling from the student

`--sample` generates the completion from `--model` instead of reading one, then
scores it — with a teacher, that is the on-policy-distillation loop: student
generates, teacher scores the student's tokens. `--save-rollout` appends the
sample to a `results.jsonl` (the save format is the load format, so saved
samples can be re-scored later with `--rollout`).

```sh
uv run distil --model mlx-community/Qwen3.5-0.8B-4bit     --prompt "What is 7 times 8?"     --sample --temperature 0.7 --seed 0 --max-tokens 256     --teacher-system "The user is quizzing you; the correct answer is 56."     --save-rollout samples.jsonl --stats
```

Sampling flags: `--max-tokens` (default 512), `--temperature` (default 1.0,
`0` = greedy), `--seed`.

## Development

```sh
uv run pytest                # unit + snapshot tests (fast ones need no model)
uv run pytest -m model       # integration tests (needs cached Qwen3.5-0.8B-4bit)
UPDATE_SNAPSHOTS=1 uv run pytest   # regenerate on-disk snapshots
```

See [DESIGN.md](DESIGN.md) for architecture and future direction (on-policy
distillation training on the teacher/student KL signal).
