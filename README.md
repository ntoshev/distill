# distil

Visualize the per-token likelihood of a text/rollout according to an LLM, using a
single forward pass with [mlx-lm](https://github.com/ml-explore/mlx-lm).

Tokens are colored on a red→yellow→green gradient by their log-probability under the
model: green = the model finds the token likely, red = surprising.

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
    --rollout outputs/evals/.../results.jsonl --index 0
```

By default the output is just the text, gradient-colored (context in dim grey).
Useful flags: `--stats` (color legend + mean logprob/perplexity), `--worst N`
(table of most surprising tokens with top-k alternatives), `--metric logprob|prob`
(color scale), `--json` (machine-readable output), `--hide-context` (omit the
uncolored prompt/context), `--color always|never|auto`.

## Development

```sh
uv run pytest                # unit + snapshot tests (fast ones need no model)
uv run pytest -m model       # integration tests (needs cached Qwen3.5-0.8B-4bit)
UPDATE_SNAPSHOTS=1 uv run pytest   # regenerate on-disk snapshots
```

See [DESIGN.md](DESIGN.md) for architecture and future direction (teacher/student
KL divergence, on-policy distillation).
