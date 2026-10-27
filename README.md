# distill

`distill` is a small MLX command-line tool for:

- visualizing token likelihood and teacher/student KL;
- sampling and saving model rollouts;
- distilling privileged context into a LoRA adapter or Qwen3 cartridge;
- memorizing raw passage continuations with a learned attention memory;
- inspecting learning and contamination across saved checkpoints.

See the [three-step demo](DEMO.md) for copyable CLI inputs and colored output:
sequence surprise, top-p teacher guidance, and single-fact training with canaries.

The current single-fact experiment learns `VIOLET-731`. Top-5 Speculative KD
creates a checkpoint where the fact is learned before any measured canary
contamination, but a small GSM8K diagnostic remains below the frozen base.
Per-step checkpoints, contamination canaries, and progress visualization make
the trade-off visible and allow useful early stopping; they do not establish
that forgetting is eliminated. See
[EXPERIMENTS.md](EXPERIMENTS.md) for the measurements and
[DESIGN.md](DESIGN.md) for implementation details.

Cartridge usage, restart semantics, limitations, and validation are documented
in [CARTRIDGE.md](CARTRIDGE.md).

## Setup

```sh
uv sync
```

Examples below use `mlx-community/Qwen3.5-0.8b-6bit`.

## Score and visualize text

```sh
# Apply the model's chat template and score only the completion.
uv run distill \
  --model mlx-community/Qwen3.5-0.8b-6bit \
  --prompt "What is 2+2?" \
  --completion "2+2 equals 5."

# Raw continuation scoring without a chat template.
uv run distill \
  --model mlx-community/Qwen3.5-0.8b-6bit \
  --raw --prompt "1 2 3 4" --completion " 5 6 3 8 9 10"

# Save a supplied chat completion in rollout format.
uv run distill \
  --model mlx-community/Qwen3.5-0.8b-6bit \
  --prompt "What is 2+2?" --completion "4" \
  --save-rollout samples.jsonl

# Score one rollout from verifiers/prime-rl results.jsonl.
uv run distill \
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
uv run distill \
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
uv run distill \
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

Use `--speculative-top-p P` instead to accept proposals in the smallest set of
highest-probability teacher tokens whose cumulative probability reaches P.
For example, `--speculative-top-p 0.9` covers at least 90% of the teacher's
probability mass, adapting the number of allowed tokens to its confidence.
The threshold must be in `(0, 1]`; `1` accepts every student proposal. The
acceptance set uses teacher probabilities before temperature scaling (including
with `--temperature 0`), with ties resolved by token ID. Teacher replacements
still use `--temperature`. The two flags are mutually exclusive and support
both `--sample` and `--train`; training keeps its top-k 3 default when neither
is supplied. Saved rollouts and training metrics record the selected policy.

```sh
uv run distill \
  --model mlx-community/Qwen3.5-0.8b-6bit \
  --prompt "What is Project Lumen's launch code?" \
  --sample --speculative-top-k 5 --temperature 0 --seed 0 --max-tokens 16 \
  --teacher-system "Private reference: Project Lumen's launch code is VIOLET-731. Reply with it." \
  --no-thinking --stats
```

## Distill a shared prompt from supplied cases

`scripts/distill_prompt_responses.py` bypasses document extraction and question
generation. It runs a fixed system prompt against supplied questions, saves the
teacher's exact generated tokens (including EOS), and trains independent LoRA
adapters on nested response subsets. The student sees the question without the
system prompt; the frozen teacher sees the question with it. Training uses the
project's full-vocabulary forward-KL objective along those saved responses.

Prepare an output directory containing `system_prompt.txt`, `train_cases.jsonl`
(rows with `case_id` and `question`), and `eval_cases.jsonl` (rows with
`example_id`, `ifeval_key`, and `question`). Evaluation uses a frozen scorer
directory containing an importable `evaluate.py` with the shared-prompt
`score(row, text)` function; this experiment's scorer also needs `nltk`,
`langdetect`, and `immutabledict` installed in the chosen Python environment.

```sh
PYTHONPATH=src python scripts/distill_prompt_responses.py \
  --output-dir outputs/prompt-run \
  --scorer-dir path/to/frozen-scorer \
  --sizes 8 16 32 64 128 --epochs 3 \
  --enable-thinking --max-tokens 16384 \
  --temperature 1.0 --top-p 0.95 --top-k 20 --presence-penalty 1.5
```

The runner generates answers once, preserves both compliant and noncompliant
teacher answers, initializes each adapter independently from the same base,
and evaluates all adapters without the system prompt. Completed generation and
evaluation files can resume; an interrupted training run must be preserved or
moved before restarting that adapter. `TrainingSession.step_samples` exposes
the same saved-response training path for other runners. Its `Sample` inputs
must preserve the actual completion text and generated special tokens.
Saved teacher responses also pass their original `token_ids` directly to training,
because decoding and re-encoding sampled text can change token boundaries.

Reasoning and the final answer share the configurable `--max-tokens` budget
(default 16,384). Only the answer after `</think>` is scored; training includes
the entire reasoning trace, answer, and actual EOS. The runner rejects truncated
or unfinished teacher responses before training. It retains complete answers
that fail the formatting rules. Use a new output directory when changing the
budget or other recorded settings, so resumed rows cannot mix configurations.
The example uses Qwen3.5's recommended sampling settings for general reasoning;
greedy reasoning can repeat indefinitely despite a larger token budget. Presence
penalties cover the entire generated continuation, excluding the prompt. Every
evaluation variant uses the same settings. Generation recreates its batch after
16 cases and synchronizes and releases unused buffers every 64 decoding steps.

The runner defaults to `--loss-chunk-size 128`, `--completion-batch-size 4`,
`--prefill-batch-size 1`, and `--prefill-step-size 512`. The Qwen3/Qwen3.5
training path computes exact full-vocabulary KL in token chunks and recomputes
layer activations during backpropagation, preserving gradients through the full
sequence. It requires a frozen output head. This avoids holding every token's
vocabulary logits at once; reducing the logical training batch alone does not
solve that allocation. These options are independent of the logical
`--batch-size 4`, which still averages four sequence gradients per update.

The smaller instruction-following diagnostic in
[`experiments/shared_prompt_forbidden/PROTOCOL.md`](experiments/shared_prompt_forbidden/PROTOCOL.md)
uses IFEval's built-in forbidden-word rule: avoid the whole word "the" in the
final answer. It freezes eight supplied training questions and eight separate
validation questions, compares thinking on/off, and gates a three-learning-rate
pilot on real-model gradient checks:

```sh
PYTHONPATH=src:. python scripts/diagnose_forbidden_prompt.py \
  --output-dir outputs/prompt-forbidden-the-diagnostic-2026-09-23
PYTHONPATH=src:. python scripts/train_forbidden_pilot.py \
  --output-dir outputs/prompt-forbidden-the-diagnostic-2026-09-23
```

Both commands require the supplied `outputs/prompt-internalization-2026-09-19`
artifact. This is a custom shared-prompt experiment, not the official IFEval
benchmark. KL uses standard float32 MLX reductions: the fused Metal KL kernel
in mlx-lm 0.31.3 produced nondeterministic losses and incorrect gradients at
the production vocabulary and chunk sizes on M4 Max.

## Continual document learning

For a single fact, supply all texts directly. The training question and held-out
question must differ; this bypasses fact and question generation:

```sh
uv run distill --model mlx-community/Qwen3.5-0.8B-6bit \
  --train "Project Lumen's launch code is VIOLET-731." \
  --prompt "What is Project Lumen's launch code? Answer with the code only." \
  --eval-prompt "Give the launch code assigned to Project Lumen. Answer with the code only." \
  --completion "VIOLET-731" \
  --canary "What is 7 plus 5? Answer with the number only." \
  --canary "What color is grass usually? Answer with the color only." \
  --adapter adapters/lumen-inline --speculative-top-p 0.9 \
  --eval-every 1 --no-thinking
```

The native checkpoint grid shows the scored held-out target beside actual generated
canary answers at each evaluation. Each token retains its own student
probability and token boundary. `p` is the geometric-mean probability, including
generated end tokens; `--show-special-tokens` reveals their spellings. New rows
and evaluation status append to terminal scrollback, preserving every checkpoint
without redrawing or hiding earlier output. This also applies to progress
evaluation. Redirected output prints the completed grid once.

Inline canaries use their initial generated answers as retention baselines;
they do not independently establish factual correctness. Use `--canaries PATH`
with reference completions when baseline correctness needs to be checked.
`learned` requires the existing held-out generation judge and patience policy.
`changed` includes punctuation drift; `leak` means an answer changed to an exact
target completion. These labels do not change the existing stopping policy.

Replay the recorded grid without loading model weights or regenerating text:

```sh
uv run distill --model mlx-community/Qwen3.5-0.8B-6bit \
  --progress adapters/lumen-inline --color always
```

The run saves inputs and exact per-token scores in `eval_metrics.jsonl` for
replay. Supplying explicit progress inputs instead evaluates saved checkpoints.

`--train` takes the trusted document itself. The tool enumerates its facts,
generates aligned train/held-out questions plus an exact source-grounded completion,
then trains only on teacher/student KL. Facts retire from the training rotation
after their greedy held-out answer passes a frozen LLM equivalence judge at four
consecutive evaluations, and return if they regress. Set `--mastery-confirmations N`
to change the required number of consecutive passes (default 4, must be positive).
Confirmations count evaluations, whose spacing is controlled by `--eval-every`;
`--iters` remains an upper limit on optimizer updates. The judge treats the
source-grounded reference as authoritative and compares values without seeing
the factual question or document, preventing its world knowledge from replacing
the reference. Training and held-out
questions share an explicit answer-only instruction, so extra explanations or
contradictory text cannot satisfy mastery. Any changed canary output stops the
run, and the best earlier safe checkpoint is selected.

```sh
uv run distill \
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
optimizer updates, a `1e-6` learning rate, and the frozen
`mlx-community/Qwen3.5-4B-6bit` mastery judge. Override the latter with
`--judge-model MODEL`. Each evaluation also records the
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
uv run distill \
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

## Evaluate an adapter with Prime/verifiers

`scripts/eval_prime.sh` starts an OpenAI-compatible MLX server, runs the
Prime/verifiers environment once with the base model and once with the adapter,
then prints a paired comparison and diff. Without `--output-dir`, intermediate
results are temporary; with it, results are saved under `base/` and
`with-adapter/`.

```sh
# Paired base-versus-adapter comparison, terminal-only results.
scripts/eval_prime.sh \
  --environment gsm8k \
  --num-examples 16 \
  --adapter outputs/fifa-first-5

# The same comparison, with both result sets saved.
scripts/eval_prime.sh \
  --environment gsm8k \
  --num-examples 16 \
  --adapter outputs/fifa-first-5 \
  --output-dir outputs/prime-gsm8k/fifa-vs-base-16

# A local environment package, such as environments/gsm8k_slice.
scripts/eval_prime.sh \
  --environment gsm8k-slice \
  --env-dir-path environments \
  --num-examples 16
```

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
uv run distill \
  --model mlx-community/Qwen3.5-0.8b-6bit \
  --progress outputs/lr-sweep-v014/lr-2e-6 \
  --prompt "What is Project Lumen's launch code? Answer with the code only." \
  --completion "VIOLET-731" --metric prob --no-thinking
```

Compare several targets and canaries in the same checkpoint grid:

```sh
uv run distill \
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

To inspect the actual generated response rather than score the fixed
`completion`, point `--progress --rollout` at a saved training rollout. It
replays that row's recorded prompt and context at every checkpoint, then scores
and colors the exact freshly generated text. The displayed output therefore
preserves the model's actual token boundaries; it is not a fixed answer colored
with another string's probabilities.

```sh
uv run distill \
  --model mlx-community/Qwen3.5-0.8b-6bit \
  --progress outputs/lr-sweep-v014/lr-2e-6 \
  --rollout outputs/lr-sweep-v014/lr-2e-6/rollouts.jsonl --index 0 \
  --temperature 0 --max-tokens 16 --seed 0 --metric prob --no-thinking
```

`--rollout` requires the saved rollout path so the sampling prompt cannot drift
from the training distribution. `--sample` remains available for ad-hoc prompt
or probe exploration. For the same distribution used by continual training:

```sh
uv run distill \
  --model mlx-community/Qwen3.5-4b-6bit \
  --progress outputs/fifa-first-5 \
  --probes examples/distillation/2026_fifa_world_cup_first_5_probes.jsonl \
  --progress-prompt train --sample --no-thinking
```

For probe sets, `--system` is the default for rows without their own `system`,
and `--metric prob|logprob` selects the rendered summary. Teacher flags enable
checkpoint-by-checkpoint KL scoring; its default metric is `kl`.

The example progress commands require the local sweep artifacts. Recreate them
with the commands in [EXPERIMENTS.md](EXPERIMENTS.md); generated sweep folders
are intentionally ignored because they contain large adapter checkpoints.

## Learn a discrete reversal prompt

The frozen-model reversal experiment learns a shared system prompt with PEZ or
GCG, comparing lengths 1, 2, 4, 8, and 16 across three optimizer seeds.
Answers come from a programmatic oracle. Evaluation uses the exported text
prompt in ordinary greedy generation, with separate fresh-list, unseen-word,
and longer-list tests.

```sh
# Exact cached snapshot used by this evaluation.
REVERSAL_MODEL=/Users/stefanpetrov/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-6bit/snapshots/fbb1092aa9daef007603ea9273cd24c225a409f1
DISTIL_BENCHMARK_SYSTEM_ID=mac16-9-m4-max-14cpu-32gpu-36gb \
uv run python scripts/run_reversal_suite.py \
  --model "$REVERSAL_MODEL" \
  --output-dir outputs/reversal-pez-fixed-20261009 --dataset-seed 20261010 \
  --methods pez --lengths 1 2 4 8 16 --seeds 0 1 2 \
  --steps 200 --batch-size 2 --learning-rate .01 \
  --pez-optimizer adafactor --pez-gradient-clip 1 \
  --initialization neutral --checkpoint-selection accuracy --eval-every 10

uv run python scripts/analyze_reversal_prompt.py \
  --output-dir outputs/reversal-pez-fixed-20261009 --methods pez
```

The suite preserves interrupted arms before restarting and reuses completed
arms with an identical configuration. It saves real generated token IDs, stop
reasons, checkpoint prompts, per-seed results, and optimizer timings. See the
[protocol](experiments/reversal_prompt/PROTOCOL.md) for the objective, tokenizer
constraints, gradient checks, and numerical limits of the quantized model.
The [completed evaluation](experiments/reversal_prompt/RESULTS.md) compares all
lengths, optimizer seeds, manual instructions, random search, and continuous
controls in the historical setup. GCG learned partial reversal; the ten-token
manual instruction was more reliable. Historical PEZ scored 0%.
The [PEZ diagnosis](experiments/reversal_prompt/PEZ_DIAGNOSIS.md) documents the
Adam correction and development ablations; the
[corrective evaluation](experiments/reversal_prompt/PEZ_RESULTS.md) reports the
separate fresh-data rerun reproduced above. A lower training loss alone does not
establish successful discrete-prompt learning.

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
the same hardware. Committed baselines are available for these systems:

| System | `DISTIL_BENCHMARK_SYSTEM_ID` |
|---|---|
| Mac15,6 — Apple M3 Pro (11-core CPU, 14-core GPU, 36 GB) | `mac15-6-m3-pro-11cpu-14gpu-36gb` |
| Mac16,9 — Apple M4 Max (14-core CPU, 32-core GPU, 36 GB) | `mac16-9-m4-max-14cpu-32gpu-36gb` |

For example, run the M3 Pro baseline with:

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

The committed timing values are below. Seconds are absolute medians; multiples
are relative benchmarks measured against the named reference in the same run.
The parenthetical percentage is the allowed regression tolerance.

| Benchmark | M3 Pro | M4 Max |
|---|---:|---:|
| Prompt/completion scoring | 0.471437 s (2%) | 0.202333 s (20%) |
| Teacher scoring, 128 tokens | 0.088454 s (5%) | 0.030341 s (20%) |
| LoRA update / teacher scoring | 14.020422x (5%) | 14.157179x (20%) |
| Training step / teacher scoring | 15.874657x (5%) | 14.809739x (20%) |
| Optimized SKD / student rollout | 2.700000x (2%) | 1.679466x (20%) |
| Optimized 4B SKD / student rollout | 1.470000x (2%) | 1.304027x (20%) |
| Generic 4B SKD / student rollout | 2.244312x (2%) | 2.113065x (20%) |
| Batched three-rollout SKD / serial | 0.603775x (15%) | 0.627939x (20%) |
| Optimized three-item training step | 1.857335 s (15%) | 0.757659 s (20%) |

The SKD rollout benchmarks are relative rather than absolute: their decode-only
milliseconds per generated token are compared with straight-student generation
in the same run. Prompt prefill and the first generated token are excluded from
both measurements. Training is instead anchored to a compute-dense
frozen-teacher forward pass over 128 tokens, with 64 prediction rows selected.
One benchmark isolates the student forward-KL LoRA update with teacher rows
precomputed, while a second adds the teacher scoring pass. Rollout generation,
generation prefill, chat rendering, tokenization, and alignment are outside both
training timers. Training costs are reported per target token as multiples of
teacher scoring per target token.

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
