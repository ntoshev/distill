# distill design

`distill` has two related purposes:

1. visualize token likelihood and teacher/student disagreement;
2. train a LoRA adapter by on-policy context distillation, while retaining
   enough checkpoints and probes to diagnose collapse.

The current implementation can internalize a synthetic fact, but it does not
solve locality. Lower learning rates, early stopping, canaries, and checkpoint
visualization make the failure measurable and partly controllable. The measured
trade-off is documented in [EXPERIMENTS.md](EXPERIMENTS.md).

## Architecture

```text
cli.py
  ├─ completion / rollout ─► templating.py ─► scoring.py ─► render.py
  ├─ sample ───────────────► sample.py ─────► scoring.py ─► render.py
  ├─ train ────────────────► generation.py ─► distill.py ─► training.py
  └─ progress ─────────────► progress.py ───► render.py
```

### Module responsibilities

- `templating.py` renders chat or raw inputs into `BuiltText(text, spans)`,
  where spans identify exactly which characters should be scored. It also owns
  the generation-prefix rendering shared by sampling, scoring, and training.
- `scoring.py` tokenizes once with offsets, computes chosen-token log
  probabilities, and optionally computes exact full-vocabulary
  `KL(teacher || student)` at aligned generated positions.
- `sample.py` generates from the student and retains the exact rendered prompt
  and generated token text used for scoring.
- `adapters.py` creates, disables, and serializes standard mlx-lm LoRA layers.
- `generation.py` compiles trusted context into auditable facts and probes.
- `distill.py` owns baseline evaluation, active rotation, stopping, and safe
  checkpoint selection.
- `training.py` provides the stateful one-update KL optimizer and low-level
  compatibility helper; serial and batched samples share rollout alignment.
- `progress.py` evaluates fixed or freshly sampled completions through one
  checkpoint traversal, with shared exact-completion scoring and result assembly.
- `render.py` provides token-level probability coloring, KL coloring, progress
  rows, sparklines, tables, and JSON output.
- `rollout.py` reads and writes verifiers-compatible `results.jsonl` files.
- `cli_config.py` owns argument parsing, validation, and mode-specific defaults.
- `cli.py` dispatches to training, progress, and scoring/sampling handlers, keeping
  model loading after validation.

## Core scoring representation

All scoring paths reduce to a complete rendered string plus character spans
marking generated content. The string is tokenized once. A token is scored when
its character range overlaps a scored span and it has preceding context.

This avoids concatenating separately tokenized prompt and completion segments.
Such concatenation is incorrect at mergeable boundaries: for example, adjacent
newlines from a Qwen generation prompt and completion may become one token.

Teacher/student scoring builds one rendered string per context. Generated token
IDs must align exactly and in order; incompatible tokenizers or boundary changes
raise an error rather than producing approximate KL values.

## Likelihood and KL visualization

For a completion token \(y_t\), single-model scoring reports

\[
\log p(y_t\mid x,y_{<t}).
\]

The default color scale is log probability. `--metric prob` maps color linearly
in probability. Teacher mode computes

\[
D_{\mathrm{KL}}\!\left(p_T(\cdot\mid c,x,y_{<t})\,\|\,
p_S(\cdot\mid x,y_{<t})\right)
\]

over the full shared vocabulary. Green means agreement; red means divergence.
When teacher and student share base weights, the teacher pass temporarily
disables LoRA rather than loading a second model.

## On-policy context distillation

For question \(x\), the current LoRA student samples a response \(\hat y\)
without privileged context. A frozen teacher evaluates the same response
prefixes with privileged context \(c\). Only LoRA parameters \(\phi\) update:

\[
\mathcal L(\phi)
= \frac{1}{|\hat y|}\sum_{t=1}^{|\hat y|}
D_{\mathrm{KL}}\!\left(
p_T(\cdot\mid c,x,\hat y_{<t})\,\|\,
p_{S,\phi}(\cdot\mid x,\hat y_{<t})
\right).
\]

The teacher does not generate a replacement answer. It supplies dense
next-token targets along the student's own sampled trajectory. `--iters` counts
successful aligned updates; failed alignments are skipped with a warning.

With `--kl-direction reverse`, the per-token objective swaps the distributions:

\[
D_{\mathrm{KL}}\!\left(
p_{S,\phi}(\cdot\mid x,\hat y_{<t})\,\|\,
p_T(\cdot\mid c,x,\hat y_{<t})
\right).
\]

The teacher logits remain fixed; gradients flow only through the student
distribution. Forward KL is the default for backward compatibility.

With `--speculative-top-k K`, rollout generation instead follows the
token-interleaved policy from Speculative Knowledge Distillation. At each
prefix, the student proposes (z_t \sim p_S). If (z_t) is among the teacher's
top-(K) tokens, it is accepted; otherwise the rollout uses a replacement
(z_t \sim p_T). Both policies then continue from the selected token. The loss
remains forward teacher-to-student KL on every aligned position of this mixed
trajectory. Metrics record accepted and replaced token counts per update.

`--speculative-top-p P` replaces the fixed token count with a probability-mass
threshold. The acceptance set is the smallest prefix of teacher tokens sorted
by descending probability (ties by ascending token ID) covering at least P.
The token that crosses P is included. Probabilities are computed from the raw
teacher logits before sampling temperature; P must be in `(0, 1]`, and P=1
accepts all proposals. For a single proposal, summing the probability mass of
tokens that precede it tests membership without sorting the vocabulary. This
policy applies to generic, shared-model, and multi-rollout decoding, and records
`speculative_top_p` in sample/run/rollout metadata and `top_p` in step metrics.

Continual evaluation batches fixed target scoring, greedy held-out answers,
and all canaries through MLX-LM's batched prefill and decode. Fixed target rows
teacher-force their canonical completion IDs; generated rows retain their actual
IDs and untempered token probabilities, so canary coloring needs no rescoring.
Per-row limits and EOS handling remain independent. Answer equivalence judging
runs as a second batch because it depends on the generated answers, with LoRA
disabled when the judge shares the student model. Startup canary generation and
validation are also batched. Quantized kernels can produce slightly different
probabilities or near-tied token choices than sequential evaluation.

Training writes:

```text
ADAPTER_PATH/
  adapter_config.json
  adapters.safetensors
  training_metrics.jsonl
  eval_metrics.jsonl
  selection.json
  run_config.json
  checkpoints/
    step-0000/
    step-0005/
    ...
  generated/
    source.json
    facts.jsonl
    probes.jsonl
```

Document compilation uses greedy generation. Each fact produces a structured
subject, one exact completion span copied from the fact, and distinct
train/held-out questions with the same answer scope. Mechanical validation
rejects unsupported fields, duplicate questions, completion leakage, missing
subjects, and document-relative wording. The exact completion span is the probe
completion; no second model cleans up or rewrites it. Both JSONL files are
persistent and can be reloaded with `--probes`.

Training rows contain rollout and loss diagnostics. Evaluation rows contain
probe probabilities, active items, exact canary drift, and confirmation output.

## Collapse monitoring and control

The implementation deliberately separates three signals:

- **learning:** fixed target completions on training and held-out prompts;
- **locality:** unrelated canary completions and contamination targets;
- **capability retention:** external evaluations such as GSM8K.

`--canaries` performs greedy generation at the shared evaluation cadence and
compares exact output against the starting-model baseline. Any change aborts
training; selection still returns the best earlier safe checkpoint.

`--progress` evaluates saved checkpoints without retraining:

- `--progress PATH --prompt ... --completion ...` renders the completion at
  every step, with each token colored by its own probability;
- `--progress PATH --probes probes.jsonl` renders one sparkline per fixed
  completion.

For a multi-token completion, each sparkline uses geometric-mean token
probability

\[
q = \exp\!\left(\frac{1}{T}\sum_{t=1}^{T}
\log p(y_t\mid x,y_{<t})\right).
\]

This is length-normalized and useful for trajectories. It is not the exact
sequence probability, which is \(q^T\), and separately scored completions do
not form a normalized distribution.

## CLI contracts

The primary modes are mutually exclusive:

- `--completion` for a supplied completion;
- `--rollout` for one saved rollout;
- `--sample` for a generated completion;
- `--train` for LoRA distillation;
- `--progress` for checkpoint analysis.

Progress mode requires exactly one of a single `--completion` or a JSONL
`--probes` file. Training takes trusted context through `--train`, requires
`--adapter` and `--canaries`, and either generates or consumes structured
learning probes. Training-only and scoring-only flags are rejected before a
model is loaded.

## Known limitations

- The single-fact experiment exhibits broad answer contamination. The present
  controls delay and expose collapse; they do not prevent it.
- Canaries are sparse sentinels, not a proof of retained general capability.
- There is no replay across documents, regularization against the starting
  policy, batching, gradient accumulation, or resume support.
- Intermediate assistant turns in thinking-model rollouts may be shown as
  context rather than scored because templates can strip prior thinking blocks.
- Full-sequence and cached generation log probabilities differ slightly under
  4-bit quantization because they use different kernels.
- Distinct teacher models must share the student's tokenizer. KL across
  different vocabularies is undefined.
- Forward `KL(teacher || student)` remains the default; reverse KL is available
  as an ablation.
- Progress analysis reloads adapters sequentially and does not cache computed
  probabilities; it avoids retraining but still performs inference.

## Testing

Fast tests cover argument validation, templating boundaries, rollout I/O,
probability/KL color scales, JSON schemas, checkpoint discovery, probe loading,
progress rendering, and training-file validation.

Model integration tests use cached `mlx-community/Qwen3.5-0.8b-6bit` weights to
check:

- fixed ANSI and JSON scoring snapshots;
- greedy generation/scoring alignment;
- near-zero KL under identical contexts;
- sample/teacher alignment;
- adapter training, checkpoint creation, reload, held-out fact recall, and KL
  reduction after 32 updates at `2e-6`.

Run all tests with `uv run pytest`. Snapshot updates are explicit through
`UPDATE_SNAPSHOTS=1 uv run pytest`.
