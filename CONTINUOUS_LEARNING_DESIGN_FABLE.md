# Continual learning: internalizing a document into LoRA weights

Status: design under discussion; per-decision status is in the table at the
end. Nothing here is implemented.

Goal: one command that takes a privileged text or markdown document and
produces a LoRA adapter whose student answers questions about that document
without seeing it:

```sh
uv run distil \
  --model mlx-community/Qwen3.5-0.8B-4bit \
  --distill examples/distillation/sakana_wiki.txt \
  --adapter-path adapters/sakana-wiki \
  --no-thinking
```

The canonical example is `examples/distillation/sakana_wiki.txt`: a one-page,
multi-fact document (founders, funding rounds, model merging, the AI
Scientist, valuation) rather than the single synthetic fact of Project Lumen.

## What exists and what is missing

The training loop already works end to end once questions exist:
`train_lora` samples student rollouts without context, scores them under a
frozen privileged-context teacher, updates only LoRA parameters, checkpoints
every step, and samples contamination canaries. `evaluate_lr_sweep.py` shows
the intended selection logic, but it is post hoc and hard-codes the Lumen
questions and the `VIOLET-731` target.

Missing for end-to-end operation:

1. **Question generation** over the privileged document (the largest gap);
2. **Target-answer probes** derived from the document, so "learned" is
   measurable without hand-written targets;
3. **An in-loop stop condition** combining loss, probe results, and canaries,
   instead of a fixed `--iters` plus post-hoc selection;
4. **Orchestration**: generate → train → monitor → stop → select → finalize.

Intuition for the pipeline: the document is turned into a syllabus (training
questions), an exam (held-out probes with target answers), and tripwires
(canaries). Training continues until the student passes the exam without the
cheat sheet, and stops before the tripwires show it is forgetting everything
else. Every intermediate artifact is written to disk so a failed run is
diagnosable with existing tools (`--progress`, probes, metrics).

Prior-work framing, kept loose: this is context distillation with synthetic
self-QA. Question synthesis for knowledge injection appears in Self-QA and
synthetic continued pretraining (EntiGraph); the rollout policy options come
from Speculative Knowledge Distillation, already implemented here.

## Design principle: judgment in prompts, mechanics in code

Wherever a decision requires judgment — is this question self-contained, is
this answer grounded in the document, does this phrasing name its subject —
the decision belongs in a model prompt, not in code. Code implements only
mechanical, deterministic operations: parsing, normalization, exact string
comparison, hard caps, file layout. When judgment quality is insufficient,
the remedy is a stronger model behind the same prompt (`--generator-model`),
never a new heuristic in code.

This keeps the code small and exactly testable, and it concentrates quality
tuning in two inspectable places: the prompt text and the generated
artifacts. The rule shapes the pipeline throughout: quality criteria live in
the generation prompts (A2), filtering is mechanical only (A3), the
contamination regex is a literal alternation of probe answers rather than a
salience heuristic (Stage C), and the escalation path for bad generations is
a better generator, not more filtering (D10).

## Pipeline overview

```text
--distill DOC
  1 chunk        DOC ─► chunks (paragraph/heading based)
  2 generate     chunks ─► candidate (question, answer) pairs   [generator model]
  3 filter       mechanical only: parse, dedupe, length/answer caps
                 (quality criteria live in the generation prompts)
  4 split        each kept item ─► train question + held-out paraphrase probe
  5 baseline     score every probe on the base student; items already known
                 start mastered and sit out of the training rotation
  6 train        existing train_lora loop over unmastered items
                 (teacher = doc in system prompt)
  7 monitor      every E steps: probe scores + canary drift + loss;
                 items retire on mastery and return on regression
  8 stop         all items mastered ∧ confirmed, or budget/locality abort
  9 select       best checkpoint by (contaminated, changed, step); copy to root
```

Stages 1–5 run before any weight update and write inspectable artifacts.
Stages 6–9 reuse the existing loop with a new evaluation callback.

## Stage A: question and probe generation (baseline)

### A1. Chunking

Split markdown on headings, then paragraphs on blank lines; merge fragments
shorter than ~200 characters into their neighbor; always keep the full
document as one additional "chunk" when it fits the context budget, so
cross-paragraph synthesis questions are possible. For `sakana_wiki.txt` this
yields three content paragraphs plus the whole document.

### A2. Candidate generation

The generator is a model prompted with each chunk. Default:
`mlx-community/Qwen3.5-2B-6bit` — a preliminary experiment showed clearly
better question and answer quality than 0.8B self-generation, and generation
is a one-time cost. Unlike the teacher, the generator only produces text, so
it has **no tokenizer-compatibility constraint**; `--generator-model` can
name any local model without affecting KL alignment.

Question prompt, per chunk (greedy is too repetitive; sample at moderate
temperature, e.g. 0.7):

```text
Here is a reference document:

{chunk}

Write {n} short factual questions that this text answers.
Rules:
- Every question must be fully self-contained: name its subject
  explicitly (never "the company", "it", "they", or any other
  reference that needs the text to resolve).
- Only ask questions whose answer is stated in the text.
- Prefer questions with short, specific answers: a name, a number,
  a date, a phrase.
One question per line, no numbering.
```

Answer generation, per surviving question, greedy with privileged context —
performed by the **teacher** (the base model with the document), not the
generator: the probe target must be an answer the teacher can actually
produce, because the teacher's distribution is what gets distilled.

```text
system: Reference:
        {chunk}
        Answer using only this reference. Be accurate and concise.
user:   {question}
```

Paraphrase generation, per kept learning question (used for the held-out
probe): "Rewrite this question with different wording; keep the same meaning
and name the same subject."

Expected shape for the Sakana document: "Who founded Sakana AI?" →
"David Ha, Llion Jones and Ren Ito", with held-out paraphrase "Name the
founders of Sakana AI." This mirrors the Lumen train/held-out pair, generated
instead of hand-written.

### A3. Mechanical filtering

The quality criteria — self-containment, groundedness, concise answers —
live in the generation prompts above, not in code. Code applies only
mechanical checks with no judgment in them:

1. **Parse**: strip bullets and numbering; drop empty lines.
2. **Dedupe**: drop duplicates after normalization (casefold, collapse
   whitespace, strip terminal punctuation).
3. **Caps**: drop questions over ~200 characters; drop items whose greedy
   teacher answer is empty or over `--max-answer-tokens` (default 32).

Every candidate and verdict is recorded in `generated/candidates.jsonl`,
and `--generate-only` exists to inspect the set before spending training
time. If questions come back ambiguous or ungrounded, the remedy is a
stronger `--generator-model`, not more filter code.

How the two semantic failure shapes surface, given no code defends against
them:

- **Wrong answers** are bounded by teacher-with-context quality, not
  generator quality, because the teacher writes the targets — and
  teacher-with-context is the ceiling of context distillation anyway. If
  the teacher misreads the document, the adapter faithfully internalizes
  the misreading; no filter could have prevented that.
- **Ambiguous questions** that slip past the prompt rules do *not* fail
  loudly: the teacher answers them (it has the context), the student
  masters them, and the stop condition is satisfied. The damage is a wrong
  storage key — the fact is attached to a generic cue like "the company"
  rather than to Sakana AI, the Lumen capital-of-France over-generalization
  built in by construction, and the matching ambiguous probe passes without
  demonstrating that the subject-anchored question works. This is the
  accepted trade for baseline simplicity; it is what to look for when
  inspecting `--generate-only` output, and the first thing to blame if a
  learned adapter answers unrelated questions with document content.

### A4. Held-out split and uniform mastery

Every kept item contributes its **original question** to the training set and
its **paraphrase** as a held-out probe `{name, prompt, completion}` in the
existing probe format. If the paraphrase fails the mechanical filters, the
original doubles as the probe and the item is flagged `no_heldout`.

No item is special-cased by provenance. Whether the base model already knows
a fact is discovered by measurement, not categorized at generation time: an
item is **mastered** at checkpoint \(t\) when its probe score
\(q_i(t) \ge \tau\), and mastered items sit out of the training rotation.
Items mastered at \(t = 0\) — real documents overlap pretraining; Sakana AI
is a real company — never enter the rotation at all. Every probe keeps being
evaluated for the entire run, so an item whose score later falls below
\(\tau\) (learning fact B disturbed fact A) returns to the rotation, and
intra-document forgetting shows up in the same metric as learning.

Probes are held-out paraphrases never trained on, so mastery measures fact
transfer, not memorization of the training question string.

### A5. Artifacts

```text
ADAPTER_PATH/generated/
  generation.json     # generator model, seed, chunking, counts, filter stats
  chunks.jsonl
  candidates.jsonl    # every candidate with filter verdicts and provenance
  train-questions.txt # accepted training questions, one per line
  probes.jsonl        # one held-out probe per item (progress-compatible)
```

Baseline probe scores are recorded as the iteration-0 row of
`eval_metrics.jsonl`; canaries remain the user-provided `--canaries` file.

`--generate-only` stops after this stage for inspection or manual editing;
`--questions` / `--probes` accept pre-existing files and skip generation.
Everything downstream consumes only these files, so generated and
hand-authored inputs are interchangeable.

## Stage B: training

Unchanged `train_lora` loop. The teacher system prompt is assembled from the
document:

```text
Private reference document:

{doc}

Answer questions using this reference. Be accurate and concise.
```

`--distill` therefore rejects `--teacher-system`. Multi-question interleaving
already exists (prompts are shuffled and cycled); the loop additionally needs
a small extension so the active rotation can shrink and grow as items master
and regress — an evaluation hook called at checkpoint cadence that returns
the updated rotation and a stop verdict, with orchestration in a new
`distill.py` module. Proposed training defaults: Speculative KD
`--speculative-top-k 5`, forward KL, `--learning-rate 1e-6` — deliberately
gentler than the `2e-6` configuration that learned Lumen cleanly at step 28;
the EXPERIMENTS.md budget table doubles the step allowance at this rate, and
optimizing for speed comes after the pipeline works. Rollout
`--max-tokens 64` (wiki answers are longer than launch codes). Rank stays at
the CLI default 16 pending a multi-fact sweep; Lumen used 4 for a single
fact.

## Stage C: monitoring, stop condition, selection

### Signals

Every `--eval-every` updates (default 5), aligned with a checkpoint:

- **Learning** — each probe \(i\) is scored by geometric-mean token
  probability of its target completion, the existing progress metric:
  \[
  q_i(t) = \exp\!\left(\frac{1}{T_i}\sum_{s=1}^{T_i}
  \log p_\phi(y_s \mid x_i, y_{<s})\right).
  \]
  One forward pass per probe — roughly an order of magnitude cheaper than
  greedy generation, which is what makes dense evaluation affordable.
- **Locality** — canary drift (`changed`) and contamination, moved from
  every step to the same eval cadence in this mode; Lumen timing showed
  canary generation dominating step time.
- **Loss** — the distillation loss, used only as a budget guard: the Lumen
  experiments show loss decreasing while recall and locality get worse, so
  loss never counts as success.

Contamination regex: derived mechanically as an alternation of the escaped
probe target answers themselves — a canary output containing one verbatim is
contaminated — overridable by the existing `--contamination-regex`.

### Mastery and stop condition

With \(\tau\) = `--learned-threshold` (default 0.5), the mastered set at
checkpoint \(t\) is

\[
M(t) = \{\, i : q_i(t) \ge \tau \,\}.
\]

The training rotation is the complement of \(M(t)\), updated at every
evaluation (mastered items retire; regressed items return).

- **Success**: \(M(t)\) contains every item for \(m\) consecutive
  evaluations (default 2), then **confirmation** by greedy generation over
  all probes with normalized-containment matching. Probability is the cheap
  dense signal; greedy confirmation guards against threshold miscalibration.
  Failed confirmation resumes training.
- **Locality abort**: any canary changed or contaminated — both budgets are
  fixed at **zero** in v1 — for \(G\) consecutive evaluations (default 3)
  while unmastered items remain: the learning window has closed.
- **Budget abort**: `--max-iters` reached (default
  \(\min(2000, \max(200, 200\,|U_0|))\) where \(U_0\) is the set of items
  unmastered at \(t=0\), following the cumulative-learning-rate heuristic in
  EXPERIMENTS.md at `1e-6`; needs calibration), or no new best \(|M(t)|\)
  and no new best loss over a patience window.

A single stubborn item — for example a subtly ungrounded answer that
survived filtering — can block the success criterion; the budget abort
bounds the damage and selection still returns the best partial checkpoint.

### Selection and finalization

Among evaluated checkpoints satisfying the success criterion, select

\[
t^\* = \arg\min_t \bigl(\text{contaminated}(t),\ \text{changed}(t),\ t\bigr),
\]

the tuple order used by `evaluate_lr_sweep.select_best_checkpoint`. With
zero canary budgets the success path reduces to the earliest confirmed
checkpoint; the tuple still orders partial checkpoints after an abort. The
selected checkpoint is **copied to the adapter root** — in this mode the
final adapter is the best checkpoint, not the last step. On abort, the best
partial checkpoint by \(|M(t)|\) (ties broken by the same tuple) is selected, `selection.json`
records `stopped: aborted` with the reason, and the exit code is nonzero.
`eval_metrics.jsonl` gets one row per evaluation (probe scores, canary
counts, loss); `selection.json` records the stop reason, selected step, and
criterion values, so `--progress` and the JSON artifacts can replay the whole
decision.

## CLI surface

`--distill PATH` joins the mutually exclusive primary modes. New flags, all
optional: `--generator-model` (default `mlx-community/Qwen3.5-2B-6bit`),
`--questions-per-chunk` (default 8), `--max-questions` (default 64),
`--max-answer-tokens` (default 32), `--generate-only`, `--questions PATH`,
`--probes PATH` (reused from progress mode), `--eval-every`,
`--learned-threshold`, `--max-iters`. Canary change and contamination
budgets are fixed at zero in v1 rather than exposed as flags. Existing
training flags keep their meaning; `--teacher-system` and `--train` are
rejected with `--distill`.

## First experiment

Run the canonical command on `sakana_wiki.txt` with the defaults above and
the four unrelated canaries. Record in EXPERIMENTS.md: generated-question
counts through each filter, how many items start mastered at \(t = 0\),
steps to success or the abort reason, the selected checkpoint, probe
trajectories via `--progress`, and the fixed GSM8K diagnostic at the
selected checkpoint. Rough cost on an M-series machine: a few minutes of 2B
generation, and roughly 1.5–2.5 s per training step at 64-token rollouts;
at `1e-6` expect several hundred steps, so on the order of 20–40 minutes.

The most likely failure mode is still generation quality: near-duplicate
questions, context-dependent phrasing, or wrong answers that the prompt
rules failed to prevent. `--generate-only` and the funnel in
`candidates.jsonl` make this inspectable before any training time is spent,
and a stronger `--generator-model` is the remedy.

## Design choices and status

| # | Decision | Status |
|---|---|---|
| D1 | Generator model | **Decided**: `mlx-community/Qwen3.5-2B-6bit` default — an experiment showed 0.8B self-generation is measurably worse; `--generator-model` overrides |
| D2 | Generation timing | **Decided**: static pre-generation for the baseline; an active-learning variant (regenerate toward low-\(q_i\) facts) is the expected later upgrade |
| D3 | Mastery signal | **Decided**: \(q_i(t)\) probability threshold — one forward pass per probe — with greedy containment confirmation only at the stop boundary |
| D4 | Teacher context scope | **Decided**: whole document in the system prompt; chunk-scoped context is a later upgrade and the only option for long documents (provenance is already recorded per question) |
| D5 | Base-known facts | **Decided**: no special treatment — uniform mastery retirement; any item the model demonstrably knows stops being trained, whether known at step 0 or learned at step 200, and returns on regression |
| D6 | Locality | **Decided**: zero changed and zero contaminated canaries; canaries run at eval cadence (default every 5 steps), with `--eval-every 1` recovering per-step density |
| D7 | Training defaults | **Decided**: SKD top-5, forward KL, `1e-6`, rank 16 — gentler than the fastest known-clean Lumen setting; optimize speed after the pipeline works |
| D8 | CLI shape | Baseline stands unless objected: one `--distill` mode plus `--generate-only` / `--questions` / `--probes` escape hatches |
| D9 | Budget formula | **Open**: \(200\,|U_0|\) steps capped at 2000 at `1e-6`; calibrate on the first sakana run |
| D10 | Question quality control | **Decided**: semantic criteria live in the generation prompts, code applies only mechanical checks (parse, dedupe, caps); the remedy for bad questions is a stronger generator, with `--generate-only` for inspection |

Notes on the less obvious rows:

- **D2**: adaptive generation is the interesting research direction (spend
  updates on the facts the student hasn't absorbed). Mastery-based
  retirement (D5) already adapts the *schedule* over a static question set;
  the future variant would also adapt the *set*.
- **D3**: the probability threshold is what makes zero-budget, dense
  monitoring affordable; the greedy confirmation step exists precisely
  because \(\tau\) is a guess until calibrated.
- **D5**: retirement means near-agreement updates on already-known facts
  never happen, dropping the earlier idea of using them as implicit replay
  regularization. If locality suffers in practice, that idea can return as
  an experiment.
- **D7**: at `1e-6` the EXPERIMENTS.md cumulative-learning-rate table
  allocates twice the steps of `2e-6` for the same movement; per-step drift
  is halved, which should widen the clean window on a multi-fact document
  the same way it did on Lumen.

## Out of scope for v1

- Sequential documents into one adapter (true continual learning across
  sessions) — v1 is one document, one adapter, from scratch.
- Multi-hop and synthesis probes; negative probes (questions the document
  does not answer, guarding against hallucination amplification).
- Replay buffers, KL-to-base regularization, batching, resume — pre-existing
  limitations in DESIGN.md that this pipeline inherits.
- Paraphrase transfer is the only generalization measured; arbitrary-question
  generalization is not claimed.

## Testing

Unit tests with an injectable generator function (no model): chunking
boundaries, candidate parsing and the mechanical filters with
positive/negative cases, mastery-set and rotation updates, probe file
emission, stop-condition and selection logic as pure functions over
synthetic eval rows, and CLI flag validation
for the new mode. One cached-model integration test: `--distill` on a
two-fact synthetic micro-document with a tiny budget, asserting the artifact
layout, at least one evaluation row, and a `selection.json` with a valid stop
reason — mirroring the existing Lumen integration test's scale.
