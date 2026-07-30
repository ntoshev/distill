# Continual learning: internalizing one document into LoRA weights

Status: implemented v1; empirical calibration remains open.

## Goal

One command takes trusted privileged context and produces a standard `mlx-lm`
LoRA adapter whose student answers factual questions about that context without
seeing it. The value follows the existing input convention: inline text,
`@file`, or `-` for stdin.

```sh
uv run distil \
  --model mlx-community/Qwen3.5-0.8b-6bit \
  --train @examples/distillation/sakana_wiki.txt \
  --generator-model mlx-community/Qwen3.5-9B-6bit \
  --adapter adapters/sakana-wiki \
  --canaries examples/distillation/canaries.jsonl \
  --no-thinking
```

The canonical source is `examples/distillation/sakana_wiki.txt`: a short,
multi-fact document containing facts about Sakana AI's founders, funding,
research, AI Scientist project, and reported valuation. It replaces Project
Lumen's single synthetic fact as the end-to-end document experiment.

The document becomes:

- a **syllabus**: generated training questions;
- an **exam**: one held-out question per fact with a source-grounded target
  completion;
- a set of **tripwires**: unrelated canaries that detect answer changes.

Training stops when the student passes the exam without the document and while
the tripwires remain unchanged.

## Scope

V1 is one document per run into one adapter. The student trains from a
**starting model**: `--model`, optionally combined with an existing adapter
through `--from-adapter`. The frozen teacher, the step-0 baseline, and the
canary references are all this starting model, so a run measures learning and
drift relative to where it started, not necessarily the raw base model. The
run is continual within itself: facts retire from training when mastered and
return if they regress. Chaining runs — distilling a second document starting
from the first document's adapter — is mechanically possible through
`--from-adapter`, but v1 makes no retention promise for the earlier document
beyond what the user's canaries cover; replay and prior-document retention
probes are a separate follow-up design.

V1 targets short factual recall. It does not claim arbitrary-question,
multi-hop, procedural, or explanatory knowledge acquisition.

`--train` assumes the source document is trusted. The baseline intentionally
does not implement semantic prompt-injection detection or sanitization; a
document containing adversarial instructions can steer generation or the
teacher and is outside this trust model.

## Resolved decisions

| Area | V1 decision |
|---|---|
| CLI | `--train CONTEXT` is the only training mode and accepts inline text, `@file`, or `-`. Supplying `--probes` selects the manual/single-target path; omitting it generates the dataset. |
| Dataset input | Use only `--probes PATH`. Each row adds `train_prompts` to the existing held-out `prompt`/`completion` probe schema. There is no `--questions` flag. |
| Starting model | `--model` plus optional `--from-adapter`. Unless an explicit `--teacher-model` is supplied, the frozen teacher, step-0 baseline, and canary references all use that complete starting model. |
| Generator | Default same as the training model; `--generator-model` overrides it. The generator need not share the student tokenizer. |
| Generation timing | Generate a static dataset before training. Adaptive question regeneration is later work. |
| Question strategy | The generator enumerates the document's distinct facts, then writes one source-copied completion, one training question, and one held-out probe per fact. No code-side chunking. |
| Quality control | Put semantic judgment in prompts. V1 applies no filtering beyond line parsing; any future quality check is a model prompt, not a code heuristic. |
| Target completions | The generator selects each fact's exact source-grounded `completion` span. Completions are evaluation targets only, never NLL/SFT targets. |
| Teacher context | The teacher receives the whole document in its system prompt; the student never receives it. V1 rejects documents that do not fit instead of truncating them. |
| Already-known facts | Use the same mastery rule for every item. Items mastered at step 0 sit out; regressed items return. If all items are mastered at step 0, save an adapter identical to the starting state and return success without optimizer updates. |
| Mastery | Geometric-mean target-token probability with greedy normalized-containment confirmation at the stop boundary. |
| Training | KL-only context distillation; forward KL by default, with speculative teacher repair. Numeric defaults live in the CLI contract. |
| Canaries | Required in `--train` and use the same JSONL probe schema. The flag assigns the retention role: `train_prompts` is ignored, and every greedy output must remain exactly equal to its step-0 baseline. Canary answers are never trained. |
| Evaluation cadence | Probes, canaries, and stop checks share `--eval-every`, and evaluation always writes a checkpoint. |
| Final adapter | Copy the selected checkpoint to the adapter root; never assume the last update is best. |
| Abort | Preserve diagnostics, select the best partial checkpoint, copy it to the root, and return nonzero with `selection.json` marked aborted. |
| Cross-session continual learning | Chaining runs through `--from-adapter` is possible but unprotected; replay and retention probes are out of scope for v1. |

## Research basis

The baseline is context distillation with synthetic self-QA. Question
synthesis for knowledge injection appears in
[Self-QA](https://arxiv.org/abs/2305.11952) and synthetic continued
pretraining ([EntiGraph](https://arxiv.org/abs/2409.07431)); the rollout
repair policy comes from
[Speculative Knowledge Distillation](https://arxiv.org/abs/2410.11325),
already implemented in this repository.

The table lists the parts of prior work that map cleanly onto the existing
repository:

| Work | Design implication |
|---|---|
| [Learning by Distilling Context](https://arxiv.org/abs/2209.15189) | A context-conditioned teacher can supervise a context-free student so contextual behavior persists in weights. |
| [On-Policy Context Distillation](https://arxiv.org/abs/2602.12275) | Student-generated trajectories reduce the train/inference mismatch. Its reverse-KL default remains an ablation because this repository's forward-KL experiments were more local on Project Lumen. |
| [New News: System-2 Fine-tuning](https://arxiv.org/abs/2505.01812) | Self-generated QA is more effective than merely training on raw new text. The student must not see the document, avoiding contextual shadowing. |
| [Self-Tuning](https://arxiv.org/abs/2406.06326) | Closed-book question formats improve extraction and recall compared with plain continued pretraining. |
| [Learning Facts at Scale with Active Reading](https://arxiv.org/abs/2508.09494) | Diverse document-derived study data can improve factual acquisition over raw repetition. |
| [SPA](https://arxiv.org/abs/2603.22213) | Carefully prompted static generation is a strong baseline; a complicated adaptive generator is not required for v1. |
| [Self-Study Reconsidered](https://arxiv.org/abs/2606.32002) | Free-form question generation can be repetitive, uneven, and sensitive to document presentation. V1 accepts this risk, exposes all generated artifacts, and uses a stronger generator rather than adding heuristic semantic filters. |

The research does not determine this repository's learning rate, probability
threshold, or update budget. Those remain empirical calibration questions for
the Sakana run.

## Current repository fit

The training substrate already exists:

- `training.py` performs on-policy context distillation into LoRA parameters;
- `sample.py` supports student rollouts and speculative teacher repair;
- `scoring.py` aligns generated student and teacher tokens exactly;
- `adapters.py` writes standard `mlx-lm` adapters;
- `training_metrics.jsonl` records updates and canary generations;
- numbered checkpoints can be written every update;
- `progress.py` scores fixed prompt/completion probes over checkpoints;
- `evaluate_lr_sweep.py` demonstrates post-hoc learned/locality checkpoint
  selection for Project Lumen.

The missing pieces are:

1. fact enumeration and question generation;
2. source-grounded probe completions and the train/held-out split;
3. baseline probe scoring and a dynamic training rotation;
4. in-loop evaluation, stopping, and checkpoint selection;
5. orchestration and persisted generation/evaluation artifacts;
6. starting-adapter support: the current self-teacher disables all LoRA
   layers, which would also disable an adapter loaded through
   `--from-adapter`; the teacher must represent the starting model, not the
   raw base.

## Design principle: judgment in prompts, mechanics in code

Semantic decisions — is a fact distinct and stated in the document, is a
question self-contained, does it name its subject — live in the generation
prompts. Code performs deterministic mechanics only: parsing, normalization,
scoring, stop and selection rules, and artifact I/O. V1 ships no question
filtering at all; if quality control ever becomes necessary, it will be a
model/prompt check, not a code heuristic.

If question quality is inadequate, the supported remedy is a stronger model
behind `--generator-model` or manual editing through the artifact escape
hatches. V1 does not grow a collection of semantic heuristics that partially
duplicate a language model.

This decision accepts an important failure mode: an ambiguous question can be
answered by the context-aware teacher, learned by the student, and passed by a
matching ambiguous probe without proving subject-anchored recall. That is why
generation is a separately inspectable stage — and the first thing to blame
if a learned adapter answers unrelated questions with document content.

## Pipeline

```text
--train CONTEXT
  │
  ├─ 1. resolve       inline text, @file, or stdin; trusted, non-empty UTF-8
  ├─ 2. validate      fits generator and teacher context
  ├─ 3. enumerate     generator lists the context's distinct facts
  ├─ 4. generate      generator writes completion and questions per fact
  ├─ 5. baseline      score every probe at step 0; retire already-mastered facts
  ├─ 6. train         current KL-only LoRA loop over the active rotation
  ├─ 7. monitor       probe probability and greedy canaries every E steps;
  │                   record loss as a diagnostic
  ├─ 8. stop          confirmed mastery, locality abort, or update-budget abort
  └─ 9. select        copy the best checkpoint to the adapter root
```

Stages 1–5 resolve or generate the structured probe dataset and write every
intermediate artifact before training. Stage 6 scores the step-0 baseline once
the student is loaded, still before any weight update.

## Stage A: document compilation

### A1. Input validation

Resolve `--train` with the same convention as other text-valued arguments:

- an ordinary value is inline context;
- `@PATH` reads UTF-8 text from a regular file;
- `-` reads UTF-8 text from stdin.

Reject missing `@` files, directories, malformed UTF-8, and empty resolved
context before loading a model. File extension is irrelevant; the generator
operates on text and can interpret plain text or markdown directly.

The full rendered teacher prompt must fit the teacher/student context window
with room for `--max-tokens`, and the resolved context must likewise fit the
generator's context for fact enumeration — the same window when
self-generating. V1 must fail with a clear size error; it must not silently
truncate the privileged document. Scoped generation and teacher contexts are
the planned long-document upgrade.

Compute and record the source SHA-256, byte count, character count, and path.
The generated artifacts restate the document's facts and are therefore
privileged. The adapter directory must be handled with the same
confidentiality as the original document.

### A2. Fact enumeration

There is no code-side chunking. The generator reads the whole document once
and enumerates the distinct facts it states:

```text
Here is a reference document:

{document}

List every distinct fact this document states.
Rules:
- One fact per line, no numbering.
- Each fact must be a single short statement that names its subject
  explicitly (never "the company", "it", "they", or another reference that
  needs the text to resolve).
- Only state facts that appear in the document.
- Do not repeat the same fact in different words.
```

Fixing the facts before asking for questions is the target-first insight:
free-form question generation gravitates toward the most salient regions of a
document and leaves the rest uncovered, while an explicit fact list makes
coverage inspectable line by line.

Each fact gets an ID and is persisted; every generated question records its
fact ID as provenance. For `sakana_wiki.txt` the expected shape is roughly
one fact per statement: founders, funding, research direction, the AI
Scientist, and the reported valuation.

### A3. Question generation and held-out split

The generator defaults to the model being trained. Generation is a one-time
preprocessing cost and has no KL token-alignment requirement, so any locally
loadable generator can be supplied. For small students an override is
recommended: a preliminary local comparison found
`mlx-community/Qwen3.5-2B-6bit` clearly better than 0.8B self-generation,
which is why the canonical command passes it explicitly.

Fact enumeration is greedy for stable coverage and uses a dedicated 512-token
output budget. For each enumerated fact, generate one structured block with an
exact source-copied `COMPLETION` span, one training question, and one held-out
question. Both stage temperatures are recorded in `generation.json`:

```text
SOURCE FACT:

{fact}

Output exactly:
SUBJECT: <explicit named subject copied from the source fact>
COMPLETION: <one exact, contiguous phrase copied from the source fact>
TRAIN: <question>
HELDOUT: <different question>
```

The `TRAIN` question enters the training rotation; `HELDOUT` becomes the probe.
Each fact is one item: mastery, rotation, and the update budget count facts.
Every fact must yield a non-empty source-copied completion, one training
question, and a distinct held-out question. Otherwise compilation fails after
writing the generation artifacts; training and evaluating on the same prompt
would not demonstrate transfer.

The prompt owns the semantic quality criteria. The code does not attempt to
prove self-containment or groundedness.

### A4. Target completion selection

The generator writes both questions and the exact target `completion` copied
from the source fact. The student receives only the question text; the source is
never present in the student context during training or evaluation.

The target completion is used for probes and stopping only. There is no
target-NLL or completion cross-entropy loss. All training gradients come from
teacher/student KL. Canary completions are likewise evaluation-only.

If the generator picks a poor completion span, the adapter can internalize that
misreading. V1 treats generation quality as the ceiling of the method; it does
not add a second semantic verifier.

### A5. Parsing and validation

Parsing is the only mechanical processing: strip bullets, list numbers, and
surrounding whitespace; drop empty lines. After parsing, every enumerated fact
must have a non-empty source-copied completion, one training question, and a
distinct held-out question. Any violation fails compilation and is recorded in
the generation artifacts; facts are never silently dropped from the document
learning claim.

There is no other filtering in v1: no semantic deduplication, no length caps,
no question budget. Exact overlap between a held-out prompt and its training
prompts is rejected as split validation; other duplicate or near-duplicate
training questions merely cost training time. Every generated line is
persisted for inspection. If generation quality proves inadequate, the remedy
is a stronger `--generator-model` or manual editing — and if an automated
check ever becomes necessary, it will be a model/prompt judgment, not code.

### A6. Generation artifacts and escape hatches

```text
ADAPTER/generated/
  generation.json
  source.json
  facts.jsonl
  probes.jsonl
```

`generation.json` records the generator model, prompt-template version,
fact/question temperatures, seed, and per-stage counts.

`facts.jsonl` records each enumerated fact with its ID, generated question
lines, and any validation error. It is an audit log, not a downstream input.

`probes.jsonl` is the single accepted dataset consumed by training,
evaluation, and `--progress`. It extends the existing probe schema with the
training prompts and fact provenance:

```json
{"name":"sakana-0001","fact_id":"fact-0001","fact":"Sakana AI was founded by David Ha, Llion Jones and Ren Ito.","train_prompts":["Who founded Sakana AI?"],"prompt":"Name the founders of Sakana AI.","completion":"David Ha, Llion Jones and Ren Ito"}
```

The existing progress loader reads `name`, `prompt`, `completion`, and optional
`system`, and ignores the added fields. Train mode additionally requires a
non-empty `train_prompts` list. It must not silently train on `prompt`, because
that field is the held-out exam question.

`--generate-only` stops after compilation. `--probes PATH` skips fact,
question, and completion generation and uses a hand-authored or previously edited
dataset. Downstream training should not care whether the probe file was
generated or written manually.

## Stage B: baseline and KL training

### B1. Step-0 baseline

Before any train-mode update, score every probe completion against the
starting student. For probe \(i\) with target tokens \(y_{1:T_i}\), define

\[
q_i(t) = \exp\!\left(
\frac{1}{T_i}\sum_{s=1}^{T_i}
\log p_{\phi_t}(y_s\mid x_i,y_{<s})
\right).
\]

This is the existing geometric-mean token-probability progress metric. It is
length normalized, cheap to compute, and does not require generation.

With mastery threshold \(\tau\), the mastered set is

\[
M(t)=\{i:q_i(t)\ge\tau\}.
\]

Items mastered at step 0 do not enter the initial training rotation. They
remain in every evaluation and return to the rotation if their score later
falls below \(\tau\). No separate “known fact” classifier or surprisal rule is
needed.

Write step 0 to `eval_metrics.jsonl` and save a step-0 checkpoint before the
first update. This is the safe fallback if the first later evaluation changes
a canary.

If every item is mastered at step 0, save an adapter identical to the
starting state — a zero-delta LoRA when training from the base model — write
`selection.json` with `stopped: already_mastered` and selected step 0,
perform no optimizer updates, and return success. This preserves the
`--train` contract that a successful invocation produces a loadable
adapter.

### B2. Training rotation

The current trainer shuffles prompts and cycles through them. Train mode
extends it with an evaluation callback that returns the active item IDs after
each evaluation:

- unmastered items remain active;
- newly mastered items retire;
- regressed items return;
- the active items are shuffled deterministically within each cycle.

This adapts the training schedule without regenerating the dataset.

### B3. KL-only objective

For an active question \(x_i\), the student generates \(\hat y\) without the
document. The frozen teacher — the starting model — evaluates the same
generated prefix with the whole document \(c\):

\[
\mathcal L(\phi)=
\frac{1}{|\hat y|}\sum_t
D_{\mathrm{KL}}\!\left(
p_T(\cdot\mid c,x_i,\hat y_{<t})
\,\|\,
p_{S,\phi}(\cdot\mid x_i,\hat y_{<t})
\right).
\]

Only LoRA parameters update. Canonical probe completions do not appear in the
loss. Forward KL is the v1 default because it produced a cleaner locality
trade-off than reverse KL in the repository's Project Lumen experiments.

Speculative top-3 generation remains enabled: a student token is retained if
it lies in the teacher's top three; otherwise a teacher-policy token replaces
it. This exposes useful suffixes before the student can generate the complete
response independently.

### B4. Training defaults

Numeric defaults are stated once, in the CLI contract below.

The default learning rate `1e-6` is deliberately gentler than the clean
`2e-6` Lumen checkpoint. At `1e-6`, the existing cumulative-learning-rate
heuristic allows twice as many updates for comparable movement. The defaults
require calibration on the first Sakana run.

When `--generator-model` names a different model, unload it before loading
the training model on memory-constrained machines; generation and training
are separate phases. The default self-generation uses one model throughout.

### B5. Starting adapter and manual-probe mode

With `--from-adapter PATH`:

1. load `--model` plus the adapter as the starting student;
2. continue updating that adapter's existing LoRA parameters rather than
   adding a second set of LoRA layers;
3. use the adapter's saved rank, layer count, and LoRA configuration;
4. reject any explicitly supplied `--lora-rank`; the source adapter already
   determines the rank;
5. require `--adapter` to differ from `--from-adapter`, preserving the
   input adapter as the last known-good state.

When no explicit `--teacher-model` is supplied, the teacher is a frozen
snapshot of the complete starting student, including the initial adapter. The
current shortcut of evaluating the shared model with LoRA disabled is valid
only for a fresh adapter trained from the raw base. With an initial adapter it
would incorrectly remove previously learned behavior from the teacher. The
implementation must therefore use a frozen model copy or explicit frozen-
adapter state swapping that is safe under MLX's lazy evaluation.

When `--teacher-model` is supplied, it remains the explicit external teacher;
`--from-adapter` initializes only the student unless a separate future teacher-
adapter option is added.

When `--probes PATH` is omitted, the generator creates the structured probe
dataset from the privileged context. When it is supplied, generation is
skipped and the file provides both `train_prompts` and the held-out
`prompt`/`completion` for each fact. Supplying `--probes` selects this manual
and single-target path within `--train`.

Both paths use the same active rotation, mastery, greedy confirmation,
`--eval-every` cadence, locality abort, and checkpoint selection. There is no
training path without probes: generated or supplied probes are required so
every successful run has an explicit behavioral stop condition.

## Stage C: monitoring and stopping

### C1. Evaluation cadence

Every `--eval-every` successful updates:

1. score all target completions and update \(M(t)\);
2. update the active training rotation;
3. greedily generate every canary;
4. record recent training loss and timing as diagnostics only;
5. evaluate success and abort conditions.

Evaluation always saves an adapter checkpoint at that step. There is no
separate checkpoint cadence: only step 0 and evaluated checkpoints are
eligible for selection.

Loss is never a mastery, abort, or checkpoint-selection signal. It is recorded
only to diagnose optimization behavior.

Dense probe scoring is much cheaper than greedy generation. Canary generation
therefore moves from every update to the shared evaluation cadence; users can
set `--eval-every 1` to recover per-step canary monitoring.

### C2. Canary baselines and change detection

`--canaries PATH` is required in v1 and uses the same JSONL schema as probes:

```json
{"name":"canary/math","prompt":"What is 7 plus 5? Answer with the number only.","completion":"12"}
```

Load canaries through the same probe parser. Extra fields such as
`train_prompts` and `fact_id` are allowed but ignored in canary mode. This
means a previous run's generated `probes.jsonl` can be passed unchanged as
`--canaries` in a later run to monitor retention of its learned facts.

Before training, greedily generate each canary's starting-model answer using
the same student system configuration as the training run. The answer must
satisfy the expected `completion` under the same answer-matching rule as
probes; otherwise fail before the first update because the supplied canary is
not valid for the starting model. Record the exact generated string as the
baseline.

At every evaluation, a canary is `changed` when its exact greedy output string
differs from its step-0 baseline. This matches the existing training change
detection: `content != baseline`.

Canaries are never included in the training rotation and never contribute a
loss. Any canary change gates stopping and selection.

The files are structurally interchangeable, but their roles are intentionally
not implicit: `--probes` means “learn these rows,” while `--canaries` means
“preserve these rows.” Collapsing both flags without an explicit role would
risk training on retention checks or treating learning targets as immutable.

### C3. Success

An evaluation is **safe** when no canary changed.
It is **mastered** when every probe satisfies \(q_i(t)\ge\tau\).

At the first evaluation that is both safe and mastered, greedily generate
answers for every held-out probe at the current checkpoint. Normalize the
generated answer and target by Unicode normalization, case folding,
whitespace collapse, and terminal-punctuation stripping. Confirmation passes
when the normalized target is contained in the normalized generated answer.

If every probe confirms, the current checkpoint is successful and training
stops. Failed confirmation resumes training; a transient threshold crossing
costs one confirmation pass, nothing more.

The probability threshold is the cheap dense learning signal. Greedy
confirmation prevents a miscalibrated threshold from being the final proof of
learning.

### C4. Locality abort

If any canary changed at an evaluation, stop immediately with a locality
abort. Even one changed output means the adapter has left the accepted
locality envelope; there is no patience window and no partial credit.

Earlier speculative-KD experiments saw transient canary changes recover, and
such a blip now ends the run — but selection still returns the best safe
checkpoint recorded before the change, so a late blip does not discard the
learning that preceded it.

### C5. Budget abort

The existing `--iters` flag is the hard update cap in train mode. Do not add
a duplicate `--max-iters` flag. When `--iters` is omitted, use the provisional
default

\[
B=\min\!\left(2000,\max\!\left(200,200|U_0|\right)\right),
\]

where \(U_0\) is the set of facts unmastered at step 0.

There is no separate plateau heuristic or patience setting. A run that makes
no useful progress ends at the same explicit update budget as every other
unsuccessful run.

### C6. Checkpoint selection

On success, the confirmed checkpoint is selected; training stops at the
first confirmed success, so it is unique.

Only safe checkpoints are eligible for selection. The checkpoint that first
shows a changed canary is diagnostic evidence, never an adapter candidate.

On abort, select the best safe partial checkpoint by:

1. largest mastered count \(|M(t)|\);
2. earliest step.

Copy the selected checkpoint's standard adapter files to `ADAPTER/`.
Write `selection.json` with `stopped: success` or `stopped: aborted`, the stop
reason, selected step, criterion values, and source checkpoint. An aborted run
returns nonzero even though its best partial adapter is preserved for
inspection.

The adapter root therefore always represents an explicitly selected
checkpoint, never merely the final attempted update.

## CLI contract

`--train CONTEXT` is the only training mode. `CONTEXT` uses the existing
text-input convention: inline text, `@file`, or `-` for stdin.

New train-mode flags:

```text
--generator-model MODEL       default: the model being trained
--questions-per-fact N        default 2; structured generation currently requires 2
--generate-only
```

Training controls:

```text
--adapter PATH                required output adapter/run directory
--from-adapter PATH           optional immutable starting student adapter
--probes PATH                 optional complete structured probe dataset
--canaries PATH               required probe JSONL in retention role
--eval-every N                default 5
--learned-threshold F         default 0.5
--iters N                     hard cap; computed when omitted
--learning-rate F             default 1e-6
--lora-rank N                 default 16 for a new adapter
--speculative-top-k N         default 3
--kl-direction forward|reverse default forward
--max-tokens N                default 64
--seed N
--no-thinking
```

When `--probes` is absent, facts, train prompts, held-out prompts, and target
completions are generated from `CONTEXT`. When present, it supplies that complete
dataset and generation is skipped. Every row therefore requires non-empty
`train_prompts`, `prompt`, and `completion` fields.

Validation rules:

- the manual workflow is `--train CONTEXT --probes PATH`;
- `--teacher-system`, `--teacher-prompt`, `--prompt`, `--sample`,
  `--completion`, `--rollout`, and `--progress` do not combine with
  `--train`;
- `--probes` skips generation and every row must contain a non-empty
  `train_prompts` list in addition to the normal probe fields; its held-out
  `prompt` must not exactly equal any training prompt in that row;
- `--canaries` uses the normal probe fields, ignores `train_prompts`, and every
  completion must match the starting model before training;
- `--generate-only` does not load or train the LoRA student after generation
  and does not combine with `--probes`;
- `--generator-model` and `--questions-per-fact` have no effect with
  `--probes` and are rejected there;
- all numeric settings must be positive;
- `--from-adapter` and `--eval-every` combine only with `--train`;
- `--from-adapter` must name a loadable adapter compatible with `--model`, and
  must not equal the output `--adapter`;
- `--from-adapter` does not combine with `--lora-rank`; the source adapter's
  saved configuration is authoritative;
- the resolved context must fit the teacher context budget;
- every evaluation automatically writes a checkpoint;
- output paths must not silently overwrite a non-empty successful adapter.

## Artifacts

```text
ADAPTER/
  adapter_config.json
  adapters.safetensors
  training_metrics.jsonl
  eval_metrics.jsonl
  selection.json
  run_config.json
  generated/
    generation.json
    source.json
    facts.jsonl
    probes.jsonl
  checkpoints/
    step-0000/
    step-0005/
    ...
```

`run_config.json` records the source hash, student model ID, starting
adapter (if any), generator model ID, prompt-template version, resolved
defaults, random seeds, training flags, canary source, and artifact paths.

`eval_metrics.jsonl` has one row for step 0 and every evaluation checkpoint:

```json
{
  "iteration": 25,
  "loss": 0.42,
  "mastered": 9,
  "total": 12,
  "active_item_ids": ["sakana-0004", "sakana-0010", "sakana-0012"],
  "probe_scores": {"sakana-0001": 0.83},
  "canary_changed": 0
}
```

The generated directory may reconstruct much of the privileged source and
must not be treated as harmless metadata.

## Implementation structure

Keep orchestration out of `cli.py`:

```text
src/distil/generation.py   fact enumeration, generation prompts, parsing
src/distil/stopping.py     mastery rotation, stop rules, selection
src/distil/distill.py      generate → baseline → train → evaluate → finalize
src/distil/training.py     evaluation callback and mutable active rotation
src/distil/cli.py          argument validation and dispatch
```

Existing sampling, scoring, progress, adapter, and rendering modules remain
the underlying primitives.

Pure data structures should represent generated items, probe scores, canary
summaries, stop verdicts, and selection rows so most behavior can be unit
tested without loading a model.

## Canonical experiment

Run the command at the top of this document and record in `EXPERIMENTS.md`:

- the enumerated facts and their count;
- parsed question counts per fact;
- generated questions and probe completions;
- number of probes mastered at step 0;
- per-probe probability trajectories;
- active-rotation changes;
- first safe all-mastered evaluation;
- greedy confirmation results;
- first canary change;
- stop reason and selected checkpoint;
- wall time by generation, training, checkpoint, probe, and canary work;
- fixed GSM8K diagnostic at the selected checkpoint.

Expected risks to inspect first:

1. enumeration that misses facts or invents unsupported ones;
2. near-duplicate questions surviving without deduplication;
3. context-dependent wording despite the prompt;
4. completion spans that are too long or inconsistent;
5. a probe that targets a different attribute of its fact than the training
   questions;
6. one stubborn fact consuming the update budget;
7. broad canary answer changes before all facts are mastered.

Do not hand-edit generated probabilities or invent records. If the checked-in
canonical fixture is later needed, it must come from a real, versioned
generation run.

## Testing

Fast tests with injectable fake generator/teacher functions:

- inline, `@file`, and stdin context resolution, including missing files,
  directories, malformed UTF-8, and empty input;
- full-document context overflow fails without truncation;
- fact and question line parsing, including bullets and empty lines;
- deterministic last-question held-out split;
- compilation fails if any fact lacks a training question, a distinct
  held-out question, or a non-empty source-copied completion;
- extended probe rows load in train mode and remain compatible with
  `--progress`;
- train mode rejects probe rows without non-empty `train_prompts` rather
  than training on their held-out `prompt`;
- train mode rejects exact overlap between a row's training and held-out
  prompts;
- step-0 mastery and active-rotation updates;
- mastered items retire and regressed items return;
- probability mastery and greedy confirmation;
- exact canary change detection against the step-0 output;
- probe JSONL accepted unchanged in canary mode, with training fields ignored;
- canaries that the starting model does not satisfy fail before training;
- immediate locality abort on any canary change and abort at the update
  budget when mastery is not reached;
- checkpoints are written only at step 0 and evaluation steps;
- successful and partial checkpoint selection;
- selected checkpoint copied to the adapter root;
- aborted run returns nonzero;
- starting-adapter runs use the starting model for the teacher, baseline,
  and canary references;
- manual `--train CONTEXT --probes PATH --from-adapter PATH` continues the
  existing LoRA without adding another adapter and never disables it in the
  default teacher;
- generated and supplied probe datasets use the same evaluation, stopping, and
  selection path;
- canary generation follows `--eval-every` rather than every update;
- CLI compatibility and early validation before model loading.

Cached-model integration tests:

1. a two-fact synthetic micro-document produces all artifacts, at least one
   evaluation row, and a valid `selection.json`;
2. the selected adapter reloads through standard `mlx-lm` APIs;
3. Project Lumen still internalizes through
   `--train @teacher.txt --probes probes.jsonl`;
4. the Sakana canonical run is an explicit slower experiment or marked model
   integration test after its budget is calibrated.

## Out of scope for v1

- retention machinery for sequential ingestion — replay buffers,
  KL-to-previous-adapter regularization, prior-document retention probes;
  chaining runs through `--from-adapter` works but is unprotected;
- target-completion NLL or supervised answer training;
- question filtering of any kind, mechanical or semantic — a future check
  would be a model/prompt judgment;
- structured claim schemas with exact evidence spans;
- automatic prompt-injection sanitization;
- documents exceeding one context window (needs scoped generation and
  teacher contexts);
- adaptive question generation during training;
- multi-hop, synthesis, negative, or unanswerable probes;
- batching, gradient accumulation, and resume support;
- semantic answer judges.

These are experiments or later milestones, not hidden requirements of the
first implementation.

## Remaining empirical calibration

No architecture or interface decision blocks implementation.

### R1. Budget and threshold calibration

The provisional values \(\tau=0.5\) and `200 * |U0|` updates capped at 2000
need measurement on the Sakana run. The first experiment should report
trajectories before changing any default.

V1 is otherwise fully specified.
