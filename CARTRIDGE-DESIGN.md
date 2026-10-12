# Cartridge backend for distill

Status: proposed design, based on the current working trees of `distill` and
`../cartridges` (2026-10-02). Commands below describe the proposed interface;
they are not implemented by this document.

## Recommendation

Add cartridges as a second backend inside the existing `distill` CLI. Preserve
`--train` as document fact learning, and add `--memorize` for verbatim passage
learning. Share document ingestion, scoring, generation, evaluation, checkpoint
inspection, and reporting; keep backend parameters and optimizers separate.

A LoRA changes how the model processes every request while attached. A cartridge
supplies a learned attention memory at the start of each request. Think of it as
learning the model's internal notes about a document rather than learning new
weight updates. Both require training. The cartridge is neither a text prompt
nor a lossless document archive, and accurate passage completion does not by
itself demonstrate question-answering ability.

Ship the existing all-attention Qwen3 implementation first. Do not advertise
Qwen3.5 support until its hybrid recurrent/attention state has an independently
implemented and tested representation.

## What exists today

| Area | Current implementation | Integration implication |
| --- | --- | --- |
| CLI | `src/distill/cli_config.py` resolves score, sample, train, and progress modes | Extend this parser instead of adding a separate executable |
| Document learning | `cli.py:_continual_cli`, `generation.py`, `distill.py:run_continual` | Reuse fact generation, question pairs, mastery judgments, rotation, and canaries |
| LoRA optimization | `training.py:TrainingSession` installs LoRA and differentiates model parameters | Introduce a separate cartridge session; changing only the save method is insufficient |
| Inference | `sample.py`, `scoring.py`, `prefill.py` largely create or assume empty initial caches | Add explicit per-policy cache construction throughout |
| Progress | `progress.py:find_checkpoints` recognizes `adapters.safetensors` | Discover backend-tagged artifacts and load the matching representation |
| Cartridge tensors | `../cartridges/src/mlx_cartridges/cartridge.py` | Reuse post-RoPE KV representation, frozen first position, FP32 masters, and native initialization |
| Source training | Sibling `train.py`, `data.py`, `memorize.py` | Port aligned source windows, NTP/KD training, practiced/shifted keys, and resume state |
| Experiments | Sibling `evaluate.py`, `checks.py`, `results/memorization_v1/REPORT.md` | Port invariants and controls; leave NarrativeQA, HELMET, and oMLX runners optional |

The sibling currently pins `Qwen3-4B-Instruct-2507-4bit` at revision
`50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`, requires ordinary `KVCache`
layers, and imports `distill` already. Adding a reverse dependency from `distill`
to that package would create a dependency cycle. Port the small reusable core
into `distill`; subsequently let the experimental package consume that core.
Do not import its dataset downloads or `.research` checkout requirements into
normal CLI execution.

The sibling's reported final 4,096-token experiments achieved 8/8 practiced
128-token continuations on each of two texts, but only 6/8 and 5/8 shifted-key
continuations. Those were refined, single-seed development runs, not evidence
that four default passes reliably converge. Its README explicitly postpones QA
tuning and SKD. Treat these as a starting implementation and measured limits.

## CLI contract

Keep `--backend lora` as the training default. `--backend cartridge` selects
cartridge fact learning. New `--memorize` defaults to cartridge and rejects an
explicit LoRA backend in the first release. Keep `--model` required, including
when loading an artifact, and validate it against the artifact identity.

Use explicit artifact names:

- `--cartridge PATH`: training output directory.
- `--cartridge-path PATH`: inference artifact for scoring or sampling.
- `--from-cartridge PATH`: initialize a new run with fresh optimizer state.
- `--resume PATH`: resume an interrupted cartridge run, including its optimizer.

Keep all existing LoRA flags and behavior. Do not reinterpret `--adapter` as a
generic output path. Artifact manifests determine the backend in progress mode;
`--cartridge-path` implies cartridge in inference. Reject conflicting explicit
backend choices before loading a model.

### Memorize a document

```sh
uv run distill \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --memorize @examples/distillation/sakana_wiki.txt \
  --cartridge cartridges/sakana-memory \
  --cartridge-positions 128 \
  --passes 4 --learning-rate 0.005 \
  --key-tokens 32 --target-tokens 128 --stride 32
```

This compiles overlapping source windows and trains on their exact continuation
tokens using NTP. It does not generate questions or invoke a mastery judge.
Input supports the existing inline text, `@file`, and stdin conventions. Start
with UTF-8 text/Markdown; PDF extraction and directory ingestion are separate
features. Reject empty or oversized input rather than silently truncating it.

`--memorize` owns the raw passage-completion template; no `--raw` flag is needed
for training. Add `--complete-key TEXT` as a fifth inference input, mutually
exclusive with `--prompt`, `--completion`, `--rollout`, and `--sample`. It renders
the recorded completion recipe and generates a continuation:

```sh
uv run distill \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --cartridge-path cartridges/sakana-memory \
  --complete-key @passage-key.txt --max-tokens 128 --temperature 0
```

The key must be a verbatim excerpt for the memorization experiment; arbitrary
paraphrased keys are not a promised retrieval interface. Preserve whitespace in
both the key and completion. Never apply chat formatting to this recipe.

### Learn document facts, as with LoRA

```sh
uv run distill \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --backend cartridge \
  --train @examples/distillation/sakana_wiki.txt \
  --generator-model mlx-community/Qwen3.5-9B-6bit \
  --cartridge cartridges/sakana-qa \
  --cartridge-positions 128 --learning-rate 0.001 \
  --canaries examples/distillation/canaries.jsonl \
  --no-thinking
```

Reuse `--generate-only`, `--probes`, `--judge-model`, `--batch-size`,
`--eval-every`, `--iters`, `--save-rollout`, and `--json`. A generator or judge
can use another architecture; only the student and its cartridge need the
cartridge architecture support. Initially restrict the distillation teacher to
the same frozen base identity. Broader teacher-model support comes later.

The student receives cartridge + question; the teacher receives document +
question. The source document must never accidentally enter the student's live
prompt. Full-vocabulary forward KL remains the objective, with the existing
top-3 proposal policy once cartridge SKD is implemented. Expose this mode only
after that implementation passes its acceptance gates; the sibling's source KD
loop is not a substitute for fact learning.

### Sample, score, and inspect

```sh
uv run distill \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --cartridge-path cartridges/sakana-qa \
  --prompt "What does Sakana AI develop?" \
  --sample --temperature 0 --max-tokens 64 --no-thinking

uv run distill \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --cartridge-path cartridges/sakana-qa \
  --prompt "What does Sakana AI develop?" \
  --completion @expected-answer.txt --metric prob --stats --no-thinking

uv run distill \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --progress cartridges/sakana-qa \
  --probes cartridges/sakana-qa/generated/probes.jsonl --no-thinking
```

Teacher-context scoring attaches the cartridge only to the student. Progress
loads the representation at each numbered checkpoint. Raw memorization progress
needs recipe-aware source-window records, not fake chat probes: allow
`--progress RUN --memory-keys RUN/generated/shifted_keys.jsonl`, mutually
exclusive with the existing progress inputs.

### Continue versus resume

```sh
# Refinement: fresh Adam, new run, existing learned memory.
uv run distill --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --memorize @document.txt --from-cartridge cartridges/memory \
  --cartridge cartridges/refined --passes 2 --learning-rate 0.001

# Exact continuation: original recipe/state, six total passes.
uv run distill --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --resume cartridges/memory --passes 6
```

Resume derives mode, source, and parameters from the run. Only an increased
total budget and presentation flags may change. `--iters` denotes total updates;
`--passes` denotes total source-window passes. Reject supplying both. Budget
extension is allowed after a fixed-budget memorization run, but do not silently
restart a fact run already stopped for success or contamination.

Refinement may use the same or a new document. Learning B from A can overwrite
A's memory; require A probes as explicit retention checks when retention is the
goal. Infer capacity from the starting artifact unless explicitly supplied and
equal. Never overwrite the source artifact. Exact resume must not initialize
FP32 masters from the lower-precision export.

### Validation and defaults

| Setting | Memorization | Cartridge fact learning |
| --- | --- | --- |
| Objective | `--objective ntp` default; optional `kd` on source trajectories | Forward KL on SKD trajectories |
| Capacity | 128 positions, including one frozen position | Same initial capacity |
| Optimizer | FP32 Adam, warmup 5 updates, gradient norm clip 1 | Same optimizer; independently tune learning rate |
| Learning rate | 0.005 pilot default | 0.001 provisional starting value, not a validated default for quality |
| Budget | 4 window passes by default | Existing adaptive fact budget or explicit `--iters` |
| Accumulation | `--batch-size 1`, token-weighted | `--batch-size 4`, equal weight per fact |
| Evaluation | Practiced and shifted keys; default every 5 updates and at end | Existing two-consecutive-evaluation mastery rule |
| Canary requirement | Optional; no retention claim without them | Required, as in current LoRA learning |

Resolve defaults after choosing both mode and backend; do not inherit LoRA's
`1e-6` rate accidentally. Persist every resolved setting. Reject LoRA rank and
adapter flags with cartridges; reject passage-window flags in fact mode and
judge/SKD/probe-generation flags in memorization mode. Initially reject mixed
LoRA + cartridge inference, multiple attached cartridges, reverse KL for
cartridges, and unsupported cache architectures. Unsupported options must fail
explicitly rather than fall back to a different algorithm.

## Representation and cache semantics

For each of L attention layers, store keys and values shaped
`[1, H_kv, P, D_head]`. Keys are already in post-RoPE coordinates. The first
position is frozen; only positions 1 through P−1 are trainable. Initialization
copies the first P native cache positions of a recorded document envelope.
Require at least P reference tokens. Persist the exact envelope and token IDs.

The trainable parameter count and inference storage are:

\[
N_\mathrm{trainable}=2L H_{\mathrm{kv}}(P-1)D_{\mathrm{head}},
\qquad
B_\mathrm{export}=2L H_{\mathrm{kv}}P D_{\mathrm{head}}b,
\]

where b is bytes per exported scalar. The sibling's 128-position Qwen3 artifact
is reported as 18 MiB. “32× compression” for 4,096 source tokens means fewer KV
positions, not a smaller file than the original text. Report tensor bytes,
training peak memory, source-token count, and prefix positions separately.

Every request creates a fresh cache from the immutable cartridge. Its offset
starts at P, and live prompt tokens begin there. Do not rotate the stored keys a
second time. Do not append an ordinary document cache before the cartridge or
reset its offset to zero. Teacher and student may have different prefix lengths;
their loss rows align by completion token IDs, not absolute sequence positions.

Use `ConcatenateKVCache` for differentiated forwards so gradients reach K/V.
Inference may use ordinary `KVCache`. Freeze and exclude base weights from the
gradient tree; use `mx.value_and_grad` on the explicit cartridge parameter tree.
Never detach a trainable prefix or its live prompt state while attempting to
reuse inference prefill optimizations during training.

Qwen3.5 recurrent state is not another pair of KV tensors. V1 should enforce the
known Qwen3/cache layout and validate dimensions, RoPE configuration, activation
dtype, tokenizer, chat template, base revision/weight identity, and quantization.
Matching vocabulary size or tensor shape alone is insufficient compatibility.

## Training and evaluation semantics

### Passage memorization

Preserve the sibling's text-first window compiler and boundary checks. Each
record includes document hash, source character spans, key, continuation,
rendered prompt, target positions, token IDs/checksums, and recipe version.
Record uncovered regions and partial windows instead of claiming full coverage.
The first key and some token-boundary regions may never become scored targets.

For source targets y and key q, NTP minimizes

\[
\mathcal L_{\mathrm{NTP}}(C)
=-\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(y_t\mid C,q,y_{<t}).
\]

Source KD replaces the one-hot target with the frozen document teacher's full
distribution on the same source trajectory. Accumulate loss sums/gradients over
all target tokens, then divide by the total token count. Score only continuation
tokens, never the cue, prompt, or frozen prefix.

Persist shuffled window order and cursor. Define a pass as visiting every window
once; consume a shorter final batch rather than silently wrapping into an extra
pass. Also report scored tokens divided by source tokens: with 128-token targets
and stride 32, one window pass is roughly four source-equivalent token passes.

Evaluate four conditions with identical target tokens: key only, full document,
initialized cartridge, and trained cartridge. Report NLL/perplexity, greedy exact
token match, positional token accuracy, first error, and literal text. Distinguish
practiced from shifted keys. Shifted keys test unseen entry points into known
text, not held-out knowledge. Keep a separate final panel if development
evaluation influences recipe choices.

Default selection is the final fixed-budget update, not the best generated
answer on the evaluation panel. If optional canaries trigger a stop, export the
last evaluated safe checkpoint and record the incomplete budget and reason.

### Fact learning

Reuse fact extraction and train/held-out question separation. Train on
teacher/student shared completion trajectories with

\[
\mathcal L_{\mathrm{KD}}(C)
=\frac{1}{T}\sum_t
D_{\mathrm{KL}}\!\left(
p_\theta(\cdot\mid d,q,y_{<t})\;\Vert\;
p_\theta(\cdot\mid C,q,y_{<t})
\right).
\]

The base parameters theta are frozen. Only C changes. For logical batches,
average each fact's mean-token loss equally to preserve current LoRA semantics;
record this reduction explicitly, since source memorization is token-weighted.

Implement SKD first with separate teacher/student caches and sequential forwards
using the same frozen model. Accept proposals using the current top-k/top-p
rules, then advance both policies with the actual accepted/replacement token.
Do not reuse LoRA-specific batched routing just because weights are shared.
Optimize that path only after sequential correctness and performance baselines.

Reuse the source-authoritative frozen judge, two consecutive mastery passes,
fact retirement/reactivation, and best-safe checkpoint selection. Probability is
a diagnostic, not a mastery criterion. Keep a final QA set outside the adaptive
mastery loop for claims about generalization.

For continuation from an existing cartridge, freeze a copy of the starting
cartridge for the teacher and add the new document to its live context. The
student updates a separate copy without that document. A fresh run's teacher
has no cartridge. Save this teacher identity in the run so resumption cannot
silently change the target policy.

### Retention and initialization

Frozen weights guarantee that detaching a cartridge can recover the base policy
under identical inference conditions. They do not guarantee unaffected answers
while the cartridge is attached.

Measure detached-base, attached-initial, and attached-trained canaries. For a
fresh fact run, compare attached outputs against the detached base from step 0.
If native initialization already changes a required canary, fail initialization
with a report; do not redefine that changed output as “safe.” A future explicit
calibration phase can address this if it blocks useful runs. For continuation,
use the starting attached artifact as the run's retention baseline and retain
the detached-base comparison as an additional diagnostic.

Any canary change stops fact learning, preserving the existing conservative
policy. A changed output is a retention alarm even if it is not a leaked source
fact. If no safe checkpoint exists, report failure without publishing a selected
root export. Do not imply that finite canary coverage proves global retention.

## Integration boundaries

Introduce small interfaces, not a framework rewrite:

```python
class InferencePolicy(Protocol):
    def new_cache(self): ...
    def identity(self) -> dict: ...

class LearningSession(Protocol):
    def step_batch(self, prompts: list[str]): ...
    def save(self, path: Path) -> Path: ...
    # Fact sessions also expose their current inference policy.
```

The inference policy includes model/tokenizer bindings as well as cache creation;
the existing LoRA policy still requires its adapter weights to be loaded.
Source-window steps belong to a separate memorization runner. Do not force
source windows into `LearningItem` or claim every backend supports every recipe.

Proposed file responsibilities:

- `backends.py`: backend identity, capability checks, inference policy, artifact
  dispatch; an adapter around the existing LoRA session.
- `cartridge.py`: tensor representation, native initialization, strict loading,
  schema-1 import, and cache construction.
- `cartridge_training.py`: FP32 cartridge optimizer, source steps, fact steps,
  gradient checks, exact checkpoint state.
- `memorization.py`: source windows, pass scheduling, practiced/shifted evaluation.
- `distill.py`: accept a session factory and inference policy; remove hard-coded
  LoRA construction and adapter-only selected-checkpoint copying.
- `cli_config.py` / `cli.py`: validate modes, dispatch recipes, resolve backend
  defaults, and preserve current invocations.
- `sample.py`, `scoring.py`, `prefill.py`: accept an optional initial-cache
  factory consistently, including short-input paths, KL pairs, and SKD.
- `progress.py`: backend-aware checkpoint loading and recipe-aware memory keys.

Keep no-artifact paths unchanged. A cache-aware sampler without cache-aware
rescoring would produce misleading token colors, so ship those together.
Preserve actual sampled token IDs, EOS, and finish reason; never reconstruct a
training trajectory by decoding then re-encoding its display string.

Do not copy the sibling's fused `mlx_lm.tuner.losses.kl_div_loss` call. This
project already documents a large-vocabulary gradient/nondeterminism failure
and uses explicit FP32 reductions in `training.py:_kl_loss`. Extract a shared
loss helper, verify analytic gradients, and bound vocabulary-logit allocations.
The existing chunked LoRA training path needs an explicit cartridge-gradient
audit before reuse; bounded inference prefill alone does not bound backprop memory.

## Artifacts and restart behavior

```text
RUN/
  run_config.json                 # backend, recipe, exact identities/defaults
  cartridge.json                 # selected inference export metadata
  cartridge.safetensors          # activation-dtype K/V, including frozen position
  selection.json                 # selected step, stop reason, success/safety
  training_metrics.jsonl
  eval_metrics.jsonl
  generated/
    source.txt                   # exact source snapshot, including stdin input
    source_manifest.json
    facts.jsonl                  # fact mode
    probes.jsonl                 # fact mode
    windows.jsonl                # memorization mode
    practiced_keys.jsonl
    shifted_keys.jsonl
  checkpoints/
    step-0000/
    step-0005/
      cartridge.json
      cartridge.safetensors
      resume.json
      resume.safetensors
  latest.json                    # last committed resumable checkpoint
```

Use versioned manifests with `backend: cartridge`, representation version,
model/tokenizer/template hashes, tensor names/shapes/dtypes, coordinate system,
frozen positions, initialization lineage, source hash, and prompt recipe.
Record resolved model revisions even when the user supplies a repository name.
Legacy LoRA directories continue loading without a new manifest.

Support loading sibling schema-1 inference exports directly after validating
their identity; save integrated runs in the new format. Treat sibling optimizer
resume formats as unsupported initially: such exports can initialize new runs,
but cannot promise exact cross-implementation resume.

Resume state includes FP32 masters, frozen tensors, Adam moments/counter, all
used RNG states, scheduler state, window cursor/order or fact rotation, mastery
streaks, canary baselines, frozen teacher identity, and metric-log commit offsets.
Save numbered checkpoints at evaluation boundaries and final updates. Commit
via a temporary directory and atomic rename, then advance `latest.json`. Recover
logs only through the committed checkpoint to avoid duplicate iterations.
The sibling currently saves its latest checkpoint at loop completion; periodic
crash-safe resume is integration work, not an existing guarantee.

Root inference export may be an earlier safe checkpoint; `latest.json` identifies
the latest resumable state. Never confuse the two. Refuse existing outputs
unless resuming explicitly. On interruption, preserve the last committed
checkpoint and return nonzero. Emit progress on stderr and one structured final
JSON result on stdout when `--json` is set.

## Progress visualization

Use the project's dense checkpoint grid: step, target completion, affected
canary answer. Color each token with its probability under that checkpoint and
the same attached cartridge. For generated answers, score exactly the generated
IDs; distinguish fixed-reference scoring from generated-answer scoring.

Display geometric-mean token probability where useful. Mark only actual
transitions (`base`, `learned`, `leak`, `recovered`). An initialized cartridge is
not the detached base: label it `initialized` and keep step 0 unambiguous. Stop
on leakage by default; show recovery only if a later actual checkpoint exists.
For memorization, start with one key/continuation and the canary that changed;
do not add a dashboard of all panels by default.

## Delivery plan and acceptance gates

1. **Inference core and import.** Port strict representation loading/native
   initialization. Add cartridge score/sample/progress plumbing. Match sibling
   exported logits and greedy token IDs. Reject Qwen3.5 clearly.
2. **Memorization MVP.** Port window compilation and NTP, periodic checkpoints,
   fresh-optimizer refinement, exact resume, and paired practiced/shifted
   evaluation. Add source KD only after its production-shape KL gradient tests.
3. **Fact-learning parity.** Add cartridge sessions, sequential SKD, the existing
   continual controller, frozen starting-cartridge teacher, and canary-safe
   selection. Verify one-fact learning, contamination handling, and a multi-fact
   document. Tune a cartridge-specific learning rate before calling it stable.
4. **Performance and expansion.** Optimize demonstrated bottlenecks, evaluate
   larger documents/capacities, then consider Qwen3.5 hybrid state, multiple
   cartridges, routing, and LoRA composition as separate work.

Required correctness tests include native-prefix/cache logit equivalence;
post-RoPE offsets; fresh-request isolation; export round trips; rejection of
model/tokenizer/shape mismatches; nonzero finite K and V gradients; unchanged
base/frozen-position tensors; token boundary/EOS fidelity; teacher/student
trajectory alignment; uneven token-weighted accumulation; exact interrupted
versus uninterrupted resume; and safe selection when initialization or a later
checkpoint fails canaries. Keep existing LoRA CLI/regression tests passing.

Quality tests must compare identical prompts and target IDs against key/task
only, initialization, full-document context, and learned cartridge controls.
Keep QA and verbatim recall results separate. Compare LoRA and cartridge on the
same supported base model rather than confounding backend with architecture.
Do not turn the sibling's small development-panel accuracy into a universal CI
threshold or require universal exact memorization.

Every implementation change affecting training or decoding must run correctness
and system-specific benchmark tests, as required by `AGENTS.md`:

```sh
uv run pytest -q -m 'not benchmark'
DISTIL_BENCHMARK_SYSTEM_ID=mac16-9-m4-max-14cpu-32gpu-36gb \
  uv run pytest -q -m benchmark
```

Use `mac15-6-m3-pro-11cpu-14gpu-36gb` on that machine, or register a baseline for
the actual system. Add cartridge benchmark cases to the existing timing harness;
do not replace LoRA baselines. Measure warmed medians for export load/cache
creation, cold and warm document prefill, cartridge prompt prefill/first token,
decode throughput, NTP/KD/SKD update cost, and peak memory. Fix model revision,
capacity, context length, completion length, dtype, seed, and sampling policy;
synchronize MLX inside timed operations. Include cache cloning costs. Report
training cost separately from per-request savings: cartridges amortize training
over repeated queries and may offer no benefit for a one-off document question.

This change is a design document only; no implementation or performance changes
are made, so these are future acceptance gates rather than claimed test results.
