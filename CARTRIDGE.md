# Qwen3 cartridges

Cartridges are a second, self-contained learning backend. LoRA remains the
training default. The cartridge backend supports the ordinary, all-attention
Qwen3 cache layout. Hybrid Qwen3.5 students, mixed LoRA/cartridge policies,
multiple cartridges, and reverse KL are rejected explicitly.

A cartridge is learned attention memory: it changes the attached policy while
leaving the base weights frozen. Removing it restores the detached base policy.
Its exported tensor size is separate from source text size and the number of
compressed KV positions; the export can be larger than the document itself.

## Passage memorization

```sh
uv run distill --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --memorize @document.txt --cartridge cartridges/memory \
  --cartridge-positions 128 --passes 4 --learning-rate 0.005 \
  --key-tokens 32 --target-tokens 128 --stride 32

uv run distill --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --cartridge-path cartridges/memory --complete-key @key.txt \
  --max-tokens 128 --temperature 0

uv run distill --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --progress cartridges/memory \
  --memory-keys cartridges/memory/generated/shifted_keys.jsonl
```

Input accepts inline text, UTF-8 `@file`, or `-` for stdin. Source snapshots and
keys preserve whitespace, including CRLF. Empty sources, sources over 32,768
tokens, and prompts exceeding model context are rejected. Native initialization
needs at least the requested number of reference tokens; smaller documents need
an explicitly smaller capacity. The first prefix position stays frozen.

Keys are verbatim excerpts. Completion uses the recorded raw recipe, with no
chat formatting. Training windows preserve character spans and exact token IDs;
unusable token boundaries, partial windows, and uncovered regions are recorded.
`--objective ntp` is the default; `--objective kd` uses a frozen full-document
teacher on the same source trajectory. Both source objectives average by target
token count. A final short batch does not wrap into another source-window pass.

Evaluations compare the same target IDs under key only, full document,
initialized memory, and learned memory. They report NLL/perplexity, greedy exact
match, positional accuracy, first error, literal text, generated IDs, and finish
reason. Up to eight practiced and eight shifted keys form the development
panel. Shifted keys test unseen entry points in the same document. They are not
held-out knowledge or a separate final quality set. The terminal progress view
starts with one key and an affected canary; JSON includes all supplied keys.

The fixed-budget final update is selected unless an optional canary changes.
In that case the run stops and exports its last evaluated safe checkpoint, if
one exists. Optional canaries compare literal greedy answers; they do not imply
global retention. No retention claim is made when canaries are omitted.

## Document facts

```sh
uv run distill --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --backend cartridge --train @document.txt --cartridge cartridges/facts \
  --cartridge-positions 128 --learning-rate 0.001 \
  --canaries examples/distillation/canaries.jsonl --no-thinking
```

Fact learning uses the existing extraction, held-out questions, frozen mastery
judge, fact rotation/reactivation, and two consecutive mastery evaluations.
Cartridge SKD runs the teacher and student sequentially with separate caches,
using the same frozen base weights. The teacher gets the document; the student
gets memory and the question. The objective is full-vocabulary forward KL,
weighted equally per fact. Generators and judges may use other architectures.
The distillation teacher must have the student's frozen base identity.

Fresh initialization is compared with detached-base canary answers. A changed
initial answer fails the run without publishing a root export. A later change
stops training and selects a previously safe checkpoint. These conservative
checks can block an otherwise useful memory. The cartridge learning rates are
pilot settings, not validated defaults guaranteeing mastery or exact recall.
Use a separate final QA set for generalization claims.

Scoring and sampling attach `--cartridge-path` only to the student, including
teacher-context KL scoring. Progress discovers the backend from checkpoint
artifacts. Its dense grid distinguishes detached `base` from `initialized`,
uses actual generated token IDs for generated-answer scoring, and shows only
recorded transitions. Normal LoRA invocations remain supported.

## Refinement and exact resume

```sh
# Fresh optimizer, new output, capacity inherited from the starting artifact.
uv run distill --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --memorize @new-document.txt --from-cartridge cartridges/memory \
  --cartridge cartridges/refined --passes 2 --learning-rate 0.001

# Original source, recipe, FP32 masters, Adam, RNG and cursor; six total passes.
uv run distill --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --resume cartridges/memory --passes 6
```

`--iters` and `--passes` are alternative total budgets. Resume accepts an
increased budget in the original units and presentation flags. Successful or
contaminated fact runs cannot restart through resume. Refinement may overwrite
old knowledge; explicit old-document retention probes are needed to measure
retention. Starting-artifact fact teachers keep a frozen copy of that memory.

Inference and resume load the recorded base revision and verify the required
`--model` against the saved identity.

Runs save a schema-2 model/tokenizer/template/weight identity, post-RoPE tensor
manifest and checksums, exact source and compiled records, metrics, and
numbered checkpoints. Checkpoints commit atomically at evaluations and the
final update. `latest.json` points to resumable state; the root export points
to the selected inference state and can be earlier. Resume verifies the
committed configuration, restores FP32 masters rather than rounded exports,
and truncates metric logs to committed byte offsets. Interruptions return
nonzero and preserve the last committed checkpoint.

Strict schema-1 inference manifests with the recorded identity and representation
fields can be imported. Optimizer resume requires the integrated schema-2
format; inference imports can initialize a fresh run.

## Validation

The tests cover native-cache equivalence and offsets, request isolation,
artifact and identity rejection, finite K/V gradients, frozen base/first
position, production-vocabulary NTP/KD gradient equivalence, token boundaries,
actual EOS/trajectory IDs, both accumulation reductions, interrupted resume,
half-precision exports with FP32 resume, and canary-safe selection. A real
quantized Qwen3 test checks native logits and greedy IDs, export/load, and
improving source likelihood after updates.

```sh
uv run pytest -q -m 'not benchmark'
DISTIL_BENCHMARK_SYSTEM_ID=mac15-6-m3-pro-11cpu-14gpu-36gb \
  uv run pytest -q -m benchmark
```

Eight additional Qwen3-4B benchmark cases retain the existing LoRA baselines.
They fix revision `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`, capacity 128,
seed 0, greedy decoding, and synchronized MLX operations. They include export
load/cache creation, fresh and cached 512-token document prefill, prefix prompt
prefill/first token, 32-token decoding, 64-target-token NTP/KD updates, and a
16-token SKD update. Timing comparisons include cache construction or cloning.
Training cost must be amortized over repeated requests; these measurements do
not establish a quality advantage over LoRA or a benefit for one-off questions.
