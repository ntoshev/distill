# distil — design

A console tool that visualizes how likely a text/rollout is according to an LLM,
token by token, using a single forward pass with mlx-lm. Green = the model finds
the token likely, red = surprising. A teacher mode scores the same completion
under two contexts and colors by exact per-token KL(teacher ‖ student).

## Architecture

All input modes reduce to one primitive: **a full text string plus a list of
character spans** marking the model-generated content to score. Teacher mode is
two of those primitives — one per context — sharing the completion string.

```
cli.py ──► templating.py  BuiltText(text, spans, warnings)   [× 2 with a teacher]
                │
                ▼
        scoring.py  score_text      ── single-model path
                    score_text_pair ── teacher/student path: aligned spans,
                                       exact full-vocab KL(teacher ‖ student)
                │
                ▼
        render.py   colored text + legend + stats + worst/disagreement table │ JSON
        rollout.py  verifiers/prime-rl results.jsonl loader + saver
        sample.py   sample the completion from the student (mlx-lm)
```

### Module responsibilities

- **templating.py** — builds `BuiltText` for the three input modes:
  - `build_raw(prompt, completion)`: plain concatenation, no template. Also
    used by sample mode, where the rendered generation prompt plus the
    generated text *is* the exact scored string.
  - `build_chat(tokenizer, prompt, completion, system=None)`: wraps into
    system/user/assistant messages, delegates to `build_rollout`.
  - `build_rollout(tokenizer, messages)`: renders the conversation through the
    model's chat template and locates each assistant turn. Span start comes
    from rendering the preceding messages with `add_generation_prompt=True`,
    span end from rendering through the assistant message; both renderings
    must be **string prefixes** of the full rendering or the turn is skipped
    with a warning (never scored approximately). Trailing template glue after
    the final end-of-turn token is excluded; the end-of-turn token itself is
    scored (the model's decision to stop is meaningful).
  - `apply_teacher_context(messages, system, prompt)`: teacher-side
    conditioning — replaces (or inserts) the system message and/or replaces
    the first user message. Assistant turns are never touched, so scored spans
    stay aligned between the two renderings.
- **scoring.py** —
  - `tokenize_with_offsets`: tokenizes the rendered text once and returns
    per-token character offsets (HF offset mapping, with a streaming-
    detokenizer fallback). Decode round-trip is asserted; a mismatch becomes a
    warning, not silent corruption.
  - `score_ids`: the numeric core. One forward pass over the whole sequence;
    row *t* of the logits predicts token *t+1*. Log-softmax is computed in
    float32 in row chunks (never materializing a full `(seq, vocab)` float32
    tensor): `logprob = chosen_logit − logsumexp(row)`. Top-k alternatives via
    `argpartition` only at scored positions.
  - `score_text`: glues the two together. A token is scored when its character
    range **overlaps** a span and it has at least one token of context.
  - `score_text_pair`: the teacher/student path. Tokenizes both renderings,
    requires the scored positions to carry **identical token ids in identical
    order** (a mismatch means the teacher context changed how the completion
    tokenizes — incompatible tokenizer or template edge case — and is an
    error, never scored approximately), then `kl_pair_ids` runs one forward
    pass per model and computes, per aligned position in float32 chunks, the
    exact full-vocabulary `KL(teacher ‖ student)` plus both models' chosen-
    token logprobs and top-k. Both chunks of log-softmax rows live together,
    so the chunk is smaller (128) than in `score_ids`. Teacher = student
    weights is the common case and shares a single loaded model.
- **sample.py** — `sample_completion` generates from the student with mlx-lm's
  `stream_generate` (temperature sampler, optional seed). It returns the
  rendered generation prompt and the decoded generated ids (end-of-turn token
  included) as strings, so scoring is `build_raw(prompt_text, completion_text)`
  — exact by construction, no re-templating — plus `content` with special
  tokens stripped for saving. A teacher scores the same `completion_text`
  after its own generation-prompt rendering: the on-policy loop.
- **rollout.py** — reads verifiers/prime-rl `results.jsonl` (one rollout per
  line; `prompt` and `completion` are OpenAI-style message lists) plus display
  metadata (`example_id`, `reward`, `task`, …). Message fields that are
  explicit nulls in the row (`reasoning_content`, `tool_calls`) are dropped
  before templating. `save_rollout` appends a sampled rollout in the same
  format — the save format is the load format, so samples re-score via
  `--rollout` and interoperate with anything that reads verifiers rollouts.
- **render.py** — rich-based output. The default output is nothing but the
  text itself, foreground-colored on a red→yellow→green truecolor gradient
  over logprob clamped to `[-8, 0]` (`--metric prob` maps linearly in
  probability instead); context tokens are dim grey. With a teacher the
  default metric is `kl`: the gradient runs in reverse over KL clamped to
  `[0, 10]` nats (green = agreement). Opt-in extras: `--stats` (color legend
  and scored/total tokens, mean logprob, perplexity, and with a teacher mean
  KL and teacher mean logprob), `--worst N` (most-surprising-tokens table, or
  in kl mode the largest teacher/student disagreements with each model's
  preferred token), and `to_json_dict` for `--json` output and snapshots.
  Warnings and rollout/teacher metadata go to stderr so stdout stays pipeable.
- **cli.py** — argparse; exactly one of `--completion`, `--rollout`, or
  `--sample`; values accept inline strings, `@file`, and `-` (stdin). Any
  `--teacher-*` flag enables two-pass scoring (`--teacher-model` defaults to
  `--model`). Sampling flags (`--max-tokens`, `--temperature`, `--seed`,
  `--save-rollout`) only combine with `--sample`; the sample is saved before
  scoring so a scoring failure never loses it. Validation happens before the
  (slow) model import so error paths are instant and testable.

## Why character spans, not token-id concatenation

Concatenating separately-tokenized segments drifts at boundaries: Qwen3.5's
generation prompt ends in `<think>\n`, and when the assistant text begins with
`\n` the two newlines retokenize as a single `\n\n` token. So the full rendered
string is tokenized exactly once, and spans are located by prefix-rendering at
the string level (where the prefix property does hold — verified for Qwen3.5).

Consequence: a token that straddles a span boundary (the `\n\n` above) is
scored under the overlap rule, even though its first character was forced by
the template. This is inherent to re-tokenizing generated text and is accepted
rather than hidden.

## Known limitations (v1)

- **Intermediate assistant turns in thinking-model rollouts** are shown as
  context, not scored: the chat template strips their `<think>` blocks on
  re-rendering, so the tokens that were actually generated aren't in the
  rendered conversation at all — exact scoring is ill-defined. Single-turn
  rollouts (the common eval case) are exact. Non-thinking templates that
  re-render turns verbatim score every assistant turn. The same applies to a
  *reloaded* saved sample: the fresh sample is scored on the exact generated
  string, the reloaded one through the template.
- **Scoring-time vs generation-time logprobs** differ by up to a few tenths of
  a nat per token under 4-bit quantization (kv-cache incremental decoding and
  a full-sequence pass take different kernel paths). The greedy round-trip
  test pins this envelope.
- Rollouts are scored under the *scorer's* chat template, which may differ
  from the template of the model that produced them — that's the point when
  comparing models, but worth remembering when numbers look odd.
- **A distinct `--teacher-model` must share the student's tokenizer**: per-
  position KL across different vocabularies is ill-defined, and the strict
  token-id alignment check rejects it. Differently *padded* vocabularies are
  handled by computing KL over the overlap.
- The KL direction is fixed at `KL(teacher ‖ student)` (teacher-probability-
  weighted disagreement, per the distillation loss); the reverse direction
  would be a render/flag addition, not a scoring change.
- No batching across rollouts; one `--index` (or one sample) at a time.

## Testing

Snapshot style: expected outputs live in `tests/snapshots/` and are committed
to git. Missing snapshots are created on first run; `UPDATE_SNAPSHOTS=1`
regenerates after intentional changes. Logprobs are rounded (2dp) before
snapshotting to absorb numeric jitter; rich output is captured with a fixed
width/truecolor recording console so ANSI snapshots are byte-stable.

- **Unit (no model):** color-scale endpoints and monotonicity for both the
  logprob and kl scales; renderer ANSI snapshots from synthetic scores (single
  and teacher-mode); JSON shape; span construction against fake chat templates
  (including a thinking-style template that exercises the skip-with-warning
  path); teacher-context message rewriting; rollout JSONL parsing against a
  fixture copied from a real `prime eval run` output and save/load
  round-tripping; CLI argument validation.
- **Integration (`-m model`, needs cached `mlx-community/Qwen3.5-0.8B-4bit`):**
  - JSON + ANSI snapshots of a fixed chat-mode scoring, of the real rollout
    fixture (the fixture was itself generated by Qwen3.5-0.8B, so it scores
    plausibly high), and of teacher-mode KL on the fixture under an
    answer-revealing system prompt.
  - **Greedy round-trip** (self-validating, no golden data): generate
    greedily, re-score the generated ids with the single-pass scorer, assert
    argmax agreement and logprob closeness. Any alignment/off-by-one bug fails
    this loudly.
  - **Identical-context KL ≈ 0** (self-validating): teacher = student with the
    same conditioning must give near-zero KL at every position — the pair
    path compared against itself.
  - **Sample → teacher alignment**: a fresh sample scored under a
    teacher-conditioned rendering must align position-for-position and produce
    finite, (near-)nonnegative KL; greedy samples must re-score as likely and
    round-trip through `save_rollout`/`load_rollout`.
  - Predictable text must outscore shuffled text; offsets must tile the
    rendered text exactly.

## Future direction: on-policy distillation training

The visualization now computes exactly the training signal: student generates
(`sample_completion`), teacher scores the student's tokens under privileged
conditioning (`score_text_pair`), and the per-token KL is what a distillation
loss would minimize. What's missing to close the loop is batched sampling and
scoring (many rollouts per step rather than one `--index` at a time) and an
optimizer step on the student — the scoring/alignment core would be reused
unchanged.
