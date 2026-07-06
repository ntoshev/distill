# distil — design

A console tool that visualizes how likely a text/rollout is according to an LLM,
token by token, using a single forward pass with mlx-lm. Green = the model finds
the token likely, red = surprising.

## Architecture

All input modes reduce to one primitive: **a full text string plus a list of
character spans** marking the model-generated content to score.

```
cli.py ──► templating.py  BuiltText(text, spans, warnings)
                │
                ▼
        scoring.py  score_text ── tokenize_with_offsets ──► token/char alignment
                │                 score_ids            ──► one forward pass
                ▼
        render.py   colored text + legend + stats + worst-tokens table │ JSON
        rollout.py  verifiers/prime-rl results.jsonl loader
```

### Module responsibilities

- **templating.py** — builds `BuiltText` for the three input modes:
  - `build_raw(prompt, completion)`: plain concatenation, no template.
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
- **rollout.py** — reads verifiers/prime-rl `results.jsonl` (one rollout per
  line; `prompt` and `completion` are OpenAI-style message lists) plus display
  metadata (`example_id`, `reward`, `task`, …). Message fields that are
  explicit nulls in the row (`reasoning_content`, `tool_calls`) are dropped
  before templating.
- **render.py** — rich-based output. Foreground color on a red→yellow→green
  gradient over logprob clamped to `[-8, 0]` (`--metric prob` maps linearly in
  probability instead). Context tokens are dim grey. Also: a color legend,
  summary stats (scored/total tokens, mean logprob, perplexity), a
  most-surprising-tokens table with the model's preferred alternatives, and
  `to_json_dict` for `--json` output and snapshots.
- **cli.py** — argparse; `--completion` XOR `--rollout`; values accept inline
  strings, `@file`, and `-` (stdin). Validation happens before the (slow)
  model import so error paths are instant and testable.

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
  re-render turns verbatim score every assistant turn.
- **Scoring-time vs generation-time logprobs** differ by up to a few tenths of
  a nat per token under 4-bit quantization (kv-cache incremental decoding and
  a full-sequence pass take different kernel paths). The greedy round-trip
  test pins this envelope.
- Rollouts are scored under the *scorer's* chat template, which may differ
  from the template of the model that produced them — that's the point when
  comparing models, but worth remembering when numbers look odd.
- No batching across rollouts; one `--index` at a time.

## Testing

Snapshot style: expected outputs live in `tests/snapshots/` and are committed
to git. Missing snapshots are created on first run; `UPDATE_SNAPSHOTS=1`
regenerates after intentional changes. Logprobs are rounded (2dp) before
snapshotting to absorb numeric jitter; rich output is captured with a fixed
width/truecolor recording console so ANSI snapshots are byte-stable.

- **Unit (no model):** color-scale endpoints and monotonicity; renderer ANSI
  snapshot from synthetic scores; JSON shape; span construction against fake
  chat templates (including a thinking-style template that exercises the
  skip-with-warning path); rollout JSONL parsing against a fixture copied from
  a real `prime eval run` output; CLI argument validation.
- **Integration (`-m model`, needs cached `mlx-community/Qwen3.5-0.8B-4bit`):**
  - JSON + ANSI snapshots of a fixed chat-mode scoring and of the real rollout
    fixture (the fixture was itself generated by Qwen3.5-0.8B, so it scores
    plausibly high).
  - **Greedy round-trip** (self-validating, no golden data): generate
    greedily, re-score the generated ids with the single-pass scorer, assert
    argmax agreement and logprob closeness. Any alignment/off-by-one bug fails
    this loudly.
  - Predictable text must outscore shuffled text; offsets must tile the
    rendered text exactly.

## Future direction: teacher/student KL (on-policy distillation)

The intended evolution, and how the current shape supports it:

- A rollout is scored under **two contexts**: student sees the plain prompt,
  teacher sees a privileged prompt (e.g. `--system-override`), while the
  completion text — and therefore its token ids — is shared. `build_rollout`
  already separates context from scored spans, so the completion positions
  align between the two passes by construction.
- `score_ids` grows a variant returning what KL needs. Exact
  `KL(teacher ‖ student)` per position needs both full distributions; compute
  it inside the chunked loop (both models' logits for the same chunk, never
  materializing full-vocab logprob tensors) or start with the cheaper sampled
  approximation `log p_student(x_t) − log p_teacher(x_t)` on the rollout
  tokens, which the current per-token logprobs already provide.
- Rendering generalizes from "one scalar per token" (logprob) to "one scalar
  per token" (KL divergence) with a different scale/legend, plus a table of
  positions where teacher and student disagree most and what each preferred.
- The typical setup is teacher = student weights with different conditioning,
  so a single loaded model can serve both passes; distinct
  `--teacher-model` comes for free since the passes are independent.
- On-policy distillation then reuses exactly this: student generates (mlx-lm
  `stream_generate`), teacher scores the student's tokens, the per-token
  KL/logprob-gap becomes the training signal. The greedy round-trip test
  already validates the "score what was generated" half of that loop.
