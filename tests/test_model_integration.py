"""Integration tests against the locally cached Qwen3.5-0.8B-4bit.

Snapshot style: expected outputs live in tests/snapshots/ and are committed to
git. Regenerate with UPDATE_SNAPSHOTS=1 after intentional changes. The forward
pass is deterministic on a given machine/mlx version; logprobs are rounded
before snapshotting to absorb tiny numeric drift.
"""

import json

import pytest
from rich.console import Console

from distil.render import RenderOptions, render, to_json_dict
from distil.rollout import load_rollout
from distil.scoring import score_ids, score_text, tokenize_with_offsets
from distil.templating import build_chat, build_raw, build_rollout

from conftest import FIXTURE_DIR, MODEL

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def chat_result(model, tokenizer):
    built = build_chat(
        tokenizer,
        prompt="What is 2+2?",
        completion="2+2 equals 4.",
        system="You are a terse assistant.",
    )
    return score_text(model, tokenizer, built.text, built.spans, model_name=MODEL,
                      warnings=built.warnings)


def test_chat_score_json_snapshot(chat_result, snapshot):
    snapshot("chat_score.json", json.dumps(to_json_dict(chat_result), indent=2) + "\n")


def test_chat_render_snapshot(chat_result, snapshot):
    console = Console(record=True, force_terminal=True, color_system="truecolor", width=100)
    render(chat_result, console, RenderOptions(stats=True, worst=5))
    snapshot("chat_render.ansi", console.export_text(styles=True))


def test_chat_render_default_snapshot(chat_result, snapshot):
    """Default output: nothing but the gradient-colored text."""
    console = Console(record=True, force_terminal=True, color_system="truecolor", width=100)
    render(chat_result, console)
    snapshot("chat_render_default.ansi", console.export_text(styles=True))


def test_chat_scored_span_is_exactly_the_assistant_turn(chat_result):
    scored_text = "".join(t.text for t in chat_result.scored)
    # the assistant turn: think block closed, completion, end-of-turn. The
    # leading "\n\n" is one token that straddles the span boundary — its first
    # char is the template's forced think-opening newline, merged with the
    # first generated newline on retokenization; overlap scoring includes it.
    assert scored_text == "\n\n</think>\n\n2+2 equals 4.<|im_end|>"


def test_rollout_score_json_snapshot(model, tokenizer, snapshot):
    r = load_rollout(FIXTURE_DIR / "results.jsonl", 0)
    built = build_rollout(tokenizer, r.messages)
    assert built.warnings == []
    result = score_text(model, tokenizer, built.text, built.spans, model_name=MODEL)
    d = to_json_dict(result)
    # keep the snapshot focused: summary plus the first/last tokens
    d["tokens"] = d["tokens"][:60] + d["tokens"][-20:]
    snapshot("rollout_score.json", json.dumps(d, indent=2, ensure_ascii=False) + "\n")


def test_greedy_roundtrip(model, tokenizer):
    """Generate greedily, then re-score the generated tokens with the scorer's
    single full-sequence forward pass. Each generated token must be recovered
    as (near-)argmax and its logprob must match the generation-time logprob.
    This exercises tokenization, the shift-by-one, and the forward pass with
    no golden data at all."""
    from mlx_lm import stream_generate

    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Count from 1 to 20, separated by commas."}],
        add_generation_prompt=True,
    )
    gen_ids, gen_lps = [], []
    for resp in stream_generate(model, tokenizer, prompt=prompt_ids, max_tokens=24):
        gen_ids.append(resp.token)
        gen_lps.append(resp.logprobs[resp.token].item())

    ids = list(prompt_ids) + gen_ids
    positions = set(range(len(prompt_ids), len(ids)))
    logprobs, alternatives = score_ids(model, ids, alt_positions=positions, top_k=1)

    deltas = [abs(logprobs[t] - gen_lps[t - len(prompt_ids)]) for t in sorted(positions)]
    # generation (kv-cache, step-by-step) and scoring (one full pass) take
    # different kernel paths; under 4-bit quantization individual tokens can
    # drift a few tenths of a nat. An alignment bug would be nats everywhere.
    assert max(deltas) < 1.0, f"scorer/generation logprob mismatch: {deltas}"
    assert sum(deltas) / len(deltas) < 0.2
    # greedy decoding picked the argmax at every step; the scorer must agree
    # (allowing near-ties to differ by numerics)
    argmax_hits = sum(alternatives[t][0][0] == ids[t] for t in positions)
    assert argmax_hits >= 0.9 * len(positions)


def test_predictable_beats_shuffled(model, tokenizer):
    def mean_lp(completion):
        built = build_raw("Here is a sequence: 1 2 3 4 5", completion)
        result = score_text(model, tokenizer, built.text, built.spans, top_k=0)
        return result.mean_logprob

    assert mean_lp(" 6 7 8 9 10 11 12") > mean_lp(" 11 7 12 6 9 8 10")


def test_offsets_roundtrip_on_templated_text(tokenizer):
    r = load_rollout(FIXTURE_DIR / "results.jsonl", 0)
    built = build_rollout(tokenizer, r.messages)
    ids, offsets = tokenize_with_offsets(tokenizer, built.text)
    assert tokenizer.decode(ids) == built.text
    assert "".join(built.text[s:e] for s, e in offsets) == built.text
