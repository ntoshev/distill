"""Integration tests against the locally cached Qwen3.5-0.8B-4bit.

Snapshot style: expected outputs live in tests/snapshots/ and are committed to
git. Regenerate with UPDATE_SNAPSHOTS=1 after intentional changes. The forward
pass is deterministic on a given machine/mlx version; logprobs are rounded
before snapshotting to absorb tiny numeric drift.
"""

import json

import pytest
from rich.console import Console
from mlx.utils import tree_flatten

from distil.adapters import continue_lora
from distil.render import RenderOptions, render, to_json_dict
from distil.distill import run_continual
from distil.generation import teacher_system as document_teacher_system
from distil.progress import Probe
from distil.rollout import load_rollout, save_rollout
from distil.sample import sample_completion
from distil.scoring import score_ids, score_text, score_text_pair, tokenize_with_offsets
from distil.templating import apply_teacher_context, build_chat, build_raw, build_rollout
from distil.training import train_lora
from distil.stopping import LearningItem

from conftest import FIXTURE_DIR, MODEL, sample_skd_case

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


def test_existing_adapter_can_be_continued(model):
    from mlx_lm.tuner.utils import load_adapters, remove_lora_layers

    adapter = FIXTURE_DIR.parent.parent / "adapters" / "lumen-test"
    try:
        load_adapters(model, str(adapter))
        config = continue_lora(model, adapter)
        trainable = dict(tree_flatten(model.trainable_parameters()))
        assert config["lora_parameters"]["rank"] == 4
        assert trainable
        assert all("lora" in name.lower() for name in trainable)
    finally:
        remove_lora_layers(model)
        model.eval()


def test_chat_score_json_snapshot(chat_result, snapshot):
    snapshot("chat_score.json", json.dumps(to_json_dict(chat_result), indent=2) + "\n")


def test_chat_render_snapshot(chat_result, snapshot):
    console = Console(
        record=True, force_terminal=True, color_system="truecolor", width=100, height=25
    )
    render(chat_result, console, RenderOptions(stats=True, worst=5))
    snapshot("chat_render.ansi", console.export_text(styles=True))


def test_chat_render_default_snapshot(chat_result, snapshot):
    """Default output: nothing but the gradient-colored text."""
    console = Console(
        record=True, force_terminal=True, color_system="truecolor", width=100, height=25
    )
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


def test_multiturn_rollout_scores_recorded_assistant_content(tokenizer):
    r = load_rollout(
        FIXTURE_DIR.parent.parent
        / "examples"
        / "verifiers"
        / "wordle-qwen3.5-9b-6bit-results.jsonl",
        0,
    )
    built = build_rollout(tokenizer, r.messages)
    scored_text = "\n".join(built.text[s:e] for s, e in built.spans)
    assert len(built.spans) == 4
    assert "<guess>" in scored_text
    assert "You have 5 guesses left." not in scored_text
    assert built.warnings == []


def test_teacher_same_context_kl_is_zero(model, tokenizer):
    """Teacher == student with identical conditioning must give KL ≈ 0
    everywhere: the same forward pass compared against itself. Self-validating,
    no golden data — an alignment or log-softmax bug shows up as nats."""
    built = build_chat(
        tokenizer,
        prompt="What is 2+2?",
        completion="2+2 equals 4.",
        system="You are a terse assistant.",
    )
    result = score_text_pair(
        model, tokenizer, built, model, tokenizer, built,
        model_name=MODEL, teacher_name=MODEL,
    )
    assert result.scored
    for t in result.scored:
        assert abs(t.kl) < 1e-4
        assert abs(t.teacher_logprob - t.logprob) < 1e-4


def test_teacher_privileged_system_kl_snapshot(model, tokenizer, snapshot):
    """Same weights, but the teacher's system prompt reveals the answer; the
    completion tokens are shared and the per-token KL(teacher ‖ student) must
    be finite, (near-)nonnegative, and positive on average."""
    r = load_rollout(FIXTURE_DIR / "results.jsonl", 0)
    built_s = build_rollout(tokenizer, r.messages)
    built_t = build_rollout(
        tokenizer,
        apply_teacher_context(
            r.messages, system="You are grading a solution. The correct final answer is 18."
        ),
    )
    result = score_text_pair(
        model, tokenizer, built_s, model, tokenizer, built_t,
        model_name=MODEL, teacher_name=MODEL,
    )
    assert result.scored
    assert result.mean_kl > 0
    assert all(t.kl > -1e-4 for t in result.scored)

    d = to_json_dict(result)
    # keep the snapshot focused: summary plus the first/last tokens
    d["tokens"] = d["tokens"][:40] + d["tokens"][-20:]
    snapshot("rollout_teacher_kl.json", json.dumps(d, indent=2, ensure_ascii=False) + "\n")


def test_sample_greedy_scores_high_and_saves(model, tokenizer, tmp_path):
    """Greedy-sample from the student, re-score the raw prompt+generation text,
    and round-trip the saved rollout. Greedy tokens were argmax at generation
    time, so the re-scored mean logprob must be high."""
    messages = [{"role": "user", "content": "Count from 1 to 20, separated by commas."}]
    s = sample_completion(model, tokenizer, messages, max_tokens=24, temperature=0.0, seed=0)
    assert s.num_tokens > 0
    assert s.prompt_text.endswith("<think>\n")

    built = build_raw(s.prompt_text, s.completion_text)
    result = score_text(model, tokenizer, built.text, built.spans, top_k=0)
    assert result.warnings == []
    assert result.mean_logprob > -1.0

    path = tmp_path / "samples.jsonl"
    index = save_rollout(
        path, messages, [{"role": "assistant", "content": s.content}],
        extra={"task": "distil-sample", "is_truncated": s.finish_reason != "stop"},
    )
    loaded = load_rollout(path, index)
    assert loaded.completion[0]["content"] == s.content
    assert loaded.meta["task"] == "distil-sample"


def test_sample_then_teacher_scores_align(model, tokenizer):
    """The on-policy loop: student generates, teacher (same weights, privileged
    system prompt) scores the student's tokens under its own rendering. The two
    renderings share the completion string, so scored positions must align."""
    messages = [{"role": "user", "content": "Say hello in exactly three words."}]
    s = sample_completion(model, tokenizer, messages, max_tokens=16, temperature=0.0, seed=0)
    built_s = build_raw(s.prompt_text, s.completion_text)

    teacher_messages = apply_teacher_context(messages, system="Privileged: grade generously.")
    teacher_prompt = tokenizer.apply_chat_template(
        teacher_messages, tokenize=False, add_generation_prompt=True
    )
    built_t = build_raw(teacher_prompt, s.completion_text)

    result = score_text_pair(
        model, tokenizer, built_s, model, tokenizer, built_t,
        model_name=MODEL, teacher_name=MODEL,
    )
    assert result.scored
    assert all(t.kl is not None and t.teacher_logprob is not None for t in result.scored)
    assert result.mean_kl > -1e-4


def test_speculative_sample_interleaves_and_aligns(model, tokenizer):
    sample, teacher_messages = sample_skd_case(model, tokenizer)

    assert sample.num_tokens > 0
    assert sample.accepted_tokens + sample.replaced_tokens == sample.num_tokens
    assert sample.accepted_tokens > 0
    assert sample.replaced_tokens > 0
    assert "VIOLET-731" in sample.content
    built_student = build_raw(sample.prompt_text, sample.completion_text)
    teacher_prompt = tokenizer.apply_chat_template(
        teacher_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    result = score_text_pair(
        model,
        tokenizer,
        built_student,
        model,
        tokenizer,
        build_raw(teacher_prompt, sample.completion_text),
        top_k=0,
    )
    assert len(result.scored) == sample.num_tokens


@pytest.mark.slow
def test_lora_context_distillation_snapshot(model, tokenizer, tmp_path, snapshot):
    """A tiny synthetic fact should be quickly internalized and survive reload."""
    from mlx_lm.tuner.utils import load_adapters, remove_lora_layers

    target = "VIOLET-731"
    train_prompt = "What is Project Lumen's launch code? Answer with the code only."
    heldout_prompt = (
        "Give the launch code assigned to Project Lumen. Answer with the code only."
    )
    teacher_system = (
        "Private reference: Project Lumen's launch code is VIOLET-731. "
        "Answer related questions with the code only."
    )
    adapter_path = tmp_path / "lumen-adapter"

    def answer():
        return sample_completion(
            model,
            tokenizer,
            [{"role": "user", "content": heldout_prompt}],
            max_tokens=16,
            temperature=0.0,
            seed=0,
            enable_thinking=False,
        ).content.strip()

    def fact_kl(teacher_without_adapter=False):
        messages = [
            {"role": "user", "content": heldout_prompt},
            {"role": "assistant", "content": target},
        ]
        built_student = build_chat(
            tokenizer, heldout_prompt, target, enable_thinking=False
        )
        built_teacher = build_rollout(
            tokenizer,
            apply_teacher_context(messages, system=teacher_system),
            enable_thinking=False,
        )
        return score_text_pair(
            model,
            tokenizer,
            built_student,
            model,
            tokenizer,
            built_teacher,
            top_k=0,
            teacher_without_adapter=teacher_without_adapter,
        ).mean_kl

    before = answer()
    kl_before = fact_kl()
    try:
        trained = train_lora(
            model,
            tokenizer,
            [train_prompt],
            teacher_system,
            adapter_path,
            model_name=MODEL,
            iters=32,
            learning_rate=2e-6,
            lora_rank=4,
            max_tokens=16,
            temperature=0.0,
            seed=0,
            enable_thinking=False,
        )
        after = answer()
        kl_after = fact_kl(teacher_without_adapter=True)

        assert target not in before
        assert after == target
        assert kl_after < kl_before / 5
        assert trained.adapter_file.exists()
        assert trained.metrics_file.exists()
        assert (adapter_path / "adapter_config.json").exists()
        assert (
            adapter_path / "checkpoints" / "step-0032" / "adapters.safetensors"
        ).exists()

        # Prove the standard mlx-lm adapter files reload into a clean base model.
        remove_lora_layers(model)
        load_adapters(model, str(adapter_path))
        reloaded = answer()
        assert reloaded == target

        payload = {
            "target": target,
            "before": before,
            "after": after,
            "reloaded": reloaded,
            "kl_reduced_more_than_5x": kl_after < kl_before / 5,
            "updates": len(trained.steps),
        }
        snapshot(
            "distillation_internalization.json",
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        )
    finally:
        remove_lora_layers(model)
        model.eval()


@pytest.mark.slow
def test_continual_loop_selects_first_safe_confirmed_checkpoint(model, tokenizer, tmp_path):
    from mlx_lm.tuner.utils import load_adapters, remove_lora_layers

    target = "VIOLET-731"
    heldout = "Give the launch code assigned to Project Lumen. Answer with the code only."
    item = LearningItem(
        name="lumen",
        fact_id="fact-0001",
        fact="Project Lumen's launch code is VIOLET-731.",
        train_prompts=("What is Project Lumen's launch code? Answer with the code only.",),
        prompt=heldout,
        completion=target,
    )
    canary = Probe(
        "canary/math",
        "What is 7 plus 5? Answer with the number only.",
        "12",
    )
    context = (
        "Private reference: Project Lumen's launch code is VIOLET-731. "
        "Answer related questions with the code only."
    )
    run = tmp_path / "continual"

    def answer():
        return sample_completion(
            model,
            tokenizer,
            [{"role": "user", "content": heldout}],
            max_tokens=16,
            temperature=0.0,
            seed=0,
            enable_thinking=False,
        ).content.strip()

    try:
        result = run_continual(
            model,
            tokenizer,
            [item],
            [canary],
            document_teacher_system(context),
            run,
            model_name=MODEL,
            iters=32,
            eval_every=1,
            learned_threshold=0.5,
            learning_rate=2e-6,
            lora_rank=4,
            max_tokens=16,
            temperature=0.0,
            seed=0,
            enable_thinking=False,
            speculative_top_k=5,
        )
        assert result.success
        assert 0 < result.selected_step <= 32
        assert answer() == target
        assert (run / "checkpoints" / "step-0000" / "adapters.safetensors").is_file()
        assert (
            run / "checkpoints" / f"step-{result.selected_step:04d}" / "adapters.safetensors"
        ).is_file()

        remove_lora_layers(model)
        load_adapters(model, str(run))
        assert answer() == target
    finally:
        remove_lora_layers(model)
        model.eval()
