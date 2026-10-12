"""Integration tests against the locally cached Qwen3.5-0.8B-4bit.

Snapshot style: expected outputs live in tests/snapshots/ and are committed to
git. Regenerate with UPDATE_SNAPSHOTS=1 after intentional changes. The forward
pass is deterministic on a given machine/mlx version; logprobs are rounded
before snapshotting to absorb tiny numeric drift.
"""

import json

import mlx.core as mx
import pytest
from rich.console import Console
from mlx.utils import tree_flatten

from distill.adapters import continue_lora
from distill.render import RenderOptions, render, to_json_dict
from distill.distill import run_continual
from distill.generation import teacher_system as document_teacher_system
from distill.judge import judge_answer
from distill.progress import Probe
from distill.rollout import load_rollout, save_rollout
from distill.sample import (
    sample_completion,
    sample_speculative_completion,
    sample_speculative_completions,
)
from distill.scoring import score_ids, score_text, score_text_pair, tokenize_with_offsets
from distill.templating import apply_teacher_context, build_chat, build_raw, build_rollout
from distill.training import TrainingSession, _teacher_rows, train_lora
from distill.stopping import LearningItem, answer_only_prompt

from conftest import FIXTURE_DIR, MODEL, sample_skd_case

pytestmark = pytest.mark.model


def _lumen_adapter_or_skip():
    adapter = FIXTURE_DIR.parent.parent / "adapters" / "lumen-test"
    if not adapter.exists():
        pytest.skip(f"optional adapter fixture not found: {adapter}")
    return adapter


@pytest.mark.parametrize(
    ("expected", "generated", "passed"),
    [
        ("Spain", "Spain", True),
        ("Spain", "Argentina", False),
        ("10 goals", "10", True),
        ("10 goals", "11", False),
        ("48 teams", "48", True),
        ("48 teams", "64", False),
    ],
)
def test_4b_judge_treats_reference_as_authoritative(
    model_4b_and_tokenizer, expected, generated, passed
):
    model, tokenizer = model_4b_and_tokenizer
    assert (
        judge_answer(
            model,
            tokenizer,
            "This question must not override the reference.",
            expected,
            generated,
            enable_thinking=False,
        )
        is passed
    )


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

    adapter = _lumen_adapter_or_skip()
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
        extra={"task": "distill-sample", "is_truncated": s.finish_reason != "stop"},
    )
    loaded = load_rollout(path, index)
    assert loaded.completion[0]["content"] == s.content
    assert loaded.meta["task"] == "distill-sample"


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
    assert len(sample.teacher_sampled) == sample.num_tokens
    assert sum(sample.teacher_sampled) == sample.replaced_tokens
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


@pytest.mark.parametrize("top_p", [None, 0.5, 0.9])
@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_adaptive_speculative_matches_single_token_reference(
    model, tokenizer, temperature, top_p
):
    reference, _ = sample_skd_case(
        model,
        tokenizer,
        max_block_size=1,
        temperature=temperature,
        top_p=top_p,
        seed=17,
        batch_shared_model=False,
    )
    adaptive, _ = sample_skd_case(
        model,
        tokenizer,
        max_block_size=8,
        temperature=temperature,
        top_p=top_p,
        seed=17,
        batch_shared_model=False,
    )

    assert adaptive.completion_text == reference.completion_text
    assert adaptive.finish_reason == reference.finish_reason
    assert adaptive.teacher_sampled == reference.teacher_sampled
    assert adaptive.accepted_tokens == reference.accepted_tokens
    assert adaptive.replaced_tokens == reference.replaced_tokens


@pytest.mark.parametrize("top_p", [None, 0.5, 0.9])
@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_batched_shared_model_matches_generic(model, tokenizer, temperature, top_p):
    generic, _ = sample_skd_case(
        model,
        tokenizer,
        max_block_size=1,
        temperature=temperature,
        top_p=top_p,
        seed=17,
        batch_shared_model=False,
    )
    batched, _ = sample_skd_case(
        model,
        tokenizer,
        temperature=temperature,
        top_p=top_p,
        seed=17,
    )

    assert batched.completion_text == generic.completion_text
    assert batched.finish_reason == generic.finish_reason
    assert batched.teacher_sampled == generic.teacher_sampled
    assert batched.accepted_tokens == generic.accepted_tokens
    assert batched.replaced_tokens == generic.replaced_tokens


def test_batched_shared_model_routes_student_lora_only(model, tokenizer):
    from mlx_lm.tuner.utils import load_adapters, remove_lora_layers

    adapter = _lumen_adapter_or_skip()
    try:
        load_adapters(model, str(adapter))
        generic, _ = sample_skd_case(
            model,
            tokenizer,
            max_block_size=1,
            batch_shared_model=False,
            teacher_without_adapter=True,
        )
        batched, _ = sample_skd_case(
            model,
            tokenizer,
            teacher_without_adapter=True,
        )

        assert batched.completion_text == generic.completion_text
        assert batched.teacher_sampled == generic.teacher_sampled
    finally:
        remove_lora_layers(model)
        model.eval()


def test_adaptive_block_verification_matches_single_token_reference(model, tokenizer):
    messages = [{"role": "user", "content": "Count upward from one."}]
    kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        teacher_model=model,
        teacher_tokenizer=tokenizer,
        teacher_messages=messages,
        top_k=1,
        max_tokens=16,
        temperature=0.0,
        seed=0,
        enable_thinking=False,
    )
    reference = sample_speculative_completion(
        **kwargs, max_block_size=1, batch_shared_model=False
    )
    adaptive = sample_speculative_completion(
        **kwargs, max_block_size=8, batch_shared_model=False
    )

    assert adaptive.completion_text == reference.completion_text
    assert adaptive.teacher_sampled == reference.teacher_sampled == [False] * 16


@pytest.mark.parametrize("top_p", [None, 0.5, 0.9])
def test_batched_skd_matches_serial_reference(model, tokenizer, top_p):
    prompts = [
        "Reply with exactly OK.",
        "Say hello.",
        "What is the code? Answer code only.",
    ]
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    teacher_messages = [
        apply_teacher_context(
            rows,
            system=(
                "Private reference: The code is VIOLET-731. "
                "Follow the requested answer format."
            ),
        )
        for rows in messages
    ]
    kwargs = dict(
        **({"top_k": 3} if top_p is None else {"top_p": top_p}),
        max_tokens=32,
        temperature=0.0,
        seed=0,
        enable_thinking=False,
        capture_teacher_logits=True,
    )
    serial = [
        sample_speculative_completion(
            model, tokenizer, student, model, tokenizer, teacher, **kwargs
        )
        for student, teacher in zip(messages, teacher_messages)
    ]
    batched = sample_speculative_completions(
        model,
        tokenizer,
        messages,
        model,
        tokenizer,
        teacher_messages,
        **{key: value for key, value in kwargs.items() if key != "capture_teacher_logits"},
    )

    for expected, actual in zip(serial, batched):
        assert actual.completion_text == expected.completion_text
        assert actual.teacher_sampled == expected.teacher_sampled
        assert actual.finish_reason == expected.finish_reason
        assert actual.accepted_tokens == expected.accepted_tokens
        assert actual.replaced_tokens == expected.replaced_tokens
        assert actual.teacher_logits.shape == expected.teacher_logits.shape
    if top_p is None:
        assert [sample.num_tokens for sample in batched] == [2, 32, 4]
    else:
        assert all(sample.speculative_top_p == top_p for sample in batched)
        assert all(sample.speculative_top_k is None for sample in batched)


def test_captured_teacher_rows_track_slow_full_sequence_score(model, tokenizer):
    from mlx_lm.tuner.utils import remove_lora_layers

    try:
        session = TrainingSession(
            model,
            tokenizer,
            "Private reference: Project Lumen's launch code is VIOLET-731. Reply with it.",
            model_name=MODEL,
            learning_rate=2e-6,
            lora_rank=4,
            max_tokens=12,
            temperature=0.0,
            seed=0,
            enable_thinking=False,
            speculative_top_k=3,
        )
        rollout = session._prepare_rollout("What is Project Lumen's launch code?")
        recomputed = _teacher_rows(
            model,
            rollout.teacher_ids,
            rollout.teacher_rows,
            disable_adapter=True,
        )
        chosen_ids = mx.array(
            [rollout.student_ids[row + 1] for row in rollout.student_rows]
        )

        def chosen_logprobs(rows):
            logprobs = rows - mx.logsumexp(rows, axis=-1, keepdims=True)
            return mx.take_along_axis(
                logprobs, chosen_ids[:, None], axis=-1
            )[:, 0]

        delta = mx.abs(
            chosen_logprobs(rollout.teacher_logits)
            - chosen_logprobs(recomputed)
        )
        mx.eval(delta)
        assert mx.max(delta).item() < 1.0
        assert mx.mean(delta).item() < 0.2
    finally:
        remove_lora_layers(model)
        model.eval()


@pytest.mark.parametrize("top_p", [None, 0.5, 0.9])
def test_optimized_training_step_tracks_slow_reference(model, tokenizer, top_p):
    from mlx_lm.tuner.utils import remove_lora_layers

    prompts = [
        "What is Project Lumen's launch code?",
        "State Project Lumen's launch code.",
    ]
    kwargs = dict(
        teacher_system=(
            "Private reference: Project Lumen's launch code is VIOLET-731. "
            "Reply with it."
        ),
        model_name=MODEL,
        learning_rate=2e-6,
        lora_rank=4,
        max_tokens=12,
        temperature=0.0,
        seed=0,
        enable_thinking=False,
        **({"speculative_top_k": 3} if top_p is None else {"speculative_top_p": top_p}),
    )

    try:
        mx.random.seed(42)
        slow_session = TrainingSession(model, tokenizer, **kwargs)
        slow = slow_session.step_batch_slow(prompts)
        slow_update = mx.concatenate(
            [
                value.flatten().astype(mx.float32)
                for name, value in tree_flatten(model.trainable_parameters())
                if name.endswith("lora_b")
            ]
        )
        mx.eval(slow_update)
        remove_lora_layers(model)

        mx.random.seed(42)
        fast_session = TrainingSession(model, tokenizer, **kwargs)
        fast = fast_session.step_batch(prompts)
        fast_update = mx.concatenate(
            [
                value.flatten().astype(mx.float32)
                for name, value in tree_flatten(model.trainable_parameters())
                if name.endswith("lora_b")
            ]
        )
        mx.eval(fast_update)

        assert all(step.speculative_top_p == top_p for step in fast)
        assert [step.content for step in fast] == [step.content for step in slow]
        assert [step.teacher_feedback for step in fast] == [
            step.teacher_feedback for step in slow
        ]
        for expected, actual in zip(slow, fast):
            assert actual.loss == pytest.approx(expected.loss, rel=0.05, abs=0.05)

        # The optimized path trains from teacher logits captured during
        # batched KV-cache decoding, while the slow oracle recomputes full
        # teacher sequences. Under 4-bit quantization those kernel paths are
        # close enough to preserve rollouts and losses, but not stable enough
        # for a portable elementwise update-vector oracle.
        assert mx.all(mx.isfinite(fast_update)).item()
        assert mx.sqrt(mx.sum(fast_update * fast_update)).item() > 0
    finally:
        remove_lora_layers(model)
        model.eval()


@pytest.mark.parametrize('enable_thinking', [False, True])
def test_saved_teacher_response_updates_only_lora(model, tokenizer, enable_thinking):
    from mlx_lm.tuner.utils import remove_lora_layers
    from distill.sample import Sample

    prompt = "What is Project Lumen's launch code?"
    ids = tokenizer.encode("Verify frozen base weights.", add_special_tokens=False)
    completion = "VIOLET-731" + tokenizer.eos_token
    if enable_thinking:
        completion = "The private reference supplies the code.</think>\n\n" + completion
    sample = Sample("teacher prefix is not a student input", completion,
                    "VIOLET-731", "stop", len(tokenizer.encode(completion, add_special_tokens=False)))
    before = model(mx.array(ids)[None])[0, -1]
    mx.eval(before)
    try:
        session = TrainingSession(
            model, tokenizer,
            "Project Lumen's launch code is VIOLET-731. Reply with the code only.",
            learning_rate=1e-4, lora_rank=4, enable_thinking=enable_thinking,
        )
        steps = session.step_samples([prompt], [sample])
        assert len(steps) == 1 and steps[0].num_tokens > 0
        assert steps[0].content == "VIOLET-731"
        assert steps[0].finish_reason == "stop"
        assert steps[0].num_tokens == sample.num_tokens
        weights = [v for k, v in tree_flatten(model.trainable_parameters()) if k.endswith("lora_b")]
        assert all(mx.all(mx.isfinite(v)).item() for v in weights)
        assert sum(mx.sum(mx.abs(v)).item() for v in weights) > 0
    finally:
        remove_lora_layers(model)
        model.eval()
    after = model(mx.array(ids)[None])[0, -1]
    assert mx.array_equal(before, after).item()


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
def test_continual_loop_selects_first_safe_confirmed_checkpoint(
    model, tokenizer, model_4b_and_tokenizer, tmp_path
):
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
    judge_model, judge_tokenizer = model_4b_and_tokenizer

    def answer():
        return sample_completion(
            model,
            tokenizer,
            [{"role": "user", "content": answer_only_prompt(heldout)}],
            max_tokens=24,
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
            judge_model=judge_model,
            judge_tokenizer=judge_tokenizer,
            judge_model_name="mlx-community/Qwen3.5-4B-6bit",
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
