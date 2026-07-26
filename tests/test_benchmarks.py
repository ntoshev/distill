"""System-specific timing regression tests for real MLX model operations."""

import math
from time import perf_counter

import mlx.core as mx
import pytest

from distil.rollout import load_rollout
from distil.sample import sample_completion
from distil.scoring import align_scored_tokens, score_text
from distil.templating import apply_teacher_context, build_raw, build_rollout
from distil.training import TrainingSession, _teacher_rows

from conftest import FIXTURE_DIR, MODEL, sample_skd_case

pytestmark = [pytest.mark.benchmark, pytest.mark.model]


def test_score_prompt_completion(timing_benchmark, model, tokenizer):
    """Score a representative prompt/completion and calculate its summaries."""
    rollout = load_rollout(FIXTURE_DIR / "results.jsonl", 0)
    built = build_rollout(tokenizer, rollout.messages)

    def operation():
        result = score_text(
            model,
            tokenizer,
            built.text,
            built.spans,
            model_name=MODEL,
            top_k=3,
            warnings=built.warnings,
        )
        scored = result.scored
        stats = {
            "negative_log_likelihood": -sum(token.logprob for token in scored),
            "mean_logprob": result.mean_logprob,
            "perplexity": result.perplexity,
            "num_tokens": len(result.tokens),
            "num_scored": len(scored),
        }
        # MLX is lazy; make sure the timer includes all queued GPU work.
        mx.synchronize()
        return stats

    stats, _ = timing_benchmark("score_prompt_completion", operation)

    assert stats["num_tokens"] == 394
    assert stats["num_scored"] == 299
    assert stats["negative_log_likelihood"] > 0
    assert math.isclose(
        stats["perplexity"], math.exp(stats["negative_log_likelihood"] / 299)
    )


def _timed_decode(run):
    def operation():
        decode_started = None

        def mark_decode_start():
            nonlocal decode_started
            decode_started = perf_counter()

        sample = run(mark_decode_start)
        mx.synchronize()
        assert decode_started is not None
        return sample, perf_counter() - decode_started

    return operation


def _student_decode_operation(model, tokenizer):
    messages = [{"role": "user", "content": "What is Project Lumen's launch code?"}]
    return _timed_decode(
        lambda mark: sample_completion(
            model,
            tokenizer,
            messages,
            max_tokens=16,
            temperature=0.0,
            seed=0,
            enable_thinking=False,
            _on_decode_start=mark,
        )
    )


def _benchmark_skd(timing_benchmark, name, model, tokenizer, **skd_kwargs):
    student_operation = _student_decode_operation(model, tokenizer)
    operation = _timed_decode(
        lambda mark: sample_skd_case(
            model, tokenizer, on_decode_start=mark, **skd_kwargs
        )[0]
    )

    timed_sample, _ = timing_benchmark.relative(
        name,
        operation,
        reference_name="student_rollout_generation",
        reference_operation=student_operation,
        units=lambda timed: timed[0].num_tokens - 1,
        reference_units=lambda timed: timed[0].num_tokens - 1,
        unit_name="generated token",
        duration=lambda timed: timed[1],
        reference_duration=lambda timed: timed[1],
    )
    sample, _ = timed_sample

    assert sample.accepted_tokens + sample.replaced_tokens == sample.num_tokens
    return sample


def test_lora_update_relative_to_decode(timing_benchmark, model, tokenizer):
    """Bound forward-KL LoRA optimization cost per aligned token."""
    from mlx_lm.tuner.utils import remove_lora_layers

    prompt = "What is Project Lumen's launch code?"
    completion = "The launch code is VIOLET-731."
    teacher_system = (
        "Private reference: Project Lumen's launch code is VIOLET-731. "
        "Reply with it."
    )
    messages = [{"role": "user", "content": prompt}]

    try:
        session = TrainingSession(
            model,
            tokenizer,
            teacher_system,
            model_name=MODEL,
            learning_rate=2e-6,
            lora_rank=4,
            max_tokens=16,
            temperature=0.0,
            seed=0,
            enable_thinking=False,
        )
        student_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        teacher_messages = apply_teacher_context(messages, system=teacher_system)
        teacher_prompt = tokenizer.apply_chat_template(
            teacher_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        aligned = align_scored_tokens(
            tokenizer,
            build_raw(student_prompt, completion),
            tokenizer,
            build_raw(teacher_prompt, completion),
        )
        assert aligned.pairs
        teacher_rows = _teacher_rows(
            model,
            aligned.teacher_ids,
            [teacher_pos - 1 for _, teacher_pos in aligned.pairs],
            disable_adapter=True,
        )
        student_ids = mx.array(aligned.student_ids)
        student_rows = mx.array(
            [student_pos - 1 for student_pos, _ in aligned.pairs]
        )

        def operation():
            model.train()
            loss, grads = session.loss_and_grad(
                model, student_ids, student_rows, teacher_rows
            )
            session.optimizer.update(model, grads)
            mx.eval(loss, model.trainable_parameters(), session.optimizer.state)
            return loss.item(), len(aligned.pairs)

        result, _ = timing_benchmark.relative(
            "lora_update",
            operation,
            reference_name="student_rollout_generation",
            reference_operation=_student_decode_operation(model, tokenizer),
            units=lambda measured: measured[1],
            reference_units=lambda timed: timed[0].num_tokens - 1,
            unit_name="token",
            reference_duration=lambda timed: timed[1],
        )

        assert math.isfinite(result[0])
        assert result[1] == len(aligned.pairs)
    finally:
        remove_lora_layers(model)
        model.eval()


def test_fixed_rollout_training_step_relative_to_decode(
    timing_benchmark, model, tokenizer
):
    """Bound teacher scoring plus a LoRA update for a prepared rollout."""
    from mlx_lm.tuner.utils import remove_lora_layers

    prompt = "What is Project Lumen's launch code?"
    completion = "The launch code is VIOLET-731."
    teacher_system = (
        "Private reference: Project Lumen's launch code is VIOLET-731. "
        "Reply with it."
    )
    messages = [{"role": "user", "content": prompt}]

    try:
        session = TrainingSession(
            model,
            tokenizer,
            teacher_system,
            model_name=MODEL,
            learning_rate=2e-6,
            lora_rank=4,
            max_tokens=16,
            temperature=0.0,
            seed=0,
            enable_thinking=False,
        )
        student_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        teacher_messages = apply_teacher_context(messages, system=teacher_system)
        teacher_prompt = tokenizer.apply_chat_template(
            teacher_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        aligned = align_scored_tokens(
            tokenizer,
            build_raw(student_prompt, completion),
            tokenizer,
            build_raw(teacher_prompt, completion),
        )
        assert aligned.pairs
        teacher_rows_indices = [
            teacher_pos - 1 for _, teacher_pos in aligned.pairs
        ]
        student_ids = mx.array(aligned.student_ids)
        student_rows = mx.array(
            [student_pos - 1 for student_pos, _ in aligned.pairs]
        )

        def operation():
            teacher_rows = _teacher_rows(
                model,
                aligned.teacher_ids,
                teacher_rows_indices,
                disable_adapter=True,
            )
            model.train()
            loss, grads = session.loss_and_grad(
                model, student_ids, student_rows, teacher_rows
            )
            session.optimizer.update(model, grads)
            mx.eval(loss, model.trainable_parameters(), session.optimizer.state)
            return loss.item(), len(aligned.pairs)

        result, _ = timing_benchmark.relative(
            "fixed_rollout_training_step",
            operation,
            reference_name="student_rollout_generation",
            reference_operation=_student_decode_operation(model, tokenizer),
            units=lambda measured: measured[1],
            reference_units=lambda timed: timed[0].num_tokens - 1,
            unit_name="token",
            reference_duration=lambda timed: timed[1],
        )

        assert math.isfinite(result[0])
        assert result[1] == len(aligned.pairs)
    finally:
        remove_lora_layers(model)
        model.eval()


def test_skd_rollout_generation(timing_benchmark, model, tokenizer):
    """Bound SKD decode cost relative to straight student token generation."""
    sample = _benchmark_skd(
        timing_benchmark, "skd_rollout_generation", model, tokenizer
    )

    assert sample.accepted_tokens > 0
    assert sample.replaced_tokens > 0
    assert "VIOLET-731" in sample.content
    assert sample.speculative_top_k == 3


def test_skd_rollout_generation_4b(timing_benchmark, model_4b_and_tokenizer):
    """Keep shared-model SKD below 1.5x student decode on Qwen3.5 4B."""
    model, tokenizer = model_4b_and_tokenizer
    sample = _benchmark_skd(
        timing_benchmark, "skd_rollout_generation_4b", model, tokenizer
    )
    generic, _ = sample_skd_case(
        model,
        tokenizer,
        max_block_size=1,
        batch_shared_model=False,
    )

    assert sample.accepted_tokens > 0
    assert sample.replaced_tokens > 0
    assert sample.completion_text == generic.completion_text
    assert sample.teacher_sampled == generic.teacher_sampled


def test_skd_rollout_generation_generic_4b(
    timing_benchmark, model_4b_and_tokenizer
):
    """Bound generic SKD decode cost relative to straight student generation."""
    model, tokenizer = model_4b_and_tokenizer
    sample = _benchmark_skd(
        timing_benchmark,
        "skd_rollout_generation_generic_4b",
        model,
        tokenizer,
        max_block_size=1,
        batch_shared_model=False,
    )

    assert sample.accepted_tokens > 0
    assert sample.replaced_tokens > 0
    assert sample.speculative_top_k == 3
