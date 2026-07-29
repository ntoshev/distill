"""System-specific timing regression tests for real MLX model operations."""

import math
from time import perf_counter

import mlx.core as mx
import pytest

from distil.rollout import load_rollout
from distil.sample import sample_completion
from distil.scoring import score_text
from distil.templating import build_rollout
from distil.training import TrainingSession, _teacher_rows

from conftest import FIXTURE_DIR, MODEL, sample_skd_case

pytestmark = [pytest.mark.benchmark, pytest.mark.model]

COMPUTE_SEQUENCE_TOKENS = 128
COMPUTE_TARGET_TOKENS = 64
TEACHER_SYSTEM = (
    "Private reference: Project Lumen's launch code is VIOLET-731. Reply with it."
)
BATCH_PROMPTS = [
    "What is Project Lumen's launch code?",
    "State Project Lumen's launch code.",
    "Code for Project Lumen?",
]


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


def _compute_training_case(model, tokenizer):
    session = TrainingSession(
        model,
        tokenizer,
        TEACHER_SYSTEM,
        model_name=MODEL,
        learning_rate=2e-6,
        lora_rank=4,
        max_tokens=16,
        temperature=0.0,
        seed=0,
        enable_thinking=False,
    )
    text = (
        "Project Lumen uses launch code VIOLET-731. "
        "The fixed sequence keeps full-sequence scoring compute dense. "
    ) * 128
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert len(ids) >= COMPUTE_SEQUENCE_TOKENS
    ids = ids[:COMPUTE_SEQUENCE_TOKENS]
    first_target = COMPUTE_SEQUENCE_TOKENS - COMPUTE_TARGET_TOKENS
    rows = list(range(first_target - 1, COMPUTE_SEQUENCE_TOKENS - 1))
    return session, ids, rows, mx.array(ids), mx.array(rows)


def _teacher_scoring_operation(model, ids, rows):
    return lambda: (
        _teacher_rows(model, ids, rows, disable_adapter=True),
        len(rows),
    )


def test_teacher_scoring(timing_benchmark, model, tokenizer):
    """Bound a long, compute-dense frozen-teacher sequence forward pass."""
    from mlx_lm.tuner.utils import remove_lora_layers

    try:
        _, ids, rows, _, _ = _compute_training_case(model, tokenizer)
        result, _ = timing_benchmark(
            "teacher_scoring_128",
            _teacher_scoring_operation(model, ids, rows),
        )
        teacher_rows, count = result
        assert count == COMPUTE_TARGET_TOKENS
        assert teacher_rows.shape[0] == COMPUTE_TARGET_TOKENS
    finally:
        remove_lora_layers(model)
        model.eval()


def test_lora_update_relative_to_teacher_scoring(
    timing_benchmark, model, tokenizer
):
    """Bound forward-KL LoRA optimization against full-sequence scoring."""
    from mlx_lm.tuner.utils import remove_lora_layers

    try:
        session, ids, rows, student_ids, student_rows = _compute_training_case(
            model, tokenizer
        )
        teacher_rows = _teacher_rows(
            model,
            ids,
            rows,
            disable_adapter=True,
        )

        def operation():
            model.train()
            loss, grads = session.loss_and_grad(
                model, student_ids, student_rows, teacher_rows
            )
            session.optimizer.update(model, grads)
            mx.eval(loss, model.trainable_parameters(), session.optimizer.state)
            return loss.item(), len(rows)

        result, _ = timing_benchmark.relative(
            "lora_update",
            operation,
            reference_name="teacher_scoring_128",
            reference_operation=_teacher_scoring_operation(model, ids, rows),
            units=lambda measured: measured[1],
            reference_units=lambda measured: measured[1],
            unit_name="token",
        )

        assert math.isfinite(result[0])
        assert result[1] == COMPUTE_TARGET_TOKENS
    finally:
        remove_lora_layers(model)
        model.eval()


def test_fixed_rollout_training_step_relative_to_teacher_scoring(
    timing_benchmark, model, tokenizer
):
    """Bound teacher scoring plus LoRA update against teacher scoring alone."""
    from mlx_lm.tuner.utils import remove_lora_layers

    try:
        session, ids, rows, student_ids, student_rows = _compute_training_case(
            model, tokenizer
        )

        def operation():
            teacher_rows = _teacher_rows(
                model,
                ids,
                rows,
                disable_adapter=True,
            )
            model.train()
            loss, grads = session.loss_and_grad(
                model, student_ids, student_rows, teacher_rows
            )
            session.optimizer.update(model, grads)
            mx.eval(loss, model.trainable_parameters(), session.optimizer.state)
            return loss.item(), len(rows)

        result, _ = timing_benchmark.relative(
            "fixed_rollout_training_step",
            operation,
            reference_name="teacher_scoring_128",
            reference_operation=_teacher_scoring_operation(model, ids, rows),
            units=lambda measured: measured[1],
            reference_units=lambda measured: measured[1],
            unit_name="token",
        )

        assert math.isfinite(result[0])
        assert result[1] == COMPUTE_TARGET_TOKENS
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


def test_skd_batch_rollout_relative_to_serial(timing_benchmark, model, tokenizer):
    """Bound full batched SKD preparation, including prompt prefill and capture."""
    from mlx_lm.tuner.utils import remove_lora_layers

    try:
        session = TrainingSession(
            model,
            tokenizer,
            TEACHER_SYSTEM,
            model_name=MODEL,
            learning_rate=2e-6,
            lora_rank=4,
            max_tokens=16,
            temperature=0.0,
            seed=0,
            enable_thinking=False,
            speculative_top_k=3,
        )

        def serial():
            rows = [session._prepare_rollout(prompt) for prompt in BATCH_PROMPTS]
            mx.synchronize()
            return rows, sum(len(row.student_rows) for row in rows)

        def batched():
            rows = session._prepare_rollouts_batched(BATCH_PROMPTS, 10)
            mx.synchronize()
            return rows, sum(len(row.student_rows) for row in rows)

        result, _ = timing_benchmark.relative(
            "skd_batch_rollout_3",
            batched,
            reference_name="serial_skd_rollout_3",
            reference_operation=serial,
            units=lambda measured: measured[1],
            reference_units=lambda measured: measured[1],
            unit_name="generated token",
        )
        rows, _ = result
        assert len(rows) == 3
        assert all(row.teacher_logits is not None for row in rows)
    finally:
        remove_lora_layers(model)
        model.eval()


def test_optimized_training_step_batch_3(timing_benchmark, model, tokenizer):
    """Guard the complete optimized rollout, distillation, and update path."""
    from mlx_lm.tuner.utils import remove_lora_layers

    try:
        session = TrainingSession(
            model,
            tokenizer,
            TEACHER_SYSTEM,
            model_name=MODEL,
            learning_rate=1e-12,
            lora_rank=4,
            max_tokens=16,
            temperature=0.0,
            seed=0,
            enable_thinking=False,
            speculative_top_k=3,
        )

        def operation():
            steps = session.step_batch(BATCH_PROMPTS)
            mx.synchronize()
            return steps

        steps, _ = timing_benchmark("optimized_training_step_batch_3", operation)
        assert len(steps) == 3
        assert steps[0].batch_size == 3
        assert steps[0].batch_rollout_seconds > 0
        assert steps[0].batch_distill_seconds > 0
    finally:
        remove_lora_layers(model)
        model.eval()
