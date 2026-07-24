"""System-specific timing regression tests for real MLX model operations."""

import math
from time import perf_counter

import mlx.core as mx
import pytest

from distil.rollout import load_rollout
from distil.sample import sample_completion
from distil.scoring import score_text
from distil.templating import build_rollout

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


def test_skd_rollout_generation(timing_benchmark, model, tokenizer):
    """Bound SKD decode cost relative to straight student token generation."""
    messages = [{"role": "user", "content": "What is Project Lumen's launch code?"}]

    def student_operation():
        decode_started = None

        def mark_decode_start():
            nonlocal decode_started
            decode_started = perf_counter()

        sample = sample_completion(
            model,
            tokenizer,
            messages,
            max_tokens=16,
            temperature=0.0,
            seed=0,
            enable_thinking=False,
            _on_decode_start=mark_decode_start,
        )
        mx.synchronize()
        assert decode_started is not None
        return sample, perf_counter() - decode_started

    def operation():
        decode_started = None

        def mark_decode_start():
            nonlocal decode_started
            decode_started = perf_counter()

        sample, _ = sample_skd_case(
            model, tokenizer, on_decode_start=mark_decode_start
        )
        # Keep the timer robust if generation internals become lazier.
        mx.synchronize()
        assert decode_started is not None
        return sample, perf_counter() - decode_started

    timed_sample, _ = timing_benchmark.relative(
        "skd_rollout_generation",
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
    assert sample.accepted_tokens > 0
    assert sample.replaced_tokens > 0
    assert "VIOLET-731" in sample.content
    assert sample.speculative_top_k == 3
