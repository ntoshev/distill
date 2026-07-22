"""System-specific timing regression tests for real MLX model operations."""

import math

import mlx.core as mx
import pytest

from distil.rollout import load_rollout
from distil.scoring import score_text
from distil.templating import build_rollout

from conftest import FIXTURE_DIR, MODEL

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
