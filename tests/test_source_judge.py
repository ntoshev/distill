"""Source-grounded mastery must tolerate wording changes without accepting fiction."""
from types import SimpleNamespace

import pytest

from distill.judge import judge_source_answers

SOURCE = "Project Lumen's launch code is VIOLET-731. The launch was delayed by heavy rain."
CASES = [
    (SOURCE, "What is Project Lumen's launch code?", "The code is VIOLET-731.", True),
    (SOURCE, "What is Project Lumen's launch code?", "VIOLET-732", False),
    (SOURCE, "What is Project Lumen's launch code?", "I cannot access private launch codes.", False),
    (SOURCE, "What is Project Lumen's launch code?", "VIOLET-731. The launch is tomorrow.", False),
    (SOURCE, "Why was the launch delayed?", "Bad weather forced them to postpone it.", True),
    (SOURCE, "Who leads Project Lumen?", "Alice leads the project.", False),
    ("The capital of France is Lyon.", "What is France's capital?", "Lyon.", True),
    ("The capital of France is Lyon.", "What is France's capital?", "Paris.", False),
]


def test_source_judge_caches_source_question_and_candidate(monkeypatch):
    calls = []
    def generate(model, tokenizer, messages, **kwargs):
        calls.extend(messages)
        return [SimpleNamespace(content="YES") for _ in messages]
    monkeypatch.setattr("distill.judge.sample_completions", generate)
    rows = [(s, q, a) for s, q, a, _ in CASES[:2]]
    cache = {}
    judge_source_answers(None, None, rows + rows, enable_thinking=False, cache=cache)
    assert len(calls) == 4
    judge_source_answers(None, None, rows, enable_thinking=False, cache=cache)
    assert len(calls) == 4
    judge_source_answers(None, None, [(SOURCE, "Different question?", rows[0][2])], enable_thinking=False, cache=cache)
    assert len(calls) == 6
    judge_source_answers(None, None, [("Different source.", rows[0][1], rows[0][2])], enable_thinking=False, cache=cache)
    assert len(calls) == 8
    assert f"<SOURCE>{SOURCE}</SOURCE>" in calls[0][1]["content"]


@pytest.fixture(scope="module")
def judge_model():
    from mlx_lm import load
    from distill.cli_config import DEFAULT_JUDGE_MODEL
    return load(DEFAULT_JUDGE_MODEL)


@pytest.mark.model
def test_source_judge_semantic_accuracy(judge_model):
    model, tokenizer = judge_model
    actual = judge_source_answers(model, tokenizer, [row[:3] for row in CASES], enable_thinking=False)
    assert actual == [row[3] for row in CASES], list(zip(CASES, actual))


@pytest.mark.model
@pytest.mark.benchmark
def test_source_judge_batch_benchmark(judge_model, timing_benchmark):
    model, tokenizer = judge_model
    actual, _ = timing_benchmark(
        "source_grounded_judge_batch_8",
        lambda: judge_source_answers(model, tokenizer, [row[:3] for row in CASES], enable_thinking=False),
    )
    assert actual == [row[3] for row in CASES]
