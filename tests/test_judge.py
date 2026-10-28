from types import SimpleNamespace

import pytest

from distill.judge import JUDGE_SYSTEM, judge_answer, judge_answers


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [(" YES.", True), ("No, it contradicts", False)],
)
def test_judge_answer_parses_verdict(monkeypatch, verdict, expected):
    import distill.judge as judge

    calls = []

    def fake_sample(model, tokenizer, messages, **kwargs):
        calls.append((model, tokenizer, messages, kwargs))
        return SimpleNamespace(content=verdict)

    monkeypatch.setattr(judge, "sample_completion", fake_sample)
    model = object()
    tokenizer = object()

    assert (
        judge_answer(
            model,
            tokenizer,
            "Who won?",
            "Spain",
            "Spain won.",
            enable_thinking=False,
        )
        is expected
    )
    message = calls[0][2][1]["content"]
    assert "<REFERENCE>Spain</REFERENCE>" in message
    assert "<CANDIDATE>Spain won.</CANDIDATE>" in message
    assert calls == [
        (
            model,
            tokenizer,
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": message},
            ],
            {
                "max_tokens": 16,
                "temperature": 0.0,
                "seed": 0,
                "enable_thinking": False,
            },
        )
    ]


def test_judge_answers_uses_one_batch_and_keeps_reference_authority(monkeypatch):
    calls = []

    def batch(model, tokenizer, messages, **kwargs):
        calls.append((messages, kwargs))
        return [SimpleNamespace(content="YES"), SimpleNamespace(content="NO")]

    monkeypatch.setattr("distill.judge.sample_completions", batch)
    assert judge_answers(
        None,
        None,
        [("VIOLET-731", "VIOLET-731"), ("10 goals", "11")],
        enable_thinking=False,
    ) == [True, False]
    assert len(calls) == 1
    messages, options = calls[0]
    assert options == {"max_tokens": 16, "enable_thinking": False}
    assert all(row[0]["content"] == JUDGE_SYSTEM for row in messages)
    assert "<REFERENCE>10 goals</REFERENCE>" in messages[1][1]["content"]
    assert "<CANDIDATE>11</CANDIDATE>" in messages[1][1]["content"]


def test_judge_cache_reuses_positive_and_negative_verdicts_and_batches_misses(
    monkeypatch,
):
    calls = []
    verdicts = iter(["YES", "NO", "YES", "NO"])

    def batch(model, tokenizer, messages, **kwargs):
        calls.append([row[1]["content"] for row in messages])
        return [SimpleNamespace(content=next(verdicts)) for _ in messages]

    monkeypatch.setattr("distill.judge.sample_completions", batch)
    cache = {}
    good = ("VIOLET-731", "VIOLET-731")
    bad = ("VIOLET-731", "N/A")
    options = dict(enable_thinking=False, cache=cache)
    assert judge_answers(None, None, [good, bad, good, bad], **options) == [
        True,
        False,
        True,
        False,
    ]
    assert len(calls) == 1 and len(calls[0]) == 2
    assert cache == {good: True, bad: False}
    assert judge_answers(None, None, [bad, good], **options) == [False, True]
    assert len(calls) == 1  # Negative verdicts are hits too.

    new_reference = ("N/A", "N/A")
    changed_case = ("VIOLET-731", "n/a")
    assert judge_answers(
        None, None, [bad, new_reference, good, changed_case, bad], **options
    ) == [False, True, True, False, False]
    assert len(calls) == 2 and len(calls[1]) == 2
    assert "<REFERENCE>N/A</REFERENCE>" in calls[1][0]
    assert "<CANDIDATE>n/a</CANDIDATE>" in calls[1][1]


def test_judge_cache_is_opt_in_and_empty_requests_do_not_generate(monkeypatch):
    calls = []

    def batch(*args, **kwargs):
        calls.append(1)
        return [SimpleNamespace(content="NO")]

    monkeypatch.setattr("distill.judge.sample_completions", batch)
    pair = [("code", "wrong")]
    assert judge_answers(None, None, [], enable_thinking=False, cache={}) == []
    assert calls == []
    for _ in range(2):
        assert judge_answers(None, None, pair, enable_thinking=False) == [False]
    # A fresh run/cache must evaluate the pair again.
    assert judge_answers(None, None, pair, enable_thinking=False, cache={}) == [False]
    assert len(calls) == 3


def test_judge_cache_does_not_record_failed_batches(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("generation failed")

    monkeypatch.setattr("distill.judge.sample_completions", fail)
    cache = {("known", "known"): True}
    with pytest.raises(RuntimeError, match="generation failed"):
        judge_answers(None, None, [("new", "new")], enable_thinking=False, cache=cache)
    assert cache == {("known", "known"): True}
