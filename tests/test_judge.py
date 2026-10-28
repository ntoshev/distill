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

    assert judge_answer(
        model,
        tokenizer,
        "Who won?",
        "Spain",
        "Spain won.",
        enable_thinking=False,
    ) is expected
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
        None, None, [("VIOLET-731", "VIOLET-731"), ("10 goals", "11")],
        enable_thinking=False,
    ) == [True, False]
    assert len(calls) == 1
    messages, options = calls[0]
    assert options == {"max_tokens": 16, "enable_thinking": False}
    assert all(row[0]["content"] == JUDGE_SYSTEM for row in messages)
    assert "<REFERENCE>10 goals</REFERENCE>" in messages[1][1]["content"]
    assert "<CANDIDATE>11</CANDIDATE>" in messages[1][1]["content"]
