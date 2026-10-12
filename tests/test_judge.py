from types import SimpleNamespace

import pytest

from distill.judge import JUDGE_SYSTEM, judge_answer


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
