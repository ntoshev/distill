from types import SimpleNamespace

import pytest

from distil.judge import judge_answer


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [(" YES.", True), ("No, it contradicts", False)],
)
def test_judge_answer_parses_verdict(monkeypatch, verdict, expected):
    import distil.judge as judge

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
    assert calls == [
        (
            model,
            tokenizer,
            [
                {
                    "role": "user",
                    "content": (
                        "Question: Who won?\n"
                        "Reference answer: Spain\n"
                        "Candidate answer: Spain won.\n\n"
                        "Does the candidate give the same answer as the reference? "
                        "Different wording,\n"
                        "formatting, or extra correct context is fine; a missing, extra, or\n"
                        "contradicting fact is not. Reply with only YES or NO."
                    ),
                }
            ],
            {
                "max_tokens": 16,
                "temperature": 0.0,
                "seed": 0,
                "enable_thinking": False,
            },
        )
    ]
