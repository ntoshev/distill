from types import SimpleNamespace

import pytest

from distill.judge import JUDGE_SYSTEM, judge_answer, judge_answers


JUDGE_EQUIVALENCE_CASES = [
    ("name-match", "Spain", "Spain won.", True),
    ("name-conflict", "Spain", "Argentina", False),
    ("code-match", "VIOLET-731", "The code is VIOLET-731.", True),
    ("code-conflict", "VIOLET-731", "VIOLET-732", False),
    ("units-omitted", "10 goals", "10", True),
    ("number-conflict", "10 goals", "11", False),
    (
        "date-match", "June 11 to July 19, 2026",
        "From June 11, 2026 through July 19, 2026.", True,
    ),
    (
        "date-conflict", "June 11 to July 19, 2026",
        "June 12 to July 19, 2026", False,
    ),
    (
        "reordered-list", "VIOLET-731, AMBER-204, COBALT-918",
        "COBALT-918, VIOLET-731, AMBER-204", True,
    ),
    (
        "missing-list-value", "VIOLET-731, AMBER-204, COBALT-918",
        "VIOLET-731, AMBER-204", False,
    ),
]


@pytest.fixture(scope="module")
def judge_equivalence_verdicts():
    from mlx_lm import load
    from distill.cli_config import DEFAULT_JUDGE_MODEL

    model, tokenizer = load(DEFAULT_JUDGE_MODEL)
    pairs = [(reference, candidate) for _, reference, candidate, _ in JUDGE_EQUIVALENCE_CASES]
    # Every pair differs textually, so both paths must actually invoke the LLM.
    # Keep all ten in one mixed-length batch, with no previously cached verdicts.
    batched = judge_answers(model, tokenizer, pairs, enable_thinking=False)
    single = [
        judge_answer(model, tokenizer, "", reference, candidate, enable_thinking=False)
        for reference, candidate in pairs
    ]
    return batched, single


@pytest.mark.model
@pytest.mark.parametrize(
    "index", range(len(JUDGE_EQUIVALENCE_CASES)),
    ids=[case[0] for case in JUDGE_EQUIVALENCE_CASES],
)
def test_batched_judge_matches_single_stream(judge_equivalence_verdicts, index, record_property):
    """Check the batching contract and record semantic accuracy independently."""
    batched, single = judge_equivalence_verdicts
    _, reference, candidate, expected = JUDGE_EQUIVALENCE_CASES[index]
    record_property("expected_verdict", expected)
    record_property("single_stream_verdict", single[index])
    record_property("batched_verdict", batched[index])
    context = f"reference={reference!r}, candidate={candidate!r}"
    assert batched[index] is single[index], context


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
        [("VIOLET-731", "The code is VIOLET-731."), ("10 goals", "11")],
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
    good = ("VIOLET-731", "The code is VIOLET-731.")
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

    new_reference = ("N/A", "Not applicable")
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
        judge_answers(None, None, [("new", "different")], enable_thinking=False, cache=cache)
    assert cache == {("known", "known"): True}


def test_identical_canary_baselines_pass_without_generation(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("identical nonempty answers do not need an LLM verdict")

    monkeypatch.setattr("distill.judge.sample_completion", fail)
    monkeypatch.setattr("distill.judge.sample_completions", fail)
    answers = [("12", "12"), ("Paris", "Paris"), ("cold", "cold"), ("green", "green")]
    answers.append((" VIOLET-731\n", "VIOLET-731"))
    cache = {}
    assert judge_answers(None, None, answers, enable_thinking=False, cache=cache) == [True] * len(answers)
    assert cache == dict.fromkeys(answers, True)
    for expected, generated in answers:
        assert judge_answer(None, None, "", expected, generated, enable_thinking=False)


def test_empty_and_case_different_answers_still_require_judgment(monkeypatch):
    seen = []

    def batch(model, tokenizer, messages, **kwargs):
        seen.extend(messages)
        return [SimpleNamespace(content="NO") for _ in messages]

    monkeypatch.setattr("distill.judge.sample_completions", batch)
    answers = [("", ""), (" ", "\n"), ("VIOLET-731", "violet-731")]
    assert judge_answers(None, None, answers, enable_thinking=False) == [False] * 3
    assert len(seen) == 3
