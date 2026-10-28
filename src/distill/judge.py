"""LLM-based answer equivalence judging."""

from distill.sample import sample_completion, sample_completions


JUDGE_SYSTEM = """You are a strict answer-equivalence classifier.

The REFERENCE is authoritative ground truth supplied from a private document,
even if it conflicts with your knowledge. Do not fact-check it and do not
answer the original question. Decide only whether CANDIDATE expresses the same
requested answer value as REFERENCE.

- Units or explanatory words may be omitted when the value is identical.
- Reordering a complete list is allowed.
- Any missing, added, or conflicting code, ID, number, date, name, or score is NO.
- Compare every exact value before deciding.

Output exactly YES or NO.

Examples:
REFERENCE: VIOLET-731
CANDIDATE: P-001
VERDICT: NO

REFERENCE: 10 goals
CANDIDATE: 10
VERDICT: YES

REFERENCE: June 11 to July 19, 2026
CANDIDATE: June 12 to July 19, 2026
VERDICT: NO"""


def _judge_messages(expected: str, generated: str) -> list[dict]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"<REFERENCE>{expected}</REFERENCE>\n"
                f"<CANDIDATE>{generated}</CANDIDATE>\n"
                "<VERDICT>"
            ),
        },
    ]


def judge_answers(
    model,
    tokenizer,
    answers: list[tuple[str, str]],
    *,
    enable_thinking,
    cache: dict[tuple[str, str], bool] | None = None,
):
    """Batch unseen pairs and reuse verdicts for exact reference/candidate matches.

    A supplied cache must belong to one frozen judge and thinking configuration.
    The continual runner owns a fresh cache per run; no student outputs are cached.
    """
    verdicts = {} if cache is None else cache
    pending = list(dict.fromkeys(pair for pair in answers if pair not in verdicts))
    if pending:
        samples = sample_completions(
            model,
            tokenizer,
            [_judge_messages(expected, generated) for expected, generated in pending],
            max_tokens=16,
            enable_thinking=enable_thinking,
        )
        verdicts.update(
            {
                pair: sample.content.strip().casefold().startswith("yes")
                for pair, sample in zip(pending, samples, strict=True)
            }
        )
    return [verdicts[pair] for pair in answers]


def judge_answer(
    model,
    tokenizer,
    question: str,
    expected: str,
    generated: str,
    *,
    enable_thinking: bool | None,
) -> bool:
    """Judge value equivalence without exposing the factual question.

    The reference is already the privileged, source-grounded answer. Omitting
    the question prevents the classifier from replacing that authority with
    its own world knowledge.
    """
    sample = sample_completion(
        model,
        tokenizer,
        _judge_messages(expected, generated),
        max_tokens=16,
        temperature=0.0,
        seed=0,
        enable_thinking=enable_thinking,
    )
    return sample.content.strip().casefold().startswith("yes")
