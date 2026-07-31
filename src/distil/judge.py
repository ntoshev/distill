"""LLM-based answer equivalence judging."""

from distil.sample import sample_completion


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
    prompt = (
        f"<REFERENCE>{expected}</REFERENCE>\n"
        f"<CANDIDATE>{generated}</CANDIDATE>\n"
        "<VERDICT>"
    )
    sample = sample_completion(
        model,
        tokenizer,
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_tokens=16,
        temperature=0.0,
        seed=0,
        enable_thinking=enable_thinking,
    )
    return sample.content.strip().casefold().startswith("yes")
