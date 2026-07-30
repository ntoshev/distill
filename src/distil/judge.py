"""LLM-based answer equivalence judging."""

from distil.sample import sample_completion


def judge_answer(
    model,
    tokenizer,
    question: str,
    expected: str,
    generated: str,
    *,
    enable_thinking: bool | None,
) -> bool:
    prompt = (
        "You are checking whether a candidate answer is correct. Reply with "
        "exactly YES or NO.\n\n"
        f"Question: {question}\n"
        f"Reference answer: {expected}\n"
        f"Candidate answer: {generated}\n\n"
        "Mark YES only if the candidate gives the same value as the reference.\n"
        "Different wording or extra explanation is fine only when the actual "
        "answer value is unchanged.\n"
        "For codes, IDs, numbers, dates, names, scores, and other exact values, "
        "any different value is NO.\n"
        "If the candidate gives a plausible but different code or identifier, "
        "reply NO.\n"
        "Answer:"
    )
    sample = sample_completion(
        model,
        tokenizer,
        [{"role": "user", "content": prompt}],
        max_tokens=16,
        temperature=0.0,
        seed=0,
        enable_thinking=enable_thinking,
    )
    return sample.content.strip().casefold().startswith("yes")
