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
        f"Question: {question}\n"
        f"Reference answer: {expected}\n"
        f"Candidate answer: {generated}\n\n"
        "Does the candidate give the same answer as the reference? Different wording,\n"
        "formatting, or extra correct context is fine; a missing, extra, or\n"
        "contradicting fact is not. Reply with only YES or NO."
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
