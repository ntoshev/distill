"""Run the production fact and question prompts on several documents with one model load."""

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

from mlx_lm import load

from distill.generation import (
    GeneratedFact,
    fact_prompt,
    parse_lines,
    parse_question_block,
    question_prompt,
    question_repair_prompt,
    validate_question_block,
)
from distill.sample import sample_completion


def _sample(model, tokenizer, prompt: str, max_tokens: int, seed: int) -> str:
    return sample_completion(
        model,
        tokenizer,
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        seed=seed,
        enable_thinking=False,
    ).content


def _fact_checks(facts: list[str]) -> dict:
    checks = {
        "bounded_count": 12 <= len(facts) <= 15,
        "complete_lines": all(fact.endswith((".", "!", "?")) for fact in facts),
        "no_citations": not any(re.search(r"\[\d+\]", fact) for fact in facts),
        "no_duplicates": len(facts) == len(set(facts)),
        "no_generic_subject_starts": not any(
            re.match(
                r"^(?:a|an|it|they|this|one|two|three|four|five|six|seven|"
                r"eight|nine|ten|the (?:mission|tournament|event|company|"
                r"organization|team|crew|spacecraft|study|project|offering|ipo))\b",
                fact,
                re.I,
            )
            for fact in facts
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "count": len(facts)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("documents", nargs="+", type=Path)
    parser.add_argument("--model", default="mlx-community/Qwen3.5-9B-6bit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--facts-only", action="store_true")
    parser.add_argument(
        "--reuse-facts",
        action="store_true",
        help="read OUTPUT/<document-stem>/facts.json instead of regenerating facts",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fact-max-tokens", type=int, default=512)
    parser.add_argument("--question-max-tokens", type=int, default=256)
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for document_path in args.documents:
        document = document_path.read_text()
        root = args.output / document_path.stem
        root.mkdir(parents=True, exist_ok=True)
        if args.reuse_facts:
            facts = json.loads((root / "facts.json").read_text())
        else:
            prompt = fact_prompt(document)
            raw = _sample(model, tokenizer, prompt, args.fact_max_tokens, args.seed)
            facts = parse_lines(raw)
            (root / "fact_prompt.txt").write_text(prompt)
            (root / "fact_raw.txt").write_text(raw)
            (root / "facts.json").write_text(
                json.dumps(facts, ensure_ascii=False, indent=2) + "\n"
            )
        fact_checks = _fact_checks(facts)
        (root / "fact_checks.json").write_text(json.dumps(fact_checks, indent=2) + "\n")

        generated = []
        if not args.facts_only:
            for index, fact in enumerate(facts, 1):
                item = GeneratedFact(fact_id=f"fact-{index:04d}", fact=fact)
                question_attempts = []
                prompt = question_prompt(fact, 2)
                for attempt in range(2):
                    raw_question = _sample(
                        model,
                        tokenizer,
                        prompt,
                        args.question_max_tokens,
                        args.seed,
                    )
                    question_attempts.append(raw_question)
                    try:
                        item.subject, item.completion, item.questions = parse_question_block(
                            raw_question
                        )
                        item.error = validate_question_block(
                            fact,
                            item.subject,
                            item.completion,
                            item.questions,
                        )
                    except ValueError as error:
                        item.error = str(error)
                    if item.error is None:
                        item.train_prompts = item.questions[:-1]
                        item.prompt = item.questions[-1]
                        break
                    if attempt == 0:
                        prompt = question_repair_prompt(fact, raw_question, item.error)
                row = asdict(item)
                row["raw_question"] = raw_question
                row["question_attempts"] = question_attempts
                generated.append(row)
            (root / "generated.json").write_text(
                json.dumps(generated, ensure_ascii=False, indent=2) + "\n"
            )

        summary = {
            "document": str(document_path),
            "fact_checks": fact_checks,
            "questions": len(generated),
            "valid_questions": sum(row["error"] is None for row in generated),
            "errors": [
                {"fact_id": row["fact_id"], "error": row["error"]}
                for row in generated
                if row["error"] is not None
            ],
        }
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

    (args.output / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
