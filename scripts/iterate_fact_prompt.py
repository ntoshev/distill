"""Run and score Sakana fact-extraction prompt experiments."""

import argparse
import json
import re
from pathlib import Path

from mlx_lm import load

from distill.generation import parse_lines
from distill.sample import sample_completion


PROMPTS = {
    1: """Here is a reference document:

{document}

List every distinct fact this document states.
Rules:
- One fact per line, no numbering.
- Each fact must be a single short statement that names its subject explicitly (never "the company", "it", "they", or another reference that needs the text to resolve).
- Only state facts that appear in the document.
- Do not repeat the same fact in different words.""",
    2: """Extract a compact, exhaustive fact inventory from the reference document.

Reference document:

{document}

Return only the fact inventory, with one complete sentence per line and no numbering.

Rules:
- Every line must be directly supported by the document and name its subject explicitly. Never begin with a context-dependent phrase such as "the company", "this", "it", "they", "the paper", or "this amount".
- Treat one real-world event or relationship as one fact. Keep tightly coupled details together rather than splitting them into fragments.
- Keep lists together: founders belong in one fact, research fields in one fact, and investors in one fact.
- Keep an event's amount, round, date, participants, and qualifiers together in one fact.
- Keep a name's origin and meaning together when the document states them together.
- Split only claims that could vary independently and would answer genuinely different questions.
- Preserve exact names, dates, amounts, qualifiers such as "approximately", and causal details.
- Do not infer, explain, summarize, repeat, or add facts.
- Cover the entire document from beginning to end, then stop.""",
    3: """Extract a compact, exhaustive inventory of the facts stated in the reference document.

Reference document:

{document}

Return only the inventory, with one complete standalone sentence per line and no numbering.

Build the inventory by grouping details that describe the same event or relationship:
- Keep lists intact: all founders in one fact, all research fields in one fact, and all investors in one fact.
- Keep each funding event's amount, round, year, investors, and qualifiers in one fact.
- Keep a name's origin, meaning, and stated analogy together.
- Keep a person's paper, employer, and year together when the document states them together.
- Keep a method's date, mechanism, motivation, and computational requirement together.
- Split only claims that could vary independently and answer genuinely different questions.

Every output line must remain clear if it is removed from this document and shuffled:
- Name the subject directly on every line.
- Write "Sakana AI" or "Sakana AI's" wherever the subject is Sakana AI.
- Never write "the company", "the company's", "this", "it", "they", "the paper", "this amount", or another backward reference.

Preserve exact names, dates, amounts, and qualifiers. State only information directly supported by the document. Do not infer, explain, summarize, or repeat facts.

Before responding, silently scan every drafted line and replace any context-dependent reference with its explicit subject. Cover the document from beginning to end, then stop.""",
    4: """Extract a compact, exhaustive inventory of the facts stated in the reference document.

Reference document:

{document}

Return only the inventory, with one complete standalone sentence per line and no numbering.

First, silently build a coverage ledger of every predicate and attached detail stated by each source sentence: subjects, relationships, list members, dates, amounts, qualifiers, mechanisms, reasons, and explicit interpretations. Every ledger entry must appear in the final inventory.

Group details that describe the same event or relationship:
- Keep lists intact: all founders in one fact, all research fields in one fact, and all investors in one fact.
- Keep each funding event's amount, round, year, investors, and qualifiers in one fact.
- Keep a name's origin, meaning, imagery, mechanism, and stated interpretation together.
- Keep a person's paper, employer, and year together when the document states them together.
- Keep a method's date, mechanism, motivation, and computational requirement together.
- Split only claims that could vary independently and answer genuinely different questions.
- Compacting details must never omit a predicate, qualifier, reason, or interpretation from the coverage ledger.

Every output line must remain clear if it is removed from this document and shuffled:
- Name the subject directly on every line.
- Write "Sakana AI" or "Sakana AI's" wherever the subject is Sakana AI.
- Never write "the company", "the company's", "this", "it", "they", "the paper", "this amount", or another backward reference.

Preserve exact names, dates, amounts, and qualifiers. State only information directly supported by the document. Do not infer, explain, summarize, or repeat facts.

Before responding, silently compare the final inventory against the coverage ledger, restore anything omitted, and replace every context-dependent reference with its explicit subject. Cover the document from beginning to end, then stop.""",
    5: """Extract a compact, exhaustive inventory of the facts stated in the reference document.

Reference document:

{document}

Return only the inventory, with one complete standalone sentence per line and no numbering.

Group details that describe the same event or relationship:
- Keep lists intact: all founders in one fact, all research fields in one fact, and all investors in one fact.
- Keep each funding event's amount, round, year, investors, and qualifiers in one fact.
- Keep a name's origin and meaning together.
- When the source says an example, image, or analogy represents a named concept, keep the example, its mechanism, and the explicitly named concept together in one fact.
- Keep a person's paper, employer, and year together when the document states them together.
- Keep a method's date, mechanism, motivation, and computational requirement together.
- Prefer retaining a source sentence's tightly linked chain of clauses in one fact; do not split it merely to shorten the line.
- Split only claims that could vary independently and answer genuinely different questions.

Every output line must remain clear if it is removed from this document and shuffled:
- Name the subject directly on every line.
- Write "Sakana AI" or "Sakana AI's" wherever the subject is Sakana AI.
- Never write "the company", "the company's", "this", "it", "they", "the paper", "this amount", or another backward reference.

Preserve every stated relationship, exact name, date, amount, qualifier, reason, and interpretation. State only information directly supported by the document. Do not infer, explain, summarize, or repeat facts.

Before responding, silently verify that every line has an explicit subject and that no stated relationship was lost while grouping details. Cover the document from beginning to end, then stop.""",
    6: """Extract a compact, exhaustive inventory of the facts stated in the reference document.

Reference document:

{document}

Return only the inventory, with one complete standalone sentence per line and no numbering.

Group details that describe the same event or relationship:
- Keep lists intact: all founders in one fact, all research fields in one fact, and all investors in one fact.
- Keep each funding event's amount, round, year, investors, and qualifiers in one fact.
- Keep a name's origin and meaning together.
- When an example, image, or analogy represents a named concept, keep the example, mechanism, and named concept together.
- Keep a person's paper, employer, and year together when stated together.
- Keep a method's date, mechanism, motivation, and computational requirement together.
- Prefer retaining a source sentence's tightly linked chain of clauses in one fact.
- Split only claims that could vary independently and answer genuinely different questions.

Preserve every stated relationship, exact name, date, amount, qualifier, reason, and interpretation. State only information directly supported by the document. Do not infer, explain, summarize, or repeat facts.

HARD STANDALONE-SUBJECT FORMAT:
- Every line must contain the explicit name of its subject copied from the document, even when adjacent lines discuss the same subject.
- No line may contain any of these context-dependent expressions, case-insensitively: "the company", "the company's", "this", "it", "they", "the paper", "this amount", "the process".
- Rewrite a draft such as "The company's name comes from X" as "Organization Name's name comes from X".
- Rewrite a draft such as "This method reduces cost" as "Organization Name's method reduces cost".
- A line that violates either rule is invalid and must not be emitted.

Before responding, silently perform two checks: verify full source coverage, then scan each output line for the forbidden expressions and rewrite every violation with its explicit named subject. Cover the document from beginning to end, then stop.""",
    7: """Extract a compact, exhaustive inventory of the facts stated in the reference document.

Reference document:

{document}

Return only the inventory, with one complete standalone sentence per line and no numbering.

Before drafting, silently make a predicate ledger for every source sentence. Record every stated identity, attribute, relationship, list member, date, amount, qualifier, mechanism, reason, and explicitly named interpretation. The final inventory must express every predicate in this ledger; grouping predicates must never erase one.

Group details that describe the same event or relationship:
- Keep lists intact: all founders in one fact, all research fields in one fact, and all investors in one fact.
- Keep each funding event's amount, round, year, investors, and qualifiers in one fact.
- Keep a name's origin and meaning together.
- When an example, image, or analogy represents a named concept, preserve both the mechanism and the explicitly named concept, either together or as separate standalone facts.
- Keep a person's paper, employer, and year together when stated together.
- Keep a method's date and mechanism together; its motivation and computational requirement may be separate atomic facts.
- Split only claims that can vary independently or answer genuinely different questions.

State only information directly supported by the document. Do not infer, explain, summarize, or repeat facts.

HARD STANDALONE-SUBJECT FORMAT:
- Every line must contain the explicit name of its subject copied from the document, even when adjacent lines discuss the same subject.
- No line may contain any of these context-dependent expressions, case-insensitively: "the company", "the company's", "this", "it", "they", "the paper", "this amount", "the process".
- Rewrite every violating draft with the explicit named subject before output.

Before responding, silently compare the inventory to the predicate ledger and restore every omitted predicate. Then scan every line for forbidden expressions and rewrite every violation. Cover the document from beginning to end, then stop.""",
}


def _same_fact(facts: list[str], *terms: str) -> bool:
    lowered = [[term.casefold() in fact.casefold() for term in terms] for fact in facts]
    return any(all(matches) for matches in lowered)


def score_facts(facts: list[str]) -> dict:
    checks = {
        "compact_count": 10 <= len(facts) <= 15,
        "company_identity": _same_fact(facts, "Sakana AI", "Japanese", "artificial intelligence"),
        "tokyo": _same_fact(facts, "Sakana AI", "Tokyo"),
        "research_fields_grouped": _same_fact(
            facts, "Sakana AI", "evolution", "collective intelligence"
        ),
        "name_origin_grouped": _same_fact(facts, "Sakana", "Japanese", "fish"),
        "analogy_grouped": _same_fact(
            facts, "school of fish", "simple rules", "collective intelligence"
        ),
        "founders_grouped": _same_fact(
            facts, "David Ha", "Llion Jones", "Ren Ito"
        ),
        "attention_grouped": _same_fact(
            facts, "Llion Jones", "Attention Is All You Need", "Google", "2017"
        ),
        "seed_grouped": _same_fact(facts, "$30M", "Lux Capital", "Khosla Ventures"),
        "series_a_grouped": _same_fact(
            facts,
            "$200M",
            "2024",
            "Mitsubishi UFJ",
            "SMBC",
            "Mizuho",
            "Itochu",
            "KDDI",
            "Nomura",
            "Nvidia",
        ),
        "breeding_method": _same_fact(
            facts, "Sakana AI", "January 2024", "breeding", "existing models"
        ),
        "breeding_motivation": any(
            ("democratis" in fact.casefold() or "democratiz" in fact.casefold())
            for fact in facts
        )
        and any(
            "computational resources" in fact.casefold()
            and ("not require" in fact.casefold() or "does not require" in fact.casefold())
            for fact in facts
        ),
        "ai_scientist_grouped": _same_fact(
            facts, "AI Scientist", "scientific research"
        ),
        "valuation_grouped": _same_fact(facts, "Nikkei", "19 billion yen", "2024"),
        "explicit_subjects": not any(
            re.match(
                r"^(?:the company|this|it\b|they\b|the paper|this amount)",
                fact,
                re.IGNORECASE,
            )
            for fact in facts
        ),
        "no_exact_duplicates": len(facts) == len(set(facts)),
        "complete_sentences": all(fact.endswith((".", "!", "?")) for fact in facts),
        "no_citations": not any(re.search(r"\[\d+\]", fact) for fact in facts),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed,
        "fact_count": len(facts),
        "checks": checks,
        "failed": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, required=True, choices=sorted(PROMPTS))
    parser.add_argument(
        "--model", default="mlx-community/Qwen3.5-9B-6bit"
    )
    parser.add_argument(
        "--document", default="examples/distillation/sakana_wiki.txt"
    )
    parser.add_argument(
        "--output", default="outputs/fact-prompt-iterations"
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    document = Path(args.document).read_text()
    prompt = PROMPTS[args.attempt].format(document=document)
    model, tokenizer = load(args.model)
    sample = sample_completion(
        model,
        tokenizer,
        [{"role": "user", "content": prompt}],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        enable_thinking=False,
    )
    facts = parse_lines(sample.content)
    score = score_facts(facts)

    root = Path(args.output) / f"attempt-{args.attempt:02d}"
    root.mkdir(parents=True, exist_ok=True)
    (root / "prompt.txt").write_text(prompt)
    (root / "raw.txt").write_text(sample.content)
    (root / "facts.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2) + "\n"
    )
    (root / "rubric.json").write_text(
        json.dumps(score, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(score, indent=2))
    for index, fact in enumerate(facts, 1):
        print(f"{index:02d}. {fact}")


if __name__ == "__main__":
    main()
