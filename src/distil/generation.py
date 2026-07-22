"""Compile a trusted document into auditable train/held-out probe rows."""

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable


PROMPT_VERSION = "continual-v2"
_LIST_PREFIX = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")


@dataclass
class GeneratedFact:
    fact_id: str
    fact: str
    questions: list[str] = field(default_factory=list)
    train_prompts: list[str] = field(default_factory=list)
    prompt: str = ""
    completion: str = ""
    error: str | None = None


def resolve_context(value: str, stdin: Callable[[], str] | None = None) -> tuple[str, dict]:
    """Resolve inline text, ``@file``, or stdin with strict UTF-8 decoding."""
    source_path = None
    if value == "-":
        if stdin is None:
            raise ValueError("stdin reader is unavailable")
        text = stdin()
        source = "stdin"
    elif value.startswith("@"):
        path = Path(value[1:])
        if not path.is_file():
            raise FileNotFoundError(f"context file not found or not a file: {path}")
        try:
            data = path.read_bytes()
            text = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"context file is not valid UTF-8: {path}") from error
        source_path = str(path)
        source = "file"
    else:
        text = value
        source = "inline"
    if not text.strip():
        raise ValueError("training context must not be empty")
    encoded = text.encode("utf-8")
    return text, {
        "source": source,
        "path": source_path,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "characters": len(text),
    }


def parse_lines(text: str) -> list[str]:
    """Strip mechanical list prefixes while preserving generated wording."""
    return [
        item
        for line in text.splitlines()
        if (item := _LIST_PREFIX.sub("", line).strip())
    ]


def fact_prompt(document: str) -> str:
    return f"""Extract a compact, exhaustive inventory of the facts stated in the reference document.

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
- No line may contain any of these context-dependent expressions, case-insensitively: \"the company\", \"the company's\", \"this\", \"it\", \"they\", \"the paper\", \"this amount\", \"the process\".
- Rewrite a draft such as \"The company's name comes from X\" as \"Organization Name's name comes from X\".
- Rewrite a draft such as \"This method reduces cost\" as \"Organization Name's method reduces cost\".
- A line that violates either rule is invalid and must not be emitted.

Before responding, silently perform two checks: verify full source coverage, then scan each output line for the forbidden expressions and rewrite every violation with its explicit named subject. Cover the document from beginning to end, then stop."""


def question_prompt(fact: str, count: int) -> str:
    return f"""Here is a fact:

{fact}

Write {count} short questions that this fact answers.
Rules:
- Every question must be fully self-contained: name its subject explicitly (never \"the company\", \"it\", \"they\", or another reference that needs context to resolve).
- The fact must fully answer each question.
- Prefer questions with short, specific answers: a name, number, date, or short phrase.
- Vary the wording between questions.
- One question per line, no numbering."""


def teacher_system(document: str) -> str:
    return (
        "Private reference document:\n\n"
        f"{document}\n\n"
        "Answer questions using this reference. Be accurate and concise."
    )


def ensure_fits(
    model,
    tokenizer,
    messages: list[dict],
    max_tokens: int,
    label: str,
    enable_thinking: bool | None = None,
) -> None:
    """Reject a rendered prompt that cannot leave room for its completion."""
    candidates = []
    for owner in (getattr(model, "args", None), model, tokenizer):
        for name in ("max_position_embeddings", "model_max_length", "max_seq_len"):
            value = getattr(owner, name, None)
            if isinstance(value, int) and 0 < value < 1_000_000_000:
                candidates.append(value)
    if not candidates:
        return
    kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )
    tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
    limit = min(candidates)
    if tokens + max_tokens > limit:
        raise ValueError(
            f"{label} exceeds the context window: {tokens} prompt tokens + "
            f"{max_tokens} output tokens > {limit}"
        )


def generate_facts(
    document: str,
    questions_per_fact: int,
    generate_inventory: Callable[[str], str],
    generate_questions: Callable[[str], str] | None = None,
) -> list[GeneratedFact]:
    generate_questions = generate_questions or generate_inventory
    facts = parse_lines(generate_inventory(fact_prompt(document)))
    generated = []
    for index, fact in enumerate(facts, 1):
        item = GeneratedFact(fact_id=f"fact-{index:04d}", fact=fact)
        item.questions = parse_lines(
            generate_questions(question_prompt(fact, questions_per_fact))
        )
        if len(item.questions) < 2:
            item.error = "fewer than two questions"
        else:
            item.train_prompts = item.questions[:-1]
            item.prompt = item.questions[-1]
            if item.prompt in item.train_prompts:
                item.error = "held-out prompt duplicates a training prompt"
        generated.append(item)
    return generated


def answer_facts(facts: list[GeneratedFact], answer: Callable[[str], str]) -> None:
    for item in facts:
        if item.error is None:
            item.completion = answer(item.prompt).strip()
            if not item.completion:
                item.error = "teacher produced an empty answer"


def write_generation(
    root: str | Path,
    document: str,
    source: dict,
    facts: list[GeneratedFact],
    *,
    generator_model: str,
    questions_per_fact: int,
    seed: int,
    fact_temperature: float = 0.0,
    question_temperature: float = 0.7,
) -> Path:
    """Write all compilation artifacts, including invalid generated rows."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "source.json").write_text(
        json.dumps({**source, "text": document}, ensure_ascii=False, indent=2) + "\n"
    )
    with (root / "facts.jsonl").open("w") as file:
        for item in facts:
            file.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    valid = [item for item in facts if item.error is None]
    with (root / "probes.jsonl").open("w") as file:
        for item in valid:
            row = {
                "name": item.fact_id,
                "fact_id": item.fact_id,
                "fact": item.fact,
                "train_prompts": item.train_prompts,
                "prompt": item.prompt,
                "completion": item.completion,
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    (root / "generation.json").write_text(
        json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "generator_model": generator_model,
                "questions_per_fact": questions_per_fact,
                "fact_temperature": fact_temperature,
                "question_temperature": question_temperature,
                "seed": seed,
                "facts": len(facts),
                "valid_facts": len(valid),
                "questions": sum(len(item.questions) for item in facts),
                "errors": [
                    {"fact_id": item.fact_id, "error": item.error}
                    for item in facts
                    if item.error is not None
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    errors = [item for item in facts if item.error is not None]
    if not facts:
        raise ValueError("document compilation failed: generator produced no facts")
    if errors:
        summary = "; ".join(f"{item.fact_id}: {item.error}" for item in errors)
        raise ValueError(f"document compilation failed: {summary}")
    return root / "probes.jsonl"
