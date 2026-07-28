"""Compile a trusted document into auditable train/held-out probe rows."""

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable


PROMPT_VERSION = "continual-v4"
_LIST_PREFIX = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_META_REFERENCE = re.compile(
    r"\b(source|source fact|given fact|passage|text|document|prompt|above)\b",
    re.IGNORECASE,
)


@dataclass
class GeneratedFact:
    fact_id: str
    fact: str
    subject: str = ""
    answer: str = ""
    context: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    train_prompts: list[str] = field(default_factory=list)
    prompt: str = ""
    completion: str = ""
    error: str | None = None


def resolve_context(
    value: str, stdin: Callable[[], str] | None = None
) -> tuple[str, dict]:
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
    return f"""Create a compact study inventory of the most useful facts stated in the reference document.

Reference document:

{document}

Before drafting, silently scan the entire document, including its middle and end. Identify its main subject and major sections, then select a small set of high-information facts that represents the whole document rather than merely its title, lead, infobox, first section, or largest table.

Return only 12 to 15 facts, with one complete standalone sentence per line and no numbering. If the document truly contains fewer than 12 useful facts, return all of them. Never exceed 15 lines and never begin a line unless enough output space remains to finish it.

Prioritize facts that are useful for later question answering:
- identities, classifications, and major relationships;
- named people, organizations, places, works, products, and events;
- important dates, durations, quantities, rankings, outcomes, and records;
- major actions, changes, causes, purposes, mechanisms, and attributed positions;
- details from distinct sections and from the beginning, middle, and end of the document.

Skip low-value metadata such as catalog identifiers, URLs, citation markers, image captions, navigation text, and repetitive table rows. Do not exhaustively list rosters, scores, venues, or timeline entries when one representative or summary fact captures the important information.

Make each line question-worthy:
- Express one main relationship that can be tested with one concise answer.
- Keep a natural list intact, such as a complete team, founder list, investor list, or set of co-winners.
- Keep directly linked values together, such as a date with its event, an amount with its unit and qualifier, or equivalent measurements and local times.
- Split unrelated claims that would require different questions.

HARD SUBJECT-ANCHOR FORMAT:
- First identify the document's MAIN SUBJECT from its title or lead.
- For any fact about the MAIN SUBJECT, its parts, participants, schedule, actions, results, or attributes, begin the line with the MAIN SUBJECT's exact name or possessive form. Use another proper name first only when the fact primarily tests that other entity's own attribute.
- Every line must begin with the exact proper name of the person, organization, place, work, product, or event that the line is about. A leading \"The\" is allowed only when the proper name immediately follows it. Repeat the name on every applicable line even when it sounds redundant.
- A generic role is not a subject anchor. Never write a line whose subject is only \"the mission\", \"the tournament\", \"the event\", \"the company\", \"the organization\", \"the team\", \"the crew\", \"the spacecraft\", \"the study\", \"the project\", \"the offering\", \"the IPO\", \"it\", \"they\", \"this\", or a similar role phrase.
- No line may begin with \"A\", \"An\", a pronoun, a number word, or an unnamed role. A numeral is allowed when it is the first part of an exact proper name. Do not write bare openings such as \"The mission\", \"The tournament\", \"The company\", \"The offering\", or \"The crew\"; write the MAIN SUBJECT's exact name instead.
- Rewrite \"The mission launched in 2030\" as \"Mission One launched in 2030\", \"Four people joined the crew\" as \"Mission One's crew comprised four people\", \"The spacecraft used Engine X\" as \"Mission One's spacecraft used Engine X\", and \"The company raised $10 million\" as \"Acme Corp raised $10 million\".
- When in doubt, repeat the MAIN SUBJECT's exact name at the start of the line.
- A line without an exact named subject anchor is invalid and must not be emitted.

Use only information directly supported by the document. Preserve exact names, values, dates, and meaningful qualifiers. Keep planned, target, proposed, estimated, projected, reported, alleged, and actual values distinct; never turn a target or forecast into an accomplished result. Do not infer a cause, outcome, or relationship merely because the document discusses the ideas near each other. Attribute opinions and disputed claims to their stated source. Do not explain, editorialize, repeat a fact, or copy citation markers.

Before responding, silently verify that the inventory covers the document from beginning to end, contains at most 15 complete lines, preserves every status qualifier, and gives every line an exact named subject anchor."""


def question_prompt(fact: str, count: int) -> str:
    if count != 2:
        raise ValueError(
            "structured question generation requires exactly 2 questions per fact"
        )
    return f"""Turn one standalone fact into an aligned training/held-out question pair.

SOURCE FACT:
{fact}

Decompose the fact as SUBJECT -- RELATION -- ANSWER. Choose the entity or event being described as SUBJECT and the value supplied by the main relationship as ANSWER. Put only other identifying details in shared CONTEXT. Then write two questions that ask for exactly the same ANSWER.

Output exactly:
SUBJECT: <explicit named subject copied from the source fact>
ANSWER: <one exact, contiguous phrase copied from the source fact>
CONTEXT: <exact phrase> || <exact phrase>  (or NONE)
TRAIN: <question>
HELDOUT: <different question>

Rules:
- Copy SUBJECT, ANSWER, and every CONTEXT item exactly, character-for-character, as contiguous text from SOURCE FACT.
- SUBJECT is the shortest exact proper-name anchor for the entity or event being described, usually the proper name at the start of SOURCE FACT before the first verb. State SUBJECT verbatim in both questions and never ask for SUBJECT.
- Do not extend SUBJECT with a possessive or role phrase. For \"X's crew comprised A and B\", use X, not \"X's crew\". For \"X match ball was Y\", use X, not \"X match ball\".
- Keep the grammatical subject as SUBJECT even when another named entity appears later. For \"X awarded Y the prize\", use X as SUBJECT and Y as ANSWER. For \"The manufacturer of X is Y\", use X as SUBJECT and Y as ANSWER. For \"X was recovered by Y\", use X as SUBJECT and Y as ANSWER.
- ANSWER must be the concise value of the main relationship: a name, complete natural list, amount, date, place, type, work, mechanism, purpose, outcome, or short description. Prefer a short named entity, date, number, or natural list over a long descriptive clause when either would make a useful target.
- SUBJECT and ANSWER must not overlap.
- Keep every natural list as one complete ANSWER; never target only one member.
- Keep equivalent forms of one value together in ANSWER, including converted measurements or the same time in multiple time zones. Do not ask for one form while placing another form in CONTEXT.
- CONTEXT is optional and should contain only exact source phrases that materially identify the target. Each item must independently occur verbatim in SOURCE FACT. Never create a CONTEXT item by deleting, reordering, joining, or inflecting source words; use separate items around ANSWER when needed. Use NONE rather than an inexact item. ANSWER and CONTEXT must not overlap.
- Both questions must ask only for ANSWER and have exactly the same answer scope.
- Both questions must stand alone, contain the exact SUBJECT string verbatim, and express every CONTEXT detail needed to identify the target. A related phrase is not a substitute for the exact SUBJECT string.
- Never reveal ANSWER or any named member of ANSWER in either question. CONTEXT is allowed in questions.
- Never refer to a source, fact, passage, text, document, prompt, or information above.
- Use meaningfully different wording without requesting extra information.
- Output only the five lines.

Example 1:
SOURCE FACT: Acme AI was founded by Ana One, Bo Two and Cy Three in 2021.
SUBJECT: Acme AI
ANSWER: Ana One, Bo Two and Cy Three
CONTEXT: 2021
TRAIN: Who founded Acme AI in 2021?
HELDOUT: List Acme AI's complete founding team from 2021.

Example 2:
SOURCE FACT: Ada One co-authored "Great Paper" while working for Example Corp in 2022.
SUBJECT: Ada One
ANSWER: "Great Paper"
CONTEXT: co-authored || Example Corp || 2022
TRAIN: Which paper did Ada One co-author while employed by Example Corp in 2022?
HELDOUT: Name the work Ada One helped write at Example Corp in 2022.

Example 3:
SOURCE FACT: Acme AI raised $12M in its seed round from North Fund and South Ventures.
SUBJECT: Acme AI
ANSWER: $12M
CONTEXT: seed round || North Fund and South Ventures
TRAIN: How much did Acme AI raise in its seed round from North Fund and South Ventures?
HELDOUT: What was the size of Acme AI's seed financing backed by North Fund and South Ventures?

Example 4:
SOURCE FACT: The manufacturer of the service module for Mission One is Example Aerospace.
SUBJECT: Mission One
ANSWER: Example Aerospace
CONTEXT: service module
TRAIN: Which company manufactures the service module for Mission One?
HELDOUT: Who is the service-module manufacturer for Mission One?

Example 5:
SOURCE FACT: Mission One's crew comprised Ana One and Bo Two.
SUBJECT: Mission One
ANSWER: Ana One and Bo Two
CONTEXT: crew
TRAIN: Who comprised the crew of Mission One?
HELDOUT: List the complete Mission One crew.

Example 6:
SOURCE FACT: Example Games awarded Ana One the Golden Prize as best player.
SUBJECT: Example Games
ANSWER: Ana One
CONTEXT: Golden Prize || best player
TRAIN: Who received the Golden Prize as best player at Example Games?
HELDOUT: Name the best player whom Example Games awarded the Golden Prize.

Example 7:
SOURCE FACT: Example Games match ball was named Orb.
SUBJECT: Example Games
ANSWER: Orb
CONTEXT: match ball
TRAIN: What was the Example Games match ball called?
HELDOUT: Name the match ball used at Example Games.

Leakage correction:
BAD HELDOUT: Identify the match ball named Orb at Example Games.
GOOD HELDOUT: Identify the name of the Example Games match ball.
The bad question is invalid because it contains ANSWER. Descriptive ANSWER phrases are also forbidden verbatim; ask for the category instead of repeating the description.

Before responding, silently run literal substring checks: every SUBJECT, ANSWER, and CONTEXT value must be found unchanged in SOURCE FACT; the exact SUBJECT string must be found in both questions; the ANSWER and every named member of it must be absent from both questions. Then verify that both questions request one identical target and that the output has exactly five nonempty lines."""


def question_repair_prompt(fact: str, previous: str, error: str) -> str:
    if error == "question omits its explicit subject":
        repair_action = (
            "Keep SUBJECT, ANSWER, and CONTEXT exactly as they are in FAILED OUTPUT. "
            "Rewrite only TRAIN and HELDOUT so each contains the exact full SUBJECT "
            "string verbatim."
        )
    elif error == "question reveals its answer":
        repair_action = (
            "Keep SUBJECT, ANSWER, and CONTEXT exactly as they are in FAILED OUTPUT. "
            "Rewrite only TRAIN and HELDOUT so neither contains ANSWER or any named "
            "member of ANSWER."
        )
    elif error == "subject, answer, or context is not copied from the fact":
        repair_action = (
            "Keep any SUBJECT and ANSWER that already occur verbatim in SOURCE FACT, "
            "and set CONTEXT: NONE. If SUBJECT or ANSWER is not verbatim, replace only "
            "that field with an exact source span."
        )
    elif error == "answer duplicates context":
        repair_action = (
            "Keep SUBJECT and ANSWER exactly as they are in FAILED OUTPUT, set "
            "CONTEXT: NONE, and rewrite the questions only if needed."
        )
    elif error == "answer duplicates the subject":
        repair_action = (
            "Keep ANSWER if it is a useful exact target. Replace SUBJECT with the "
            "leading named entity before any possessive suffix or role phrase, set "
            "CONTEXT: NONE, and rewrite both questions."
        )
    else:
        repair_action = (
            "Preserve every field that already satisfies the validation error, and "
            "change only the invalid field or question."
        )
    return f"""Repair a generated training/held-out question pair that failed deterministic validation.

SOURCE FACT:
{fact}

FAILED OUTPUT:
{previous}

VALIDATION ERROR:
{error}

REQUIRED REPAIR ACTION:
{repair_action}

Return a corrected block with exactly five nonempty lines:
SUBJECT: <shortest leading proper-name anchor copied exactly from SOURCE FACT>
ANSWER: <one concise exact contiguous phrase copied from SOURCE FACT>
CONTEXT: <exact contiguous phrase> || <exact contiguous phrase>  (or NONE)
TRAIN: <question>
HELDOUT: <different question>

Non-negotiable repair rules:
- Follow REQUIRED REPAIR ACTION exactly; do not replace fields that it says to keep.
- When a new SUBJECT is required, copy the leading proper name from SOURCE FACT. If SOURCE FACT begins with \"X's\", SUBJECT is exactly X before the apostrophe. Never choose a proper name that first appears later in the sentence.
- Every SUBJECT, ANSWER, and CONTEXT value must occur verbatim in SOURCE FACT. Use CONTEXT: NONE instead of paraphrasing, deleting, joining, reordering, or inflecting source words.
- Both questions must contain the exact SUBJECT string verbatim.
- Neither question may contain ANSWER or any named member of ANSWER. If the failed output leaked ANSWER, rewrite the entire question to ask for its category without naming it.
- Both questions must ask for exactly the same single ANSWER and must not refer to a source or text.
- Output only the corrected five-line block."""


def parse_question_block(text: str) -> tuple[str, str, list[str], list[str]]:
    """Parse the exact five-line question-generation contract."""
    fields = {}
    for line in text.strip().splitlines():
        if ":" not in line:
            raise ValueError(f"invalid question line: {line!r}")
        key, value = line.split(":", 1)
        key, value = key.strip().upper(), value.strip()
        if key in fields:
            raise ValueError(f"duplicate question field: {key}")
        fields[key] = value
    expected = {"SUBJECT", "ANSWER", "CONTEXT", "TRAIN", "HELDOUT"}
    if set(fields) != expected or not all(fields.values()):
        raise ValueError(f"invalid question fields: {sorted(fields)}")
    context = (
        []
        if fields["CONTEXT"].upper() == "NONE"
        else [value.strip() for value in fields["CONTEXT"].split("||")]
    )
    if any(not value for value in context):
        raise ValueError("empty context component")
    return (
        fields["SUBJECT"],
        fields["ANSWER"],
        context,
        [fields["TRAIN"], fields["HELDOUT"]],
    )


def validate_question_block(
    fact: str,
    subject: str,
    answer: str,
    context: list[str],
    questions: list[str],
) -> str | None:
    normalized_fact = fact.casefold()
    if any(
        value.casefold() not in normalized_fact for value in [subject, answer, *context]
    ):
        return "subject, answer, or context is not copied from the fact"
    if (
        answer.casefold() == subject.casefold()
        or answer.casefold() in subject.casefold()
        or subject.casefold() in answer.casefold()
    ):
        return "answer duplicates the subject"
    if answer.casefold() in {value.casefold() for value in context}:
        return "answer duplicates context"
    if questions[0].casefold() == questions[1].casefold():
        return "held-out prompt duplicates a training prompt"
    subject_terms = {
        term
        for term in re.findall(r"[^\W_]+", subject.casefold())
        if term not in {"a", "an", "and", "at", "for", "in", "of", "on", "s", "the"}
    }
    if any(
        not subject_terms.issubset(set(re.findall(r"[^\W_]+", question.casefold())))
        for question in questions
    ):
        return "question omits its explicit subject"
    if any(answer.casefold() in question.casefold() for question in questions):
        return "question reveals its answer"
    if any(_META_REFERENCE.search(question) for question in questions):
        return "question contains a document-relative reference"
    return None


def sanitize_question_context(fact: str, answer: str, context: list[str]) -> list[str]:
    """Drop optional context fields that are inexact or duplicate the answer."""
    normalized_fact = fact.casefold()
    normalized_answer = answer.casefold()
    return [
        value
        for value in context
        if value.casefold() in normalized_fact and value.casefold() != normalized_answer
    ]


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
        prompt = question_prompt(fact, questions_per_fact)
        previous = ""
        for attempt in range(2):
            previous = generate_questions(prompt)
            try:
                item.subject, item.answer, item.context, item.questions = (
                    parse_question_block(previous)
                )
                item.context = sanitize_question_context(
                    fact, item.answer, item.context
                )
                item.error = validate_question_block(
                    fact, item.subject, item.answer, item.context, item.questions
                )
            except ValueError as error:
                item.error = str(error)
            if item.error is None:
                item.train_prompts = item.questions[:-1]
                item.prompt = item.questions[-1]
                break
            if attempt == 0:
                prompt = question_repair_prompt(fact, previous, item.error)
        generated.append(item)
    return generated


def answer_facts(facts: list[GeneratedFact], answer: Callable[[str], str]) -> None:
    for item in facts:
        if item.error is None:
            item.completion = item.answer.strip() or answer(item.prompt).strip()
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
    question_temperature: float = 0.0,
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
                "subject": item.subject,
                "answer": item.answer,
                "context": item.context,
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
