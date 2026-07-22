"""Run and score Sakana question-generation prompt experiments."""

import argparse
import json
import re
from pathlib import Path

from mlx_lm import load

from distil.sample import sample_completion


PROMPTS = {
    1: """Here is a fact:

{fact}

Write 2 short questions that this fact answers.
Rules:
- Every question must be fully self-contained: name its subject explicitly (never "the company", "it", "they", or another reference that needs context to resolve).
- The fact must fully answer each question.
- Prefer questions with short, specific answers: a name, number, date, or short phrase.
- Vary the wording between questions.
- One question per line, no numbering.""",
    2: """Create a training/held-out question pair for this fact:

{fact}

First identify the complete answer proposition expressed by the fact. Then write two differently worded questions that both require that same complete proposition as their answer.

Return exactly one JSON object and nothing else:
{{"answer_requirements":["exact phrase copied from the fact"],"questions":["training question","held-out question"]}}

Rules:
- `answer_requirements` must list every answer detail needed to preserve the fact's complete proposition, excluding the subject itself. Copy each requirement exactly from the fact.
- Both questions must request every listed requirement. They must not target different attributes or permit a partial answer.
- Each question must be fully self-contained and name the fact's subject explicitly.
- Do not reveal any answer requirement in either question.
- The questions must be meaningfully different paraphrases, not duplicates.
- Keep the questions concise, but explicitly request all needed fields when the fact contains an amount, date, list, mechanism, reason, qualifier, or interpretation.
- Do not add information absent from the fact.""",
    3: """Create a training/held-out question pair for this fact:

{fact}

Partition the fact into:
- `givens`: subject names and contextual details that may be stated in the questions;
- `answer_requirements`: the answer values that both questions must ask the model to supply.

Return exactly one JSON object and nothing else:
{{"givens":["exact phrase copied from the fact"],"answer_requirements":["exact phrase copied from the fact"],"questions":["training question","held-out question"]}}

Rules:
- Copy every given and answer requirement exactly from the fact.
- Together, the givens and answer requirements must preserve the fact's complete proposition; do not silently discard an amount, date, list member, qualifier, mechanism, reason, or interpretation.
- Both questions must state or clearly identify the same givens and request every answer requirement.
- Neither question may state or reveal an answer requirement.
- Both questions must therefore have the same complete answer, not merely concern the same broad fact.
- Each question must be fully self-contained and explicitly name the fact's subject.
- The questions must be concise, distinct paraphrases.
- Do not add information absent from the fact.""",
    4: """Create an exhaustive set of aligned training/held-out question pairs for this fact:

{fact}

Decompose the fact into the smallest useful answer targets needed to test every informative detail. Create one group per answer target.

Return exactly one JSON object and nothing else:
{{"groups":[{{"answer_requirements":["exact answer phrase copied from the fact"],"questions":["training question","held-out question"]}}]}}

Rules:
- The groups' answer requirements must collectively cover every informative detail in the fact except the subject itself and grammatical glue.
- Keep details together when they form one natural answer, such as a complete list of founders or investors.
- Split independently queryable details into separate groups, such as a company's type and its location, or a funding amount and its investors.
- Every answer requirement must be copied exactly from the fact.
- Each group must contain exactly two concise, distinct questions.
- Both questions in a group must have exactly the same complete answer requirements.
- Each question must be fully self-contained and explicitly name the fact's subject.
- Neither question may state or reveal its group's answer requirements.
- Do not create a group whose answer is merely the subject.
- Do not add information absent from the fact.""",
    5: """Create an exhaustive set of aligned training/held-out question pairs for this fact:

{fact}

Decompose the fact into natural answer targets. Every informative detail in the fact, except the subject and grammatical glue, must be tested as an answer in at least one group. A detail appearing only inside a question does not count as covered.

For each answer target, output exactly this three-line block:
ANSWER: <one exact phrase copied from the fact>
TRAIN: <self-contained training question>
HELDOUT: <different self-contained held-out question>

Separate groups with a line containing only ---.

Rules:
- Keep a natural list, such as all founders or all investors, as one answer.
- Split independently queryable details, such as company type and location, into separate groups.
- Every ANSWER must be one exact, contiguous phrase copied from the fact.
- TRAIN and HELDOUT within a group must have exactly the same complete answer.
- Both questions must explicitly name the fact's subject.
- Neither question may state or reveal its ANSWER.
- Do not create a group whose answer is merely the subject.
- Do not omit a fact detail, duplicate an answer target, or add unsupported information.
- Output only the blocks, with no introduction, explanation, numbering, or code fence.""",
    6: """Create an exhaustive, non-redundant set of aligned training/held-out question pairs for this fact:

{fact}

Decompose the fact into natural answer targets. Every informative detail except the subject and grammatical glue must be tested as an answer in exactly one group.

For each answer target, output exactly this three-line block:
ANSWER: <one exact phrase copied from the fact>
TRAIN: <self-contained training question>
HELDOUT: <different self-contained held-out question>

Separate groups with a line containing only ---.

Rules:
- Never create an ANSWER for the fact's subject or for an entity used only to identify the event.
- Keep every natural list intact as one answer. For example, generate one founders group for the complete founder list, never one group per founder; do the same for investors.
- No ANSWER may overlap another ANSWER: do not generate both a whole phrase and one of its members or subphrases.
- Split independently queryable non-list details, such as company type and location, into separate groups.
- Every ANSWER must be one exact, contiguous phrase copied from the fact.
- TRAIN and HELDOUT within a group must have exactly the same complete answer.
- Both questions must explicitly name the fact's subject and be understandable without the source fact.
- Neither question may state or reveal its ANSWER.
- Do not omit a fact detail, duplicate a target, or add unsupported information.
- Output only the blocks, with no introduction, explanation, numbering, or code fence.""",
    7: """Create an exhaustive set of aligned training/held-out question pairs for this fact:

{fact}

Decompose the fact into the smallest useful answer targets needed to test every informative detail. Create one group per answer target.

For each group, output exactly:
ANSWER: <one exact answer phrase copied from the fact>
TRAIN: <training question>
HELDOUT: <held-out question>

Separate groups with a line containing only ---.

Rules:
- The groups' answers must collectively cover every informative detail in the fact except the subject itself and grammatical glue.
- Keep details together when they form one natural answer, such as a complete list of founders or investors.
- Split independently queryable details into separate groups, such as a company's type and its location, or a funding amount and its investors.
- Every answer must be copied exactly from the fact.
- Each group must contain exactly two concise, distinct questions.
- Both questions in a group must have exactly the same complete answer.
- Each question must be fully self-contained and explicitly name the fact's subject.
- Neither question may state or reveal its group's answer.
- Do not create a group whose answer is merely the subject.
- Do not add information absent from the fact.
- Output only the blocks, with no introduction, explanation, numbering, or code fence.""",
    8: """Extract a minimal, non-redundant set of answer targets from this fact:

{fact}

Output one line per target:
ANSWER: <one exact, contiguous phrase copied from the fact>

Rules:
- Cover every informative detail except the subject and grammatical glue.
- Never output the fact's subject as an answer.
- Keep a natural list intact as one target; never output its individual members too.
- Split independently queryable details such as type, location, amount, date, and investors.
- Targets must not duplicate or overlap one another.
- Output only ANSWER lines.""",
    9: """Create one aligned training/held-out question pair for the source fact.

SOURCE FACT:
{fact}

Select one complete answer span containing every informative detail after the subject. Do not split the fact into separate targets. Then write two questions that both explicitly request every component of that same answer.

Output exactly:
ANSWER: <one exact, contiguous span copied from the source fact>
TRAIN: <question>
HELDOUT: <different question>

Rules:
- ANSWER must exclude the subject but preserve all informative predicate details, including lists, amounts, dates, round types, mechanisms, purposes, and qualifications.
- Both questions must require the whole ANSWER. If it contains multiple fields, explicitly ask for every field in both questions.
- Keep complete founder and investor lists together.
- Each question must stand alone and explicitly name the subject.
- Do not state or reveal any part of ANSWER in a question.
- Use meaningfully different wording while keeping identical answer scope.
- Output only the three lines.

Example 1:
SOURCE FACT:
Acme AI is a Canadian robotics company based in Montreal.
ANSWER: a Canadian robotics company based in Montreal
TRAIN: What kind of company is Acme AI, including its nationality and field, and where is it based?
HELDOUT: How would you classify Acme AI by country and industry, and in which city is it located?

Example 2:
SOURCE FACT:
Acme AI raised $12M in its seed round from North Fund and South Ventures.
ANSWER: $12M in its seed round from North Fund and South Ventures
TRAIN: How much did Acme AI raise, in which funding round, and from which investors?
HELDOUT: State the amount, financing stage, and complete investor list for Acme AI's funding round.""",
    10: """Create one aligned training/held-out question pair for the source fact.

SOURCE FACT:
{fact}

First identify every informative answer component other than the subject. Then write two questions that both request all components.

Output exactly:
ANSWER: <exact component> || <exact component>
TRAIN: <question>
HELDOUT: <different question>

Rules:
- Copy each answer component as one exact, contiguous phrase from the source fact.
- Together the components must cover every informative detail other than the subject and grammatical glue.
- Keep a natural list intact as one component: one complete founders list or one complete investors list, never separate members.
- Include all applicable names, types, locations, amounts, dates, round types, actions, mechanisms, purposes, and qualifications.
- Both questions must explicitly request every component and therefore have identical complete answers.
- Each question must stand alone and explicitly name the subject.
- Never include an answer component or any of its specific names, numbers, dates, or values in either question. Ask for them instead.
- Generic category words such as company, investors, date, method, purpose, or funding round are allowed.
- Use meaningfully different wording and do not request unsupported details.
- Output only the three lines.

Example 1:
SOURCE FACT:
Acme AI is a Canadian robotics company based in Montreal.
ANSWER: Canadian || robotics company || Montreal
TRAIN: What is Acme AI's nationality and industry, and where is it based?
HELDOUT: State Acme AI's country, business field, and headquarters city.

Example 2:
SOURCE FACT:
Acme AI raised $12M in its seed round from North Fund and South Ventures in 2023.
ANSWER: $12M || seed round || North Fund and South Ventures || 2023
TRAIN: How much did Acme AI raise, in which funding round, from which investors, and in what year?
HELDOUT: State the amount, financing stage, complete investor list, and date of Acme AI's raise.

Bad question for Example 2:
Which investors, including North Fund, backed Acme AI?
This is invalid because it reveals part of the answer.""",
    11: """Create one aligned training/held-out question pair for the source fact.

SUBJECT CONTEXT (state this; never ask for it):
{subject}

SOURCE FACT:
{fact}

Identify every informative answer component other than the subject context. Then write two questions that both request all components.

Output exactly:
ANSWER: <exact component> || <exact component>
TRAIN: <question>
HELDOUT: <different question>

Rules:
- Copy each component as one exact, contiguous phrase from the source fact.
- Together the components must cover every informative predicate detail: introduced names, classifications, locations, amounts, dates, round types, actions, works, affiliations, mechanisms, purposes, and qualifications.
- Never put the subject context in ANSWER and never ask who or what the subject is.
- Keep each natural list intact as one component. A complete founders or investors list must never be split into members.
- Both questions must explicitly request every component and have identical complete answers.
- Each question must stand alone and explicitly state the subject context.
- Never include an answer component or any of its specific names, numbers, dates, or values in either question. Ask for them by category instead.
- Generic category words such as model, paper, employer, investors, date, method, purpose, or funding round are allowed.
- Use meaningfully different wording and do not request unsupported details.
- Output only the three lines.

Example 1:
SUBJECT CONTEXT: Acme AI
SOURCE FACT: Acme AI was founded by Ana One, Bo Two and Cy Three.
ANSWER: Ana One, Bo Two and Cy Three
TRAIN: Who founded Acme AI?
HELDOUT: List the complete founding team of Acme AI.

Example 2:
SUBJECT CONTEXT: Acme AI
SOURCE FACT: Acme AI is developing a model called Lab Bot, which automates experiment design.
ANSWER: Lab Bot || automates experiment design
TRAIN: What is Acme AI's developing model called, and what does it do?
HELDOUT: Give the name and function of the model Acme AI is developing.

Example 3:
SUBJECT CONTEXT: Ada One
SOURCE FACT: Ada One co-authored "Great Paper" while working for Example Corp in 2022.
ANSWER: co-authored || "Great Paper" || Example Corp || 2022
TRAIN: What did Ada One do, which work was involved, where was Ada One employed, and when did it happen?
HELDOUT: State Ada One's action, the work's title, the employer, and the year.""",
    12: """Create one aligned training/held-out question pair for the source fact.

SUBJECT CONTEXT:
{subject}

SOURCE FACT:
{fact}

Choose one useful answer target. Put any other source-fact details needed to identify the event in shared context. Then write two questions that ask only for the same answer target.

Output exactly:
ANSWER: <one exact, contiguous phrase copied from the source fact>
CONTEXT: <exact phrase> || <exact phrase>  (or NONE)
TRAIN: <question>
HELDOUT: <different question>

Rules:
- ANSWER must be a concise, meaningful target: a name, complete natural list, amount, date, place, type, work, mechanism, purpose, or short description.
- Never choose the subject context as ANSWER.
- Keep founders or investors together as one complete list; never target one member of a natural list.
- CONTEXT may contain exact source phrases that both questions can state or rely on. ANSWER and CONTEXT must not overlap.
- Every informative source detail must be represented by either ANSWER, CONTEXT, or the subject context.
- Both questions must ask only for ANSWER and have exactly the same answer scope.
- Both questions must stand alone, explicitly name the subject, and express the same relevant context.
- Never reveal ANSWER or part of ANSWER in either question. CONTEXT is allowed in questions.
- Use meaningfully different wording without requesting extra information.
- Output only the four lines.

Example 1:
SUBJECT CONTEXT: Acme AI
SOURCE FACT: Acme AI was founded by Ana One, Bo Two and Cy Three in 2021.
ANSWER: Ana One, Bo Two and Cy Three
CONTEXT: 2021
TRAIN: Who founded Acme AI in 2021?
HELDOUT: List Acme AI's complete founding team from 2021.

Example 2:
SUBJECT CONTEXT: Ada One
SOURCE FACT: Ada One co-authored "Great Paper" while working for Example Corp in 2022.
ANSWER: "Great Paper"
CONTEXT: co-authored || Example Corp || 2022
TRAIN: Which paper did Ada One co-author while employed by Example Corp in 2022?
HELDOUT: Name the work Ada One helped write at Example Corp in 2022.

Example 3:
SUBJECT CONTEXT: Acme AI
SOURCE FACT: Acme AI raised $12M in its seed round from North Fund and South Ventures.
ANSWER: $12M
CONTEXT: seed round || North Fund and South Ventures
TRAIN: How much did Acme AI raise in its seed round from North Fund and South Ventures?
HELDOUT: What was the size of Acme AI's seed financing backed by North Fund and South Ventures?""",
}

PROMPTS[13] = PROMPTS[12].replace(
    "- Both questions must stand alone, explicitly name the subject, and express the same relevant context.",
    """- Both questions must stand alone, explicitly name the subject, and express every CONTEXT detail needed to identify the target.
- Never refer to a source, fact, passage, text, document, prompt, or information above. The reader will see only the question.""",
)

PROMPTS[14] = """Create one aligned training/held-out question pair for the source fact.

SOURCE FACT:
{fact}

Identify the explicit subject, choose one useful answer target, and put other details needed to identify the event in shared context. Then write two questions that ask only for the same answer target.

Output exactly:
SUBJECT: <explicit named subject copied from the source fact>
ANSWER: <one exact, contiguous phrase copied from the source fact>
CONTEXT: <exact phrase> || <exact phrase>  (or NONE)
TRAIN: <question>
HELDOUT: <different question>

Rules:
- SUBJECT is the named entity being described. State it in both questions; never ask for it.
- ANSWER must be one concise, meaningful target: a name, complete natural list, amount, date, place, type, work, mechanism, purpose, or short description.
- SUBJECT and ANSWER must not overlap.
- Keep founders or investors together as one complete list; never target one member of a natural list.
- CONTEXT may contain exact source phrases that both questions can state or rely on. ANSWER and CONTEXT must not overlap.
- Every informative source detail must be represented by SUBJECT, ANSWER, or CONTEXT.
- Both questions must ask only for ANSWER and have exactly the same answer scope.
- Both questions must stand alone, explicitly name SUBJECT, and express every CONTEXT detail needed to identify the target.
- Never reveal ANSWER or part of ANSWER in either question. CONTEXT is allowed in questions.
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
HELDOUT: What was the size of Acme AI's seed financing backed by North Fund and South Ventures?"""


PAIR_PROMPT = """Write two questions whose complete answer is the given target.

SOURCE FACT:
{fact}

ANSWER TARGET:
{answer}

Output exactly:
TRAIN: <question>
HELDOUT: <different question>

Rules:
- Both questions must ask for exactly the answer target, with the same scope.
- The source fact must fully answer both questions.
- Explicitly name the subject; each question must stand alone.
- Do not state, quote, or reveal the answer target.
- Use genuinely different wording, not a cosmetic rewrite.
- Do not ask for any additional answer.
- Output only the two lines."""


def _normalize(text: str) -> str:
    return re.sub(r"[^\w$]+", " ", text).strip().casefold()


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    return json.loads(text)


def _parse_blocks(text: str, *, split_answers: bool = False) -> dict:
    groups = []
    for raw_block in re.split(r"^---\s*$", text.strip(), flags=re.MULTILINE):
        lines = [line.strip() for line in raw_block.splitlines() if line.strip()]
        if not lines:
            continue
        fields = {}
        for line in lines:
            if ":" not in line:
                raise ValueError(f"invalid block line: {line!r}")
            key, value = line.split(":", 1)
            fields[key.strip().upper()] = value.strip()
        if set(fields) != {"ANSWER", "TRAIN", "HELDOUT"}:
            raise ValueError(f"invalid block fields: {sorted(fields)}")
        groups.append(
            {
                "answer_requirements": (
                    [value.strip() for value in fields["ANSWER"].split("||")]
                    if split_answers
                    else [fields["ANSWER"]]
                ),
                "questions": [fields["TRAIN"], fields["HELDOUT"]],
            }
        )
    return {"groups": groups}


def _parse_targets(text: str) -> list[str]:
    targets = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"\s*ANSWER:\s*(.+?)\s*", line, re.IGNORECASE)
        if not match:
            raise ValueError(f"invalid answer line: {line!r}")
        targets.append(match.group(1))
    if not targets:
        raise ValueError("no answer targets")
    return targets


def _parse_pair(text: str) -> list[str]:
    fields = {}
    for line in text.strip().splitlines():
        if ":" not in line:
            raise ValueError(f"invalid pair line: {line!r}")
        key, value = line.split(":", 1)
        fields[key.strip().upper()] = value.strip()
    if set(fields) != {"TRAIN", "HELDOUT"}:
        raise ValueError(f"invalid pair fields: {sorted(fields)}")
    return [fields["TRAIN"], fields["HELDOUT"]]


def _parse_context_block(text: str, *, require_subject: bool = False) -> dict:
    fields = {}
    for line in text.strip().splitlines():
        if ":" not in line:
            raise ValueError(f"invalid context line: {line!r}")
        key, value = line.split(":", 1)
        fields[key.strip().upper()] = value.strip()
    expected = {"ANSWER", "CONTEXT", "TRAIN", "HELDOUT"}
    if require_subject:
        expected.add("SUBJECT")
    if set(fields) != expected:
        raise ValueError(f"invalid context fields: {sorted(fields)}")
    context = [] if fields["CONTEXT"].upper() == "NONE" else [
        value.strip() for value in fields["CONTEXT"].split("||")
    ]
    return {
        "coverage_mode": "answer_or_context",
        "groups": [
            {
                "answer_requirements": [fields["ANSWER"]],
                "givens": context,
                "subject_context": fields.get("SUBJECT"),
                "questions": [fields["TRAIN"], fields["HELDOUT"]],
            }
        ],
    }


def generate_two_stage(model, tokenizer, fact: str, args) -> tuple[dict, dict]:
    target_sample = sample_completion(
        model,
        tokenizer,
        [{"role": "user", "content": PROMPTS[8].format(fact=fact)}],
        max_tokens=192,
        temperature=0.0,
        seed=args.seed,
        enable_thinking=False,
    )
    targets = _parse_targets(target_sample.content)
    groups = []
    pair_raw = []
    for index, answer in enumerate(targets):
        pair_sample = sample_completion(
            model,
            tokenizer,
            [
                {
                    "role": "user",
                    "content": PAIR_PROMPT.format(fact=fact, answer=answer),
                }
            ],
            max_tokens=128,
            temperature=args.temperature,
            seed=args.seed + index,
            enable_thinking=False,
        )
        pair_raw.append(pair_sample.content)
        groups.append(
            {
                "answer_requirements": [answer],
                "questions": _parse_pair(pair_sample.content),
            }
        )
    return {"groups": groups}, {"targets": target_sample.content, "pairs": pair_raw}


def _subject_terms(fact: str) -> tuple[str, ...]:
    if fact.startswith("Llion Jones"):
        return ("Llion Jones",)
    if fact.startswith("The Nikkei"):
        return ("Sakana AI", "Nikkei")
    return ("Sakana AI",)


def _subject_label(fact: str) -> str:
    if fact.startswith("Sakana AI Co, Ltd."):
        return "Sakana AI Co, Ltd."
    if fact.startswith("Llion Jones"):
        return "Llion Jones"
    if fact.startswith("The Nikkei"):
        return "The Nikkei and Sakana AI"
    return "Sakana AI"


def mechanical_score(fact: str, row: dict) -> dict:
    if "groups" in row:
        groups = row.get("groups")
        if not isinstance(groups, list) or not groups:
            return {"passed": False, "schema": False}
        group_scores = [_score_group(fact, group) for group in groups]
        targets = [
            tuple(map(_normalize, group.get("answer_requirements", [])))
            for group in groups
            if isinstance(group, dict)
        ]
        checks = {
            "schema": all(score.get("schema", False) for score in group_scores),
            "groups_pass": all(score.get("passed", False) for score in group_scores),
            "distinct_targets": len(targets) == len(set(targets)),
            "non_overlapping_targets": not any(
                left != right
                and (" ".join(left) in " ".join(right) or " ".join(right) in " ".join(left))
                for index, left in enumerate(targets)
                for right in targets[index + 1 :]
            ),
        }
        return {"passed": all(checks.values()), **checks, "group_scores": group_scores}

    requirements = row.get("answer_requirements")
    givens = row.get("givens", [])
    questions = row.get("questions")
    schema = (
        isinstance(givens, list)
        and all(isinstance(value, str) and value.strip() for value in givens)
        and
        isinstance(requirements, list)
        and requirements
        and all(isinstance(value, str) and value.strip() for value in requirements)
        and isinstance(questions, list)
        and len(questions) == 2
        and all(isinstance(value, str) and value.strip() for value in questions)
    )
    if not schema:
        return {"passed": False, "schema": False}

    normalized_fact = _normalize(fact)
    grounded = all(
        _normalize(value) in normalized_fact for value in [*givens, *requirements]
    )
    disjoint = not (set(map(_normalize, givens)) & set(map(_normalize, requirements)))
    distinct = _normalize(questions[0]) != _normalize(questions[1])
    subjects = _subject_terms(fact)
    self_contained = all(
        all(term.casefold() in question.casefold() for term in subjects)
        for question in questions
    )
    no_leak = all(
        all(
            len(_normalize(requirement)) <= 3
            or _normalize(requirement) not in _normalize(question)
            for requirement in requirements
        )
        for question in questions
    )
    checks = {
        "schema": True,
        "grounded_requirements": grounded,
        "givens_answers_disjoint": disjoint,
        "two_distinct_questions": distinct,
        "self_contained": self_contained,
        "no_answer_leak": no_leak,
    }
    return {"passed": all(checks.values()), **checks}


def _score_group(fact: str, group: dict) -> dict:
    if not isinstance(group, dict):
        return {"passed": False, "schema": False}
    requirements = group.get("answer_requirements")
    givens = group.get("givens", [])
    questions = group.get("questions")
    schema = (
        isinstance(givens, list)
        and all(isinstance(value, str) and value.strip() for value in givens)
        and
        isinstance(requirements, list)
        and requirements
        and all(isinstance(value, str) and value.strip() for value in requirements)
        and isinstance(questions, list)
        and len(questions) == 2
        and all(isinstance(value, str) and value.strip() for value in questions)
    )
    if not schema:
        return {"passed": False, "schema": False}
    normalized_fact = _normalize(fact)
    subject_context = group.get("subject_context")
    subjects = (subject_context,) if subject_context else _subject_terms(fact)
    subject_answers = {"sakana ai", "sakana ai co ltd", "llion jones", "the nikkei", "nikkei"}
    if subject_context:
        subject_answers.add(_normalize(subject_context))
    checks = {
        "schema": True,
        "grounded_requirements": all(
            _normalize(value) in normalized_fact for value in [*givens, *requirements]
        ),
        "grounded_subject": not subject_context
        or _normalize(subject_context) in normalized_fact,
        "givens_answers_disjoint": not (
            set(map(_normalize, givens)) & set(map(_normalize, requirements))
        ),
        "two_distinct_questions": _normalize(questions[0]) != _normalize(questions[1]),
        "self_contained": all(
            all(term.casefold() in question.casefold() for term in subjects)
            for question in questions
        ),
        "no_meta_reference": not any(
            re.search(
                r"\b(source|source fact|given fact|passage|text|document|prompt|above)\b",
                question,
                re.IGNORECASE,
            )
            for question in questions
        ),
        "no_answer_leak": all(
            all(
                len(_normalize(requirement)) <= 3
                or _normalize(requirement) not in _normalize(question)
                for requirement in requirements
            )
            for question in questions
        ),
        "not_subject_answer": not any(
            _normalize(requirement) in subject_answers for requirement in requirements
        ),
    }
    return {"passed": all(checks.values()), **checks}


def judge_prompt(fact: str, row: dict) -> str:
    if row.get("coverage_mode") == "answer_or_context":
        return f"""Judge this generated question pair against its source fact.

SOURCE FACT:
{fact}

GENERATED:
{json.dumps(row.get('groups'), ensure_ascii=False)}

Return exactly one JSON object:
{{"fact_details_preserved":true,"same_target_in_both":true,"fact_fully_answers_both":true,"both_questions_self_contained":true,"answer_not_leaked":true,"questions_natural":true}}

Use strict standards:
- `fact_details_preserved` is false unless every informative source detail appears as the answer, shared context, or subject context in both questions.
- `same_target_in_both` is false if the questions request different attributes or scopes.
- `answer_not_leaked` is false if either question states all or part of the answer target.
- Context values may appear in questions and do not count as answer leakage."""
    if "groups" in row:
        return f"""Judge these generated question groups against their source fact.

SOURCE FACT:
{fact}

GROUPS:
{json.dumps(row.get('groups'), ensure_ascii=False)}

Return exactly one JSON object:
{{"full_fact_coverage":true,"natural_target_decomposition":true,"same_target_within_every_group":true,"fact_fully_answers_every_question":true,"all_questions_self_contained":true,"no_answer_leak":true}}

Use strict standards:
- `full_fact_coverage` is false if any informative detail other than the subject is not tested as an answer requirement in at least one group. Merely stating a detail inside a question does not count.
- `natural_target_decomposition` is false if a group combines unrelated answer types or splits a natural list such as founders or investors.
- `same_target_within_every_group` is false if two questions in any group ask for different attributes or scopes.
- `no_answer_leak` is false if a question states its own group's answer requirement."""
    return f"""Judge this generated question pair against its source fact.

SOURCE FACT:
{fact}

ANSWER REQUIREMENTS:
{json.dumps(row.get('answer_requirements'), ensure_ascii=False)}

GIVENS:
{json.dumps(row.get('givens', []), ensure_ascii=False)}

QUESTIONS:
{json.dumps(row.get('questions'), ensure_ascii=False)}

Return exactly one JSON object:
{{"same_complete_target":true,"fact_fully_answers_both":true,"both_require_all_requirements":true,"self_contained":true,"no_answer_leak":true}}

Use strict standards:
- `same_complete_target` is false if the questions ask for different attributes, scopes, or answer types.
- `both_require_all_requirements` is false if either question can be answered correctly while omitting any listed requirement.
- `self_contained` is false if a question needs the source fact to resolve its subject.
- `no_answer_leak` is false if a question states an answer requirement instead of asking for it. Values listed as givens are allowed in questions."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, required=True, choices=sorted(PROMPTS))
    parser.add_argument("--model", default="mlx-community/Qwen3.5-9B-6bit")
    parser.add_argument(
        "--facts",
        default="outputs/sakana-wiki-fact-prompt-v2-20260722/generated/facts.jsonl",
    )
    parser.add_argument("--output", default="outputs/question-prompt-iterations")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    facts = [json.loads(line)["fact"] for line in Path(args.facts).read_text().splitlines()]
    model, tokenizer = load(args.model)
    results = []
    for fact in facts:
        raw = None
        try:
            if args.attempt == 8:
                row, raw = generate_two_stage(model, tokenizer, fact, args)
            else:
                prompt = PROMPTS[args.attempt].format(
                    fact=fact, subject=_subject_label(fact)
                )
                sample = sample_completion(
                    model,
                    tokenizer,
                    [{"role": "user", "content": prompt}],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    seed=args.seed,
                    enable_thinking=False,
                )
                raw = sample.content
                row = (
                    _parse_context_block(
                        sample.content, require_subject=args.attempt >= 14
                    )
                    if args.attempt >= 12
                    else _parse_blocks(sample.content, split_answers=args.attempt >= 10)
                    if args.attempt >= 5
                    else _parse_json(sample.content)
                )
            mechanical = mechanical_score(fact, row)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            row = {"parse_error": str(error), "raw": raw}
            mechanical = {"passed": False, "schema": False}

        judge = {}
        if mechanical.get("schema"):
            judged = sample_completion(
                model,
                tokenizer,
                [{"role": "user", "content": judge_prompt(fact, row)}],
                max_tokens=96,
                temperature=0.0,
                seed=0,
                enable_thinking=False,
            )
            try:
                judge = _parse_json(judged.content)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                judge = {"parse_error": str(error), "raw": judged.content}
        judge_passed = bool(judge) and all(value is True for value in judge.values())
        results.append(
            {
                "fact": fact,
                "raw": raw,
                "generated": row,
                "mechanical": mechanical,
                "judge": judge,
                "passed": mechanical.get("passed", False) and judge_passed,
            }
        )

    summary = {
        "attempt": args.attempt,
        "temperature": args.temperature,
        "total": len(results),
        "passed": sum(result["passed"] for result in results),
        "all_passed": all(result["passed"] for result in results),
    }
    root = Path(args.output) / f"attempt-{args.attempt:02d}"
    root.mkdir(parents=True, exist_ok=True)
    prompt_text = PROMPTS[args.attempt]
    if args.attempt == 8:
        prompt_text += "\n\n--- PAIR PROMPT ---\n\n" + PAIR_PROMPT
    (root / "prompt.txt").write_text(prompt_text)
    (root / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    )
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    for index, result in enumerate(results, 1):
        print(f"\n{index:02d}. {result['fact']}")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
