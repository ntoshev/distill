import json

import pytest

from distill.generation import (
    ensure_fits,
    fact_prompt,
    generate_facts,
    parse_question_block,
    question_prompt,
    question_repair_prompt,
    resolve_context,
    validate_question_block,
    write_generation,
)


def test_resolve_context_supports_inline_file_and_stdin(tmp_path):
    assert resolve_context("hello")[0] == "hello"
    path = tmp_path / "source.md"
    path.write_text("document")
    text, source = resolve_context(f"@{path}")
    assert text == "document"
    assert source["path"] == str(path)
    assert source["bytes"] == 8
    assert resolve_context("-", stdin=lambda: "pipe")[0] == "pipe"


def test_resolve_context_rejects_missing_invalid_and_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_context(f"@{tmp_path / 'missing'}")
    bad = tmp_path / "bad"
    bad.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        resolve_context(f"@{bad}")
    with pytest.raises(ValueError, match="empty"):
        resolve_context("  ")


def test_parse_and_compile_use_last_question_as_heldout():
    outputs = iter(
        [
            "1. Alpha is red.\n- Beta is blue.",
            "SUBJECT: Alpha\nCOMPLETION: red\n"
            "TRAIN: What color is Alpha?\nHELDOUT: Name Alpha's color.",
            "SUBJECT: Beta\nCOMPLETION: blue\n"
            "TRAIN: What color is Beta?\nHELDOUT: How is Beta colored?",
        ]
    )
    facts = generate_facts("doc", 2, lambda _: next(outputs))
    assert [fact.fact for fact in facts] == ["Alpha is red.", "Beta is blue."]
    assert facts[0].train_prompts == ["What color is Alpha?"]
    assert facts[0].prompt == "Name Alpha's color."
    assert facts[1].completion == "blue"


def test_fact_and_question_generation_can_use_distinct_policies():
    inventories = []
    questions = []

    def inventory(prompt):
        inventories.append(prompt)
        return "Alpha is red."

    def question(prompt):
        questions.append(prompt)
        return (
            "SUBJECT: Alpha\nCOMPLETION: red\n"
            "TRAIN: What color is Alpha?\nHELDOUT: Name Alpha's color."
        )

    facts = generate_facts("doc", 2, inventory, question)
    assert len(inventories) == 1
    assert len(questions) == 1
    assert facts[0].train_prompts == ["What color is Alpha?"]


def test_selected_source_completion_is_used_without_teacher_cleanup():
    facts = generate_facts(
        "doc",
        2,
        lambda _: "Alpha is red.",
        lambda _: (
            "SUBJECT: Alpha\nCOMPLETION: red\n"
            "TRAIN: What color is Alpha?\nHELDOUT: Name Alpha's color."
        ),
    )
    assert facts[0].completion == "red"


def test_generation_artifacts_use_completion_only_schema(tmp_path):
    facts = generate_facts(
        "doc",
        2,
        lambda _: "Alpha is red.",
        lambda _: (
            "SUBJECT: Alpha\nCOMPLETION: red\n"
            "TRAIN: What color is Alpha?\nHELDOUT: Name Alpha's color."
        ),
    )
    write_generation(
        tmp_path,
        "doc",
        {"sha256": "x"},
        facts,
        generator_model="g",
        questions_per_fact=2,
        seed=0,
    )
    fact_row = json.loads((tmp_path / "facts.jsonl").read_text())
    probe_row = json.loads((tmp_path / "probes.jsonl").read_text())
    assert "answer" not in fact_row
    assert "context" not in fact_row
    assert "fact_id" not in fact_row
    assert "answer" not in probe_row
    assert "context" not in probe_row
    assert "subject" not in probe_row
    assert "fact_id" not in probe_row
    assert probe_row["completion"] == "red"


def test_fact_prompt_requires_bounded_whole_document_study_facts():
    prompt = fact_prompt("DOCUMENT")
    assert "silently scan the entire document" in prompt
    assert "Return only 12 to 15 facts" in prompt
    assert "beginning, middle, and end" in prompt
    assert "Keep a natural list intact" in prompt
    assert (
        "never begin a line unless enough output space remains to finish it" in prompt
    )
    assert "HARD SUBJECT-ANCHOR FORMAT" in prompt
    assert "A generic role is not a subject anchor" in prompt
    assert (
        'A leading "The" is allowed only when the proper name immediately follows'
        in prompt
    )
    assert "never turn a target or forecast into an accomplished result" in prompt


def test_question_prompt_and_parser_use_auditable_aligned_fields():
    prompt = question_prompt("Alpha is red.", 2)
    assert "SUBJECT:" in prompt
    assert "Both questions must ask only for COMPLETION" in prompt
    assert "Never refer to a source, fact, passage" in prompt
    assert "The manufacturer of X is Y" in prompt
    assert "exactly four nonempty lines" in prompt
    assert "shortest exact proper-name anchor" in prompt
    assert "silently run literal substring checks" in prompt
    repair = question_repair_prompt("Alpha is red.", "bad", "question reveals its completion")
    assert "VALIDATION ERROR:\nquestion reveals its completion" in repair
    assert "COMPLETION: <one concise exact contiguous phrase" in repair
    omitted = question_repair_prompt(
        "Alpha is red.", "bad", "question omits its explicit subject"
    )
    assert "Keep SUBJECT and COMPLETION exactly" in omitted
    copied = question_repair_prompt(
        "Alpha is red.",
        "bad",
        "subject or completion is not copied from the fact",
    )
    assert "replace only that field with an exact source span" in copied
    parsed = parse_question_block(
        "SUBJECT: Alpha\nCOMPLETION: red\n"
        "TRAIN: What color is Alpha?\nHELDOUT: Name Alpha's color."
    )
    assert parsed == (
        "Alpha",
        "red",
        ["What color is Alpha?", "Name Alpha's color."],
    )
    assert validate_question_block("Alpha is red.", *parsed) is None


def test_question_generation_rejects_unsupported_counts_and_answer_leaks():
    with pytest.raises(ValueError, match="exactly 2"):
        question_prompt("Alpha is red.", 3)
    error = validate_question_block(
        "Alpha is red.",
        "Alpha",
        "red",
        ["Is Alpha red?", "What color is Alpha?"],
    )
    assert error == "question reveals its completion"


def test_question_validation_allows_natural_subject_reordering():
    parsed = (
        "SpaceX IPO voting control",
        "Elon Musk",
        [
            "Who retained voting control of the SpaceX IPO?",
            "Which person held the SpaceX IPO voting control?",
        ],
    )
    fact = "SpaceX IPO voting control remained with Elon Musk."
    assert validate_question_block(fact, parsed[0], parsed[1], parsed[2]) is None


def test_question_generation_repairs_one_invalid_attempt():
    outputs = iter(
        [
            "SUBJECT: Alpha\nCOMPLETION: red\n"
            "TRAIN: Is Alpha red?\nHELDOUT: What color is Alpha?",
            "SUBJECT: Alpha\nCOMPLETION: red\n"
            "TRAIN: What color is Alpha?\nHELDOUT: Name Alpha's color.",
        ]
    )
    prompts = []

    def questions(prompt):
        prompts.append(prompt)
        return next(outputs)

    facts = generate_facts("doc", 2, lambda _: "Alpha is red.", questions)
    assert facts[0].error is None
    assert len(prompts) == 2
    assert "VALIDATION ERROR:\nquestion reveals its completion" in prompts[1]


def test_generation_artifacts_record_validation_errors(tmp_path):
    facts = generate_facts(
        "doc",
        2,
        lambda _: "A is red.",
        lambda _: (
            "SUBJECT: A\nCOMPLETION: red\n"
            "TRAIN: What color is A?\nHELDOUT: What color is A?"
        ),
    )
    with pytest.raises(ValueError, match="duplicates"):
        write_generation(
            tmp_path,
            "doc",
            {"sha256": "x"},
            facts,
            generator_model="g",
            questions_per_fact=2,
            seed=0,
        )
    row = json.loads((tmp_path / "facts.jsonl").read_text())
    assert row["error"]
    generation = json.loads((tmp_path / "generation.json").read_text())
    assert generation["fact_temperature"] == 0.0
    assert generation["question_temperature"] == 0.0


def test_context_overflow_is_rejected_without_truncation():
    class Args:
        max_position_embeddings = 8

    class Model:
        args = Args()

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return "one two three four five"

        def encode(self, text, add_special_tokens=False):
            return text.split()

    with pytest.raises(ValueError, match=r"5 prompt tokens \+ 4 output tokens > 8"):
        ensure_fits(Model(), Tokenizer(), [], 4, "teacher")
