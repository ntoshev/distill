import json

import pytest

from distil.generation import (
    answer_facts,
    ensure_fits,
    fact_prompt,
    generate_facts,
    parse_question_block,
    parse_lines,
    question_prompt,
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
            "SUBJECT: Alpha\nANSWER: red\nCONTEXT: NONE\n"
            "TRAIN: What color is Alpha?\nHELDOUT: Name Alpha's color.",
            "SUBJECT: Beta\nANSWER: blue\nCONTEXT: NONE\n"
            "TRAIN: What color is Beta?\nHELDOUT: How is Beta colored?",
        ]
    )
    facts = generate_facts("doc", 2, lambda _: next(outputs))
    answer_facts(facts, lambda prompt: "red" if "Alpha" in prompt else "blue")
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
            "SUBJECT: Alpha\nANSWER: red\nCONTEXT: NONE\n"
            "TRAIN: What color is Alpha?\nHELDOUT: Name Alpha's color."
        )

    facts = generate_facts("doc", 2, inventory, question)
    assert len(inventories) == 1
    assert len(questions) == 1
    assert facts[0].train_prompts == ["What color is Alpha?"]


def test_selected_source_answer_is_used_without_teacher_cleanup():
    facts = generate_facts(
        "doc",
        2,
        lambda _: "Alpha is red.",
        lambda _: (
            "SUBJECT: Alpha\nANSWER: red\nCONTEXT: NONE\n"
            "TRAIN: What color is Alpha?\nHELDOUT: Name Alpha's color."
        ),
    )
    answer_facts(facts, lambda _: pytest.fail("teacher should not be called"))
    assert facts[0].completion == "red"


def test_fact_prompt_requires_compact_standalone_event_facts():
    prompt = fact_prompt("DOCUMENT")
    assert "Keep lists intact" in prompt
    assert "funding event's amount, round, year, investors" in prompt
    assert "HARD STANDALONE-SUBJECT FORMAT" in prompt
    assert 'No line may contain any of these context-dependent expressions' in prompt
    assert "verify full source coverage" in prompt


def test_question_prompt_and_parser_use_auditable_aligned_fields():
    prompt = question_prompt("Alpha is red.", 2)
    assert "SUBJECT:" in prompt
    assert "Both questions must ask only for ANSWER" in prompt
    assert "Never refer to a source, fact, passage" in prompt
    parsed = parse_question_block(
        "SUBJECT: Alpha\nANSWER: red\nCONTEXT: NONE\n"
        "TRAIN: What color is Alpha?\nHELDOUT: Name Alpha's color."
    )
    assert parsed == (
        "Alpha",
        "red",
        [],
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
        [],
        ["Is Alpha red?", "What color is Alpha?"],
    )
    assert error == "question reveals its answer"


def test_generation_artifacts_record_validation_errors(tmp_path):
    facts = generate_facts(
        "doc",
        2,
        lambda _: "A is red.",
        lambda _: (
            "SUBJECT: A\nANSWER: red\nCONTEXT: NONE\n"
            "TRAIN: What color is A?\nHELDOUT: What color is A?"
        ),
    )
    answer_facts(facts, lambda _: "answer")
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
