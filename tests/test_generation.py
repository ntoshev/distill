import json

import pytest

from distil.generation import (
    answer_facts,
    ensure_fits,
    generate_facts,
    parse_lines,
    resolve_context,
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
            "* What color is Alpha?\n2) Name Alpha's color.",
            "What color is Beta?\nHow is Beta colored?",
        ]
    )
    facts = generate_facts("doc", 2, lambda _: next(outputs))
    answer_facts(facts, lambda prompt: "red" if "Alpha" in prompt else "blue")
    assert [fact.fact for fact in facts] == ["Alpha is red.", "Beta is blue."]
    assert facts[0].train_prompts == ["What color is Alpha?"]
    assert facts[0].prompt == "Name Alpha's color."
    assert facts[1].completion == "blue"


def test_generation_artifacts_record_validation_errors(tmp_path):
    facts = generate_facts("doc", 2, lambda _: "A fact." if _.startswith("Here is a reference") else "Same?\nSame?")
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
    assert (tmp_path / "generation.json").is_file()


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
