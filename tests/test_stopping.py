import json

import pytest

from distil.stopping import (
    ActiveRotation,
    answer_matches,
    load_learning_items,
    mastered,
    normalize_answer,
)


def _write(path, row):
    path.write_text(json.dumps(row) + "\n")


def test_learning_items_require_training_prompts_and_a_real_holdout(tmp_path):
    path = tmp_path / "probes.jsonl"
    _write(path, {"name": "x", "prompt": "held", "completion": "answer"})
    with pytest.raises(ValueError, match="train_prompts"):
        load_learning_items(path)
    _write(
        path,
        {"name": "x", "prompt": "same", "completion": "answer", "train_prompts": ["same"]},
    )
    with pytest.raises(ValueError, match="duplicates"):
        load_learning_items(path)


def test_answer_matching_normalizes_but_does_not_require_exact_output():
    assert normalize_answer("  VIOLET-731. \n") == "violet-731"
    assert answer_matches("The answer is VIOLET-731!", "violet-731")
    assert not answer_matches("VIOLET-732", "VIOLET-731")


def test_mastery_and_rotation_retire_and_restore_items(tmp_path):
    path = tmp_path / "probes.jsonl"
    path.write_text(
        json.dumps({"name": "a", "prompt": "ha", "completion": "A", "train_prompts": ["a1", "a2"]})
        + "\n"
        + json.dumps({"name": "b", "prompt": "hb", "completion": "B", "train_prompts": ["b1"]})
        + "\n"
    )
    items = load_learning_items(path)
    assert mastered({"a": 0.5, "b": 0.49}, 0.5) == {"a"}
    rotation = ActiveRotation(items, seed=0)
    rotation.update({"b"})
    assert rotation.next() == ("b", "b1")
    rotation.update({"a", "b"})
    seen = {rotation.next(), rotation.next(), rotation.next()}
    assert seen == {("a", "a1"), ("a", "a2"), ("b", "b1")}
