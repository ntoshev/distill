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
    first = rotation.next_batch(2)
    second = rotation.next_batch(2)
    assert {name for name, _ in first} == {"a", "b"}
    assert {name for name, _ in second} == {"a", "b"}
    assert {
        prompt
        for name, prompt in (*first, *second)
        if name == "a"
    } == {"a1", "a2"}


def test_rotation_preserves_queue_when_active_set_is_unchanged(tmp_path):
    path = tmp_path / "probes.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "name": name,
                    "prompt": f"held-{name}",
                    "completion": name.upper(),
                    "train_prompts": [f"train-{name}"],
                }
            )
            for name in ("a", "b", "c")
        )
        + "\n"
    )
    rotation = ActiveRotation(load_learning_items(path), seed=0)
    rotation.update({"a", "b", "c"})
    first = rotation.next()
    remaining = list(rotation.queue)

    rotation.update({"a", "b", "c"})

    assert rotation.queue == remaining
    assert {first, rotation.next(), rotation.next()} == {
        ("a", "train-a"),
        ("b", "train-b"),
        ("c", "train-c"),
    }


def test_rotation_batches_distinct_facts(tmp_path):
    path = tmp_path / "probes.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "name": name,
                    "prompt": f"held-{name}",
                    "completion": name.upper(),
                    "train_prompts": [f"{name}-1", f"{name}-2"],
                }
            )
            for name in ("a", "b", "c", "d")
        )
        + "\n"
    )
    rotation = ActiveRotation(load_learning_items(path), seed=0)
    rotation.update({"a", "b", "c", "d"})

    for _ in range(4):
        batch = rotation.next_batch(3)
        assert len(batch) == 3
        assert len({name for name, _ in batch}) == 3

    with pytest.raises(ValueError, match="batch size"):
        rotation.next_batch(0)
