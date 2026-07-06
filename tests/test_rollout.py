"""Unit tests for parsing verifiers/prime-rl results.jsonl. No model needed."""

import pytest

from distil.rollout import load_rollout

from conftest import FIXTURE_DIR


def test_load_real_fixture():
    r = load_rollout(FIXTURE_DIR / "results.jsonl", 0)
    assert r.prompt[0]["role"] == "system"
    assert r.prompt[1]["role"] == "user"
    assert r.completion[0]["role"] == "assistant"
    assert "boxed" in r.completion[0]["content"]
    assert r.meta["reward"] == 1.0
    assert r.meta["answer"] == "18"
    assert r.messages == r.prompt + r.completion


def test_index_out_of_range():
    with pytest.raises(IndexError):
        load_rollout(FIXTURE_DIR / "results.jsonl", 5)


def test_not_a_rollout(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"prompt": "just a string", "completion": []}\n')
    with pytest.raises(ValueError, match="prompt"):
        load_rollout(bad, 0)
