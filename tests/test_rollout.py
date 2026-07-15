"""Unit tests for parsing verifiers/prime-rl results.jsonl. No model needed."""

import pytest

from distil.rollout import load_rollout, resolve_rollout_path, save_rollout

from conftest import FIXTURE_DIR

EXAMPLE_ROLLOUT = FIXTURE_DIR.parent.parent / "examples" / "verifiers" / "gsm8k-results.jsonl"
GENERATED_9B_WORDLE_ROLLOUT = (
    FIXTURE_DIR.parent.parent
    / "examples"
    / "verifiers"
    / "wordle-qwen3.5-9b-6bit-results.jsonl"
)


def test_load_real_fixture():
    r = load_rollout(FIXTURE_DIR / "results.jsonl", 0)
    assert r.prompt[0]["role"] == "system"
    assert r.prompt[1]["role"] == "user"
    assert r.completion[0]["role"] == "assistant"
    assert "boxed" in r.completion[0]["content"]
    assert r.meta["reward"] == 1.0
    assert r.meta["answer"] == "18"
    assert r.messages == r.prompt + r.completion


def test_load_documented_example_rollout():
    r = load_rollout(EXAMPLE_ROLLOUT, 1)
    assert r.prompt[0]["role"] == "system"
    assert r.prompt[1]["role"] == "user"
    assert r.completion[0]["role"] == "assistant"
    assert r.meta["reward"] == 1.0
    assert r.meta["answer"] == "3"


def test_load_generated_9b_wordle_rollout():
    r = load_rollout(GENERATED_9B_WORDLE_ROLLOUT, 0)
    assert r.prompt[0]["role"] == "system"
    assert [m["role"] for m in r.completion] == [
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert r.meta["reward"] == 0.41000000000000003
    assert r.meta["answer"] == "story"


def test_index_out_of_range():
    with pytest.raises(IndexError):
        load_rollout(FIXTURE_DIR / "results.jsonl", 5)


def test_not_a_rollout(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"prompt": "just a string", "completion": []}\n')
    with pytest.raises(ValueError, match="prompt"):
        load_rollout(bad, 0)


def test_save_rollout_round_trips(tmp_path):
    path = tmp_path / "samples.jsonl"
    prompt = [{"role": "user", "content": "q"}]
    extra = {"task": "distil-sample", "model": "m", "temperature": 0.7, "is_truncated": False}
    assert save_rollout(path, prompt, [{"role": "assistant", "content": "a"}], extra=extra) == 0
    assert save_rollout(path, prompt, [{"role": "assistant", "content": "b"}]) == 1

    first = load_rollout(path, 0)
    assert first.prompt == prompt
    assert first.completion[0]["content"] == "a"
    assert first.meta["task"] == "distil-sample"  # save format is the load format
    assert first.meta["is_truncated"] is False
    assert load_rollout(path, 1).completion[0]["content"] == "b"


def test_resolve_rollout_path_expands_ellipsis(tmp_path):
    result = tmp_path / "outputs" / "evals" / "env--model" / "run-id" / "results.jsonl"
    result.parent.mkdir(parents=True)
    result.write_text("")

    assert resolve_rollout_path(tmp_path / "outputs" / "evals" / "..." / "results.jsonl") == result


def test_resolve_rollout_path_rejects_ambiguous_ellipsis(tmp_path):
    for run in ("one", "two"):
        result = tmp_path / "outputs" / "evals" / "env--model" / run / "results.jsonl"
        result.parent.mkdir(parents=True)
        result.write_text("")

    with pytest.raises(ValueError, match="matched 2 files"):
        resolve_rollout_path(tmp_path / "outputs" / "evals" / "..." / "results.jsonl")
