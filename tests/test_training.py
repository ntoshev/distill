"""Fast training helpers that do not load a model."""

import pytest

from distil.training import load_prompts


def test_load_prompts_uses_nonempty_lines(tmp_path):
    path = tmp_path / "questions.txt"
    path.write_text("first question\n\n  second question  \n")
    assert load_prompts(path) == ["first question", "second question"]


def test_load_prompts_rejects_missing_and_empty_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_prompts(tmp_path / "missing.txt")
    empty = tmp_path / "empty.txt"
    empty.write_text(" \n")
    with pytest.raises(ValueError, match="empty"):
        load_prompts(empty)
