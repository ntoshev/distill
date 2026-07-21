"""Fast training helpers that do not load a model."""

import pytest
import mlx.core as mx

from distil.sample import _select_interleaved_token
from distil.training import _kl_loss, load_prompts


def test_kl_loss_directions_match_manual_distributions():
    student = mx.array([[2.0, 0.0, -1.0]])
    teacher = mx.array([[0.0, 1.0, 0.0]])
    student_logp = student - mx.logsumexp(student, axis=-1, keepdims=True)
    teacher_logp = teacher - mx.logsumexp(teacher, axis=-1, keepdims=True)
    forward = (mx.exp(teacher_logp) * (teacher_logp - student_logp)).sum(axis=-1)
    reverse = (mx.exp(student_logp) * (student_logp - teacher_logp)).sum(axis=-1)

    assert mx.allclose(_kl_loss(student, teacher, "forward"), forward).item()
    assert mx.allclose(_kl_loss(student, teacher, "reverse"), reverse).item()
    assert not mx.allclose(forward, reverse).item()


def test_kl_loss_rejects_unknown_direction():
    with pytest.raises(ValueError, match="unknown KL direction"):
        _kl_loss(mx.zeros((1, 2)), mx.zeros((1, 2)), "sideways")


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


def test_load_prompts_rejects_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a file"):
        load_prompts(tmp_path)


def test_interleaved_token_accepts_only_inside_teacher_top_k():
    greedy = lambda values: mx.argmax(values, axis=-1)
    student = mx.array([0.0, 10.0, 1.0])
    teacher = mx.array([10.0, 9.0, 0.0])

    token, accepted = _select_interleaved_token(
        student,
        teacher,
        top_k=1,
        student_sampler=greedy,
        teacher_sampler=greedy,
    )
    assert (token, accepted) == (0, False)

    token, accepted = _select_interleaved_token(
        student,
        teacher,
        top_k=2,
        student_sampler=greedy,
        teacher_sampler=greedy,
    )
    assert (token, accepted) == (1, True)
