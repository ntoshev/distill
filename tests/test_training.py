"""Fast training helpers that do not load a model."""

import json

import pytest
import mlx.core as mx

from distil.progress import Probe, ProbePoint
from distil.sample import _select_interleaved_token
from distil.stopping import LearningItem
from distil.training import TrainStep, _kl_loss, load_prompts


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


def test_continual_training_uses_canary_limit_and_saves_rollouts(tmp_path, monkeypatch):
    import distil.distill as distill

    state = {"learned": False}
    answer_calls = []

    class Model:
        def eval(self):
            pass

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        def save(self, path):
            path.mkdir(parents=True, exist_ok=True)
            (path / "adapter_config.json").write_text("{}")
            (path / "adapters.safetensors").write_bytes(b"")
            return path / "adapters.safetensors"

        def step(self, prompt):
            state["learned"] = True
            return TrainStep(1, prompt, 0.1, 1, "stop", "VIOLET-731")

    def fake_answer(model, tokenizer, probe, **kwargs):
        answer_calls.append((probe.name, kwargs["max_tokens"]))
        return probe.completion

    def fake_score(model, tokenizer, probe, step, enable_thinking):
        probability = 0.9 if state["learned"] else 0.1
        return ProbePoint(step, -1.0, probability, -1.0, probability, [])

    monkeypatch.setattr(distill, "TrainingSession", Session)
    monkeypatch.setattr(distill, "_answer", fake_answer)
    monkeypatch.setattr(distill, "score_probe", fake_score)

    rollouts = tmp_path / "rollouts.jsonl"
    result = distill.run_continual(
        Model(),
        object(),
        [LearningItem("fact", "heldout", "VIOLET-731", ("train",))],
        [Probe("canary", "2+2?", "4")],
        "teacher context",
        tmp_path / "adapter",
        model_name="model",
        iters=1,
        eval_every=1,
        canary_max_tokens=7,
        save_rollout_path=rollouts,
    )
    assert result.success
    assert all(limit == 7 for name, limit in answer_calls if name == "canary")
    row = json.loads(rollouts.read_text())
    assert row["item_id"] == "fact"
    assert row["prompt"][-1]["content"] == "train"
    assert row["completion"][0]["content"] == "VIOLET-731"
