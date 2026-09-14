"""Fast training helpers that do not load a model."""

import json

import pytest
import mlx.core as mx

from distil.progress import Probe, ProbePoint
from distil.sample import _proposal_is_accepted
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
    teacher = mx.array([10.0, 9.0, 0.0])

    assert not _proposal_is_accepted(1, teacher, top_k=1)
    assert _proposal_is_accepted(1, teacher, top_k=2)


@pytest.mark.parametrize("captured_rows", [None, 2, 3])
@pytest.mark.parametrize("empty", [False, True])
def test_shared_rollout_alignment_preserves_rows_text_and_teacher_logits(
    monkeypatch, captured_rows, empty
):
    import distil.training as training
    from distil.sample import Sample
    from distil.scoring import AlignedTokens

    class TeacherTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "question"}]
            assert kwargs == {"tokenize": False, "add_generation_prompt": True,
                              "enable_thinking": False}
            return "TTT"

    session = training.TrainingSession.__new__(training.TrainingSession)
    session.tokenizer = object()  # The student's saved prefix must not be re-rendered.
    session.teacher_tokenizer = TeacherTokenizer()
    session.enable_thinking = False
    logits = None if captured_rows is None else mx.zeros((captured_rows, 4))
    sample = Sample("P", "ab", "ab", "length", 2, teacher_logits=logits)

    def align(tokenizer, student, teacher_tokenizer, teacher):
        assert (student.text, student.spans) == ("Pab", [(1, 3)])
        assert (teacher.text, teacher.spans) == ("TTTab", [(3, 5)])
        return AlignedTokens(
            [10, 11, 12], [(0, 1), (1, 2), (2, 3)],
            [20, 21, 22, 11, 12], [(i, i + 1) for i in range(5)],
            [] if empty else [(1, 3), (2, 4)],
        )

    monkeypatch.setattr(training, "align_scored_tokens", align)
    result = session._align_rollout("question", [{"role": "user", "content": "question"}], sample)
    if empty:
        assert result is None
    else:
        assert result.sample is sample
        assert result.student_rows == [0, 1]
        assert result.teacher_rows == [2, 3]
        assert result.token_texts == ["a", "b"]
        assert result.teacher_logits is (logits if captured_rows == 2 else None)


@pytest.mark.parametrize("separate_judge", [False, True])
def test_continual_training_uses_canary_limit_and_saves_rollouts(
    tmp_path, monkeypatch, separate_judge
):
    import distil.distill as distill

    state = {"learned": False}
    answer_calls = []
    eval_reports = []
    train_reports = []
    feedback_capture_calls = []
    scored_prompts = []
    judge_calls = []

    class Model:
        def eval(self):
            pass

        def named_modules(self):
            return []

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        def save(self, path):
            path.mkdir(parents=True, exist_ok=True)
            (path / "adapter_config.json").write_text("{}")
            (path / "adapters.safetensors").write_bytes(b"")
            return path / "adapters.safetensors"

        def step_batch(self, prompts, capture_feedback=False):
            feedback_capture_calls.append(capture_feedback)
            state["learned"] = True
            return [
                TrainStep(
                    1,
                    prompt,
                    0.1,
                    1,
                    "stop",
                    "VIOLET-731",
                    batch_size=len(prompts),
                    batch_index=index,
                    batch_loss=0.1,
                )
                for index, prompt in enumerate(prompts)
            ]

    def fake_answer(model, tokenizer, probe, **kwargs):
        answer_calls.append((probe.name, kwargs["max_tokens"]))
        return probe.completion

    def fake_score(model, tokenizer, probe, step, enable_thinking):
        scored_prompts.append(probe.prompt)
        probability = 0.9 if state["learned"] else 0.1
        return ProbePoint(step, -1.0, probability, -1.0, probability, [])

    def fake_judge(model, tokenizer, question, expected, generated, **kwargs):
        judge_calls.append((question, kwargs.get("system")))
        return generated == expected

    monkeypatch.setattr(distill, "TrainingSession", Session)
    monkeypatch.setattr(distill, "_answer", fake_answer)
    monkeypatch.setattr(
        distill,
        "judge_answer",
        fake_judge,
    )
    monkeypatch.setattr(distill, "score_probe", fake_score)

    rollouts = tmp_path / "rollouts.jsonl"
    student = Model()
    result = distill.run_continual(
        student,
        object(),
        [LearningItem("fact", "heldout", "VIOLET-731", ("train",))],
        [Probe("canary", "2+2?", "4")],
        "teacher context",
        tmp_path / "adapter",
        model_name="model",
        teacher_model=Model() if separate_judge else student,
        teacher_tokenizer=object() if separate_judge else None,
        iters=1,
        eval_every=1,
        canary_max_tokens=7,
        save_rollout_path=rollouts,
        report=eval_reports.append,
        train_report=train_reports.append,
    )
    assert result.success
    assert all(limit == 7 for name, limit in answer_calls if name == "canary")
    assert all(limit == 32 for name, limit in answer_calls if name == "fact")
    assert scored_prompts == [
        "heldout\n\nAnswer with only the requested answer. Do not add explanation or formatting.",
        "heldout\n\nAnswer with only the requested answer. Do not add explanation or formatting.",
    ]
    row = json.loads(rollouts.read_text())
    assert row["item_id"] == "fact"
    assert row["batch_size"] == 1
    assert row["batch_index"] == 0
    assert row["batch_loss"] == 0.1
    assert row["prompt"][-1]["content"] == "train"
    assert row["completion"][0]["content"] == "VIOLET-731"
    metric = json.loads(
        (tmp_path / "adapter" / "training_metrics.jsonl").read_text()
    )
    assert metric["batch"] == {
        "size": 1,
        "index": 0,
        "loss": 0.1,
        "train_seconds": 0.0,
    }
    config = json.loads((tmp_path / "adapter" / "run_config.json").read_text())
    assert config["batch_size"] == 4
    assert config["mastery_criterion"] == "greedy_llm_judge"
    assert config["mastery_patience"] == 2
    assert train_reports[0] == {
        "event": "question",
        "iteration": 1,
        "item_id": "fact",
        "prompt": "train",
        "batch_size": 1,
    }
    assert train_reports[1]["event"] == "step"
    assert train_reports[1]["timing"] == {
        "rollout_seconds": 0.0,
        "distill_seconds": 0.0,
        "optimizer_seconds": 0.0,
        "total_seconds": 0.0,
    }
    assert eval_reports[-1]["training_rollout"]["prompt"] == "train"
    assert eval_reports[-1]["training_rollout"]["content"] == "VIOLET-731"
    assert eval_reports[-1]["confirmations"]["fact"] == {
        "content": "VIOLET-731",
        "passed": True,
        "streak": 2,
    }
    assert feedback_capture_calls == [True]
    assert judge_calls == [
        ("2+2?", None),
        ("heldout", None),
        ("heldout", None),
    ]


def test_continual_mastery_requires_consecutive_whole_answer_passes(
    tmp_path, monkeypatch
):
    import distil.distill as distill

    reports = []
    item_answers = iter(["wrong", "VIOLET-731", "wrong", "VIOLET-731", "VIOLET-731"])

    class Model:
        def eval(self):
            pass

        def named_modules(self):
            return []

    class Session:
        def __init__(self, *args, **kwargs):
            self.iteration = 0

        def save(self, path):
            path.mkdir(parents=True, exist_ok=True)
            (path / "adapter_config.json").write_text("{}")
            (path / "adapters.safetensors").write_bytes(b"")
            return path / "adapters.safetensors"

        def step_batch(self, prompts, capture_feedback=False):
            self.iteration += 1
            return [
                TrainStep(
                    self.iteration,
                    prompt,
                    0.1,
                    1,
                    "stop",
                    "VIOLET-731",
                    batch_size=len(prompts),
                    batch_index=index,
                    batch_loss=0.1,
                )
                for index, prompt in enumerate(prompts)
            ]

    def fake_answer(model, tokenizer, probe, **kwargs):
        return "4" if probe.name == "canary" else next(item_answers)

    def fake_score(model, tokenizer, probe, step, enable_thinking):
        return ProbePoint(step, -1.0, 0.9, -1.0, 0.9, [])

    monkeypatch.setattr(distill, "TrainingSession", Session)
    monkeypatch.setattr(distill, "_answer", fake_answer)
    monkeypatch.setattr(
        distill,
        "judge_answer",
        lambda model, tokenizer, question, expected, generated, **kwargs: generated
        == expected,
    )
    monkeypatch.setattr(distill, "score_probe", fake_score)

    result = distill.run_continual(
        Model(),
        object(),
        [LearningItem("fact", "heldout", "VIOLET-731", ("train",))],
        [Probe("canary", "2+2?", "4")],
        "teacher context",
        tmp_path / "adapter",
        model_name="model",
        iters=4,
        eval_every=1,
        report=reports.append,
    )

    assert result.success
    assert result.selected_step == 4
    assert [row["mastery_streaks"]["fact"] for row in reports] == [0, 1, 0, 1, 2]
    assert [row["mastered"] for row in reports] == [0, 0, 0, 0, 1]
