"""Fast training helpers that do not load a model."""

import json

import pytest
import mlx.core as mx

from distill.progress import Probe, ProbePoint
from distill.sample import _proposal_is_accepted
from distill.stopping import LearningItem
from distill.training import TrainStep, _kl_loss, load_prompts


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


@pytest.mark.parametrize('rows', [128, 256])
def test_forward_kl_large_vocab_has_repeatable_analytic_gradient(rows):
    """Production chunk/dense sizes exposed a race in the fused Metal kernel."""
    mx.random.seed(42)
    student = 5 * mx.random.normal((rows, 248320))
    teacher = student + mx.random.normal(student.shape)
    expected = (mx.softmax(student, axis=-1) - mx.softmax(teacher, axis=-1)) / rows
    mx.eval(student, teacher, expected)
    previous = None
    for _ in range(3):
        loss, grad = mx.value_and_grad(
            lambda s: _kl_loss(s, teacher, 'forward').mean())(student)
        mx.eval(loss, grad)
        relative_error = mx.linalg.norm(grad - expected) / mx.linalg.norm(expected)
        assert relative_error.item() < 1e-5
        if previous is not None:
            assert loss.item() == previous[0]
            assert mx.array_equal(grad, previous[1]).item()
        previous = loss.item(), grad


@pytest.mark.parametrize('direction', ['forward', 'reverse'])
def test_kl_keeps_teacher_fixed(direction):
    student = mx.array([[1., 2., 3.]])
    teacher = mx.array([[3., 2., 1.]])
    gradient = mx.grad(lambda t: _kl_loss(student, t, direction).sum())(teacher)
    assert mx.array_equal(gradient, mx.zeros_like(teacher)).item()


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


def test_saved_samples_use_student_prefix_and_preserve_teacher_completion(monkeypatch):
    import distill.training as training
    from distill.sample import Sample
    from types import SimpleNamespace

    session = training.TrainingSession.__new__(training.TrainingSession)
    session.tokenizer = object()
    session.student_system = None
    session.teacher_system = "private instructions"
    session.enable_thinking = False
    sample = Sample("TEACHER PREFIX", "answer<eos>", "answer", "stop", 2,
                    teacher_logits=mx.zeros((2, 4)))
    seen = []

    def render(tokenizer, messages, thinking):
        assert messages == [{"role": "user", "content": "question"}]
        assert thinking is False
        return "STUDENT PREFIX"

    def align(prompt, messages, actual):
        assert messages == [{"role": "system", "content": "private instructions"},
                            {"role": "user", "content": "question"}]
        assert actual.prompt_text == "STUDENT PREFIX"
        assert actual.completion_text == "answer<eos>"
        assert actual.content == "answer" and actual.finish_reason == "stop"
        assert actual.teacher_logits is None
        seen.append(actual)
        return SimpleNamespace(prepare_seconds=0.0)

    def update(prepared, **kwargs):
        assert len(prepared) == 1 and kwargs["optimized"] is False
        return ["updated"]

    monkeypatch.setattr(training, "render_generation_prompt", render)
    monkeypatch.setattr(session, "_align_rollout", align)
    monkeypatch.setattr(session, "_update_prepared_batch", update)
    monkeypatch.setattr(training, "sample_completion", lambda *a, **k: pytest.fail("must not generate"))
    assert session.step_samples(["question"], [sample]) == ["updated"]
    assert sample.prompt_text == "TEACHER PREFIX" and sample.teacher_logits is not None
    assert len(seen) == 1


@pytest.mark.parametrize("prompts,samples", [([], []), (["q"], [])])
def test_saved_samples_reject_invalid_batch(prompts, samples):
    from distill.training import TrainingSession
    session = TrainingSession.__new__(TrainingSession)
    with pytest.raises(ValueError, match="matching non-empty"):
        session.step_samples(prompts, samples)


def test_saved_noncanonical_token_splits_are_preserved(monkeypatch):
    import distill.training as training
    from distill.sample import Sample

    class Tokenizer:
        def encode(self, text, **kwargs):
            return {'student': [10], 'teacher': [20, 21], 'ab': [99]}[text]
        def decode(self, ids):
            return ''.join({11: 'a', 12: 'b'}[i] for i in ids)

    session = training.TrainingSession.__new__(training.TrainingSession)
    session.tokenizer = session.teacher_tokenizer = Tokenizer()
    session.enable_thinking = True
    monkeypatch.setattr(training, 'render_generation_prompt', lambda *a: 'teacher')
    sample = Sample('student', 'ab', 'ab', 'length', 2, token_ids=[11, 12])
    result = session._align_rollout('question', [], sample)
    assert result.student_ids == [10, 11, 12]
    assert result.teacher_ids == [20, 21, 11, 12]
    assert result.student_rows == [0, 1]
    assert result.teacher_rows == [1, 2]
    sample.num_tokens = 1
    with pytest.raises(ValueError, match='token count'):
        session._align_rollout('question', [], sample)


def test_interleaved_token_accepts_only_inside_teacher_top_k():
    teacher = mx.array([10.0, 9.0, 0.0])

    assert not _proposal_is_accepted(1, teacher, top_k=1)
    assert _proposal_is_accepted(1, teacher, top_k=2)


@pytest.mark.parametrize("captured_rows", [None, 2, 3])
@pytest.mark.parametrize("empty", [False, True])
def test_shared_rollout_alignment_preserves_rows_text_and_teacher_logits(
    monkeypatch, captured_rows, empty
):
    import distill.training as training
    from distill.sample import Sample
    from distill.scoring import AlignedTokens

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


@pytest.mark.parametrize("policy", [{}, {"speculative_top_k": None}, {"speculative_top_p": 0.9}])
@pytest.mark.parametrize("separate_judge", [False, True])
def test_continual_training_uses_canary_limit_and_saves_rollouts(
    tmp_path, monkeypatch, separate_judge, policy, stub_continual_evaluation
):
    import distill.distill as distill

    top_p = policy.get("speculative_top_p")
    top_k = policy.get("speculative_top_k", 3 if top_p is None else None)

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
            assert kwargs["speculative_top_p"] == top_p
            assert kwargs["speculative_top_k"] == top_k

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
                    speculative_top_k=top_k,
                    speculative_top_p=top_p,
                    batch_size=len(prompts),
                    batch_index=index,
                    batch_loss=0.1,
                )
                for index, prompt in enumerate(prompts)
            ]

    def fake_answer(model, tokenizer, probe, **kwargs):
        answer_calls.append((probe.name, kwargs["max_tokens"]))
        return probe.completion

    def fake_score(model, tokenizer, probe, step, enable_thinking, **kwargs):
        scored_prompts.append(probe.prompt)
        probability = 0.9 if state["learned"] else 0.1
        return ProbePoint(step, -1.0, probability, -1.0, probability, [])

    def fake_judge(model, tokenizer, question, expected, generated, **kwargs):
        judge_calls.append((expected, generated))
        return generated == expected

    monkeypatch.setattr(distill, "TrainingSession", Session)
    batch_calls = stub_continual_evaluation(
        answer=fake_answer, score=fake_score, judge=fake_judge,
    )

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
        **policy,
        iters=1,
        eval_every=1,
        mastery_patience=2,
        canary_max_tokens=7,
        save_rollout_path=rollouts,
        report=eval_reports.append,
        train_report=train_reports.append,
    )
    assert result.success
    assert [(step, len(probes), limits) for step, probes, limits in batch_calls] == [
        (-1, 1, [7]), (0, 3, [32, 32, 7]), (1, 3, [32, 32, 7]),
    ]
    assert all(limit == 7 for name, limit in answer_calls if name == "canary")
    assert all(limit == 32 for name, limit in answer_calls if name == "fact")
    assert scored_prompts == [
        "heldout\n\nAnswer with only the requested answer. Do not add explanation or formatting.",
        "heldout\n\nAnswer with only the requested answer. Do not add explanation or formatting.",
    ]
    row = json.loads(rollouts.read_text())
    if top_p is not None:
        assert row["speculative_top_p"] == top_p
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
    assert config["speculative_top_p"] == top_p
    assert config["speculative_top_k"] == top_k
    if top_p is not None:
        assert metric["sampling"]["top_p"] == top_p
        assert eval_reports[-1]["training_rollout"]["speculative_top_p"] == top_p
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
        ("4", "4"),
        ("VIOLET-731", "VIOLET-731"),
        ("VIOLET-731", "VIOLET-731"),
    ]


def test_continual_mastery_requires_consecutive_whole_answer_passes(
    tmp_path, monkeypatch, stub_continual_evaluation
):
    import distill.distill as distill

    reports = []
    item_answers = iter(["wrong", "VIOLET-731", "wrong", *["VIOLET-731"] * 4])

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

    def fake_score(model, tokenizer, probe, step, enable_thinking, **kwargs):
        return ProbePoint(step, -1.0, 0.9, -1.0, 0.9, [])

    monkeypatch.setattr(distill, "TrainingSession", Session)
    stub_continual_evaluation(
        answer=fake_answer, score=fake_score,
        judge=lambda model, tokenizer, question, expected, generated, **kw: generated == expected,
    )

    result = distill.run_continual(
        Model(),
        object(),
        [LearningItem("fact", "heldout", "VIOLET-731", ("train",))],
        [Probe("canary", "2+2?", "4")],
        "teacher context",
        tmp_path / "adapter",
        model_name="model",
        iters=6,
        eval_every=1,
        report=reports.append,
    )

    assert result.success
    assert result.selected_step == 6
    assert [row["mastery_streaks"]["fact"] for row in reports] == [0, 1, 0, 1, 2, 3, 4]
    assert [row["mastered"] for row in reports] == [0, 0, 0, 0, 0, 0, 1]
