"""Probe loading, checkpoint discovery, and progress JSON need no model."""

import json

import pytest

from distil.progress import (
    Probe,
    ProbePoint,
    ProgressResult,
    find_checkpoints,
    load_probes,
    progress_to_json,
    sample_progress,
    score_progress,
    score_probe,
)
from distil.scoring import ScoreResult, TokenScore, TopToken


def test_load_probes(tmp_path):
    path = tmp_path / "probes.jsonl"
    path.write_text(
        '{"name":"fact","prompt":"Code?","completion":"VIOLET-731"}\n'
        '{"name":"canary","prompt":"2+2?","completion":"4","system":"Be terse"}\n'
    )
    probes = load_probes(path)
    assert probes == [
        Probe("fact", "Code?", "VIOLET-731"),
        Probe("canary", "2+2?", "4", "Be terse"),
    ]


def test_load_probes_can_expand_training_prompts(tmp_path):
    path = tmp_path / "probes.jsonl"
    path.write_text(
        json.dumps({
            "name": "fact",
            "prompt": "Heldout?",
            "completion": "VIOLET-731",
            "train_prompts": ["Train one?", "Train two?"],
        }) + "\n"
    )
    probes = load_probes(path, prompt_mode="train")
    assert probes == [
        Probe("fact/train-1", "Train one?", "VIOLET-731"),
        Probe("fact/train-2", "Train two?", "VIOLET-731"),
    ]


@pytest.mark.parametrize(
    "content, message",
    [
        ("\n", "empty"),
        ('{"name":"x","prompt":"p"}\n', "invalid probe"),
        (
            '{"name":"x","prompt":"p","completion":"a"}\n'
            '{"name":"x","prompt":"q","completion":"b"}\n',
            "duplicate probe name",
        ),
    ],
)
def test_load_probes_rejects_bad_files(tmp_path, content, message):
    path = tmp_path / "probes.jsonl"
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        load_probes(path)


def test_load_probes_rejects_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a file"):
        load_probes(tmp_path)


def test_find_checkpoints_accepts_run_root_and_sorts_numerically(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    for name in ("step-0010", "step-0002", "step-bad"):
        directory = checkpoints / name
        directory.mkdir(parents=True)
        (directory / "adapters.safetensors").touch()
    expected = [(2, checkpoints / "step-0002"), (10, checkpoints / "step-0010")]
    assert find_checkpoints(tmp_path) == expected
    assert find_checkpoints(checkpoints) == expected


def test_progress_json_is_serializable():
    probe = Probe("fact", "Code?", "VIOLET-731")
    point = ProbePoint(0, -1.0, 0.37, -2.0, 0.14, [{"text": "V", "logprob": -1.0}])
    payload = progress_to_json(ProgressResult("run", [0], [probe], {"fact": [point]}))
    assert payload["steps"] == [0]
    assert payload["series"]["fact"][0]["mean_probability"] == 0.37
    json.dumps(payload)


@pytest.mark.parametrize("run,evaluator", [
    (score_progress, "score_probe"), (sample_progress, "sample_probe"),
])
@pytest.mark.parametrize("has_zero", [False, True])
def test_progress_includes_base_exactly_once(tmp_path, monkeypatch, run, evaluator, has_zero):
    checkpoints = tmp_path / "checkpoints"
    for step in ((0, 5) if has_zero else (5,)):
        directory = checkpoints / f"step-{step:04d}"
        directory.mkdir(parents=True)
        (directory / "adapters.safetensors").touch()

    loaded = []
    monkeypatch.setattr("distil.progress.load_adapters", lambda model, path: loaded.append(path))
    monkeypatch.setattr("distil.progress.remove_lora_layers", lambda model: None)
    monkeypatch.setattr(
        f"distil.progress.{evaluator}",
        lambda model, tokenizer, probe, step, enable_thinking, **kwargs: ProbePoint(
            step, -1.0, 0.37, -1.0, 0.37, []
        ),
    )
    class Model:
        def eval(self):
            pass

    result = run(Model(), object(), tmp_path, [Probe("x", "p", "c")])
    assert result.steps == [0, 5]
    assert [point.step for point in result.series["x"]] == [0, 5]
    assert len(loaded) == (2 if has_zero else 1)


@pytest.mark.parametrize("run,evaluator", [
    (score_progress, "score_probe"), (sample_progress, "sample_probe"),
])
@pytest.mark.parametrize("failure", ["load", "evaluate", "callback"])
def test_progress_cleans_up_adapter_on_failure(tmp_path, monkeypatch, run, evaluator, failure):
    checkpoint = tmp_path / "step-0000"
    checkpoint.mkdir()
    (checkpoint / "adapters.safetensors").touch()
    events = []

    class Model:
        def eval(self):
            events.append("eval")

    def event(name):
        def call(*args, **kwargs):
            if failure == name:
                raise RuntimeError(name)
            return ProbePoint(0, -1.0, 0.37, -1.0, 0.37, [])
        return call

    monkeypatch.setattr("distil.progress.load_adapters", event("load"))
    monkeypatch.setattr(f"distil.progress.{evaluator}", event("evaluate"))
    monkeypatch.setattr("distil.progress.remove_lora_layers", lambda model: events.append("remove"))
    with pytest.raises(RuntimeError, match=failure):
        run(Model(), object(), tmp_path, [Probe("x", "p", "c")], on_point=event("callback"))
    assert events[-2:] == ["remove", "eval"]


def test_score_progress_reports_points_incrementally(tmp_path, monkeypatch):
    checkpoints = tmp_path / "checkpoints"
    for step in (0, 5):
        directory = checkpoints / f"step-{step:04d}"
        directory.mkdir(parents=True)
        (directory / "adapters.safetensors").touch()

    monkeypatch.setattr("distil.progress.load_adapters", lambda model, path: None)
    monkeypatch.setattr("distil.progress.remove_lora_layers", lambda model: None)
    monkeypatch.setattr(
        "distil.progress.score_probe",
        lambda model, tokenizer, probe, step, enable_thinking, **kwargs: ProbePoint(
            step, -1.0, 0.37, -1.0, 0.37, []
        ),
    )

    class Model:
        def eval(self):
            pass

    starts = []
    points = []
    score_progress(
        Model(),
        object(),
        tmp_path,
        [Probe("x", "p", "c")],
        on_start=lambda result: starts.append((result.steps, len(result.series["x"]))),
        on_point=lambda probe, point: points.append((probe.name, point.step)),
    )

    assert starts == [([0, 5], 0)]
    assert points == [("x", 0), ("x", 5)]


def test_score_probe_applies_default_and_teacher_context(monkeypatch):
    import distil.progress

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "|".join(f"{m['role']}:{m['content']}" for m in messages) + "|assistant:"

    captured = {}

    def fake_pair(model, tokenizer, built, teacher_model, teacher_tokenizer, built_teacher, **kwargs):
        captured["student"] = built.text
        captured["teacher"] = built_teacher.text
        token = TokenScore(
            1, "A", 0, 1, True, -1.0,
            [TopToken(1, "A", -1.0)],
            teacher_logprob=-0.5,
            kl=0.2,
            teacher_top_k=[TopToken(2, "B", -0.3)],
        )
        return ScoreResult(built.text, [token], "student", [], "teacher")

    monkeypatch.setattr(distil.progress, "score_text_pair", fake_pair)
    point = score_probe(
        object(),
        Tokenizer(),
        Probe("x", "question", "A"),
        0,
        None,
        top_k=1,
        default_system="student system",
        teacher_model=object(),
        teacher_tokenizer=Tokenizer(),
        teacher_system="teacher system",
    )
    assert captured["student"].startswith("system:student system|user:question")
    assert captured["teacher"].startswith("system:teacher system|user:question")
    assert point.mean_teacher_logprob == -0.5
    assert point.mean_kl == 0.2
    assert point.tokens[0]["teacher_top"] == [["B", -0.3]]


def test_sample_progress_scores_each_generated_completion(tmp_path, monkeypatch):
    import distil.progress
    from distil.sample import Sample

    checkpoints = tmp_path / "checkpoints"
    for step in (0, 5):
        directory = checkpoints / f"step-{step:04d}"
        directory.mkdir(parents=True)
        (directory / "adapters.safetensors").touch()

    class Model:
        def eval(self):
            pass

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "prompt>"

    loaded = []
    samples = []
    scored = []
    monkeypatch.setattr("distil.progress.load_adapters", lambda model, path: loaded.append(path))
    monkeypatch.setattr("distil.progress.remove_lora_layers", lambda model: None)

    def fake_sample(model, tokenizer, messages, **kwargs):
        text = f"answer-{len(samples)}"
        samples.append((messages, kwargs))
        return Sample("prompt>", text, text, "stop", 1)

    def fake_score(model, tokenizer, text, spans, **kwargs):
        scored.append((text, spans))
        token = TokenScore(1, text[-1], len("prompt>answer-"), len(text), True, -0.5)
        return ScoreResult(text, [token], "student", [])

    monkeypatch.setattr(distil.progress, "sample_completion", fake_sample)
    monkeypatch.setattr(distil.progress, "score_text", fake_score)
    result = sample_progress(
        Model(),
        Tokenizer(),
        tmp_path,
        [Probe("x", "question", "")],
        max_tokens=7,
        temperature=0.0,
        seed=3,
    )

    assert result.steps == [0, 5]
    assert len(loaded) == 2
    assert samples[0][0] == [{"role": "user", "content": "question"}]
    assert samples[0][1]["temperature"] == 0.0
    assert samples[0][1]["seed"] == 3
    assert scored[0] == ("prompt>answer-0", [(7, 15)])
    points = result.series["x"]
    assert [point.completion for point in points] == ["answer-0", "answer-1"]
    assert points[0].finish_reason == "stop"


def test_sample_probe_teacher_scores_exact_generated_string(monkeypatch):
    import distil.progress as progress
    from distil.sample import Sample

    class TeacherTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0] == {"role": "system", "content": "private"}
            assert kwargs["enable_thinking"] is False
            return "teacher>"

    sample = Sample("exact student>\n", "\nfresh<eos>", "fresh", "stop", 3)
    monkeypatch.setattr(progress, "sample_completion", lambda *a, **kw: sample)

    def score(model, tokenizer, built, teacher, teacher_tokenizer, teacher_built, **kwargs):
        assert built.text == sample.prompt_text + sample.completion_text
        assert built.spans == [(len(sample.prompt_text), len(built.text))]
        assert teacher_built.text == "teacher>" + sample.completion_text
        assert kwargs == {"top_k": 2, "teacher_without_adapter": True}
        return ScoreResult(built.text, [
            TokenScore(1, "\nfresh", 0, 6, True, -0.5, teacher_logprob=-0.2, kl=0.1),
        ], "student", [], "teacher")

    monkeypatch.setattr(progress, "score_text_pair", score)
    point = progress.sample_probe(
        object(), object(), Probe("p", "question", "UNUSED TARGET"), 5, False,
        max_tokens=3, temperature=0.0, top_k=2, teacher_model=object(),
        teacher_tokenizer=TeacherTokenizer(), teacher_system="private", teacher_without_adapter=True,
    )
    assert (point.completion, point.content, point.num_tokens) == ("\nfresh<eos>", "fresh", 3)
    assert point.mean_teacher_logprob == -0.2


def test_sample_progress_replays_saved_rollout_messages(tmp_path, monkeypatch):
    import distil.progress
    from distil.sample import Sample

    checkpoints = tmp_path / "checkpoints"
    directory = checkpoints / "step-0000"
    directory.mkdir(parents=True)
    (directory / "adapters.safetensors").touch()

    class Model:
        def eval(self):
            pass

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "prompt>"

    captured = {}
    monkeypatch.setattr("distil.progress.load_adapters", lambda model, path: None)
    monkeypatch.setattr("distil.progress.remove_lora_layers", lambda model: None)

    def fake_sample(model, tokenizer, messages, **kwargs):
        captured["messages"] = messages
        return Sample("prompt>", "answer", "answer", "stop", 1)

    monkeypatch.setattr(
        distil.progress,
        "sample_completion",
        fake_sample,
    )
    monkeypatch.setattr(
        distil.progress,
        "score_text",
        lambda model, tokenizer, text, spans, **kwargs: ScoreResult(
            text, [TokenScore(1, "answer", 7, 13, True, -0.5)], "student", []
        ),
    )

    saved_messages = (
        {"role": "system", "content": "Saved system."},
        {"role": "user", "content": "Saved question?"},
        {"role": "assistant", "content": "Earlier turn."},
        {"role": "user", "content": "Final question?"},
    )
    sample_progress(
        Model(),
        Tokenizer(),
        tmp_path,
        [Probe("rollout-0", "Final question?", "saved", messages=saved_messages)],
    )

    assert captured["messages"] == list(saved_messages)


def test_sample_progress_reports_points_incrementally(tmp_path, monkeypatch):
    import distil.progress
    from distil.sample import Sample

    checkpoints = tmp_path / "checkpoints"
    for step in (0, 5):
        directory = checkpoints / f"step-{step:04d}"
        directory.mkdir(parents=True)
        (directory / "adapters.safetensors").touch()

    class Model:
        def eval(self):
            pass

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "prompt>"

    monkeypatch.setattr("distil.progress.load_adapters", lambda model, path: None)
    monkeypatch.setattr("distil.progress.remove_lora_layers", lambda model: None)
    monkeypatch.setattr(
        distil.progress,
        "sample_completion",
        lambda *args, **kwargs: Sample("prompt>", "answer", "answer", "stop", 1),
    )
    monkeypatch.setattr(
        distil.progress,
        "score_text",
        lambda model, tokenizer, text, spans, **kwargs: ScoreResult(
            text, [TokenScore(1, "answer", 7, 13, True, -0.5)], "student", []
        ),
    )

    starts = []
    points = []
    sample_progress(
        Model(),
        Tokenizer(),
        tmp_path,
        [Probe("x", "p", "")],
        on_start=lambda result: starts.append((result.steps, len(result.series["x"]))),
        on_point=lambda probe, point: points.append((probe.name, point.step, point.content)),
    )

    assert starts == [([0, 5], 0)]
    assert points == [("x", 0, "answer"), ("x", 5, "answer")]
