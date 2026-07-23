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


def test_progress_uses_saved_step_zero_instead_of_duplicating_base(tmp_path, monkeypatch):
    checkpoints = tmp_path / "checkpoints"
    for step in (0, 5):
        directory = checkpoints / f"step-{step:04d}"
        directory.mkdir(parents=True)
        (directory / "adapters.safetensors").touch()

    loaded = []
    monkeypatch.setattr("distil.progress.load_adapters", lambda model, path: loaded.append(path))
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

    result = score_progress(Model(), object(), tmp_path, [Probe("x", "p", "c")])
    assert result.steps == [0, 5]
    assert [point.step for point in result.series["x"]] == [0, 5]
    assert len(loaded) == 2


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
