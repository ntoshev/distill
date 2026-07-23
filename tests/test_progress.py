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
)


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
