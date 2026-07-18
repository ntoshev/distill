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
