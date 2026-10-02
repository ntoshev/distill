"""Probability-mass acceptance and its CLI/metadata contracts."""

import json
import random

import mlx.core as mx
import pytest

from distil.cli import main, _speculative_metadata
from distil.cli_config import build_parser, resolve_config
from distil.sample import Sample, _proposal_is_accepted, sample_speculative_text
from distil.training import (
    TrainStep, TrainingSession, _step_payload, save_training_rollout, train_lora,
)


@pytest.mark.parametrize("threshold,expected", [
    (0.01, {1}), (0.5, {1}), (0.6, {1, 3}), (0.75, {1, 3}),
    (0.8, {1, 3, 0}), (1.0, {0, 1, 2, 3}),
])
def test_top_p_keeps_smallest_nucleus_including_crossing_token(threshold, expected):
    # Deliberately unsorted, distinct probabilities; exact 0.5/0.75 boundaries.
    scores = mx.log(mx.array([0.15625, 0.5, 0.09375, 0.25]))
    accepted = {i for i in range(4) if _proposal_is_accepted(i, scores, top_p=threshold)}
    assert accepted == expected
    assert not _proposal_is_accepted(-1, scores, top_p=threshold)
    assert not _proposal_is_accepted(4, scores, top_p=threshold)


def test_top_p_adapts_support_and_keeps_at_least_one_token():
    concentrated = mx.array([1000.0, 0.0, -1000.0, -2000.0])
    uniform = mx.zeros(4)
    assert sum(_proposal_is_accepted(i, concentrated, top_p=0.75) for i in range(4)) == 1
    assert sum(_proposal_is_accepted(i, uniform, top_p=0.75) for i in range(4)) == 3
    assert all(_proposal_is_accepted(i, concentrated, top_p=1.0) for i in range(4))


@pytest.mark.parametrize("value", [0.0, -0.1, 1.01, float("nan"), float("inf")])
def test_top_p_validated_by_sampling_and_training_before_model_access(value):
    with pytest.raises(ValueError, match=r"--speculative-top-p must be in \(0, 1\]"):
        _proposal_is_accepted(0, mx.zeros(4), top_p=value)
    with pytest.raises(ValueError, match="--speculative-top-p"):
        sample_speculative_text(None, None, "s", None, None, "t", top_p=value)
    with pytest.raises(ValueError, match="--speculative-top-p"):
        TrainingSession(None, None, "context", speculative_top_p=value)
    with pytest.raises(ValueError, match="--speculative-top-p"):
        train_lora(None, None, ["p"], "context", "run", speculative_top_p=value)


def test_top_p_and_top_k_are_mutually_exclusive_in_python_api():
    with pytest.raises(ValueError, match="only one"):
        _proposal_is_accepted(0, mx.zeros(4), top_k=2, top_p=0.9)
    with pytest.raises(ValueError, match="only one"):
        TrainingSession(None, None, "context", speculative_top_k=2, speculative_top_p=0.9)


@pytest.mark.parametrize("mode", [
    ["--sample", "--prompt", "p", "--teacher-system", "t"],
    ["--train", "context", "--adapter", "run", "--canaries", "canaries.jsonl"],
])
def test_cli_top_p_disables_implicit_training_top_k(mode):
    argv = ["--model", "m", *mode, "--speculative-top-p", "0.9"]
    args = build_parser().parse_args(argv)
    config = resolve_config(args, argv)
    assert args.speculative_top_p == 0.9
    if config.training is not None:
        assert config.training.speculative_top_k is None
        assert config.training.speculative_top_p == 0.9


@pytest.mark.parametrize("flags,error", [
    (["--sample", "--prompt", "p"], "requires a --teacher-* flag"),
    (["--completion", "answer"], "only combines with --sample or --train"),
    (["--progress", "run", "--probes", "p", "--sample"], "does not combine with --progress"),
    (["--sample", "--prompt", "p", "--teacher-system", "t", "--speculative-top-k", "3"], "not allowed with argument"),
])
def test_cli_top_p_rejects_invalid_combinations(flags, error, capsys):
    argv = ["--model", "m", *flags, "--speculative-top-p", "0.9"]
    try:
        result = main(argv)
    except SystemExit as exc:
        result = exc.code
    assert result == 2
    assert error in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-0.1", "1.01", "nan", "inf"])
def test_cli_top_p_rejects_invalid_threshold(value, capsys):
    assert main(["--model", "m", "--sample", "--prompt", "p", "--teacher-system", "t",
                 "--speculative-top-p", value]) == 2
    assert "--speculative-top-p must be in (0, 1]" in capsys.readouterr().err


@pytest.mark.parametrize("raw", [False, True])
def test_cli_top_p_dispatch_and_saved_metadata(monkeypatch, capsys, tmp_path, raw):
    import mlx_lm
    import distil.cli as cli

    monkeypatch.setattr(mlx_lm, "load", lambda *a, **k: (object(), object()))

    def sample(*args, **kwargs):
        assert kwargs["top_p"] == 0.9
        assert kwargs.get("top_k") is None
        return Sample("s>", "answer", "answer", "stop", 1,
                      speculative_top_p=0.9, accepted_tokens=0, replaced_tokens=1,
                      teacher_sampled=[True])

    monkeypatch.setattr(cli, "sample_speculative_text" if raw else "sample_speculative_completion", sample)
    monkeypatch.setattr(cli, "render_generation_prompt", lambda *a, **k: "t>")
    monkeypatch.setattr(cli, "score_text_pair", lambda *a, **k: object())
    monkeypatch.setattr(cli, "to_json_dict", lambda result: {})
    rollout = tmp_path / "rollout.jsonl"
    flags = ["--raw"] if raw else ["--save-rollout", str(rollout)]
    assert main(["--model", "m", "--sample", "--prompt", "s>",
                 "--teacher-prompt", "t>", "--speculative-top-p", "0.9",
                 "--json", *flags]) == 0
    output = capsys.readouterr()
    assert "0 accepted, 1 replaced" in output.err
    assert json.loads(output.out)["sample"]["speculative_top_p"] == 0.9
    if not raw:
        assert json.loads(rollout.read_text())["speculative_top_p"] == 0.9


def test_top_p_training_metrics_and_rollout_metadata(tmp_path):
    step = TrainStep(1, "p", 0.1, 2, "stop", "answer", speculative_top_p=0.9,
                     accepted_tokens=1, replaced_tokens=1)
    sampling = _step_payload(step)["sampling"]
    assert sampling["policy"] == "speculative"
    assert sampling["top_p"] == 0.9 and sampling["top_k"] is None
    assert sampling["acceptance_rate"] == 0.5
    path = tmp_path / "rollout.jsonl"
    save_training_rollout(path, [{"role": "user", "content": "p"}], step,
                          model_name="m", temperature=0, max_tokens=2, seed=0)
    assert json.loads(path.read_text())["speculative_top_p"] == 0.9
    assert _speculative_metadata(Sample("p", "a", "a", "stop", 1)) == {}


def test_top_p_keeps_most_likely_token_for_threshold_below_float32_range():
    assert _proposal_is_accepted(1, mx.array([0.0, 10.0]), top_p=1e-100)
    assert not _proposal_is_accepted(0, mx.array([0.0, 10.0]), top_p=1e-100)


def test_top_p_breaks_boundary_ties_by_token_id():
    scores = mx.zeros(4)
    assert [_proposal_is_accepted(i, scores, top_p=0.5) for i in range(4)] == [True, True, False, False]


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_top_p_matches_sorted_nucleus_oracle(dtype):
    rng = random.Random(17)
    scores = mx.array([rng.uniform(-8, 8) for _ in range(64)], dtype=dtype)
    probabilities = mx.softmax(scores.astype(mx.float32)).tolist()
    order = sorted(range(64), key=lambda i: (-probabilities[i], i))
    for threshold in (0.001, 0.25, 0.5, 0.9, 0.999, 1.0):
        expected = set()
        total = 0
        for i in order:
            expected.add(i)
            total += probabilities[i]
            if total >= threshold:
                break
        actual = {i for i in range(64) if _proposal_is_accepted(i, scores, top_p=threshold)}
        assert actual == expected
