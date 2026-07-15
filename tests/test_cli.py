"""CLI argument validation. These run before any model load, so no model needed."""

from distil.cli import main


def test_requires_completion_or_rollout(capsys):
    assert main(["--model", "m"]) == 2


def test_rejects_both_inputs(capsys):
    assert main(["--model", "m", "--completion", "x", "--rollout", "r.jsonl"]) == 2


def test_rejects_completion_and_sample(capsys):
    assert main(["--model", "m", "--completion", "x", "--sample", "--prompt", "p"]) == 2


def test_raw_excludes_system(capsys):
    assert main(["--model", "m", "--raw", "--system", "s", "--completion", "x"]) == 2


def test_raw_excludes_teacher(capsys):
    assert main(["--model", "m", "--raw", "--completion", "x", "--teacher-system", "s"]) == 2


def test_raw_excludes_no_thinking(capsys):
    assert main(["--model", "m", "--raw", "--completion", "x", "--no-thinking"]) == 2


def test_sample_requires_prompt(capsys):
    assert main(["--model", "m", "--sample"]) == 2


def test_sampling_flags_require_sample(capsys):
    for flag in (["--seed", "1"], ["--temperature", "0.7"], ["--max-tokens", "8"],
                 ["--save-rollout", "s.jsonl"]):
        assert main(["--model", "m", "--completion", "x", *flag]) == 2


def test_metric_kl_requires_teacher(capsys):
    assert main(["--model", "m", "--completion", "x", "--metric", "kl"]) == 2
    assert "teacher" in capsys.readouterr().err


def test_bad_rollout_path_exits_before_model_load(capsys):
    assert main(["--model", "m", "--rollout", "outputs/evals/.../results.jsonl"]) == 2
    assert "rollout pattern matched no files" in capsys.readouterr().err
