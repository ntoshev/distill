"""CLI argument validation. These run before any model load, so no model needed."""

from distil.cli import main


def test_requires_an_input_mode(capsys):
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


def test_training_requires_teacher_context_and_adapter(tmp_path, capsys):
    prompts = tmp_path / "questions.txt"
    prompts.write_text("What is the code?\n")
    assert main(["--model", "m", "--train", str(prompts)]) == 2
    assert "teacher-system" in capsys.readouterr().err
    assert main([
        "--model", "m", "--train", str(prompts), "--teacher-system", "context"
    ]) == 2
    assert "adapter-path" in capsys.readouterr().err


def test_training_rejects_prompt_and_teacher_prompt(tmp_path, capsys):
    prompts = tmp_path / "questions.txt"
    prompts.write_text("What is the code?\n")
    base = [
        "--model", "m", "--train", str(prompts),
        "--teacher-system", "context", "--adapter-path", "adapter",
    ]
    assert main([*base, "--prompt", "q"]) == 2
    assert main([*base, "--teacher-prompt", "q"]) == 2


def test_training_flags_require_train(capsys):
    for flag in (["--iters", "1"], ["--learning-rate", "1e-4"], ["--lora-rank", "4"]):
        assert main(["--model", "m", "--completion", "x", *flag]) == 2


def test_training_rejects_nonpositive_settings(tmp_path):
    prompts = tmp_path / "questions.txt"
    prompts.write_text("What is the code?\n")
    base = [
        "--model", "m", "--train", str(prompts),
        "--teacher-system", "context", "--adapter-path", "adapter",
    ]
    for flag in (["--iters", "0"], ["--learning-rate", "0"], ["--lora-rank", "0"]):
        assert main([*base, *flag]) == 2


def test_empty_training_file_exits_before_model_load(tmp_path, capsys):
    prompts = tmp_path / "empty.txt"
    prompts.write_text("\n")
    assert main([
        "--model", "m", "--train", str(prompts),
        "--teacher-system", "context", "--adapter-path", "adapter",
    ]) == 2
    assert "empty" in capsys.readouterr().err


def test_metric_kl_requires_teacher(capsys):
    assert main(["--model", "m", "--completion", "x", "--metric", "kl"]) == 2
    assert "teacher" in capsys.readouterr().err


def test_bad_rollout_path_exits_before_model_load(capsys):
    assert main(["--model", "m", "--rollout", "outputs/evals/.../results.jsonl"]) == 2
    assert "rollout pattern matched no files" in capsys.readouterr().err
