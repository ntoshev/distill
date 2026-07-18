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
    for flag in (
        ["--iters", "1"],
        ["--learning-rate", "1e-4"],
        ["--lora-rank", "4"],
        ["--checkpoint-every", "1"],
        ["--canaries", "canaries.txt"],
        ["--canary-max-tokens", "8"],
        ["--contamination-regex", "VIO"],
    ):
        assert main(["--model", "m", "--completion", "x", *flag]) == 2


def test_training_rejects_nonpositive_settings(tmp_path):
    prompts = tmp_path / "questions.txt"
    prompts.write_text("What is the code?\n")
    base = [
        "--model", "m", "--train", str(prompts),
        "--teacher-system", "context", "--adapter-path", "adapter",
    ]
    for flag in (
        ["--iters", "0"],
        ["--learning-rate", "0"],
        ["--lora-rank", "0"],
        ["--checkpoint-every", "0"],
        ["--canary-max-tokens", "0"],
    ):
        assert main([*base, *flag]) == 2


def test_training_validates_canary_options_before_model_load(tmp_path, capsys):
    prompts = tmp_path / "questions.txt"
    prompts.write_text("What is the code?\n")
    canaries = tmp_path / "canaries.txt"
    canaries.write_text("What is two plus two?\n")
    base = [
        "--model", "m", "--train", str(prompts),
        "--teacher-system", "context", "--adapter-path", "adapter",
    ]
    assert main([*base, "--contamination-regex", "VIO"]) == 2
    assert "requires --canaries" in capsys.readouterr().err
    assert main([*base, "--canaries", str(canaries), "--contamination-regex", "["]) == 2
    assert "invalid --contamination-regex" in capsys.readouterr().err


def test_empty_canary_file_exits_before_model_load(tmp_path, capsys):
    prompts = tmp_path / "questions.txt"
    prompts.write_text("What is the code?\n")
    canaries = tmp_path / "canaries.txt"
    canaries.write_text("\n")
    assert main([
        "--model", "m", "--train", str(prompts),
        "--teacher-system", "context", "--adapter-path", "adapter",
        "--canaries", str(canaries),
    ]) == 2
    assert "empty" in capsys.readouterr().err


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


def test_progress_requires_completion_or_probes(capsys):
    assert main(["--model", "m", "--progress", "run"]) == 2
    assert "exactly one" in capsys.readouterr().err


def test_progress_rejects_completion_and_probes(capsys):
    assert main([
        "--model", "m", "--progress", "run", "--completion", "x", "--probes", "p"
    ]) == 2
    assert "exactly one" in capsys.readouterr().err


def test_probes_require_progress(capsys):
    assert main(["--model", "m", "--completion", "x", "--probes", "p.jsonl"]) == 2
    assert "only combines with --progress" in capsys.readouterr().err


def test_progress_validates_probe_file_before_model_load(tmp_path, capsys):
    probes = tmp_path / "probes.jsonl"
    probes.write_text("\n")
    assert main([
        "--model", "m", "--progress", "run", "--probes", str(probes)
    ]) == 2
    assert "empty" in capsys.readouterr().err


def test_progress_rejects_other_mode_flags(tmp_path):
    probes = tmp_path / "probes.jsonl"
    probes.write_text('{"name":"x","prompt":"p","completion":"c"}\n')
    base = ["--model", "m", "--progress", "run", "--probes", str(probes)]
    for flag in (["--adapter-path", "a"], ["--system", "s"], ["--metric", "prob"]):
        assert main([*base, *flag]) == 2


def test_progress_accepts_prompt_completion_and_suppresses_download_bars(
    monkeypatch, capsys
):
    import huggingface_hub.utils
    import mlx_lm
    import distil.progress

    disabled = []
    monkeypatch.setattr(
        huggingface_hub.utils, "disable_progress_bars", lambda: disabled.append(True)
    )
    monkeypatch.setattr(mlx_lm, "load", lambda *args, **kwargs: (object(), object()))

    def fake_score(model, tokenizer, root, probes, **kwargs):
        point = distil.progress.ProbePoint(0, -1.0, 0.37, -1.0, 0.37, [])
        return distil.progress.ProgressResult(str(root), [0], probes, {"target": [point]})

    monkeypatch.setattr(distil.progress, "score_progress", fake_score)
    assert main([
        "--model", "m", "--progress", "run", "--prompt", "Code?",
        "--completion", "VIOLET-731", "--json",
    ]) == 0
    assert disabled == [True]
    assert '"completion": "VIOLET-731"' in capsys.readouterr().out


def test_bad_rollout_path_exits_before_model_load(capsys):
    assert main(["--model", "m", "--rollout", "outputs/evals/.../results.jsonl"]) == 2
    assert "rollout pattern matched no files" in capsys.readouterr().err
