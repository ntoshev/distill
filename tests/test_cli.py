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


def test_speculative_sampling_requires_teacher(capsys):
    assert main([
        "--model", "m", "--sample", "--prompt", "p", "--speculative-top-k", "5"
    ]) == 2
    assert "requires a --teacher-* flag" in capsys.readouterr().err


def test_speculative_top_k_requires_sample_or_train(capsys):
    assert main([
        "--model", "m", "--completion", "x", "--speculative-top-k", "5"
    ]) == 2
    assert "only combines with --sample or --train" in capsys.readouterr().err


def test_speculative_sampling_dispatches_to_interleaved_policy(monkeypatch, capsys):
    import mlx_lm
    import distil.cli as cli
    from distil.sample import Sample

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "teacher prompt"

    model = object()
    tokenizer = Tokenizer()
    monkeypatch.setattr(mlx_lm, "load", lambda *args, **kwargs: (model, tokenizer))
    captured = {}

    def fake_speculative(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return Sample(
            prompt_text="student prompt",
            completion_text="VIOLET-731",
            content="VIOLET-731",
            finish_reason="stop",
            num_tokens=3,
            speculative_top_k=5,
            accepted_tokens=2,
            replaced_tokens=1,
        )

    monkeypatch.setattr(cli, "sample_speculative_completion", fake_speculative)
    monkeypatch.setattr(cli, "score_text_pair", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "to_json_dict", lambda result: {"tokens": []})

    assert main([
        "--model", "m",
        "--sample",
        "--prompt", "What is Project Lumen's launch code?",
        "--teacher-system", "The code is VIOLET-731.",
        "--speculative-top-k", "5",
        "--max-tokens", "16",
        "--temperature", "0",
        "--seed", "0",
        "--no-thinking",
        "--json",
    ]) == 0
    assert captured["kwargs"]["top_k"] == 5
    assert captured["kwargs"]["max_tokens"] == 16
    assert captured["args"][5][0]["content"] == "The code is VIOLET-731."
    output = capsys.readouterr()
    assert "2 accepted, 1 replaced" in output.err
    assert '"accepted_tokens": 2' in output.out


def test_training_requires_adapter_and_canaries(capsys):
    assert main(["--model", "m", "--train", "context"]) == 2
    assert "--adapter" in capsys.readouterr().err
    assert main(["--model", "m", "--train", "context", "--adapter", "run"]) == 2
    assert "--canaries" in capsys.readouterr().err


def test_training_rejects_prompt_and_teacher_prompts(tmp_path, capsys):
    base = [
        "--model", "m", "--train", "context", "--adapter", "adapter",
        "--canaries", str(tmp_path / "canaries.jsonl"),
    ]
    assert main([*base, "--prompt", "q"]) == 2
    assert main([*base, "--teacher-prompt", "q"]) == 2
    assert main([*base, "--teacher-system", "q"]) == 2


def test_training_flags_require_train(capsys):
    for flag in (
        ["--iters", "1"],
        ["--learning-rate", "1e-4"],
        ["--lora-rank", "4"],
        ["--checkpoint-every", "1"],
        ["--canaries", "canaries.txt"],
        ["--canary-max-tokens", "8"],
        ["--contamination-regex", "VIO"],
        ["--kl-direction", "reverse"],
        ["--from-adapter", "old"],
        ["--generator-model", "g"],
        ["--questions-per-fact", "2"],
        ["--generate-only"],
        ["--eval-every", "5"],
        ["--learned-threshold", "0.5"],
        ["--adapter", "run"],
    ):
        assert main(["--model", "m", "--completion", "x", *flag]) == 2


def test_training_rejects_nonpositive_settings(tmp_path):
    base = [
        "--model", "m", "--train", "context", "--adapter", "adapter",
        "--canaries", str(tmp_path / "canaries.jsonl"),
    ]
    for flag in (
        ["--iters", "0"],
        ["--learning-rate", "0"],
        ["--lora-rank", "0"],
        ["--checkpoint-every", "0"],
        ["--canary-max-tokens", "0"],
        ["--speculative-top-k", "0"],
        ["--eval-every", "0"],
        ["--learned-threshold", "0"],
    ):
        assert main([*base, *flag]) == 2


def test_speculative_top_k_rejects_nonpositive_sample_value(capsys):
    assert main([
        "--model", "m", "--sample", "--prompt", "p",
        "--teacher-system", "t", "--speculative-top-k", "0",
    ]) == 2
    assert "must be positive" in capsys.readouterr().err


def test_training_validates_continual_options_before_model_load(tmp_path, capsys):
    canaries = tmp_path / "canaries.jsonl"
    canaries.write_text('{"name":"math","prompt":"2+2?","completion":"4"}\n')
    base = [
        "--model", "m", "--train", "context", "--adapter", "adapter",
        "--canaries", str(canaries),
    ]
    assert main([*base, "--questions-per-fact", "1"]) == 2
    assert "at least 2" in capsys.readouterr().err
    assert main([*base, "--from-adapter", "adapter"]) == 2
    assert "must differ" in capsys.readouterr().err


def test_empty_canary_file_exits_before_model_load(tmp_path, capsys):
    canaries = tmp_path / "canaries.jsonl"
    canaries.write_text("\n")
    assert main([
        "--model", "m", "--train", "context", "--adapter", "adapter",
        "--canaries", str(canaries),
    ]) == 2
    assert "empty" in capsys.readouterr().err


def test_empty_context_file_exits_before_model_load(tmp_path, capsys):
    context = tmp_path / "empty.txt"
    context.write_text("\n")
    canaries = tmp_path / "canaries.jsonl"
    canaries.write_text('{"name":"math","prompt":"2+2?","completion":"4"}\n')
    assert main([
        "--model", "m", "--train", f"@{context}", "--adapter", "adapter",
        "--canaries", str(canaries),
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
    assert "only combines with --progress or --train" in capsys.readouterr().err


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
