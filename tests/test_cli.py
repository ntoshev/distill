"""CLI argument validation. These run before any model load, so no model needed."""

import json

import pytest

from distill.cli import DEFAULT_JUDGE_MODEL, DEFAULT_TEMPERATURE, main


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


def test_index_requires_rollout(capsys):
    assert main(["--model", "m", "--completion", "x", "--index", "2"]) == 2
    assert "only combines with --rollout" in capsys.readouterr().err


def test_sampling_flags_require_sample(capsys):
    for flag in (["--seed", "1"], ["--temperature", "0.7"], ["--max-tokens", "8"]):
        assert main(["--model", "m", "--completion", "x", *flag]) == 2


def test_system_only_sample_and_raw_sample_use_shared_sampling_paths(monkeypatch, capsys):
    import mlx_lm
    import distill.cli as cli
    from distill.sample import Sample

    monkeypatch.setattr(mlx_lm, "load", lambda *args, **kwargs: (object(), object()))
    calls = []

    def sampled(kind, prefix):
        calls.append((kind, prefix))
        return Sample("rendered" if kind == "chat" else prefix, "answer", "answer", "stop", 1)

    monkeypatch.setattr(
        cli, "sample_completion", lambda model, tokenizer, messages, **kwargs:
        sampled("chat", messages)
    )
    monkeypatch.setattr(
        cli, "sample_text", lambda model, tokenizer, prompt, **kwargs:
        sampled("raw", prompt)
    )
    monkeypatch.setattr(cli, "score_text", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "to_json_dict", lambda result: {"tokens": []})

    assert main(["--model", "m", "--sample", "--system", "Be terse", "--json"]) == 0
    assert main([
        "--model", "m", "--raw", "--sample", "--prompt", "1 2 3", "--json"
    ]) == 0
    assert calls == [
        ("chat", [
            {"role": "system", "content": "Be terse"},
            {"role": "user", "content": ""},
        ]),
        ("raw", "1 2 3"),
    ]
    capsys.readouterr()


def test_raw_teacher_prompt_scores_same_completion_under_new_prefix(monkeypatch, capsys):
    import mlx_lm
    import distill.cli as cli
    from distill.sample import Sample

    monkeypatch.setattr(mlx_lm, "load", lambda *args, **kwargs: (object(), object()))
    monkeypatch.setattr(
        cli,
        "sample_text",
        lambda *args, **kwargs: Sample("student: ", "answer", "answer", "stop", 1),
    )
    captured = {}

    def fake_pair(*args, **kwargs):
        captured["student"] = args[2]
        captured["teacher"] = args[5]
        return object()

    monkeypatch.setattr(cli, "score_text_pair", fake_pair)
    monkeypatch.setattr(cli, "to_json_dict", lambda result: {"tokens": []})
    assert main([
        "--model", "m", "--raw", "--sample", "--prompt", "student: ",
        "--teacher-prompt", "teacher: ", "--json",
    ]) == 0
    assert captured["student"].text == "student: answer"
    assert captured["teacher"].text == "teacher: answer"
    capsys.readouterr()


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
    import distill.cli as cli
    from distill.sample import Sample

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
            teacher_sampled=[False, True, False],
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
    assert "teacher replacements underlined" in output.err
    assert '"accepted_tokens": 2' in output.out


def test_training_requires_adapter_and_canaries(capsys):
    assert main(["--model", "m", "--train", "context"]) == 2
    assert "--adapter" in capsys.readouterr().err
    assert main(["--model", "m", "--train", "context", "--adapter", "run"]) == 2
    assert "--canaries" in capsys.readouterr().err


def test_training_accepts_json_rollouts_and_canary_limit(monkeypatch):
    import distill.cli as cli

    captured = {}
    monkeypatch.setattr(
        cli,
        "_continual_cli",
        lambda args, errcon, fail: captured.update(vars(args)) or 0,
    )
    assert main([
        "--model", "m", "--train", "context", "--adapter", "run",
        "--canaries", "canaries.jsonl", "--canary-max-tokens", "9",
        "--save-rollout", "rollouts.jsonl", "--json",
    ]) == 0
    assert captured["canary_max_tokens"] == 9
    assert captured["batch_size"] is None
    assert captured["save_rollout"] == "rollouts.jsonl"
    assert captured["json"] is True
    assert captured["judge_model"] == DEFAULT_JUDGE_MODEL


@pytest.mark.parametrize("top_p", [None, 0.9])
@pytest.mark.parametrize("mastery_patience", [None, 2])
def test_training_json_reports_result_and_forwards_rollout_options(
    tmp_path, monkeypatch, capsys, top_p, mastery_patience
):
    import mlx_lm
    import distill.distill

    probes = tmp_path / "probes.jsonl"
    probes.write_text(json.dumps({
        "name": "fact",
        "prompt": "heldout",
        "completion": "answer",
        "train_prompts": ["train"],
    }) + "\n")
    canaries = tmp_path / "canaries.jsonl"
    canaries.write_text('{"name":"math","prompt":"2+2?","completion":"4"}\n')
    monkeypatch.setattr(mlx_lm, "load", lambda *args, **kwargs: (object(), object()))
    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        root = tmp_path / "adapter"
        return distill.distill.ContinualResult(
            True, "mastered", 3, root / "adapters.safetensors", root / "selection.json"
        )

    monkeypatch.setattr(distill.distill, "run_continual", fake_run)
    rollouts = tmp_path / "rollouts.jsonl"
    assert main([
        "--model", "m", "--train", "context", "--adapter", str(tmp_path / "adapter"),
        "--probes", str(probes), "--canaries", str(canaries),
        "--canary-max-tokens", "9", "--save-rollout", str(rollouts),
        "--judge-model", "judge", "--json",
        *([] if top_p is None else ["--speculative-top-p", str(top_p)]),
        *([] if mastery_patience is None else ["--mastery-confirmations", str(mastery_patience)]),
    ]) == 0
    assert captured["canary_max_tokens"] == 9
    assert captured["batch_size"] == 4
    assert captured["mastery_patience"] == (4 if mastery_patience is None else mastery_patience)
    assert captured["save_rollout_path"] == str(rollouts)
    assert captured["speculative_top_k"] == (3 if top_p is None else None)
    assert captured["speculative_top_p"] == top_p
    assert captured["learning_rate"] == 1e-6
    assert captured["temperature"] == DEFAULT_TEMPERATURE
    assert captured["max_tokens"] == 64
    assert captured["seed"] == 0
    assert captured["judge_model_name"] == "judge"
    assert captured["judge_model"] is not None
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["selected_step"] == 3


def test_continuation_shares_tokenizer_with_separate_frozen_teacher(tmp_path, monkeypatch):
    import mlx_lm
    import distill.distill

    starting_adapter = tmp_path / "starting"
    output = tmp_path / "continued"
    output.mkdir()
    (output / "selection.json").write_text('{"stopped": "success"}')
    loaded = []

    def load(model_name, *, adapter_path):
        assert model_name == "m"
        assert adapter_path == str(starting_adapter)
        pair = (object(), object())
        loaded.append(pair)
        return pair

    def run(model, tokenizer, *args, **kwargs):
        assert len(loaded) == 2
        assert (model, tokenizer) == loaded[0]
        assert kwargs["teacher_model"] is loaded[1][0]
        assert kwargs["teacher_model"] is not model
        # Exact generated IDs can be aligned across the two weight copies.
        assert kwargs["teacher_tokenizer"] is tokenizer
        assert kwargs["from_adapter"] == str(starting_adapter)
        return distill.distill.ContinualResult(
            True, "mastered", 6, output / "adapters.safetensors", output / "selection.json"
        )

    monkeypatch.setattr(mlx_lm, "load", load)
    monkeypatch.setattr(distill.distill, "run_continual", run)
    assert main([
        "--model", "m", "--train", "private context",
        "--prompt", "training question", "--eval-prompt", "held-out question",
        "--completion", "answer", "--canary", "2+2?",
        "--from-adapter", str(starting_adapter), "--adapter", str(output),
        "--judge-model", "m", "--speculative-top-p", "0.9", "--json",
    ]) == 0


def test_generate_only_preserves_distinct_generator_limits(tmp_path, monkeypatch, capsys):
    import mlx_lm
    import distill.cli as cli
    import distill.generation as generation
    from distill.sample import Sample

    monkeypatch.setattr(mlx_lm, "load", lambda *args, **kwargs: (object(), object()))
    calls = []

    def sample(model, tokenizer, messages, **kwargs):
        calls.append((messages, kwargs))
        return Sample("prefix", "generated", "generated", "stop", 1)

    def facts(document, count, inventory, questions):
        assert (document, count) == ("context", 2)
        assert inventory("inventory") == questions("questions") == "generated"
        return [generation.GeneratedFact(
            "fact-0001", "fact", completion="answer", train_prompts=["train"], prompt="heldout",
        )]

    monkeypatch.setattr(cli, "sample_completion", sample)
    monkeypatch.setattr(generation, "generate_facts", facts)
    assert main([
        "--model", "m", "--train", "context", "--adapter", str(tmp_path),
        "--generate-only", "--max-tokens", "16", "--json",
    ]) == 0
    assert [kwargs["max_tokens"] for _, kwargs in calls] == [512, 256]
    assert [kwargs["enable_thinking"] for _, kwargs in calls] == [None, False]
    assert all(kwargs["temperature"] == 0.0 and kwargs["seed"] == 0 for _, kwargs in calls)
    assert "generated_probes" in json.loads(capsys.readouterr().out)


def test_raw_speculative_dispatch_preserves_teacher_and_sampling_options(monkeypatch, capsys):
    import mlx_lm
    import distill.cli as cli
    from distill.sample import Sample

    model, tokenizer = object(), object()
    monkeypatch.setattr(mlx_lm, "load", lambda *args, **kwargs: (model, tokenizer))

    def sample(*args, **kwargs):
        assert args == (model, tokenizer, "student>", model, tokenizer, "teacher>")
        assert kwargs == {"max_tokens": 7, "temperature": 0.0, "seed": 3,
                          "top_k": 2, "teacher_without_adapter": False}
        return Sample("student>", "answer", "answer", "stop", 1, 2, 1, 0)

    monkeypatch.setattr(cli, "sample_speculative_text", sample)
    monkeypatch.setattr(cli, "score_text_pair", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "to_json_dict", lambda result: {})
    assert main([
        "--model", "m", "--raw", "--sample", "--prompt", "student>",
        "--teacher-prompt", "teacher>", "--speculative-top-k", "2",
        "--max-tokens", "7", "--temperature", "0", "--seed", "3", "--json",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["sample"]["accepted_tokens"] == 1


def test_training_reuses_student_as_judge_when_model_names_match(
    tmp_path, monkeypatch
):
    import mlx_lm
    import distill.distill

    probes = tmp_path / "probes.jsonl"
    probes.write_text(json.dumps({
        "name": "fact",
        "prompt": "heldout",
        "completion": "answer",
        "train_prompts": ["train"],
    }) + "\n")
    canaries = tmp_path / "canaries.jsonl"
    canaries.write_text('{"name":"math","prompt":"2+2?","completion":"4"}\n')
    student = object()
    tokenizer = object()
    loads = []

    def fake_load(*args, **kwargs):
        loads.append((args, kwargs))
        return student, tokenizer

    monkeypatch.setattr(mlx_lm, "load", fake_load)
    captured = {}

    def fake_run(*args, **kwargs):
        captured["student"] = args[0]
        captured["tokenizer"] = args[1]
        captured.update(kwargs)
        root = tmp_path / "adapter"
        return distill.distill.ContinualResult(
            True, "mastered", 3, root / "adapters.safetensors", root / "selection.json"
        )

    monkeypatch.setattr(distill.distill, "run_continual", fake_run)
    assert main([
        "--model", "same-model", "--train", "context", "--adapter", str(tmp_path / "adapter"),
        "--probes", str(probes), "--canaries", str(canaries),
        "--judge-model", "same-model", "--json",
    ]) == 0

    assert loads == [(('same-model',), {"adapter_path": None})]
    assert captured["judge_model"] is student
    assert captured["judge_tokenizer"] is tokenizer


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
        ["--batch-size", "4"],
        ["--learning-rate", "1e-4"],
        ["--lora-rank", "4"],
        ["--checkpoint-every", "1"],
        ["--canaries", "canaries.txt"],
        ["--canary-max-tokens", "8"],
        ["--contamination-regex", "VIO"],
        ["--kl-direction", "reverse"],
        ["--from-adapter", "old"],
        ["--generator-model", "g"],
        ["--judge-model", "j"],
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
        ["--batch-size", "0"],
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
    assert "exactly 2" in capsys.readouterr().err
    assert main([*base, "--questions-per-fact", "3"]) == 2
    assert "exactly 2" in capsys.readouterr().err
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


def test_progress_prompt_requires_progress(capsys):
    assert main([
        "--model", "m", "--completion", "x", "--progress-prompt", "test"
    ]) == 2
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
    for flag in (["--adapter-path", "a"],):
        assert main([*base, *flag]) == 2


def test_progress_prompt_train_requires_probes(capsys):
    assert main([
        "--model", "m", "--progress", "run", "--prompt", "p",
        "--completion", "c", "--progress-prompt", "train",
    ]) == 2
    assert "requires --probes" in capsys.readouterr().err


def test_progress_prompt_train_expands_probe_train_prompts(tmp_path, monkeypatch, capsys):
    import huggingface_hub.utils
    import mlx_lm
    import distill.progress

    probes = tmp_path / "probes.jsonl"
    probes.write_text(json.dumps({
        "name": "fact",
        "prompt": "Heldout?",
        "completion": "VIOLET-731",
        "train_prompts": ["Train one?", "Train two?"],
    }) + "\n")
    monkeypatch.setattr(huggingface_hub.utils, "disable_progress_bars", lambda: None)
    monkeypatch.setattr(mlx_lm, "load", lambda *args, **kwargs: (object(), object()))
    captured = {}

    def fake_score_progress(model, tokenizer, root, probes, **kwargs):
        captured["probes"] = probes
        result = distill.progress.ProgressResult(
            str(root), [0], probes, {probe.name: [] for probe in probes}
        )
        kwargs["on_start"](result)
        return result

    monkeypatch.setattr(distill.progress, "score_progress", fake_score_progress)
    assert main([
        "--model", "m", "--progress", "run", "--probes", str(probes),
        "--progress-prompt", "train",
    ]) == 0
    assert [probe.name for probe in captured["probes"]] == ["fact/train-1", "fact/train-2"]
    assert captured["probes"][0].prompt == "Train one?"


def test_progress_accepts_prompt_completion_and_suppresses_download_bars(
    monkeypatch, capsys
):
    import huggingface_hub.utils
    import mlx_lm
    import distill.progress

    disabled = []
    monkeypatch.setattr(
        huggingface_hub.utils, "disable_progress_bars", lambda: disabled.append(True)
    )
    monkeypatch.setattr(mlx_lm, "load", lambda *args, **kwargs: (object(), object()))

    def fake_score(model, tokenizer, root, probes, **kwargs):
        point = distill.progress.ProbePoint(0, -1.0, 0.37, -1.0, 0.37, [])
        return distill.progress.ProgressResult(
            str(root), [0], probes, {"target": [point]}, teacher_model="m"
        )

    monkeypatch.setattr(distill.progress, "score_progress", fake_score)
    assert main([
        "--model", "m", "--progress", "run", "--prompt", "Code?",
        "--completion", "VIOLET-731", "--json",
    ]) == 0
    assert disabled == [True]
    assert '"completion": "VIOLET-731"' in capsys.readouterr().out


def test_progress_sample_forwards_sampling_defaults(monkeypatch, capsys):
    import huggingface_hub.utils
    import mlx_lm
    import distill.progress

    monkeypatch.setattr(huggingface_hub.utils, "disable_progress_bars", lambda: None)
    monkeypatch.setattr(mlx_lm, "load", lambda *args, **kwargs: (object(), object()))
    captured = {}

    def fake_sample_progress(model, tokenizer, root, probes, **kwargs):
        captured["root"] = root
        captured["probes"] = probes
        captured.update(kwargs)
        point = distill.progress.ProbePoint(
            0, -1.0, 0.37, -1.0, 0.37, [],
            completion="sampled", content="sampled", finish_reason="stop", num_tokens=1
        )
        return distill.progress.ProgressResult(str(root), [0], probes, {"sample": [point]})

    monkeypatch.setattr(distill.progress, "sample_progress", fake_sample_progress)
    assert main([
        "--model", "m", "--progress", "run", "--prompt", "When?",
        "--sample", "--metric", "prob", "--no-thinking", "--json",
    ]) == 0
    assert captured["root"] == "run"
    assert captured["probes"][0] == distill.progress.Probe("sample", "When?", "")
    assert captured["temperature"] == DEFAULT_TEMPERATURE
    assert captured["max_tokens"] == 512
    assert captured["enable_thinking"] is False
    payload = json.loads(capsys.readouterr().out)
    assert payload["series"]["sample"][0]["completion"] == "sampled"


def test_progress_rollout_replays_saved_prompt_and_scores_displayed_completion(
    tmp_path, monkeypatch, capsys
):
    import huggingface_hub.utils
    import mlx_lm
    import distill.progress

    monkeypatch.setattr(huggingface_hub.utils, "disable_progress_bars", lambda: None)
    monkeypatch.setattr(mlx_lm, "load", lambda *args, **kwargs: (object(), object()))
    captured = {}
    rollout_path = tmp_path / "rollouts.jsonl"
    rollout_path.write_text(json.dumps({
        "prompt": [
            {"role": "system", "content": "Saved system."},
            {"role": "user", "content": "Saved question?"},
        ],
        "completion": [{"role": "assistant", "content": "Saved answer."}],
    }) + "\n")

    def fake_sample_progress(model, tokenizer, root, probes, **kwargs):
        captured["root"] = root
        captured["probes"] = probes
        captured.update(kwargs)
        point = distill.progress.ProbePoint(
            0,
            -0.2,
            0.82,
            -0.2,
            0.82,
            [{"text": "fresh", "logprob": -0.2, "probability": 0.82}],
            completion="fresh",
            content="fresh",
            finish_reason="stop",
            num_tokens=1,
        )
        result = distill.progress.ProgressResult(
            str(root), [0], probes, {probes[0].name: []}
        )
        kwargs["on_start"](result)
        result.series[probes[0].name].append(point)
        kwargs["on_point"](probes[0], point)
        return result

    monkeypatch.setattr(distill.progress, "sample_progress", fake_sample_progress)

    assert main([
        "--model", "m", "--progress", "run", "--rollout", str(rollout_path),
        "--temperature", "0", "--max-tokens", "8", "--seed", "3", "--metric", "prob",
    ]) == 0
    assert captured["root"] == "run"
    assert captured["probes"] == [distill.progress.Probe(
        "rollout-0",
        "Saved question?",
        "Saved answer.",
        messages=(
            {"role": "system", "content": "Saved system."},
            {"role": "user", "content": "Saved question?"},
        ),
    )]
    assert captured["temperature"] == 0.0
    assert captured["max_tokens"] == 8
    assert captured["seed"] == 3
    output = capsys.readouterr().out
    assert "· generated" in output
    assert "fresh" in output
    assert "p=82.0%" in output


def test_progress_rejects_sample_and_rollout_together(tmp_path, capsys):
    rollout_path = tmp_path / "rollouts.jsonl"
    rollout_path.write_text('{"prompt": [], "completion": []}\n')
    assert main([
        "--model", "m", "--progress", "run", "--rollout", str(rollout_path),
        "--sample", "--prompt", "When?",
    ]) == 2
    assert "does not combine" in capsys.readouterr().err


def test_progress_rollout_rejects_prompt_override(tmp_path, capsys):
    rollout_path = tmp_path / "rollouts.jsonl"
    rollout_path.write_text('{"prompt": [], "completion": []}\n')
    assert main([
        "--model", "m", "--progress", "run", "--rollout", str(rollout_path),
        "--prompt", "Not the saved prompt",
    ]) == 2
    assert "replays the saved prompt" in capsys.readouterr().err


def test_progress_sample_can_sample_every_training_probe(tmp_path, monkeypatch, capsys):
    import huggingface_hub.utils
    import mlx_lm
    import distill.progress

    probes_path = tmp_path / "probes.jsonl"
    probes_path.write_text(
        json.dumps({
            "name": "fact",
            "prompt": "Held out?",
            "completion": "Short target",
            "train_prompts": ["Train one?", "Train two?"],
        }) + "\n"
    )
    monkeypatch.setattr(huggingface_hub.utils, "disable_progress_bars", lambda: None)
    monkeypatch.setattr(mlx_lm, "load", lambda *args, **kwargs: (object(), object()))
    captured = {}

    def fake_sample_progress(model, tokenizer, root, probes, **kwargs):
        captured["probes"] = probes
        points = {
            probe.name: [
                distill.progress.ProbePoint(
                    0, -1.0, 0.37, -1.0, 0.37, [],
                    completion="A full sampled sentence.",
                    content="A full sampled sentence.",
                    finish_reason="stop",
                    num_tokens=5,
                )
            ]
            for probe in probes
        }
        return distill.progress.ProgressResult(str(root), [0], probes, points)

    monkeypatch.setattr(distill.progress, "sample_progress", fake_sample_progress)
    assert main([
        "--model", "m", "--progress", "run", "--probes", str(probes_path),
        "--progress-prompt", "train", "--sample", "--json",
    ]) == 0
    assert captured["probes"] == [
        distill.progress.Probe("fact/train-1", "Train one?", "Short target"),
        distill.progress.Probe("fact/train-2", "Train two?", "Short target"),
    ]
    payload = json.loads(capsys.readouterr().out)
    assert payload["series"]["fact/train-1"][0]["completion"] == "A full sampled sentence."


def test_progress_sample_auto_uses_run_teacher_context(tmp_path, monkeypatch, capsys):
    import huggingface_hub.utils
    import mlx_lm
    import distill.progress

    source = tmp_path / "facts.txt"
    source.write_text("The answer is VIOLET-731.")
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_config.json").write_text(json.dumps({
        "source": {"source": "file", "path": str(source)}
    }))
    monkeypatch.setattr(huggingface_hub.utils, "disable_progress_bars", lambda: None)
    model = object()
    tokenizer = object()
    monkeypatch.setattr(mlx_lm, "load", lambda *args, **kwargs: (model, tokenizer))
    captured = {}

    def fake_sample_progress(model, tokenizer, root, probes, **kwargs):
        captured.update(kwargs)
        point = distill.progress.ProbePoint(
            0, -2.0, 0.14, -2.0, 0.14, [],
            mean_teacher_logprob=-0.1,
            mean_kl=0.2,
            completion="sampled",
            content="sampled",
            finish_reason="stop",
            num_tokens=1,
        )
        result = distill.progress.ProgressResult(
            str(root), [0], probes, {"sample": []}, teacher_model="m"
        )
        kwargs["on_start"](result)
        result.series["sample"].append(point)
        kwargs["on_point"](probes[0], point)
        return result

    monkeypatch.setattr(distill.progress, "sample_progress", fake_sample_progress)

    assert main([
        "--model", "m", "--progress", str(run), "--prompt", "When?",
        "--sample", "--metric", "prob", "--no-thinking",
    ]) == 0
    assert captured["teacher_model"] is model
    assert "Private reference document" in captured["teacher_system"]
    assert "The answer is VIOLET-731." in captured["teacher_system"]
    output = capsys.readouterr().out
    assert "· generated" in output
    assert "p=14.0%" in output


def test_progress_sample_defaults_to_student_probability_with_auto_teacher(tmp_path, monkeypatch, capsys):
    import huggingface_hub.utils
    import mlx_lm
    import distill.progress

    source = tmp_path / "facts.txt"
    source.write_text("The answer is VIOLET-731.")
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_config.json").write_text(json.dumps({
        "source": {"source": "file", "path": str(source)}
    }))
    monkeypatch.setattr(huggingface_hub.utils, "disable_progress_bars", lambda: None)
    model = object()
    tokenizer = object()
    monkeypatch.setattr(mlx_lm, "load", lambda *args, **kwargs: (model, tokenizer))

    def fake_sample_progress(model, tokenizer, root, probes, **kwargs):
        point = distill.progress.ProbePoint(
            0, -2.0, 0.14, -2.0, 0.14,
            [{"text": "sampled", "logprob": -2.0, "probability": 0.14, "kl": 0.2}],
            mean_teacher_logprob=-0.1,
            mean_kl=0.2,
            completion="sampled",
            content="sampled",
            finish_reason="stop",
            num_tokens=1,
        )
        result = distill.progress.ProgressResult(
            str(root), [0], probes, {"sample": []}, teacher_model="m"
        )
        kwargs["on_start"](result)
        result.series["sample"].append(point)
        kwargs["on_point"](probes[0], point)
        return result

    monkeypatch.setattr(distill.progress, "sample_progress", fake_sample_progress)

    assert main([
        "--model", "m", "--progress", str(run), "--prompt", "When?",
        "--sample", "--no-thinking",
    ]) == 0
    output = capsys.readouterr().out
    assert "sample · generated" in output
    assert "p=14.0%" in output
    assert "KL=" not in output


def test_progress_completion_auto_uses_run_teacher_context(tmp_path, monkeypatch, capsys):
    import huggingface_hub.utils
    import mlx_lm
    import distill.progress

    source = tmp_path / "facts.txt"
    source.write_text("The answer is VIOLET-731.")
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_config.json").write_text(json.dumps({
        "source": {"source": "file", "path": str(source)}
    }))
    monkeypatch.setattr(huggingface_hub.utils, "disable_progress_bars", lambda: None)
    model = object()
    tokenizer = object()
    monkeypatch.setattr(mlx_lm, "load", lambda *args, **kwargs: (model, tokenizer))
    captured = {}

    def fake_score_progress(model, tokenizer, root, probes, **kwargs):
        captured.update(kwargs)
        point = distill.progress.ProbePoint(
            0, -2.0, 0.14, -2.0, 0.14, [], mean_teacher_logprob=-0.1, mean_kl=0.2
        )
        return distill.progress.ProgressResult(
            str(root), [0], probes, {"target": [point]}, teacher_model="m"
        )

    monkeypatch.setattr(distill.progress, "score_progress", fake_score_progress)
    assert main([
        "--model", "m", "--progress", str(run), "--prompt", "When?",
        "--completion", "VIOLET-731", "--json",
    ]) == 0
    assert captured["teacher_model"] is model
    assert "Private reference document" in captured["teacher_system"]
    assert "The answer is VIOLET-731." in captured["teacher_system"]
    payload = json.loads(capsys.readouterr().out)
    assert payload["teacher_model"] == "m"


def test_progress_probes_stream_rows(monkeypatch, tmp_path, capsys):
    import huggingface_hub.utils
    import mlx_lm
    import distill.progress

    probes_path = tmp_path / "probes.jsonl"
    probes_path.write_text(
        '{"name":"a","prompt":"A?","completion":"one"}\n'
        '{"name":"b","prompt":"B?","completion":"two"}\n'
    )
    monkeypatch.setattr(huggingface_hub.utils, "disable_progress_bars", lambda: None)
    monkeypatch.setattr(mlx_lm, "load", lambda *args, **kwargs: (object(), object()))

    def fake_score_progress(model, tokenizer, root, probes, **kwargs):
        result = distill.progress.ProgressResult(
            str(root), [0], probes, {probe.name: [] for probe in probes}
        )
        kwargs["on_start"](result)
        for index, probe in enumerate(probes):
            point = distill.progress.ProbePoint(
                0, -1.0, 0.37, -1.0, 0.37,
                [{"text": str(index), "logprob": -1.0, "probability": 0.37}],
            )
            result.series[probe.name].append(point)
            kwargs["on_point"](probe, point)
        return result

    monkeypatch.setattr(distill.progress, "score_progress", fake_score_progress)

    assert main([
        "--model", "m", "--progress", "run", "--probes", str(probes_path)
    ]) == 0
    output = capsys.readouterr().out
    assert "· scored" in output
    assert "a" in output
    assert "b" in output
    assert "p=37.0%" in output


def test_progress_top_k_is_forwarded_and_emitted_in_json(monkeypatch, capsys):
    import huggingface_hub.utils
    import mlx_lm
    import distill.progress

    monkeypatch.setattr(huggingface_hub.utils, "disable_progress_bars", lambda: None)
    monkeypatch.setattr(mlx_lm, "load", lambda *args, **kwargs: (object(), object()))
    captured = {}

    def fake_score(model, tokenizer, root, probes, **kwargs):
        captured.update(kwargs)
        tokens = [{
            "text": "V",
            "logprob": -1.0,
            "probability": 0.37,
            "top": [["V", -1.0], ["A", -1.5]],
        }]
        point = distill.progress.ProbePoint(0, -1.0, 0.37, -1.0, 0.37, tokens)
        return distill.progress.ProgressResult(str(root), [0], probes, {"target": [point]})

    monkeypatch.setattr(distill.progress, "score_progress", fake_score)
    assert main([
        "--model", "m", "--progress", "run", "--prompt", "Code?",
        "--completion", "VIOLET-731", "--top-k", "2", "--json",
    ]) == 0
    assert captured["top_k"] == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["series"]["target"][0]["tokens"][0]["top"][1] == ["A", -1.5]


def test_progress_top_k_requires_json(capsys):
    assert main([
        "--model", "m", "--progress", "run", "--prompt", "Code?",
        "--completion", "VIOLET-731", "--top-k", "2",
    ]) == 2
    assert "requires --json" in capsys.readouterr().err


def test_progress_accepts_teacher_metric_and_default_probe_system(
    tmp_path, monkeypatch, capsys
):
    import huggingface_hub.utils
    import mlx_lm
    import distill.progress

    probes = tmp_path / "probes.jsonl"
    probes.write_text('{"name":"x","prompt":"p","completion":"c"}\n')
    monkeypatch.setattr(huggingface_hub.utils, "disable_progress_bars", lambda: None)
    model = object()
    tokenizer = object()
    monkeypatch.setattr(mlx_lm, "load", lambda *args, **kwargs: (model, tokenizer))
    captured = {}

    def fake_score(model, tokenizer, root, probes, **kwargs):
        captured.update(kwargs)
        point = distill.progress.ProbePoint(
            0, -1.0, 0.37, -1.0, 0.37, [], mean_kl=0.2
        )
        return distill.progress.ProgressResult(
            str(root), [0], probes, {"x": [point]}, teacher_model="m"
        )

    monkeypatch.setattr(distill.progress, "score_progress", fake_score)
    assert main([
        "--model", "m", "--progress", "run", "--probes", str(probes),
        "--system", "student default", "--teacher-system", "teacher context",
        "--metric", "kl", "--json",
    ]) == 0
    assert captured["default_system"] == "student default"
    assert captured["teacher_model"] is model
    assert captured["teacher_system"] == "teacher context"
    capsys.readouterr()


def test_rollout_accepts_student_system_and_prompt_overrides(
    tmp_path, monkeypatch, capsys
):
    import mlx_lm
    import distill.cli as cli
    from distill.templating import BuiltText

    path = tmp_path / "rollout.jsonl"
    path.write_text(json.dumps({
        "prompt": [
            {"role": "system", "content": "old system"},
            {"role": "user", "content": "old prompt"},
        ],
        "completion": [{"role": "assistant", "content": "answer"}],
    }) + "\n")
    monkeypatch.setattr(mlx_lm, "load", lambda *args, **kwargs: (object(), object()))
    captured = {}

    def fake_build(tokenizer, messages, **kwargs):
        captured["messages"] = messages
        return BuiltText("rendered", [(0, 8)])

    monkeypatch.setattr(cli, "build_rollout", fake_build)
    monkeypatch.setattr(cli, "score_text", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "to_json_dict", lambda result: {"tokens": []})

    assert main([
        "--model", "m", "--rollout", str(path),
        "--system", "new system", "--prompt", "new prompt", "--json",
    ]) == 0
    assert captured["messages"] == [
        {"role": "system", "content": "new system"},
        {"role": "user", "content": "new prompt"},
        {"role": "assistant", "content": "answer"},
    ]


def test_completion_can_be_saved_as_rollout(tmp_path, monkeypatch, capsys):
    import mlx_lm
    import distill.cli as cli
    from distill.rollout import load_rollout
    from distill.templating import BuiltText

    monkeypatch.setattr(mlx_lm, "load", lambda *args, **kwargs: (object(), object()))
    monkeypatch.setattr(cli, "build_rollout", lambda *args, **kwargs: BuiltText("x", [(0, 1)]))
    monkeypatch.setattr(cli, "score_text", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "to_json_dict", lambda result: {"tokens": []})
    path = tmp_path / "saved.jsonl"

    assert main([
        "--model", "m", "--prompt", "Question?", "--completion", "Answer.",
        "--save-rollout", str(path), "--json",
    ]) == 0
    rollout = load_rollout(path)
    assert rollout.prompt == [{"role": "user", "content": "Question?"}]
    assert rollout.completion == [{"role": "assistant", "content": "Answer."}]
    assert rollout.meta["task"] == "distill-completion"
    capsys.readouterr()


def test_bad_rollout_path_exits_before_model_load(capsys):
    assert main(["--model", "m", "--rollout", "outputs/evals/.../results.jsonl"]) == 2
    assert "rollout pattern matched no files" in capsys.readouterr().err
