"""Inline inputs and truthful live/replayed checkpoint displays."""

import io
import json
import math

import pytest
from rich.console import Console

from distill.cli import main
from distill.cli_config import build_parser, resolve_config
from distill.progress import (
    Probe,
    ProbePoint,
    ProgressResult,
    evaluation_transition,
    recorded_progress,
    progress_to_json,
)
from distill.render import render_checkpoint_grid


def test_inline_fact_requires_holdout_and_preserves_questions():
    argv = [
        "--model",
        "m",
        "--train",
        "The code is VIOLET-731.",
        "--adapter",
        "run",
        "--prompt",
        "Code?",
        "--eval-prompt",
        "Which code?",
        "--completion",
        "VIOLET-731",
        "--canary",
        "Grass color?",
        "--speculative-top-p",
        "0.9",
    ]
    args = build_parser().parse_args(argv)
    config = resolve_config(args, argv)
    assert config.mode == "train" and config.training.speculative_top_p == 0.9
    assert args.canary == ["Grass color?"]
    args.eval_prompt = None
    with pytest.raises(ValueError, match="--eval-prompt"):
        resolve_config(args, argv)
    args.eval_prompt = args.prompt
    with pytest.raises(ValueError, match="must differ"):
        resolve_config(args, argv)


def test_inline_cli_bypasses_generator_and_persists_inputs(
    tmp_path, monkeypatch, capsys
):
    import mlx_lm
    import distill.distill as controller
    import distill.generation as generation
    from distill.stopping import load_learning_items
    from distill.progress import load_probes

    monkeypatch.setattr(mlx_lm, "load", lambda *a, **kw: (object(), object()))
    monkeypatch.setattr(generation, "ensure_fits", lambda *a, **kw: None)
    monkeypatch.setattr(
        generation,
        "generate_facts",
        lambda *a, **kw: pytest.fail("must not generate inline questions"),
    )
    seen = {}

    def run(model, tokenizer, items, canaries, context, root, **options):
        seen.update(items=items, canaries=canaries, options=options)
        return controller.ContinualResult(
            True, "mastered", 2, None, root / "selection.json"
        )

    # The CLI accepts paths as strings, as well as Path in the library.
    def wrapped(*args, **kwargs):
        from pathlib import Path

        args = list(args)
        args[5] = Path(args[5])
        return run(*args, **kwargs)

    monkeypatch.setattr(controller, "run_continual", wrapped)
    root = tmp_path / "run"
    assert (
        main(
            [
                "--model",
                "m",
                "--train",
                "Code is VIOLET-731.",
                "--adapter",
                str(root),
                "--prompt",
                "Code?",
                "--eval-prompt",
                "Which code?",
                "--completion",
                "VIOLET-731",
                "--canary",
                "Grass color?",
                "--json",
            ]
        )
        == 0
    )
    assert seen["items"][0].train_prompts == ("Code?",)
    assert seen["items"][0].prompt == "Which code?"
    assert seen["canaries"][0].dynamic and seen["canaries"][0].completion == ""
    assert seen["options"]["capture_progress"]
    assert load_learning_items(root / "generated/probes.jsonl") == seen["items"]
    assert load_probes(root / "generated/canaries.jsonl") == seen["canaries"]
    assert json.loads(capsys.readouterr().out)["success"]


def test_grid_keeps_exact_token_colors_and_optionally_hides_specials():
    points = [
        ProbePoint(
            0,
            -1,
            math.exp(-1),
            -2,
            math.exp(-2),
            [
                {"text": "Green", "logprob": -0.1, "special": False},
                {"text": "<|im_end|>", "logprob": -1.9, "special": True},
            ],
            completion="Green<|im_end|>",
            content="Green",
            label="base",
        )
    ]
    result = ProgressResult(
        "run", [0], [Probe("grass", "Color?", "", dynamic=True)], {"grass": points}
    )
    output = io.StringIO()
    console = Console(
        file=output, width=120, force_terminal=True, color_system="truecolor"
    )
    render_checkpoint_grid(result, console)
    text = output.getvalue()
    assert "<|im_end|>" not in text and "p=36.8%" in text
    assert "grass · generated" in text and "Green" in text
    assert "\x1b[" in text
    output.seek(0)
    output.truncate()
    render_checkpoint_grid(result, console, show_special_tokens=True)
    assert "<|im_end|>" in output.getvalue()
    assert points[0].tokens[1]["text"] == "<|im_end|>"


def test_transitions_do_not_call_punctuation_drift_leak_or_probability_mastery():
    base = {"iteration": 0, "mastered": 0, "total": 1, "canary_changed": 0}
    high_probability = dict(base, iteration=1, probability_mastered=1)
    changed = dict(
        high_probability,
        iteration=2,
        canary_changed=1,
        canaries=[{"content": "Green.", "contaminated": False}],
    )
    assert evaluation_transition(base) == "base"
    assert evaluation_transition(high_probability, base) is None
    assert evaluation_transition(changed, high_probability) == "changed"
    assert (
        evaluation_transition(
            dict(changed, iteration=3, canaries=[{"contaminated": True}]), changed
        )
        == "leak"
    )
    assert (
        evaluation_transition(dict(high_probability, iteration=4), changed)
        == "recovered"
    )
    assert evaluation_transition(dict(high_probability, mastered=1), base) == "learned"


def test_recorded_replay_never_loads_model(tmp_path, monkeypatch, capsys):
    import mlx_lm

    point = ProbePoint(
        0,
        -1,
        math.exp(-1),
        -1,
        math.exp(-1),
        [{"text": "V", "logprob": -1}],
        label="base",
    )
    result = ProgressResult(
        str(tmp_path), [0], [Probe("target", "Code?", "V")], {"target": [point]}
    )
    (tmp_path / "eval_metrics.jsonl").write_text(
        json.dumps({"progress": progress_to_json(result)}) + "\n"
    )
    monkeypatch.setattr(
        mlx_lm, "load", lambda *a, **kw: pytest.fail("replay must not load model")
    )
    assert recorded_progress(tmp_path).series["target"][0].tokens == point.tokens
    assert main(["--model", "m", "--progress", str(tmp_path), "--color", "never"]) == 0
    assert "target · scored" in capsys.readouterr().out


def test_training_records_actual_canary_tokens_and_keeps_locality_stop(
    tmp_path, monkeypatch
):
    import distill.distill as controller
    from distill.stopping import LearningItem
    from distill.training import TrainStep

    class Model:
        def eval(self):
            pass

        def named_modules(self):
            return []

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        def save(self, path):
            path.mkdir(parents=True)
            (path / "adapter_config.json").write_text("{}")
            (path / "adapters.safetensors").write_bytes(b"")

        def step_batch(self, prompts, **kwargs):
            return [
                TrainStep(1, prompts[0], 0.1, 1, "stop", "VIOLET-731", batch_loss=0.1)
            ]

    monkeypatch.setattr(controller, "TrainingSession", Session)
    monkeypatch.setattr(
        controller,
        "_answer",
        lambda m, t, p, **kw: "Green" if p.name == "grass" else "wrong",
    )
    monkeypatch.setattr(
        controller,
        "judge_answer",
        lambda m, t, p, expected, actual, **kw: expected == actual,
    )
    monkeypatch.setattr(
        controller,
        "score_probe",
        lambda m, t, p, step, *a, **kw: ProbePoint(
            step,
            -0.1,
            math.exp(-0.1),
            -0.1,
            math.exp(-0.1),
            [{"text": p.completion, "logprob": -0.1}],
        ),
    )
    calls = []

    def sample(m, t, probe, step, *args, **kwargs):
        calls.append((step, kwargs["max_tokens"]))
        content = "Green" if step == 0 else "Green."
        tokens = [
            {"text": content, "logprob": -0.4},
            {"text": "<|im_end|>", "logprob": -0.6, "special": True},
        ]
        return ProbePoint(
            step,
            -0.5,
            math.exp(-0.5),
            -1,
            math.exp(-1),
            tokens,
            completion=content + "<|im_end|>",
            content=content,
            num_tokens=2,
            finish_reason="stop",
        )

    monkeypatch.setattr(controller, "sample_probe", sample)
    result = controller.run_continual(
        Model(),
        object(),
        [LearningItem("target", "Heldout?", "VIOLET-731", ("Train?",))],
        [Probe("grass", "Grass?", "", dynamic=True)],
        "Reference",
        tmp_path / "run",
        model_name="m",
        iters=1,
        eval_every=1,
        capture_progress=True,
        canary_max_tokens=7,
    )
    assert (
        not result.success
        and result.stop_reason == "locality"
        and result.selected_step == 0
    )
    assert calls == [(0, 7), (1, 7)]
    replay = recorded_progress(tmp_path / "run")
    assert replay.series["grass"][1].content == "Green."
    assert replay.series["grass"][1].tokens[0]["text"] == "Green."
    assert replay.series["grass"][1].label == "changed"
    assert replay.series["target"][1].tokens[0]["text"] == "VIOLET-731"
    assert (
        replay.series["target"][1].label != "learned"
    )  # High probability did not pass recall.
