import json
import mlx.core as mx
from cartridge_helpers import tiny_model
from distill.cartridge import model_identity
from distill.cli import main

SOURCE = "The launch code is violet. The river is quiet. The sky is blue.\n"


def bind(monkeypatch):
    def load(name, revision=None):
        model, tokenizer = tiny_model()
        return model, tokenizer, model_identity(model, tokenizer, name)

    monkeypatch.setattr("distill.cli._load_cartridge_base", load)


def test_memorize_complete_resume_and_raw_memory_progress(
    tmp_path, monkeypatch, capsys
):
    bind(monkeypatch)
    root = tmp_path / "run"
    common = ["--model", "tiny", "--json"]
    assert (
        main(
            [
                *common,
                "--memorize",
                SOURCE,
                "--cartridge",
                str(root),
                "--cartridge-positions",
                "4",
                "--key-tokens",
                "5",
                "--target-tokens",
                "8",
                "--stride",
                "8",
                "--iters",
                "1",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["selected_step"] == 1
    assert (
        main(
            [
                *common,
                "--cartridge-path",
                str(root),
                "--complete-key",
                "The launch ",
                "--max-tokens",
                "3",
                "--temperature",
                "0",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["sample"]["token_ids"]) == payload["sample"]["num_tokens"]
    assert main([*common, "--resume", str(root), "--iters", "2"]) == 0
    assert json.loads(capsys.readouterr().out)["selected_step"] == 2
    assert (
        main(
            [
                *common,
                "--progress",
                str(root),
                "--memory-keys",
                str(root / "generated/practiced_keys.jsonl"),
            ]
        )
        == 0
    )
    progress = json.loads(capsys.readouterr().out)
    assert progress["steps"] == [-1, 0, 1, 2]
    first = next(iter(progress["series"].values()))
    assert [p.get("label") for p in first[:2]] == ["base", "initialized"]


def test_multifact_cli_and_exact_fact_resume(tmp_path, monkeypatch, capsys, stub_continual_evaluation):
    bind(monkeypatch)
    from distill.progress import ProbePoint
    import distill.distill as controller

    stub_continual_evaluation(
        answer=lambda m, t, p, **kw: "base" if p.name == "canary" else "wrong",
        judge=lambda m, t, p, c, a, **kw: a == "base",
        sample=lambda m, t, p, step, *a, **kw: ProbePoint(
            step, -0.1, 0.9, -0.1, 0.9, [], completion="base", content="base"
        ),
        score=lambda m, t, p, s, *a, **kw: ProbePoint(s, -2.0, 0.1, -2.0, 0.1, []),
    )
    probes = tmp_path / "probes.jsonl"
    probes.write_text(
        "".join(
            json.dumps(
                {
                    "name": f"fact-{i}",
                    "prompt": f"heldout {i}",
                    "completion": "answer",
                    "train_prompts": [f"train {i}"],
                }
            )
            + "\n"
            for i in range(2)
        )
    )
    canaries = tmp_path / "canaries.jsonl"
    canaries.write_text(
        json.dumps({"name": "canary", "prompt": "q", "completion": "base"}) + "\n"
    )
    common = [
        "--model",
        "tiny",
        "--backend",
        "cartridge",
        "--train",
        SOURCE,
        "--probes",
        str(probes),
        "--canaries",
        str(canaries),
        "--judge-model",
        "tiny",
        "--cartridge-positions",
        "4",
        "--eval-every",
        "2",
        "--max-tokens",
        "4",
        "--temperature",
        "0",
        "--json",
    ]
    full = tmp_path / "full"
    split = tmp_path / "split"
    assert main([*common, "--cartridge", str(full), "--iters", "3"]) == 1
    capsys.readouterr()
    assert main([*common, "--cartridge", str(split), "--iters", "2"]) == 1
    capsys.readouterr()
    assert (
        main(["--model", "tiny", "--resume", str(split), "--iters", "3", "--json"]) == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["selected_step"] == 3
    left = mx.load(str(full / "checkpoints/step-0003/resume.safetensors"))
    right = mx.load(str(split / "checkpoints/step-0003/resume.safetensors"))
    assert left.keys() == right.keys() and all(
        mx.array_equal(left[k], right[k]).item() for k in left
    )
    config = json.loads((split / "run_config.json").read_text())
    assert config["reduction"] == "equal-fact" and config["source_hash"]
    assert SOURCE not in (split / "training_metrics.jsonl").read_text()


def test_utf8_files_preserve_crlf_and_key_whitespace(tmp_path, monkeypatch, capsys):
    bind(monkeypatch)
    document = SOURCE.replace("\n", "\r\n")
    source = tmp_path / "source.txt"
    source.write_bytes(document.encode("utf-8"))
    root = tmp_path / "memory"
    common = ["--model", "tiny", "--json"]
    assert (
        main(
            [
                *common,
                "--memorize",
                "@" + str(source),
                "--cartridge",
                str(root),
                "--cartridge-positions",
                "4",
                "--key-tokens",
                "5",
                "--target-tokens",
                "8",
                "--stride",
                "8",
                "--iters",
                "1",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (root / "generated/source.txt").read_bytes() == source.read_bytes()
    assert main([*common, "--resume", str(root), "--iters", "2"]) == 0
    capsys.readouterr()
    key = tmp_path / "key.txt"
    key.write_bytes(b" sky \r\n")
    assert (
        main(
            [
                *common,
                "--cartridge-path",
                str(root),
                "--complete-key",
                "@" + str(key),
                "--max-tokens",
                "1",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert "".join(t["text"] for t in payload["tokens"]).startswith(" sky \r\n")


def test_inference_loads_the_recorded_base_revision(tmp_path, monkeypatch):
    from argparse import Namespace
    from distill.cli import _load_models
    from distill.cli_config import CommandConfig, SamplingOptions
    from distill.cartridge import Cartridge

    model, tokenizer = tiny_model()
    identity = model_identity(model, tokenizer, "tiny")
    identity["revision"] = "fixed-revision"
    memory = Cartridge.initialize(model, tokenizer, SOURCE, 4, identity=identity)
    memory.save(tmp_path)
    seen = []

    def load(name, revision=None):
        seen.append((name, revision))
        return model, tokenizer, identity

    monkeypatch.setattr("distill.cli._load_cartridge_base", load)
    args = Namespace(
        model="tiny",
        config=CommandConfig(
            "score", SamplingOptions(), None, False, backend="cartridge"
        ),
        cartridge_path=str(tmp_path),
        progress=None,
        adapter_path=None,
        teacher_model=None,
    )
    _load_models(args)
    assert seen == [("tiny", "fixed-revision")]
