from pathlib import Path
from types import SimpleNamespace
import json
from distill.distill import run_continual
from distill.progress import Probe, ProbePoint
from distill.stopping import LearningItem
from distill.training import TrainStep


def run_case(tmp_path, monkeypatch, *, unsafe_init=False, leak_at=None):
    import distill.distill as controller

    class Model:
        def eval(self):
            pass

    class Session:
        iteration = 0

        def save(self, path):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            (path / "cartridge.json").write_text("{}")
            (path / "cartridge.safetensors").write_text("memory")
            return path / "cartridge.safetensors"

        def save_resume(self, path, state):
            (path / "resume.json").write_text(json.dumps({"state": state}))
            (path / "resume.safetensors").write_text("masters")

        @property
        def inference_policy(self):
            return SimpleNamespace(new_cache=lambda: ["attached"])

        def step_batch(self, prompts, **kw):
            self.iteration += 1
            return [
                TrainStep(self.iteration, p, 1.0, 1, "length", "answer", batch_loss=1.0)
                for p in prompts
            ]

    from contextlib import nullcontext

    monkeypatch.setattr(controller, "lora_disabled", lambda model: nullcontext())
    session = Session()

    def answer(model, tokenizer, probe, **kwargs):
        if probe.name == "canary":
            attached = kwargs.get("cache_factory") is not None
            return (
                "changed"
                if attached
                and (
                    unsafe_init
                    or (leak_at is not None and session.iteration >= leak_at)
                )
                else "base"
            )
        return "incorrect"

    monkeypatch.setattr(controller, "_answer", answer)
    monkeypatch.setattr(
        controller,
        "score_probe",
        lambda m, t, p, s, *a, **kw: ProbePoint(s, -2.0, 0.1, -2.0, 0.1, []),
    )
    monkeypatch.setattr(
        controller, "judge_answer", lambda m, t, p, c, a, **kw: a == "base"
    )
    result = run_continual(
        Model(),
        object(),
        [LearningItem("fact", "heldout", "answer", ("train",))],
        [Probe("canary", "question", "base")],
        "source document",
        tmp_path / "run",
        model_name="tiny",
        iters=3,
        eval_every=2,
        session_factory=lambda *a, **kw: session,
        backend="cartridge",
    )
    return result, tmp_path / "run"


def test_initialization_alarm_does_not_publish_an_unsafe_root_export(
    tmp_path, monkeypatch
):
    result, root = run_case(tmp_path, monkeypatch, unsafe_init=True)
    assert not result.success and result.selected_step is None
    assert not (root / "cartridge.safetensors").exists()
    assert json.loads((root / "selection.json").read_text())["reason"] == "locality"


def test_later_canary_alarm_selects_last_safe_checkpoint_and_keeps_latest(
    tmp_path, monkeypatch
):
    result, root = run_case(tmp_path, monkeypatch, leak_at=2)
    assert not result.success and result.selected_step == 0
    assert (root / "cartridge.safetensors").exists()
    assert json.loads((root / "latest.json").read_text())["step"] == 2


def test_final_noninterval_update_is_evaluated_and_committed(tmp_path, monkeypatch):
    result, root = run_case(tmp_path, monkeypatch)
    assert json.loads((root / "latest.json").read_text())["step"] == 3
    assert [
        json.loads(line)["iteration"]
        for line in (root / "eval_metrics.jsonl").read_text().splitlines()
    ] == [0, 2, 3]
