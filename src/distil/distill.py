"""Probe-driven continual context distillation orchestration."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

from distil.progress import Probe, score_probe
from distil.sample import sample_completion
from distil.stopping import (
    ActiveRotation,
    LearningItem,
    answer_matches,
    mastered,
)
from distil.training import TrainingSession, _step_payload, save_training_rollout


@dataclass(frozen=True)
class ContinualResult:
    success: bool
    stop_reason: str
    selected_step: int
    adapter_file: Path
    selection_file: Path


def _prompt_messages(prompt: str, system: str | None = None) -> list[dict]:
    messages = [] if system is None else [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": prompt})
    return messages


def _messages(probe: Probe, default_system: str | None = None) -> list[dict]:
    return _prompt_messages(
        probe.prompt, probe.system if probe.system is not None else default_system
    )


def _answer(
    model,
    tokenizer,
    probe: Probe,
    *,
    max_tokens: int,
    enable_thinking: bool | None,
    default_system: str | None = None,
) -> str:
    return sample_completion(
        model,
        tokenizer,
        _messages(probe, default_system),
        max_tokens=max_tokens,
        temperature=0.0,
        seed=0,
        enable_thinking=enable_thinking,
    ).content.strip()


def _copy_selected(checkpoint: Path, root: Path) -> Path:
    for name in ("adapter_config.json", "adapters.safetensors"):
        source = checkpoint / name
        if not source.is_file():
            raise FileNotFoundError(f"selected checkpoint is incomplete: {source}")
        shutil.copy2(source, root / name)
    return root / "adapters.safetensors"


def run_continual(
    model,
    tokenizer,
    items: list[LearningItem],
    canaries: list[Probe],
    teacher_system: str,
    adapter_path: str | Path,
    *,
    model_name: str,
    generator_model: str | None = None,
    student_system: str | None = None,
    teacher_model=None,
    teacher_tokenizer=None,
    from_adapter: str | Path | None = None,
    iters: int | None = None,
    eval_every: int = 5,
    learned_threshold: float = 0.5,
    learning_rate: float = 1e-6,
    lora_rank: int = 16,
    max_tokens: int = 64,
    canary_max_tokens: int = 32,
    temperature: float = 1.0,
    seed: int = 0,
    enable_thinking: bool | None = None,
    speculative_top_k: int | None = 5,
    kl_direction: str = "forward",
    source: dict | None = None,
    probes_path: str | Path | None = None,
    canaries_path: str | Path | None = None,
    save_rollout_path: str | Path | None = None,
    report: Callable[[dict], None] | None = None,
    warn: Callable[[str], None] | None = None,
) -> ContinualResult:
    if not items:
        raise ValueError("training needs at least one probe")
    if not canaries:
        raise ValueError("--canaries must contain at least one probe")
    if from_adapter is not None and (teacher_model is None or teacher_model is model):
        raise ValueError(
            "starting-adapter training requires a separate frozen starting-model teacher"
        )
    if eval_every <= 0:
        raise ValueError("--eval-every must be positive")
    if canary_max_tokens <= 0:
        raise ValueError("--canary-max-tokens must be positive")
    if not 0 < learned_threshold <= 1:
        raise ValueError("--learned-threshold must be in (0, 1]")

    root = Path(adapter_path)
    root.mkdir(parents=True, exist_ok=True)
    selection_file = root / "selection.json"
    if selection_file.is_file():
        previous = json.loads(selection_file.read_text())
        if previous.get("stopped") in {"success", "already_mastered"}:
            raise ValueError(f"refusing to overwrite successful adapter run: {root}")

    metrics_file = root / "training_metrics.jsonl"
    eval_file = root / "eval_metrics.jsonl"
    metrics_file.write_text("")
    eval_file.write_text("")

    model.eval()
    canary_baselines = {
        probe.name: _answer(
            model,
            tokenizer,
            probe,
            max_tokens=canary_max_tokens,
            enable_thinking=enable_thinking,
            default_system=student_system,
        )
        for probe in canaries
    }
    invalid = [
        probe.name
        for probe in canaries
        if not answer_matches(canary_baselines[probe.name], probe.completion)
    ]
    if invalid:
        raise ValueError(
            "starting model does not satisfy canary completion: " + ", ".join(invalid)
        )

    session = TrainingSession(
        model,
        tokenizer,
        teacher_system,
        model_name=model_name,
        student_system=student_system,
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        from_adapter=from_adapter,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        enable_thinking=enable_thinking,
        speculative_top_k=speculative_top_k,
        kl_direction=kl_direction,
        warn=warn,
    )
    checkpoints = root / "checkpoints"
    if checkpoints.exists():
        shutil.rmtree(checkpoints)
    step0 = checkpoints / "step-0000"
    session.save(step0)

    run_config = {
        "model": model_name,
        "generator_model": generator_model or model_name,
        "from_adapter": str(from_adapter) if from_adapter is not None else None,
        "teacher_model": model_name if teacher_model is None or teacher_model is model else "external",
        "source": source,
        "probes": str(probes_path) if probes_path is not None else None,
        "canaries": str(canaries_path) if canaries_path is not None else None,
        "rollouts": str(save_rollout_path) if save_rollout_path is not None else None,
        "eval_every": eval_every,
        "learned_threshold": learned_threshold,
        "iters": iters,
        "learning_rate": learning_rate,
        "lora_rank": None if from_adapter is not None else lora_rank,
        "max_tokens": max_tokens,
        "canary_max_tokens": canary_max_tokens,
        "temperature": temperature,
        "seed": seed,
        "speculative_top_k": speculative_top_k,
        "kl_direction": kl_direction,
        "enable_thinking": enable_thinking,
    }
    config_file = root / "run_config.json"

    item_probes = {
        item.name: Probe(
            item.name,
            item.prompt,
            item.completion,
            item.system if item.system is not None else student_system,
        )
        for item in items
    }
    safe_rows: list[dict] = []
    last_loss = None

    def evaluate(step: int, checkpoint: Path, confirm: bool = True) -> tuple[dict, set[str]]:
        started = perf_counter()
        scores = {
            name: score_probe(model, tokenizer, probe, step, enable_thinking).mean_probability
            for name, probe in item_probes.items()
        }
        learned = mastered(scores, learned_threshold)
        canary_rows = []
        for probe in canaries:
            content = _answer(
                model,
                tokenizer,
                probe,
                max_tokens=canary_max_tokens,
                enable_thinking=enable_thinking,
                default_system=student_system,
            )
            baseline = canary_baselines[probe.name]
            canary_rows.append(
                {
                    "name": probe.name,
                    "baseline": baseline,
                    "content": content,
                    "changed": content != baseline,
                }
            )
        confirmations = {}
        failed_confirmation: set[str] = set()
        if confirm and len(learned) == len(items) and not any(
            row["changed"] for row in canary_rows
        ):
            for name, probe in item_probes.items():
                content = _answer(
                    model,
                    tokenizer,
                    probe,
                    max_tokens=max_tokens,
                    enable_thinking=enable_thinking,
                    default_system=student_system,
                )
                passed = answer_matches(content, probe.completion)
                confirmations[name] = {"content": content, "passed": passed}
                if not passed:
                    failed_confirmation.add(name)
        active = ({item.name for item in items} - learned) | failed_confirmation
        row = {
            "iteration": step,
            "loss": last_loss,
            "mastered": len(learned),
            "total": len(items),
            "active_item_ids": sorted(active),
            "probe_scores": scores,
            "canary_changed": sum(entry["changed"] for entry in canary_rows),
            "canaries": canary_rows,
            "confirmations": confirmations,
            "checkpoint": str(checkpoint),
            "evaluation_seconds": perf_counter() - started,
        }
        with eval_file.open("a") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        if not row["canary_changed"]:
            safe_rows.append(row)
        if report is not None:
            report(row)
        return row, active

    baseline, active = evaluate(0, step0, confirm=False)
    if baseline["mastered"] == len(items):
        run_config["resolved_iters"] = 0
        config_file.write_text(json.dumps(run_config, ensure_ascii=False, indent=2) + "\n")
        stopped, reason, selected = "already_mastered", "already_mastered", baseline
    else:
        budget = iters if iters is not None else min(2000, max(200, 200 * len(active)))
        run_config["resolved_iters"] = budget
        config_file.write_text(json.dumps(run_config, ensure_ascii=False, indent=2) + "\n")
        rotation = ActiveRotation(items, seed)
        rotation.update(active)
        stopped = reason = ""
        selected = baseline
        for _ in range(budget):
            item_id, prompt = rotation.next()
            step = session.step(prompt)
            last_loss = step.loss
            payload = _step_payload(step)
            payload["item_id"] = item_id
            with metrics_file.open("a") as file:
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")
            if save_rollout_path is not None:
                save_training_rollout(
                    save_rollout_path,
                    _prompt_messages(prompt, student_system),
                    step,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    item_id=item_id,
                )
            if step.iteration % eval_every != 0:
                continue
            checkpoint = checkpoints / f"step-{step.iteration:04d}"
            session.save(checkpoint)
            row, active = evaluate(step.iteration, checkpoint)
            if row["canary_changed"]:
                stopped, reason = "aborted", "locality"
                break
            if row["mastered"] == len(items) and all(
                value["passed"] for value in row["confirmations"].values()
            ):
                stopped, reason, selected = "success", "mastered", row
                break
            rotation.update(active)
        else:
            stopped, reason = "aborted", "budget"

        if stopped == "aborted":
            selected = max(safe_rows, key=lambda row: (row["mastered"], -row["iteration"]))

    selected_checkpoint = Path(selected["checkpoint"])
    adapter_file = _copy_selected(selected_checkpoint, root)
    selection = {
        "stopped": stopped,
        "reason": reason,
        "selected_step": selected["iteration"],
        "mastered": selected["mastered"],
        "total": selected["total"],
        "source_checkpoint": str(selected_checkpoint),
    }
    selection_file.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n")
    return ContinualResult(
        success=stopped in {"success", "already_mastered"},
        stop_reason=reason,
        selected_step=selected["iteration"],
        adapter_file=adapter_file,
        selection_file=selection_file,
    )
