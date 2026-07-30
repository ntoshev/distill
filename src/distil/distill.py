"""Probe-driven continual context distillation orchestration."""

import json
import math
import shutil
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

from distil.adapters import lora_disabled
from distil.judge import judge_answer
from distil.progress import Probe, score_probe
from distil.sample import sample_completion
from distil.stopping import (
    ActiveRotation,
    LearningItem,
    answer_only_prompt,
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
    batch_size: int = 4,
    learned_threshold: float = 0.5,
    mastery_patience: int = 2,
    mastery_max_tokens: int = 32,
    learning_rate: float = 1e-6,
    lora_rank: int = 16,
    max_tokens: int = 64,
    canary_max_tokens: int = 32,
    temperature: float = 0.0,
    seed: int = 0,
    enable_thinking: bool | None = None,
    speculative_top_k: int | None = 3,
    kl_direction: str = "forward",
    source: dict | None = None,
    probes_path: str | Path | None = None,
    canaries_path: str | Path | None = None,
    save_rollout_path: str | Path | None = None,
    report: Callable[[dict], None] | None = None,
    train_report: Callable[[dict], None] | None = None,
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
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if canary_max_tokens <= 0:
        raise ValueError("--canary-max-tokens must be positive")
    if mastery_patience <= 0:
        raise ValueError("mastery_patience must be positive")
    if mastery_max_tokens <= 0:
        raise ValueError("mastery_max_tokens must be positive")
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
    separate_judge = teacher_model is not None and teacher_model is not model
    judge_model = teacher_model if separate_judge else model
    judge_tokenizer = (teacher_tokenizer or tokenizer) if separate_judge else tokenizer
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
        if not judge_answer(
            judge_model,
            judge_tokenizer,
            probe.prompt,
            probe.completion,
            canary_baselines[probe.name],
            enable_thinking=enable_thinking,
        )
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
        "batch_size": batch_size,
        "learned_threshold": learned_threshold,
        "mastery_criterion": "greedy_llm_judge",
        "mastery_patience": mastery_patience,
        "mastery_max_tokens": mastery_max_tokens,
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
            answer_only_prompt(item.prompt),
            item.completion,
            item.system if item.system is not None else student_system,
        )
        for item in items
    }
    items_by_name = {item.name: item for item in items}
    mastery_streaks = {item.name: 0 for item in items}
    safe_rows: list[dict] = []
    last_loss = None
    latest_training_rollout: dict | None = None

    def evaluate(step: int, checkpoint: Path) -> tuple[dict, set[str]]:
        started = perf_counter()
        scores = {
            name: score_probe(model, tokenizer, probe, step, enable_thinking).mean_probability
            for name, probe in item_probes.items()
        }
        probability_learned = mastered(scores, learned_threshold)
        confirmations = {}
        for name, probe in item_probes.items():
            content = _answer(
                model,
                tokenizer,
                probe,
                max_tokens=mastery_max_tokens,
                enable_thinking=enable_thinking,
                default_system=student_system,
            )
            item = items_by_name[name]
            judge_context = nullcontext() if separate_judge else lora_disabled(model)
            with judge_context:
                passed = judge_answer(
                    judge_model,
                    judge_tokenizer,
                    item.prompt,
                    item.completion,
                    content,
                    enable_thinking=enable_thinking,
                )
            mastery_streaks[name] = mastery_streaks[name] + 1 if passed else 0
            confirmations[name] = {
                "content": content,
                "passed": passed,
                "streak": mastery_streaks[name],
            }
        learned = {
            name for name, streak in mastery_streaks.items() if streak >= mastery_patience
        }
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
        active = {item.name for item in items} - learned
        row = {
            "iteration": step,
            "loss": last_loss,
            "mastered": len(learned),
            "total": len(items),
            "active_item_ids": sorted(active),
            "probe_scores": scores,
            "probability_mastered": len(probability_learned),
            "probability_mastered_item_ids": sorted(probability_learned),
            "mastery_streaks": dict(mastery_streaks),
            "canary_changed": sum(entry["changed"] for entry in canary_rows),
            "canaries": canary_rows,
            "confirmations": confirmations,
            "checkpoint": str(checkpoint),
            "evaluation_seconds": perf_counter() - started,
            "training_rollout": latest_training_rollout,
        }
        with eval_file.open("a") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        if not row["canary_changed"]:
            safe_rows.append(row)
        if report is not None:
            report(row)
        return row, active

    baseline, active = evaluate(0, step0)
    if baseline["mastered"] == len(items):
        run_config["resolved_iters"] = 0
        config_file.write_text(json.dumps(run_config, ensure_ascii=False, indent=2) + "\n")
        stopped, reason, selected = "already_mastered", "already_mastered", baseline
    else:
        example_budget = min(2000, max(200, 200 * len(active)))
        budget = iters if iters is not None else math.ceil(example_budget / batch_size)
        run_config["resolved_iters"] = budget
        config_file.write_text(json.dumps(run_config, ensure_ascii=False, indent=2) + "\n")
        rotation = ActiveRotation(items, seed)
        rotation.update(active)
        stopped = reason = ""
        selected = baseline
        reported_first_question = False
        for update_number in range(1, budget + 1):
            # Keep on-policy rollouts in the original training-prompt distribution.
            # The answer-only contract is reserved for held-out mastery evaluation.
            batch = rotation.next_batch(batch_size)
            if not reported_first_question and train_report is not None:
                train_report(
                    {
                        "event": "question",
                        "iteration": 1,
                        "item_id": batch[0][0],
                        "prompt": batch[0][1],
                        "batch_size": len(batch),
                    }
                )
            reported_first_question = True
            steps = session.step_batch(
                [prompt for _, prompt in batch],
                capture_feedback=update_number % eval_every == 0,
            )
            last_loss = steps[0].batch_loss
            with metrics_file.open("a") as file:
                for (item_id, prompt), step in zip(batch, steps):
                    payload = _step_payload(step)
                    payload["item_id"] = item_id
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
            iteration = steps[0].iteration
            first_item_id, _ = batch[0]
            first_step = steps[0]
            latest_training_rollout = {
                "iteration": iteration,
                "item_id": first_item_id,
                "prompt": first_step.prompt,
                "content": first_step.content,
                "teacher_feedback": first_step.teacher_feedback or [],
                "speculative_top_k": first_step.speculative_top_k,
                "accepted_tokens": first_step.accepted_tokens,
                "replaced_tokens": first_step.replaced_tokens,
            }
            if train_report is not None:
                train_report(
                    {
                        "event": "step",
                        "iteration": iteration,
                        "loss": last_loss,
                        "batch_size": len(steps),
                        "num_tokens": sum(step.num_tokens for step in steps),
                        "timing": {
                            "rollout_seconds": first_step.batch_rollout_seconds,
                            "distill_seconds": first_step.batch_distill_seconds,
                            "optimizer_seconds": first_step.optimizer_seconds,
                            "total_seconds": first_step.batch_train_seconds,
                        },
                    }
                )
            if iteration % eval_every != 0:
                continue
            checkpoint = checkpoints / f"step-{iteration:04d}"
            session.save(checkpoint)
            row, active = evaluate(iteration, checkpoint)
            if row["canary_changed"]:
                stopped, reason = "aborted", "locality"
                break
            if row["mastered"] == len(items):
                stopped, reason, selected = "success", "mastered", row
                break
            rotation.update(active)
        else:
            stopped, reason = "aborted", "budget"

        if stopped == "aborted":
            selected = max(
                safe_rows,
                key=lambda row: (
                    row["mastered"],
                    sum(row["mastery_streaks"].values()),
                    row["iteration"],
                ),
            )

    selected_checkpoint = Path(selected["checkpoint"])
    adapter_file = _copy_selected(selected_checkpoint, root)
    selection = {
        "stopped": stopped,
        "reason": reason,
        "selected_step": selected["iteration"],
        "mastered": selected["mastered"],
        "total": selected["total"],
        "mastery_criterion": "greedy_llm_judge",
        "mastery_patience": mastery_patience,
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
