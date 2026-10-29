"""Probe-driven continual context distillation orchestration."""

import json
import math
import shutil
from contextlib import nullcontext
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Callable

from distill.adapters import lora_disabled
from distill.cli_config import resolve_speculative_policy
from distill.judge import judge_answers
from distill.progress import (
    Probe,
    ProgressResult,
    evaluate_probes,
    progress_to_json,
    evaluation_transition,
)
from distill.stopping import (
    ActiveRotation,
    LearningItem,
    answer_only_prompt,
    mastered,
)
from distill.training import TrainingSession, _step_payload, save_training_rollout
from distill.templating import prompt_messages


class _DefaultSpeculativeTopK(Enum):
    VALUE = 3


@dataclass(frozen=True)
class ContinualResult:
    success: bool
    stop_reason: str
    selected_step: int | None
    adapter_file: Path | None
    selection_file: Path


def _copy_selected(checkpoint: Path, root: Path) -> Path:
    if (checkpoint / "cartridge.json").exists():
        from distill.cartridge_checkpoint import copy_export

        return copy_export(checkpoint, root)
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
    judge_model=None,
    judge_tokenizer=None,
    judge_model_name: str | None = None,
    from_adapter: str | Path | None = None,
    iters: int | None = None,
    eval_every: int = 5,
    batch_size: int = 4,
    learned_threshold: float = 0.5,
    mastery_patience: int = 4,
    mastery_max_tokens: int = 32,
    learning_rate: float = 1e-6,
    lora_rank: int = 16,
    max_tokens: int = 64,
    canary_max_tokens: int = 32,
    temperature: float = 0.0,
    seed: int = 0,
    enable_thinking: bool | None = None,
    speculative_top_k: int
    | None
    | _DefaultSpeculativeTopK = _DefaultSpeculativeTopK.VALUE,
    speculative_top_p: float | None = None,
    kl_direction: str = "forward",
    source: dict | None = None,
    probes_path: str | Path | None = None,
    canaries_path: str | Path | None = None,
    save_rollout_path: str | Path | None = None,
    report: Callable[[dict], None] | None = None,
    train_report: Callable[[dict], None] | None = None,
    warn: Callable[[str], None] | None = None,
    session_factory=None,
    inference_policy=None,
    backend="lora",
    artifact_config=None,
    resume_checkpoint=None,
    capture_progress: bool = False,
) -> ContinualResult:
    # Explicit None still disables speculation; only an omitted K defaults to 3.
    if speculative_top_k is _DefaultSpeculativeTopK.VALUE:
        speculative_top_k = 3 if speculative_top_p is None else None
    resolve_speculative_policy(speculative_top_k, speculative_top_p)
    if not items:
        raise ValueError("training needs at least one probe")
    if not canaries:
        raise ValueError("--canaries must contain at least one probe")
    names = [item.name for item in items] + [p.name for p in canaries]
    if len(names) != len(set(names)):
        raise ValueError("training targets and canaries must have distinct names")
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

    if backend == "cartridge" and session_factory is None:
        raise ValueError("cartridge fact learning requires a cartridge session factory")
    root = Path(adapter_path)
    if (
        backend == "cartridge"
        and not resume_checkpoint
        and root.exists()
        and any(p.name != "generated" for p in root.iterdir())
    ):
        raise ValueError(f"refusing existing cartridge output: {root}")
    root.mkdir(parents=True, exist_ok=True)
    selection_file = root / "selection.json"
    metrics_file = root / "training_metrics.jsonl"
    eval_file = root / "eval_metrics.jsonl"
    if not resume_checkpoint:
        selection_file.unlink(missing_ok=True)
        metrics_file.write_text("")
        eval_file.write_text("")

    model.eval()
    judge_model = judge_model or teacher_model or model
    separate_judge = judge_model is not model
    if not separate_judge:
        judge_tokenizer = tokenizer
    elif judge_tokenizer is None and judge_model is teacher_model:
        judge_tokenizer = teacher_tokenizer or tokenizer
    elif judge_tokenizer is None:
        raise ValueError("a separate judge model requires its tokenizer")
    # Judge weights stay frozen (LoRA is disabled when sharing the student).
    # Scope verdicts to this run so models and configurations cannot share them.
    judge_cache: dict[tuple[str, str], bool] = {}
    baseline_points = evaluate_probes(
        model,
        tokenizer,
        [replace(probe, dynamic=True) for probe in canaries],
        -1,
        enable_thinking,
        max_tokens=[canary_max_tokens] * len(canaries),
        default_system=student_system,
        **({"cache_factory": inference_policy.new_cache} if inference_policy else {}),
    )
    canary_baselines = {
        probe.name: point.content.strip()
        for probe, point in zip(canaries, baseline_points)
    }
    judge_context = nullcontext() if separate_judge else lora_disabled(model)
    with judge_context:
        baseline_verdicts = judge_answers(
            judge_model,
            judge_tokenizer,
            [
                (
                    probe.completion or canary_baselines[probe.name],
                    canary_baselines[probe.name],
                )
                for probe in canaries
            ],
            enable_thinking=enable_thinking,
            cache=judge_cache,
        )
    invalid = [probe.name for probe, passed in zip(canaries, baseline_verdicts) if not passed]
    if invalid:
        raise ValueError(
            "starting model does not satisfy canary completion: " + ", ".join(invalid)
        )

    session = (session_factory or TrainingSession)(
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
        speculative_top_p=speculative_top_p,
        kl_direction=kl_direction,
        warn=warn,
    )
    checkpoints = root / "checkpoints"
    if checkpoints.exists() and not resume_checkpoint:
        shutil.rmtree(checkpoints)
    step0 = checkpoints / "step-0000"
    if backend == "lora":
        session.save(step0)

    run_config = {
        "model": model_name,
        "student_system": student_system,
        "generator_model": generator_model or model_name,
        "from_adapter": str(from_adapter) if from_adapter is not None else None,
        "teacher_model": model_name
        if teacher_model is None or teacher_model is model
        else "external",
        "judge_model": judge_model_name
        or ("external" if separate_judge else model_name),
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
        "speculative_top_p": speculative_top_p,
        "kl_direction": kl_direction,
        "enable_thinking": enable_thinking,
        "capture_progress": capture_progress,
    }
    if artifact_config:
        run_config.update(artifact_config)
    if backend == "cartridge":
        run_config.update(
            schema_version=2,
            backend="cartridge",
            recipe="facts",
            objective="forward-kl-skd",
            reduction="equal-fact",
            warmup=5,
            gradient_clip=1.0,
            logit_chunk_size=32,
        )
        run_config.pop("lora_rank", None)
        run_config.pop("from_adapter", None)
    config_file = root / "run_config.json"
    if backend == "cartridge" and not resume_checkpoint:
        config_file.write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2) + "\n"
        )

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
    previous_evaluation = None

    rotation = ActiveRotation(items, seed)
    rotation.update(set(items_by_name))
    resume_state = None
    if resume_checkpoint:
        resume_state = session.restore(Path(resume_checkpoint))
        from distill.cartridge_checkpoint import recover_logs, validate_resume_config

        validate_resume_config(root, resume_state)
        recover_logs(root, resume_state["log_offsets"])
        rotation.restore(resume_state["rotation"])
        mastery_streaks.update(resume_state["mastery_streaks"])
        canary_baselines = resume_state["canary_baselines"]
        safe_rows = resume_state["safe_rows"]
        last_loss = resume_state["last_loss"]
        latest_training_rollout = resume_state["latest_training_rollout"]
        previous_evaluation = resume_state["last_evaluation"]

    def evaluate(step: int, checkpoint: Path) -> tuple[dict, set[str]]:
        nonlocal previous_evaluation
        started = perf_counter()
        policy = session.inference_policy if backend == "cartridge" else None
        cache_options = {"cache_factory": policy.new_cache} if policy else {}
        targets = list(item_probes.values())
        points = evaluate_probes(
            model,
            tokenizer,
            [
                *targets,
                *(replace(p, dynamic=True) for p in targets),
                *(replace(p, dynamic=True) for p in canaries),
            ],
            step,
            enable_thinking,
            max_tokens=[mastery_max_tokens] * (2 * len(targets))
            + [canary_max_tokens] * len(canaries),
            default_system=student_system,
            **cache_options,
        )
        target_points = dict(zip(item_probes, points[: len(targets)]))
        answers = points[len(targets) : 2 * len(targets)]
        canary_points = dict(
            zip((p.name for p in canaries), points[2 * len(targets) :])
        )
        scores = {name: point.mean_probability for name, point in target_points.items()}
        probability_learned = mastered(scores, learned_threshold)
        confirmations = {}
        contents = [point.content.strip() for point in answers]
        judge_context = nullcontext() if separate_judge else lora_disabled(model)
        with judge_context:
            verdicts = judge_answers(
                judge_model,
                judge_tokenizer,
                [
                    (probe.completion, content)
                    for probe, content in zip(targets, contents)
                ],
                enable_thinking=enable_thinking,
                cache=judge_cache,
            )
        for name, content, passed in zip(item_probes, contents, verdicts):
            mastery_streaks[name] = mastery_streaks[name] + 1 if passed else 0
            confirmations[name] = {
                "content": content,
                "passed": passed,
                "streak": mastery_streaks[name],
            }
        learned = {
            name
            for name, streak in mastery_streaks.items()
            if streak >= mastery_patience
        }
        canary_rows = []
        for probe in canaries:
            content = canary_points[probe.name].content.strip()
            baseline = canary_baselines[probe.name]
            changed = content.casefold() != baseline.casefold()
            canary_rows.append(
                {
                    "name": probe.name,
                    "baseline": baseline,
                    "content": content,
                    "changed": changed,
                    "contaminated": changed
                    and any(
                        content.casefold() == item.completion.strip().casefold()
                        and baseline.casefold() != item.completion.strip().casefold()
                        for item in items
                    ),
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
        if capture_progress:
            label = evaluation_transition(row, previous_evaluation)
            probes = []
            points = {**target_points, **canary_points}
            used_names = set(names)
            for probe, answer in zip(targets, answers):
                rollout_name = probe.name + "/rollout"
                while rollout_name in used_names:
                    rollout_name += "/rollout"
                used_names.add(rollout_name)
                probes.extend([
                    replace(probe, role="target"),
                    replace(probe, name=rollout_name, dynamic=True, role="rollout"),
                ])
                points[rollout_name] = answer
            probes.extend(replace(p, dynamic=True, role="canary") for p in canaries)
            row["progress"] = progress_to_json(
                ProgressResult(
                    str(root),
                    [step],
                    probes,
                    {p.name: [replace(points[p.name], label=label)] for p in probes},
                    training_rollouts={
                        step: latest_training_rollout.get(
                            "rollouts", [latest_training_rollout]
                        ) if latest_training_rollout else []
                    },
                )
            )
        previous_evaluation = row
        with eval_file.open("a") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        if not row["canary_changed"]:
            safe_rows.append(row)
        if backend == "cartridge":
            rotation.update(active)
            from distill.cartridge_checkpoint import commit_checkpoint

            commit_checkpoint(
                session,
                root,
                {
                    "rotation": rotation.state(),
                    "mastery_streaks": dict(mastery_streaks),
                    "canary_baselines": canary_baselines,
                    "safe_rows": safe_rows,
                    "last_evaluation": row,
                    "last_loss": last_loss,
                    "latest_training_rollout": latest_training_rollout,
                },
            )
        if report is not None:
            report(row)
        return row, active

    if resume_state:
        baseline = resume_state["last_evaluation"]
        active = set(rotation.active)
    else:
        baseline, active = evaluate(0, step0)
    if baseline["canary_changed"]:
        stopped, reason, selected = "aborted", "locality", None
        run_config["resolved_iters"] = iters or 0
        config_file.write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2) + "\n"
        )
    elif baseline["mastered"] == len(items):
        run_config["resolved_iters"] = 0
        config_file.write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2) + "\n"
        )
        stopped, reason, selected = "already_mastered", "already_mastered", baseline
    else:
        example_budget = min(2000, max(200, 200 * len(active)))
        budget = iters if iters is not None else math.ceil(example_budget / batch_size)
        run_config["resolved_iters"] = budget
        config_file.write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2) + "\n"
        )
        rotation.update(active)
        stopped = reason = ""
        selected = baseline
        reported_first_question = False
        for update_number in range(
            session.iteration + 1 if resume_state else 1, budget + 1
        ):
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
                capture_feedback=update_number % eval_every == 0
                or (backend == "cartridge" and update_number == budget),
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
                            prompt_messages(prompt, student_system),
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
                **(
                    {"speculative_top_p": first_step.speculative_top_p}
                    if first_step.speculative_top_p is not None
                    else {}
                ),
                "accepted_tokens": first_step.accepted_tokens,
                "replaced_tokens": first_step.replaced_tokens,
                "rollouts": [
                    {
                        "iteration": step.iteration,
                        "item_id": item_id,
                        "prompt": step.prompt,
                        "content": step.content,
                        "tokens": step.rollout_tokens or [],
                    }
                    for (item_id, _), step in zip(batch, steps)
                ],
            }
            if train_report is not None:
                train_report(
                    {
                        "event": "step",
                        "iteration": iteration,
                        "loss": last_loss,
                        "batch_size": len(steps),
                        "num_tokens": sum(step.num_tokens for step in steps),
                        "accepted_tokens": sum(
                            step.accepted_tokens or 0 for step in steps
                        ),
                        "replaced_tokens": sum(
                            step.replaced_tokens or 0 for step in steps
                        ),
                        "timing": {
                            "rollout_seconds": first_step.batch_rollout_seconds,
                            "distill_seconds": first_step.batch_distill_seconds,
                            "optimizer_seconds": first_step.optimizer_seconds,
                            "total_seconds": first_step.batch_train_seconds,
                        },
                    }
                )
            if iteration % eval_every != 0 and (
                backend != "cartridge" or iteration != budget
            ):
                continue
            checkpoint = checkpoints / f"step-{iteration:04d}"
            if backend == "lora":
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

    selected_checkpoint = Path(selected["checkpoint"]) if selected else None
    adapter_file = (
        _copy_selected(selected_checkpoint, root) if selected_checkpoint else None
    )
    selection = {
        "stopped": stopped,
        "reason": reason,
        "selected_step": selected["iteration"] if selected else None,
        "mastered": selected["mastered"] if selected else 0,
        "total": selected["total"] if selected else len(items),
        "mastery_criterion": "greedy_llm_judge",
        "mastery_patience": mastery_patience,
        "source_checkpoint": str(selected_checkpoint),
    }
    if backend == "cartridge" and hasattr(session, "cartridge"):
        import mlx.core as mx

        selection.update(
            tensor_bytes=session.cartridge.nbytes,
            prefix_positions=session.cartridge.positions,
            trainable_parameters=session.cartridge.trainable_count,
            source_tokens=run_config.get("source_tokens"),
            peak_memory_bytes=mx.get_peak_memory(),
            backend=backend,
        )
    selection_file.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n"
    )
    return ContinualResult(
        success=stopped in {"success", "already_mastered"},
        stop_reason=reason,
        selected_step=selected["iteration"] if selected else None,
        adapter_file=adapter_file,
        selection_file=selection_file,
    )
