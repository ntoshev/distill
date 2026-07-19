"""On-policy context distillation into a LoRA adapter."""

import json
import random
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm.tuner.losses import kl_div_loss

from distil.adapters import add_lora, lora_disabled, save_lora_adapter
from distil.rollout import save_rollout
from distil.sample import Sample, sample_completion, sample_speculative_completion
from distil.scoring import align_scored_tokens
from distil.templating import apply_teacher_context, build_raw


@dataclass
class TrainStep:
    iteration: int
    prompt: str
    loss: float
    num_tokens: int
    finish_reason: str | None
    content: str
    train_seconds: float = 0.0
    checkpoint_seconds: float = 0.0
    canary_seconds: float = 0.0
    total_seconds: float = 0.0
    checkpoint_file: Path | None = None
    canaries: list["CanaryResult"] | None = None
    speculative_top_k: int | None = None
    accepted_tokens: int | None = None
    replaced_tokens: int | None = None


@dataclass
class CanaryResult:
    prompt: str
    baseline: str
    content: str
    changed: bool
    contaminated: bool


@dataclass
class TrainResult:
    steps: list[TrainStep]
    adapter_file: Path
    metrics_file: Path
    canary_baseline_seconds: float = 0.0


def load_prompts(path: str | Path) -> list[str]:
    """Load one training prompt per non-empty line."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"training prompt file not found or not a file: {path}")
    prompts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"training prompt file is empty: {path}")
    return prompts


def _distillation_loss(model, student_ids, student_rows, teacher_rows):
    logits = model(student_ids[None])[0]
    vocab = min(logits.shape[-1], teacher_rows.shape[-1])
    student = logits[student_rows, :vocab].astype(mx.float32)
    teacher = teacher_rows[:, :vocab]
    return kl_div_loss(student, teacher).mean()


def _teacher_rows(model, ids: list[int], rows: list[int], disable_adapter: bool):
    context = lora_disabled(model) if disable_adapter else nullcontext()
    model.eval()
    with context:
        logits = model(mx.array(ids)[None])[0]
        selected = logits[mx.array(rows)].astype(mx.float32)
        # MLX is lazy: force the adapter-disabled graph before restoring scales.
        mx.eval(selected)
    return selected


def _messages(prompt: str, system: str | None) -> list[dict]:
    messages = [] if system is None else [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": prompt})
    return messages


def _sample_canaries(
    model,
    tokenizer,
    prompts: list[str],
    baselines: list[str],
    *,
    system: str | None,
    max_tokens: int,
    enable_thinking: bool | None,
    contamination_pattern: re.Pattern[str] | None,
) -> list[CanaryResult]:
    results = []
    for prompt, baseline in zip(prompts, baselines):
        sample = sample_completion(
            model,
            tokenizer,
            _messages(prompt, system),
            max_tokens=max_tokens,
            temperature=0.0,
            seed=0,
            enable_thinking=enable_thinking,
        )
        content = sample.content.strip()
        results.append(
            CanaryResult(
                prompt=prompt,
                baseline=baseline,
                content=content,
                changed=content != baseline,
                contaminated=bool(
                    contamination_pattern is not None
                    and contamination_pattern.search(content)
                ),
            )
        )
    return results


def _step_payload(step: TrainStep) -> dict:
    canaries = step.canaries or []
    return {
        "iteration": step.iteration,
        "prompt": step.prompt,
        "loss": step.loss,
        "num_tokens": step.num_tokens,
        "finish_reason": step.finish_reason,
        "content": step.content,
        "sampling": {
            "policy": "speculative" if step.speculative_top_k is not None else "student",
            "top_k": step.speculative_top_k,
            "accepted_tokens": step.accepted_tokens,
            "replaced_tokens": step.replaced_tokens,
            "acceptance_rate": (
                step.accepted_tokens / (step.accepted_tokens + step.replaced_tokens)
                if step.accepted_tokens is not None
                and step.replaced_tokens is not None
                and step.accepted_tokens + step.replaced_tokens > 0
                else None
            ),
        },
        "timing": {
            "train_seconds": step.train_seconds,
            "checkpoint_seconds": step.checkpoint_seconds,
            "canary_seconds": step.canary_seconds,
            "total_seconds": step.total_seconds,
        },
        "checkpoint_file": (
            str(step.checkpoint_file) if step.checkpoint_file is not None else None
        ),
        "canary_summary": {
            "total": len(canaries),
            "changed": sum(result.changed for result in canaries),
            "contaminated": sum(result.contaminated for result in canaries),
        },
        "canaries": [
            {
                "prompt": result.prompt,
                "baseline": result.baseline,
                "content": result.content,
                "changed": result.changed,
                "contaminated": result.contaminated,
            }
            for result in canaries
        ],
    }


def train_lora(
    model,
    tokenizer,
    prompts: list[str],
    teacher_system: str,
    adapter_path: str | Path,
    *,
    model_name: str = "",
    student_system: str | None = None,
    teacher_model=None,
    teacher_tokenizer=None,
    iters: int = 100,
    learning_rate: float = 2e-6,
    lora_rank: int = 16,
    max_tokens: int = 512,
    temperature: float = 1.0,
    seed: int = 0,
    enable_thinking: bool | None = None,
    save_rollout_path: str | Path | None = None,
    checkpoint_every: int = 1,
    canary_prompts: list[str] | None = None,
    canary_max_tokens: int = 32,
    contamination_regex: str | None = None,
    speculative_top_k: int | None = None,
    report: Callable[[TrainStep], None] | None = None,
    warn: Callable[[str], None] | None = None,
) -> TrainResult:
    """Train a LoRA student against a frozen, privileged-context teacher.

    Sampling is on-policy under the current adapter unless
    ``speculative_top_k`` enables teacher-corrected interleaving. With
    self-distillation, the teacher is the same model evaluated with LoRA
    disabled.
    """
    if not prompts:
        raise ValueError("training needs at least one prompt")
    if not teacher_system:
        raise ValueError("training needs a non-empty teacher system prompt")
    if iters <= 0:
        raise ValueError("--iters must be positive")
    if learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")
    if canary_max_tokens <= 0:
        raise ValueError("--canary-max-tokens must be positive")
    if speculative_top_k is not None and speculative_top_k <= 0:
        raise ValueError("--speculative-top-k must be positive")

    try:
        contamination_pattern = (
            re.compile(contamination_regex, re.IGNORECASE)
            if contamination_regex is not None
            else None
        )
    except re.error as error:
        raise ValueError(f"invalid --contamination-regex: {error}") from error

    teacher_model = teacher_model or model
    teacher_tokenizer = teacher_tokenizer or tokenizer
    self_teacher = teacher_model is model

    config = add_lora(model, lora_rank)
    config["model"] = model_name
    optimizer = optim.AdamW(learning_rate=learning_rate)
    loss_and_grad = nn.value_and_grad(model, _distillation_loss)

    adapter_path = Path(adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)
    metrics_file = adapter_path / "training_metrics.jsonl"
    metrics_file.write_text("")

    canary_prompts = list(canary_prompts or [])
    canary_baseline_seconds = 0.0
    canary_baselines: list[str] = []
    if canary_prompts:
        baseline_started = perf_counter()
        model.eval()
        with lora_disabled(model):
            for prompt in canary_prompts:
                sample = sample_completion(
                    model,
                    tokenizer,
                    _messages(prompt, student_system),
                    max_tokens=canary_max_tokens,
                    temperature=0.0,
                    seed=0,
                    enable_thinking=enable_thinking,
                )
                canary_baselines.append(sample.content.strip())
        canary_baseline_seconds = perf_counter() - baseline_started

    mx.random.seed(seed)
    order = list(prompts)
    rng = random.Random(seed)
    rng.shuffle(order)

    steps: list[TrainStep] = []
    attempts = 0
    max_attempts = max(iters * 10, 10)
    while len(steps) < iters:
        step_started = perf_counter()
        if attempts >= max_attempts:
            raise RuntimeError(
                f"only completed {len(steps)} updates after {attempts} attempts; "
                "too many rollout alignment failures"
            )
        prompt = order[attempts % len(order)]
        attempts += 1
        messages = _messages(prompt, student_system)
        teacher_messages = apply_teacher_context(messages, system=teacher_system)

        model.eval()
        if speculative_top_k is None:
            sample: Sample = sample_completion(
                model,
                tokenizer,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=None,
                enable_thinking=enable_thinking,
            )
        else:
            sample = sample_speculative_completion(
                model,
                tokenizer,
                messages,
                teacher_model,
                teacher_tokenizer,
                teacher_messages,
                top_k=speculative_top_k,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=None,
                enable_thinking=enable_thinking,
                teacher_without_adapter=self_teacher,
            )
        if sample.num_tokens == 0:
            continue

        built_student = build_raw(sample.prompt_text, sample.completion_text)
        template_kwargs = (
            {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        )
        teacher_prompt_text = teacher_tokenizer.apply_chat_template(
            teacher_messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        built_teacher = build_raw(teacher_prompt_text, sample.completion_text)

        try:
            aligned = align_scored_tokens(
                tokenizer, built_student, teacher_tokenizer, built_teacher
            )
        except ValueError as error:
            if warn is not None:
                warn(f"skipping rollout for {prompt!r}: {error}")
            continue
        if not aligned.pairs:
            if warn is not None:
                warn(f"skipping rollout for {prompt!r}: no generated tokens to train on")
            continue

        teacher_rows = _teacher_rows(
            teacher_model,
            aligned.teacher_ids,
            [teacher_pos - 1 for _, teacher_pos in aligned.pairs],
            disable_adapter=self_teacher,
        )
        student_ids = mx.array(aligned.student_ids)
        student_rows = mx.array([student_pos - 1 for student_pos, _ in aligned.pairs])

        model.train()
        loss, grads = loss_and_grad(model, student_ids, student_rows, teacher_rows)
        optimizer.update(model, grads)
        mx.eval(loss, model.trainable_parameters(), optimizer.state)

        train_seconds = perf_counter() - step_started
        iteration = len(steps) + 1
        checkpoint_file = None
        checkpoint_seconds = 0.0
        if iteration % checkpoint_every == 0:
            checkpoint_started = perf_counter()
            checkpoint_file = save_lora_adapter(
                model,
                adapter_path / "checkpoints" / f"step-{iteration:04d}",
                config,
            )
            checkpoint_seconds = perf_counter() - checkpoint_started

        canary_results = None
        canary_seconds = 0.0
        if canary_prompts:
            canary_started = perf_counter()
            model.eval()
            canary_results = _sample_canaries(
                model,
                tokenizer,
                canary_prompts,
                canary_baselines,
                system=student_system,
                max_tokens=canary_max_tokens,
                enable_thinking=enable_thinking,
                contamination_pattern=contamination_pattern,
            )
            canary_seconds = perf_counter() - canary_started

        step = TrainStep(
            iteration=iteration,
            prompt=prompt,
            loss=loss.item(),
            num_tokens=len(aligned.pairs),
            finish_reason=sample.finish_reason,
            content=sample.content,
            train_seconds=train_seconds,
            checkpoint_seconds=checkpoint_seconds,
            canary_seconds=canary_seconds,
            total_seconds=perf_counter() - step_started,
            checkpoint_file=checkpoint_file,
            canaries=canary_results,
            speculative_top_k=sample.speculative_top_k,
            accepted_tokens=sample.accepted_tokens,
            replaced_tokens=sample.replaced_tokens,
        )
        steps.append(step)
        with metrics_file.open("a") as f:
            f.write(json.dumps(_step_payload(step), ensure_ascii=False) + "\n")
        if save_rollout_path is not None:
            save_rollout(
                save_rollout_path,
                messages,
                [{"role": "assistant", "content": sample.content}],
                extra={
                    "task": "distil-train",
                    "model": model_name,
                    "iteration": step.iteration,
                    "loss": step.loss,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "seed": seed,
                    "speculative_top_k": sample.speculative_top_k,
                    "accepted_tokens": sample.accepted_tokens,
                    "replaced_tokens": sample.replaced_tokens,
                    "is_truncated": sample.finish_reason != "stop",
                },
            )
        if report is not None:
            report(step)

    model.eval()
    adapter_file = save_lora_adapter(model, adapter_path, config)
    return TrainResult(
        steps=steps,
        adapter_file=adapter_file,
        metrics_file=metrics_file,
        canary_baseline_seconds=canary_baseline_seconds,
    )
