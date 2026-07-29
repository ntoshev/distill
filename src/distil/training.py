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
from mlx.utils import tree_map
from mlx_lm.tuner.losses import kl_div_loss

from distil.adapters import add_lora, continue_lora, lora_disabled, save_lora_adapter
from distil.prefill import DEFAULT_PREFILL_STEP_SIZE, select_logits
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
    kl_direction: str = "forward"
    batch_size: int = 1
    batch_index: int = 0
    batch_loss: float | None = None
    batch_train_seconds: float = 0.0
    rollout_seconds: float = 0.0
    distill_seconds: float = 0.0
    optimizer_seconds: float = 0.0
    batch_rollout_seconds: float = 0.0
    batch_distill_seconds: float = 0.0
    teacher_feedback: list[dict] | None = None


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


def _kl_loss(student_logits, teacher_logits, direction: str):
    """Per-token KL in the requested direction, with teacher logits fixed."""
    if direction == "forward":
        return kl_div_loss(student_logits, teacher_logits)
    if direction == "reverse":
        student_logprobs = student_logits - mx.logsumexp(
            student_logits, axis=-1, keepdims=True
        )
        teacher_logprobs = teacher_logits - mx.logsumexp(
            teacher_logits, axis=-1, keepdims=True
        )
        return (
            mx.exp(student_logprobs) * (student_logprobs - teacher_logprobs)
        ).sum(axis=-1)
    raise ValueError(f"unknown KL direction: {direction}")


def _distillation_loss(
    model, student_ids, student_rows, teacher_rows, kl_direction="forward"
):
    logits = model(student_ids[None])[0]
    vocab = min(logits.shape[-1], teacher_rows.shape[-1])
    student = logits[student_rows, :vocab].astype(mx.float32)
    teacher = teacher_rows[:, :vocab]
    return _kl_loss(student, teacher, kl_direction).mean()


def _teacher_rows(
    model,
    ids: list[int],
    rows: list[int],
    disable_adapter: bool,
    prefill_step_size: int = DEFAULT_PREFILL_STEP_SIZE,
):
    context = lora_disabled(model) if disable_adapter else nullcontext()
    model.eval()
    with context:
        selected = select_logits(model, ids, rows, step_size=prefill_step_size)
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
        "kl_direction": step.kl_direction,
        "batch": {
            "size": step.batch_size,
            "index": step.batch_index,
            "loss": step.batch_loss if step.batch_loss is not None else step.loss,
            "train_seconds": step.batch_train_seconds,
        },
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
            "rollout_seconds": step.rollout_seconds,
            "distill_seconds": step.distill_seconds,
            "optimizer_seconds": step.optimizer_seconds,
            "train_seconds": step.train_seconds,
            "batch_rollout_seconds": step.batch_rollout_seconds,
            "batch_distill_seconds": step.batch_distill_seconds,
            "checkpoint_seconds": step.checkpoint_seconds,
            "canary_seconds": step.canary_seconds,
            "total_seconds": step.total_seconds,
        },
        "teacher_feedback": step.teacher_feedback or [],
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


def save_training_rollout(
    path: str | Path,
    messages: list[dict],
    step: TrainStep,
    *,
    model_name: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    item_id: str | None = None,
) -> int:
    return save_rollout(
        path,
        messages,
        [{"role": "assistant", "content": step.content}],
        extra={
            "task": "distil-train",
            "model": model_name,
            "iteration": step.iteration,
            **({"item_id": item_id} if item_id is not None else {}),
            "loss": step.loss,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "speculative_top_k": step.speculative_top_k,
            "accepted_tokens": step.accepted_tokens,
            "replaced_tokens": step.replaced_tokens,
            "kl_direction": step.kl_direction,
            "batch_size": step.batch_size,
            "batch_index": step.batch_index,
            "batch_loss": step.batch_loss,
            "is_truncated": step.finish_reason != "stop",
        },
    )


@dataclass
class _PreparedRollout:
    prompt: str
    sample: Sample
    student_ids: list[int]
    student_rows: list[int]
    teacher_ids: list[int]
    teacher_rows: list[int]
    token_texts: list[str]
    prepare_seconds: float


class TrainingSession:
    """Stateful LoRA optimizer that performs one successful KL update at a time."""

    def __init__(
        self,
        model,
        tokenizer,
        teacher_system: str,
        *,
        model_name: str = "",
        student_system: str | None = None,
        teacher_model=None,
        teacher_tokenizer=None,
        learning_rate: float = 2e-6,
        lora_rank: int = 16,
        from_adapter: str | Path | None = None,
        max_tokens: int = 512,
        temperature: float = 1.0,
        seed: int = 0,
        enable_thinking: bool | None = None,
        speculative_top_k: int | None = None,
        kl_direction: str = "forward",
        warn: Callable[[str], None] | None = None,
    ):
        if not teacher_system:
            raise ValueError("training needs a non-empty teacher system prompt")
        if learning_rate <= 0:
            raise ValueError("--learning-rate must be positive")
        if max_tokens <= 0:
            raise ValueError("--max-tokens must be positive")
        if speculative_top_k is not None and speculative_top_k <= 0:
            raise ValueError("--speculative-top-k must be positive")
        if kl_direction not in {"forward", "reverse"}:
            raise ValueError("--kl-direction must be 'forward' or 'reverse'")

        self.model = model
        self.tokenizer = tokenizer
        self.teacher_model = teacher_model or model
        self.teacher_tokenizer = teacher_tokenizer or tokenizer
        self.self_teacher = self.teacher_model is model
        self.teacher_system = teacher_system
        self.student_system = student_system
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.speculative_top_k = speculative_top_k
        self.kl_direction = kl_direction
        self.warn = warn
        self.iteration = 0

        self.config = (
            continue_lora(model, from_adapter)
            if from_adapter is not None
            else add_lora(model, lora_rank)
        )
        self.config["model"] = model_name
        self.config["kl_direction"] = kl_direction
        self.optimizer = optim.AdamW(learning_rate=learning_rate)
        self.loss_and_grad = nn.value_and_grad(
            model,
            lambda model, student_ids, student_rows, teacher_rows: _distillation_loss(
                model, student_ids, student_rows, teacher_rows, kl_direction
            ),
        )
        mx.random.seed(seed)

    def save(self, path: str | Path) -> Path:
        self.model.eval()
        return save_lora_adapter(self.model, path, self.config)

    def _prepare_rollout(
        self, prompt: str, max_attempts: int = 10
    ) -> _PreparedRollout:
        messages = _messages(prompt, self.student_system)
        teacher_messages = apply_teacher_context(messages, system=self.teacher_system)
        for _ in range(max_attempts):
            started = perf_counter()
            self.model.eval()
            if self.speculative_top_k is None:
                sample: Sample = sample_completion(
                    self.model,
                    self.tokenizer,
                    messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    seed=None,
                    enable_thinking=self.enable_thinking,
                )
            else:
                sample = sample_speculative_completion(
                    self.model,
                    self.tokenizer,
                    messages,
                    self.teacher_model,
                    self.teacher_tokenizer,
                    teacher_messages,
                    top_k=self.speculative_top_k,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    seed=None,
                    enable_thinking=self.enable_thinking,
                    teacher_without_adapter=self.self_teacher,
                )
            if sample.num_tokens == 0:
                continue

            built_student = build_raw(sample.prompt_text, sample.completion_text)
            template_kwargs = (
                {} if self.enable_thinking is None else {"enable_thinking": self.enable_thinking}
            )
            teacher_prompt_text = self.teacher_tokenizer.apply_chat_template(
                teacher_messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            built_teacher = build_raw(teacher_prompt_text, sample.completion_text)
            try:
                aligned = align_scored_tokens(
                    self.tokenizer,
                    built_student,
                    self.teacher_tokenizer,
                    built_teacher,
                )
            except ValueError as error:
                if self.warn is not None:
                    self.warn(f"skipping rollout for {prompt!r}: {error}")
                continue
            if not aligned.pairs:
                continue
            return _PreparedRollout(
                prompt=prompt,
                sample=sample,
                student_ids=aligned.student_ids,
                student_rows=[
                    student_pos - 1 for student_pos, _ in aligned.pairs
                ],
                teacher_ids=aligned.teacher_ids,
                teacher_rows=[
                    teacher_pos - 1 for _, teacher_pos in aligned.pairs
                ],
                token_texts=[
                    built_student.text[
                        aligned.student_offsets[student_pos][0] :
                        aligned.student_offsets[student_pos][1]
                    ]
                    for student_pos, _ in aligned.pairs
                ],
                prepare_seconds=perf_counter() - started,
            )
        raise RuntimeError(f"could not produce an aligned rollout for {prompt!r}")

    def step_batch(
        self,
        prompts: list[str] | tuple[str, ...],
        max_attempts: int = 10,
        capture_feedback: bool = False,
    ) -> list[TrainStep]:
        """Apply one optimizer update from independently scored prompt rollouts."""
        if not prompts:
            raise ValueError("training batch must contain at least one prompt")

        batch_started = perf_counter()
        # Generate every rollout before computing gradients so all batch members
        # come from the same pre-update student policy.
        prepared = [
            self._prepare_rollout(prompt, max_attempts=max_attempts)
            for prompt in prompts
        ]
        accumulated_grads = None
        losses: list[float] = []
        distill_seconds: list[float] = []
        teacher_feedback: list[list[dict]] = []
        for rollout_index, rollout in enumerate(prepared):
            gradient_started = perf_counter()
            teacher_rows = _teacher_rows(
                self.teacher_model,
                rollout.teacher_ids,
                rollout.teacher_rows,
                disable_adapter=self.self_teacher,
            )
            feedback = []
            if capture_feedback and rollout_index == 0:
                chosen_ids = mx.array(
                    [rollout.student_ids[row + 1] for row in rollout.student_rows]
                )
                teacher_logprobs = teacher_rows.astype(mx.float32)
                teacher_logprobs = teacher_logprobs - mx.logsumexp(
                    teacher_logprobs, axis=-1, keepdims=True
                )
                chosen_teacher_logprobs = mx.take_along_axis(
                    teacher_logprobs, chosen_ids[:, None], axis=-1
                )[:, 0]
                mx.eval(chosen_teacher_logprobs)
                sampled = rollout.sample.teacher_sampled or []
                feedback = [
                    {
                        "text": text,
                        "logprob": logprob,
                        "teacher_sampled": bool(
                            index < len(sampled) and sampled[index]
                        ),
                    }
                    for index, (text, logprob) in enumerate(
                        zip(rollout.token_texts, chosen_teacher_logprobs.tolist())
                    )
                ]
            teacher_feedback.append(feedback)
            student_ids = mx.array(rollout.student_ids)
            student_rows = mx.array(rollout.student_rows)
            self.model.train()
            loss, grads = self.loss_and_grad(
                self.model, student_ids, student_rows, teacher_rows
            )
            mx.eval(loss, grads)
            losses.append(loss.item())
            accumulated_grads = (
                grads
                if accumulated_grads is None
                else tree_map(
                    lambda accumulated, current: accumulated + current,
                    accumulated_grads,
                    grads,
                )
            )
            mx.eval(accumulated_grads)
            distill_seconds.append(perf_counter() - gradient_started)
            mx.clear_cache()

        batch_size = len(prepared)
        accumulated_grads = tree_map(lambda grad: grad / batch_size, accumulated_grads)
        optimizer_started = perf_counter()
        self.optimizer.update(self.model, accumulated_grads)
        mx.eval(self.model.trainable_parameters(), self.optimizer.state)
        optimizer_seconds = perf_counter() - optimizer_started
        self.iteration += 1
        batch_loss = sum(losses) / batch_size
        batch_seconds = perf_counter() - batch_started
        batch_rollout_seconds = sum(rollout.prepare_seconds for rollout in prepared)
        batch_distill_seconds = sum(distill_seconds)
        mx.clear_cache()
        return [
            TrainStep(
                iteration=self.iteration,
                prompt=rollout.prompt,
                loss=losses[index],
                num_tokens=len(rollout.student_rows),
                finish_reason=rollout.sample.finish_reason,
                content=rollout.sample.content,
                train_seconds=(rollout.prepare_seconds + distill_seconds[index]),
                total_seconds=batch_seconds,
                speculative_top_k=rollout.sample.speculative_top_k,
                accepted_tokens=rollout.sample.accepted_tokens,
                replaced_tokens=rollout.sample.replaced_tokens,
                kl_direction=self.kl_direction,
                batch_size=batch_size,
                batch_index=index,
                batch_loss=batch_loss,
                batch_train_seconds=batch_seconds,
                rollout_seconds=rollout.prepare_seconds,
                distill_seconds=distill_seconds[index],
                optimizer_seconds=optimizer_seconds,
                batch_rollout_seconds=batch_rollout_seconds,
                batch_distill_seconds=batch_distill_seconds,
                teacher_feedback=teacher_feedback[index],
            )
            for index, rollout in enumerate(prepared)
        ]

    def step(self, prompt: str, max_attempts: int = 10) -> TrainStep:
        return self.step_batch([prompt], max_attempts=max_attempts)[0]


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
    kl_direction: str = "forward",
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
    if kl_direction not in {"forward", "reverse"}:
        raise ValueError("--kl-direction must be 'forward' or 'reverse'")

    try:
        contamination_pattern = (
            re.compile(contamination_regex, re.IGNORECASE)
            if contamination_regex is not None
            else None
        )
    except re.error as error:
        raise ValueError(f"invalid --contamination-regex: {error}") from error

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
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        enable_thinking=enable_thinking,
        speculative_top_k=speculative_top_k,
        kl_direction=kl_direction,
        warn=warn,
    )

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

    order = list(prompts)
    rng = random.Random(seed)
    rng.shuffle(order)

    steps: list[TrainStep] = []
    while len(steps) < iters:
        step_started = perf_counter()
        prompt = order[len(steps) % len(order)]
        messages = _messages(prompt, student_system)
        step = session.step(prompt)
        iteration = step.iteration
        checkpoint_file = None
        checkpoint_seconds = 0.0
        if iteration % checkpoint_every == 0:
            checkpoint_started = perf_counter()
            checkpoint_file = session.save(
                adapter_path / "checkpoints" / f"step-{iteration:04d}"
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

        step.checkpoint_file = checkpoint_file
        step.checkpoint_seconds = checkpoint_seconds
        step.canaries = canary_results
        step.canary_seconds = canary_seconds
        step.total_seconds = perf_counter() - step_started
        steps.append(step)
        with metrics_file.open("a") as f:
            f.write(json.dumps(_step_payload(step), ensure_ascii=False) + "\n")
        if save_rollout_path is not None:
            save_training_rollout(
                save_rollout_path,
                messages,
                step,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
        if report is not None:
            report(step)

    model.eval()
    adapter_file = session.save(adapter_path)
    return TrainResult(
        steps=steps,
        adapter_file=adapter_file,
        metrics_file=metrics_file,
        canary_baseline_seconds=canary_baseline_seconds,
    )
