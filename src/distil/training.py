"""On-policy context distillation into a LoRA adapter."""

import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm.tuner.losses import kl_div_loss

from distil.adapters import add_lora, lora_disabled, save_lora_adapter
from distil.rollout import save_rollout
from distil.sample import Sample, sample_completion
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


@dataclass
class TrainResult:
    steps: list[TrainStep]
    adapter_file: Path


def load_prompts(path: str | Path) -> list[str]:
    """Load one training prompt per non-empty line."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"training prompt file not found: {path}")
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
    learning_rate: float = 1e-5,
    lora_rank: int = 16,
    max_tokens: int = 512,
    temperature: float = 1.0,
    seed: int = 0,
    enable_thinking: bool | None = None,
    save_rollout_path: str | Path | None = None,
    report: Callable[[TrainStep], None] | None = None,
    warn: Callable[[str], None] | None = None,
) -> TrainResult:
    """Train a LoRA student against a frozen, privileged-context teacher.

    Sampling is on-policy under the current adapter. With self-distillation,
    the teacher is the same model evaluated with LoRA disabled.
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

    teacher_model = teacher_model or model
    teacher_tokenizer = teacher_tokenizer or tokenizer
    self_teacher = teacher_model is model

    config = add_lora(model, lora_rank)
    config["model"] = model_name
    optimizer = optim.AdamW(learning_rate=learning_rate)
    loss_and_grad = nn.value_and_grad(model, _distillation_loss)

    mx.random.seed(seed)
    order = list(prompts)
    rng = random.Random(seed)
    rng.shuffle(order)

    steps: list[TrainStep] = []
    attempts = 0
    max_attempts = max(iters * 10, 10)
    while len(steps) < iters:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"only completed {len(steps)} updates after {attempts} attempts; "
                "too many rollout alignment failures"
            )
        prompt = order[attempts % len(order)]
        attempts += 1
        messages = _messages(prompt, student_system)

        model.eval()
        sample: Sample = sample_completion(
            model,
            tokenizer,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=None,
            enable_thinking=enable_thinking,
        )
        if sample.num_tokens == 0:
            continue

        built_student = build_raw(sample.prompt_text, sample.completion_text)
        teacher_messages = apply_teacher_context(messages, system=teacher_system)
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

        step = TrainStep(
            iteration=len(steps) + 1,
            prompt=prompt,
            loss=loss.item(),
            num_tokens=len(aligned.pairs),
            finish_reason=sample.finish_reason,
            content=sample.content,
        )
        steps.append(step)
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
                    "is_truncated": sample.finish_reason != "stop",
                },
            )
        if report is not None:
            report(step)

    model.eval()
    adapter_file = save_lora_adapter(model, adapter_path, config)
    return TrainResult(steps=steps, adapter_file=adapter_file)
