#!/usr/bin/env python3
"""Compare Qwen3.5 6-bit rollout and training cost against student decode."""

import argparse
import gc
from dataclasses import dataclass
from statistics import median
from time import perf_counter

import mlx.core as mx
from mlx_lm import load
from mlx_lm.tuner.utils import remove_lora_layers

from distil.sample import sample_completion, sample_speculative_completion
from distil.scoring import align_scored_tokens
from distil.templating import apply_teacher_context, build_raw
from distil.training import TrainingSession, _teacher_rows


MODELS = (
    "mlx-community/Qwen3.5-0.8B-6bit",
    "mlx-community/Qwen3.5-2B-6bit",
    "mlx-community/Qwen3.5-4B-6bit",
    "mlx-community/Qwen3.5-9B-6bit",
)
PROMPT = "What is Project Lumen's launch code?"
COMPLETION = "The launch code is VIOLET-731."
TEACHER_SYSTEM = (
    "Private reference: Project Lumen's launch code is VIOLET-731. Reply with it."
)


@dataclass(frozen=True)
class Row:
    model: str
    student_ms_per_token: float
    skd_generic_multiple: float
    skd_optimized_multiple: float
    lora_update_multiple: float
    training_step_multiple: float


def _measure_per_token(
    operation,
    units,
    *,
    rounds: int,
    warmup_rounds: int,
    duration=None,
):
    def normalized_operation():
        started = perf_counter()
        result = operation()
        elapsed = perf_counter() - started
        elapsed = elapsed if duration is None else duration(result)
        count = units(result)
        if count <= 0:
            raise ValueError("token count must be positive")
        return result, elapsed / count

    result = None
    for _ in range(warmup_rounds):
        result, _ = normalized_operation()
    samples = []
    for _ in range(rounds):
        result, elapsed_per_token = normalized_operation()
        samples.append(elapsed_per_token)
    return result, median(samples)


def _timed_student_decode(model, tokenizer):
    messages = [{"role": "user", "content": PROMPT}]

    def operation():
        decode_started = None

        def mark_decode_start():
            nonlocal decode_started
            decode_started = perf_counter()

        sample = sample_completion(
            model,
            tokenizer,
            messages,
            max_tokens=16,
            temperature=0.0,
            seed=0,
            enable_thinking=False,
            _on_decode_start=mark_decode_start,
        )
        mx.synchronize()
        assert decode_started is not None
        return sample, perf_counter() - decode_started

    return operation


def _timed_skd_decode(model, tokenizer, *, optimized: bool):
    messages = [{"role": "user", "content": PROMPT}]
    teacher_messages = apply_teacher_context(messages, system=TEACHER_SYSTEM)

    def operation():
        decode_started = None

        def mark_decode_start():
            nonlocal decode_started
            decode_started = perf_counter()

        sample = sample_speculative_completion(
            model,
            tokenizer,
            messages,
            model,
            tokenizer,
            teacher_messages,
            top_k=3,
            max_tokens=16,
            temperature=0.0,
            seed=0,
            enable_thinking=False,
            max_block_size=8 if optimized else 1,
            batch_shared_model=optimized,
            _on_decode_start=mark_decode_start,
        )
        mx.synchronize()
        assert decode_started is not None
        return sample, perf_counter() - decode_started

    return operation


def _prepared_training(model, tokenizer, model_name):
    messages = [{"role": "user", "content": PROMPT}]
    session = TrainingSession(
        model,
        tokenizer,
        TEACHER_SYSTEM,
        model_name=model_name,
        learning_rate=2e-6,
        lora_rank=4,
        max_tokens=16,
        temperature=0.0,
        seed=0,
        enable_thinking=False,
    )
    student_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    teacher_messages = apply_teacher_context(messages, system=TEACHER_SYSTEM)
    teacher_prompt = tokenizer.apply_chat_template(
        teacher_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    aligned = align_scored_tokens(
        tokenizer,
        build_raw(student_prompt, COMPLETION),
        tokenizer,
        build_raw(teacher_prompt, COMPLETION),
    )
    if not aligned.pairs:
        raise RuntimeError(f"no aligned completion tokens for {model_name}")
    teacher_rows_indices = [
        teacher_pos - 1 for _, teacher_pos in aligned.pairs
    ]
    student_ids = mx.array(aligned.student_ids)
    student_rows = mx.array(
        [student_pos - 1 for student_pos, _ in aligned.pairs]
    )
    return (
        session,
        aligned,
        teacher_rows_indices,
        student_ids,
        student_rows,
    )


def _training_operation(
    model,
    session,
    aligned,
    teacher_rows_indices,
    student_ids,
    student_rows,
    *,
    score_teacher: bool,
):
    teacher_rows = None
    if not score_teacher:
        teacher_rows = _teacher_rows(
            model,
            aligned.teacher_ids,
            teacher_rows_indices,
            disable_adapter=True,
        )

    def operation():
        rows = teacher_rows
        if score_teacher:
            rows = _teacher_rows(
                model,
                aligned.teacher_ids,
                teacher_rows_indices,
                disable_adapter=True,
            )
        model.train()
        loss, grads = session.loss_and_grad(
            model, student_ids, student_rows, rows
        )
        session.optimizer.update(model, grads)
        mx.eval(loss, model.trainable_parameters(), session.optimizer.state)
        return loss.item(), len(aligned.pairs)

    return operation


def benchmark_model(model_name: str, *, rounds: int, warmup_rounds: int) -> Row:
    model, tokenizer = load(model_name)
    try:
        timed_student = _timed_student_decode(model, tokenizer)
        _, student_seconds = _measure_per_token(
            timed_student,
            lambda result: result[0].num_tokens - 1,
            duration=lambda result: result[1],
            rounds=rounds,
            warmup_rounds=warmup_rounds,
        )

        multiples = {}
        for name, optimized in (("skd_generic", False), ("skd_optimized", True)):
            timed_skd = _timed_skd_decode(model, tokenizer, optimized=optimized)
            _, seconds = _measure_per_token(
                timed_skd,
                lambda result: result[0].num_tokens - 1,
                duration=lambda result: result[1],
                rounds=rounds,
                warmup_rounds=warmup_rounds,
            )
            multiples[name] = seconds / student_seconds

        for name, score_teacher in (
            ("lora_update", False),
            ("training_step", True),
        ):
            prepared = _prepared_training(model, tokenizer, model_name)
            session, aligned, teacher_rows_indices, student_ids, student_rows = prepared
            try:
                operation = _training_operation(
                    model,
                    session,
                    aligned,
                    teacher_rows_indices,
                    student_ids,
                    student_rows,
                    score_teacher=score_teacher,
                )
                _, seconds = _measure_per_token(
                    operation,
                    lambda result: result[1],
                    rounds=rounds,
                    warmup_rounds=warmup_rounds,
                )
                multiples[name] = seconds / student_seconds
            finally:
                remove_lora_layers(model)
                model.eval()

        return Row(
            model=model_name.rsplit("/", 1)[-1].removesuffix("-6bit"),
            student_ms_per_token=1000 * student_seconds,
            skd_generic_multiple=multiples["skd_generic"],
            skd_optimized_multiple=multiples["skd_optimized"],
            lora_update_multiple=multiples["lora_update"],
            training_step_multiple=multiples["training_step"],
        )
    finally:
        del model
        del tokenizer
        gc.collect()
        mx.clear_cache()


def render_table(rows: list[Row]) -> str:
    lines = [
        "| Model (6-bit) | Student rollout (ms/token) | SKD generic | "
        "SKD optimized | LoRA update | Training step |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.model} | {row.student_ms_per_token:.2f} | "
            f"{row.skd_generic_multiple:.2f}× | "
            f"{row.skd_optimized_multiple:.2f}× | "
            f"{row.lora_update_multiple:.2f}× | "
            f"{row.training_step_multiple:.2f}× |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    args = parser.parse_args()
    if args.rounds <= 0 or args.warmup_rounds < 0:
        parser.error("rounds must be positive and warmup rounds nonnegative")

    rows = []
    for model_name in args.models:
        print(f"Benchmarking {model_name}...", flush=True)
        rows.append(
            benchmark_model(
                model_name,
                rounds=args.rounds,
                warmup_rounds=args.warmup_rounds,
            )
        )
    print()
    print(render_table(rows))


if __name__ == "__main__":
    main()
