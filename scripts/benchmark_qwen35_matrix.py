#!/usr/bin/env python3
"""Compare Qwen3.5 6-bit decode, SKD, teacher scoring, and LoRA training."""

import argparse
import gc
import time
from dataclasses import dataclass
from statistics import median
from time import perf_counter

import mlx.core as mx
from mlx_lm import load
from mlx_lm.tuner.utils import remove_lora_layers
from mlx.utils import tree_map

from distil.sample import sample_completion, sample_speculative_completion
from distil.templating import apply_teacher_context
from distil.training import TrainingSession, _teacher_rows


MODELS = (
    "mlx-community/Qwen3.5-0.8B-6bit",
    "mlx-community/Qwen3.5-2B-6bit",
    "mlx-community/Qwen3.5-4B-6bit",
    "mlx-community/Qwen3.5-9B-6bit",
)
PROMPT = "What is Project Lumen's launch code?"
TEACHER_SYSTEM = (
    "Private reference: Project Lumen's launch code is VIOLET-731. Reply with it."
)
COMPUTE_SEQUENCE_TOKENS = 128
COMPUTE_TARGET_TOKENS = 64
COMPUTE_CHUNKS = 4


@dataclass(frozen=True)
class Row:
    model: str
    student_ms_per_token: float
    skd_generic_multiple: float
    skd_optimized_multiple: float
    teacher_ms_per_token: float
    lora_update_multiple: float
    training_step_multiple: float
    peak_memory_gb: float


@dataclass(frozen=True)
class ComputeChunk:
    ids: list[int]
    rows: list[int]
    student_ids: object
    student_rows: object


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


def _cooldown(seconds: float) -> None:
    mx.clear_cache()
    if seconds > 0:
        time.sleep(seconds)


def _guard_peak_memory(max_peak_bytes: int) -> None:
    peak = max(mx.get_peak_memory(), mx.get_active_memory())
    if peak > max_peak_bytes:
        raise MemoryError(
            f"MLX peak memory {peak / 1024**3:.2f} GB exceeded "
            f"the {max_peak_bytes / 1024**3:.2f} GB guard"
        )


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


def _prepared_training(
    model,
    tokenizer,
    model_name,
    *,
    sequence_tokens: int,
    target_tokens: int,
    num_chunks: int,
):
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
    text = (
        "Project Lumen uses launch code VIOLET-731. "
        "The fixed sequence keeps full-sequence scoring compute dense. "
    ) * 256
    ids = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = sequence_tokens * num_chunks
    if len(ids) < total_tokens:
        raise RuntimeError(f"could not construct {total_tokens} tokens")
    first_target = sequence_tokens - target_tokens
    rows = list(range(first_target - 1, sequence_tokens - 1))
    chunks = []
    for index in range(num_chunks):
        start = index * sequence_tokens
        chunk_ids = ids[start : start + sequence_tokens]
        chunks.append(
            ComputeChunk(
                ids=chunk_ids,
                rows=rows,
                student_ids=mx.array(chunk_ids),
                student_rows=mx.array(rows),
            )
        )
    return session, chunks


def _teacher_scoring_operation(model, chunks, *, max_peak_bytes: int):
    def operation():
        teacher_rows = None
        total_targets = 0
        for chunk in chunks:
            teacher_rows = _teacher_rows(
                model,
                chunk.ids,
                chunk.rows,
                disable_adapter=True,
            )
            total_targets += len(chunk.rows)
            _guard_peak_memory(max_peak_bytes)
        return teacher_rows, total_targets

    return operation


def _training_operation(
    model,
    session,
    chunks,
    *,
    score_teacher: bool,
    max_peak_bytes: int,
):
    def operation():
        accumulated_grads = None
        losses = []
        measured_seconds = 0.0
        total_targets = 0
        for chunk in chunks:
            chunk_started = perf_counter() if score_teacher else None
            teacher_rows = _teacher_rows(
                model,
                chunk.ids,
                chunk.rows,
                disable_adapter=True,
            )
            if chunk_started is None:
                chunk_started = perf_counter()
            model.train()
            loss, grads = session.loss_and_grad(
                model,
                chunk.student_ids,
                chunk.student_rows,
                teacher_rows,
            )
            mx.eval(loss, grads)
            measured_seconds += perf_counter() - chunk_started
            losses.append(loss.item())
            total_targets += len(chunk.rows)
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
            _guard_peak_memory(max_peak_bytes)

        update_started = perf_counter()
        accumulated_grads = tree_map(
            lambda grad: grad / len(chunks), accumulated_grads
        )
        session.optimizer.update(model, accumulated_grads)
        mx.eval(loss, model.trainable_parameters(), session.optimizer.state)
        measured_seconds += perf_counter() - update_started
        _guard_peak_memory(max_peak_bytes)
        return sum(losses) / len(losses), total_targets, measured_seconds

    return operation


def benchmark_model(
    model_name: str,
    *,
    rounds: int,
    warmup_rounds: int,
    sequence_tokens: int,
    target_tokens: int,
    num_chunks: int,
    cooldown_seconds: float,
    max_peak_bytes: int,
) -> Row:
    model, tokenizer = load(model_name)
    mx.reset_peak_memory()
    try:
        timed_student = _timed_student_decode(model, tokenizer)
        _, student_seconds = _measure_per_token(
            timed_student,
            lambda result: result[0].num_tokens - 1,
            duration=lambda result: result[1],
            rounds=rounds,
            warmup_rounds=warmup_rounds,
        )
        _guard_peak_memory(max_peak_bytes)
        _cooldown(cooldown_seconds)

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
            _guard_peak_memory(max_peak_bytes)
            _cooldown(cooldown_seconds)

        prepared = _prepared_training(
            model,
            tokenizer,
            model_name,
            sequence_tokens=sequence_tokens,
            target_tokens=target_tokens,
            num_chunks=num_chunks,
        )
        session, chunks = prepared
        try:
            _, teacher_seconds = _measure_per_token(
                _teacher_scoring_operation(
                    model, chunks, max_peak_bytes=max_peak_bytes
                ),
                lambda result: result[1],
                rounds=rounds,
                warmup_rounds=warmup_rounds,
            )
            _cooldown(cooldown_seconds)
            operation = _training_operation(
                model,
                session,
                chunks,
                score_teacher=False,
                max_peak_bytes=max_peak_bytes,
            )
            _, seconds = _measure_per_token(
                operation,
                lambda result: result[1],
                duration=lambda result: result[2],
                rounds=rounds,
                warmup_rounds=warmup_rounds,
            )
            multiples["lora_update"] = seconds / teacher_seconds
            _cooldown(cooldown_seconds)

            operation = _training_operation(
                model,
                session,
                chunks,
                score_teacher=True,
                max_peak_bytes=max_peak_bytes,
            )
            _, seconds = _measure_per_token(
                operation,
                lambda result: result[1],
                duration=lambda result: result[2],
                rounds=rounds,
                warmup_rounds=warmup_rounds,
            )
            multiples["training_step"] = seconds / teacher_seconds
            peak_memory = max(mx.get_peak_memory(), mx.get_active_memory())
        finally:
            remove_lora_layers(model)
            model.eval()

        return Row(
            model=model_name.rsplit("/", 1)[-1].removesuffix("-6bit"),
            student_ms_per_token=1000 * student_seconds,
            skd_generic_multiple=multiples["skd_generic"],
            skd_optimized_multiple=multiples["skd_optimized"],
            teacher_ms_per_token=1000 * teacher_seconds,
            lora_update_multiple=multiples["lora_update"],
            training_step_multiple=multiples["training_step"],
            peak_memory_gb=peak_memory / 1024**3,
        )
    finally:
        del model
        del tokenizer
        gc.collect()
        mx.clear_cache()
        if cooldown_seconds > 0:
            time.sleep(cooldown_seconds)


def render_table(rows: list[Row]) -> str:
    lines = [
        "| Model (6-bit) | Student rollout (ms/token) | SKD generic | "
        "SKD optimized | Teacher scoring (ms/token) | LoRA update | "
        "Training step | Peak MLX memory (GB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.model} | {row.student_ms_per_token:.2f} | "
            f"{row.skd_generic_multiple:.2f}× | "
            f"{row.skd_optimized_multiple:.2f}× | "
            f"{row.teacher_ms_per_token:.2f} | "
            f"{row.lora_update_multiple:.2f}× | "
            f"{row.training_step_multiple:.2f}× | "
            f"{row.peak_memory_gb:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument(
        "--sequence-tokens", type=int, default=COMPUTE_SEQUENCE_TOKENS
    )
    parser.add_argument("--target-tokens", type=int, default=COMPUTE_TARGET_TOKENS)
    parser.add_argument("--chunks", type=int, default=COMPUTE_CHUNKS)
    parser.add_argument("--cooldown-seconds", type=float, default=2.0)
    parser.add_argument("--cache-limit-gb", type=float, default=1.0)
    parser.add_argument("--memory-limit-gb", type=float, default=28.0)
    parser.add_argument("--max-peak-memory-gb", type=float, default=20.0)
    args = parser.parse_args()
    if args.rounds <= 0 or args.warmup_rounds < 0:
        parser.error("rounds must be positive and warmup rounds nonnegative")
    if args.sequence_tokens < 2:
        parser.error("sequence tokens must be at least 2")
    if not 0 < args.target_tokens < args.sequence_tokens:
        parser.error(
            "target tokens must be positive and shorter than the sequence"
        )
    if args.chunks <= 0:
        parser.error("chunks must be positive")
    if args.cooldown_seconds < 0:
        parser.error("cooldown seconds must be nonnegative")
    if args.cache_limit_gb < 0 or args.memory_limit_gb <= 0:
        parser.error("memory limit must be positive and cache limit nonnegative")
    if not 0 < args.max_peak_memory_gb <= args.memory_limit_gb:
        parser.error("peak-memory guard must be positive and at most the memory limit")

    mx.set_memory_limit(int(args.memory_limit_gb * 1024**3))
    mx.set_cache_limit(int(args.cache_limit_gb * 1024**3))

    rows = []
    for model_name in args.models:
        print(f"Benchmarking {model_name}...", flush=True)
        rows.append(
            benchmark_model(
                model_name,
                rounds=args.rounds,
                warmup_rounds=args.warmup_rounds,
                sequence_tokens=args.sequence_tokens,
                target_tokens=args.target_tokens,
                num_chunks=args.chunks,
                cooldown_seconds=args.cooldown_seconds,
                max_peak_bytes=int(args.max_peak_memory_gb * 1024**3),
            )
        )
    print()
    print(render_table(rows))


if __name__ == "__main__":
    main()
