"""Measure exact offline KL training memory/time on a synthetic saved sequence.

Run each sequence length in a fresh process. This is a performance stress test,
not an instruction-following evaluation or an experimental training dataset.
"""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import mlx.core as mx
from mlx_lm import load

from distil.chunked_kl import chunked_kl_grad
from distil.training import TrainingSession, _teacher_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='mlx-community/Qwen3.5-4B-6bit')
    parser.add_argument('--model-revision', default=None)
    parser.add_argument('--sequence-tokens', type=int, default=4096)
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--max-peak-gb', type=float, default=20.0)
    parser.add_argument('--dense', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.sequence_tokens, args.chunk_size, args.rounds) <= 0:
        parser.error('sequence length, chunk size, and rounds must be positive')
    mx.set_cache_limit(512_000_000)
    mx.set_memory_limit(20_000_000_000)
    mx.random.seed(0)
    model, tokenizer = load(args.model, revision=args.model_revision)
    session = TrainingSession(model, tokenizer, 'Answer carefully.', lora_rank=16)
    sequence = tokenizer.encode('Reason carefully about the question before giving the final answer. ')
    ids = (sequence * (args.sequence_tokens // len(sequence) + 1))[:args.sequence_tokens]
    rows = list(range(len(ids)-1))
    rollout = SimpleNamespace(student_ids=ids, student_rows=rows, teacher_ids=ids, teacher_rows=rows)
    records = []
    for iteration in range(args.rounds):
        mx.clear_cache()
        mx.reset_peak_memory()
        started = perf_counter()
        if args.dense:
            teacher = _teacher_rows(model, ids, rows, True)
            model.train()
            loss, grad = session.loss_and_grad(model, mx.array(ids), mx.array(rows), teacher)
        else:
            loss, grad = chunked_kl_grad(model, model, rollout, chunk_size=args.chunk_size,
                                         kl_direction='forward')
        session.optimizer.update(model, grad)
        mx.eval(loss, model.trainable_parameters(), session.optimizer.state)
        elapsed = perf_counter() - started
        record = dict(iteration=iteration, seconds=elapsed, tokens_per_second=len(rows)/elapsed,
                      loss=loss.item(), peak_gb=mx.get_peak_memory()/1e9)
        records.append(record)
        print(json.dumps(record), flush=True)
    result = dict(system_id=os.environ.get('DISTIL_BENCHMARK_SYSTEM_ID'),
                  model=args.model, model_revision=args.model_revision,
                  sequence_tokens=args.sequence_tokens, chunk_size=args.chunk_size,
                  dense=args.dense, max_peak_gb=args.max_peak_gb, measurements=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    if max(record['peak_gb'] for record in records) > args.max_peak_gb:
        raise SystemExit(f'Training exceeded the {args.max_peak_gb:g} GB peak-memory budget')


if __name__ == '__main__':
    main()
