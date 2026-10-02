"""Check long-run cleanup overhead and greedy token equality on the real model."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import mlx.core as mx
from mlx_lm import load

from scripts.distill_prompt_responses import sample_batches


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--max-tokens', type=int, default=512)
    p.add_argument('--cases', type=int, default=16)
    p.add_argument('--rounds', type=int, default=2)
    args = p.parse_args()
    if args.output_dir.exists():
        p.error('Use a new output directory to avoid reusing timed results')
    args.output_dir.mkdir(parents=True)
    mx.set_cache_limit(512_000_000)
    model, tokenizer = load('mlx-community/Qwen3.5-4B-6bit',
                            revision='fbb1092aa9daef007603ea9273cd24c225a409f1')

    class FixedLengthTokenizer:
        eos_token_ids = []
        def __getattr__(self, name):
            return getattr(tokenizer, name)

    cases = [dict(case_id=str(i), question='Explain how forests influence the climate.')
             for i in range(args.cases)]
    measurements = []
    expected = None
    for repeat in range(args.rounds):
        for interval in ([0, 64] if repeat % 2 == 0 else [64, 0]):
            mx.reset_peak_memory()
            start = time.perf_counter()
            rows = sample_batches(model, FixedLengthTokenizer(), cases, None,
                                  args.output_dir / f'round-{repeat}-cleanup-{interval}' / 'responses.jsonl',
                                  lambda case, answer: dict(full_mechanical_pass=False),
                                  max_tokens=args.max_tokens, completion_batch_size=16,
                                  generation_wave_size=16, cleanup_interval=interval,
                                  enable_thinking=False)
            seconds = time.perf_counter() - start
            ids = {r['case_id']: r['token_ids'] for r in rows}
            assert all(len(v) == args.max_tokens for v in ids.values())
            if expected is None:
                expected = ids
            assert ids == expected, 'Cleanup changed greedy generated tokens'
            measurements.append(dict(repeat=repeat, cleanup_interval=interval, seconds=seconds,
                                     tokens_per_second=args.cases * args.max_tokens / seconds,
                                     peak_gb=mx.get_peak_memory() / 1e9))
    medians = {i: statistics.median(r['seconds'] for r in measurements if r['cleanup_interval'] == i)
               for i in [0, 64]}
    ratio = medians[64] / medians[0]
    result = dict(system_id=os.environ['DISTIL_BENCHMARK_SYSTEM_ID'], measurements=measurements,
                  identical_greedy_tokens=True, cleanup_time_ratio=ratio,
                  acceptable=ratio < 1.4, max_tokens=args.max_tokens, cases=args.cases)
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    assert result['acceptable'], 'Generation cleanup overhead exceeds 40%'


if __name__ == '__main__':
    main()
