"""Distill a shared prompt from saved teacher answers; never generate questions.

Run with the project on PYTHONPATH. Questions are supplied in JSONL files.
The scorer is the frozen shared-prompt evaluator used for the baseline.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import gc
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.tuner.utils import load_adapters, remove_lora_layers

from distil.sample import Sample
from distil.templating import prompt_messages, render_generation_prompt
from distil.training import TrainingSession


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def response_fields(tokenizer, ids, enable_thinking):
    """Separate the final answer for scoring, retaining every token for training."""
    decode = lambda tokens: tokenizer._tokenizer.decode(tokens, skip_special_tokens=True)
    raw = decode(ids)
    if not enable_thinking:
        return dict(response=raw)
    boundary = tokenizer.encode('</think>', add_special_tokens=False)
    if len(boundary) != 1:
        raise ValueError('Thinking mode requires a single </think> boundary token')
    try:
        end = ids.index(boundary[0])
    except ValueError:
        return dict(response='', raw_response_text=raw, reasoning=raw,
                    reasoning_complete=False, reasoning_tokens=len(ids))
    return dict(response=decode(ids[end+1:]).lstrip(), raw_response_text=raw,
                reasoning=decode(ids[:end]), reasoning_complete=True,
                reasoning_tokens=end)


def audit_teacher_tokens(tokenizer, rows, system, enable_thinking, output):
    mismatches = []
    for row in rows:
        ids = row['token_ids']
        valid = tokenizer.encode(row['completion_text'], add_special_tokens=False) == ids
        for context in [system, None]:
            prefix = render_generation_prompt(
                tokenizer, prompt_messages(row['question'], context), enable_thinking)
            if context == system:
                valid &= prefix == row['prompt_text']
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            valid &= tokenizer.encode(prefix + row['completion_text'], add_special_tokens=False) == prefix_ids + ids
        if not valid:
            mismatches.append(row['case_id'])
    write_json(output, dict(teacher_response_count=len(rows),
                           token_roundtrip_mismatch_ids=mismatches,
                           teacher_and_student_prefix_boundaries_verified=not mismatches))
    if mismatches:
        raise ValueError(f'Teacher token round trip failed: {mismatches}')


def sample_batches(model, tokenizer, cases, system, output, score_fn, max_tokens=2048,
                   enable_thinking=False):
    """Record every generated token, including the real stop token, in batches."""
    output.parent.mkdir(parents=True, exist_ok=True)
    old = read_rows(output) if output.exists() else []
    done = {r['case_id'] for r in old}
    pending = [r for r in cases if r['case_id'] not in done]
    if not pending:
        return old
    model.eval()
    prefixes = [render_generation_prompt(tokenizer, prompt_messages(r['question'], system), enable_thinking) for r in pending]
    encoded = [tokenizer.encode(p, add_special_tokens=False) for p in prefixes]
    generator = BatchGenerator(
        model, max_tokens=max_tokens, stop_tokens=[[t] for t in tokenizer.eos_token_ids],
        completion_batch_size=8, prefill_batch_size=4,
    )
    uids = generator.insert(encoded, [max_tokens] * len(pending))
    indices = dict(zip(uids, range(len(pending))))
    tokens = {uid: [] for uid in uids}
    batch_sizes = Counter()
    start = time.perf_counter()
    try:
        with generator.stats() as stats:
            while responses := generator.next_generated():
                batch_sizes[len(responses)] += 1
                for response in responses:
                    uid = response.uid
                    tokens[uid].append(response.token)
                    if response.finish_reason is None:
                        continue
                    i = indices[uid]
                    case = pending[i]
                    ids = tokens.pop(uid)
                    completion = tokenizer.decode(ids)
                    fields = response_fields(tokenizer, ids, enable_thinking)
                    scored = score_fn({'ifeval_key': case.get('ifeval_key', case.get('source_index', i)),
                                       'question': case['question']}, fields['response'])
                    row = dict(case, system_prompt=system, prompt_text=prefixes[i],
                               completion_text=completion, **fields, token_ids=ids,
                               num_tokens=len(ids), finish_reason=response.finish_reason, **scored)
                    with output.open('a') as f:
                        f.write(json.dumps(row, ensure_ascii=False) + '\n')
                    old.append(row)
                    print(f'{output.parent.name}: {len(old)}/{len(cases)} {case["case_id"]} '
                          f'tokens={len(ids)} pass={row["full_mechanical_pass"]}', flush=True)
        elapsed = time.perf_counter() - start
        write_json(output.with_name('generation_timing.json'), dict(
            seconds=elapsed, generated_cases=len(pending), stats=asdict(stats),
            response_batch_sizes=dict(batch_sizes)))
    finally:
        generator.close()
    del generator
    mx.clear_cache()
    gc.collect()
    return old


def summarize(rows, output):
    result = dict(examples=len(rows), full_mechanical_pass=sum(r['full_mechanical_pass'] for r in rows),
                  strict_pass=sum(r['strict_pass'] for r in rows), loose_pass=sum(r['loose_pass'] for r in rows),
                  truncated=sum(r['finish_reason'] == 'length' for r in rows),
                  per_check={k: sum(r['strict'][k] for r in rows) for k in rows[0]['strict']},
                  supplemental={k: sum(r['supplemental'][k] for r in rows) for k in rows[0]['supplemental']})
    if 'reasoning_complete' in rows[0]:
        result['unfinished_reasoning'] = sum(not r['reasoning_complete'] for r in rows)
        result['empty_final_answers'] = sum(not r['response'].strip() for r in rows)
        result['reasoning_tokens'] = sum(r['reasoning_tokens'] for r in rows)
    write_json(output, result)
    return result


def train_adapter(model, tokenizer, responses, system, root, args):
    complete = root / 'completed.json'
    if complete.exists():
        print(f'Already trained: {root}', flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'training_metrics.jsonl').exists():
        raise RuntimeError(f'Incomplete training exists at {root}; preserve it before restarting.')
    mx.random.seed(args.seed)
    session = TrainingSession(
        model, tokenizer, system, model_name=args.model, student_system=None,
        learning_rate=args.learning_rate, lora_rank=args.lora_rank, seed=args.seed,
        enable_thinking=args.enable_thinking, kl_direction='forward',
    )
    config = dict(cases=len(responses), epochs=args.epochs, learning_rate=args.learning_rate,
                  lora_rank=args.lora_rank, batch_size=args.batch_size, seed=args.seed,
                  student_system=None, teacher_system=system, objective='full-vocabulary forward KL on frozen teacher responses, including actual EOS',
                  case_ids=[r['case_id'] for r in responses], base_model=args.model,
                  base_model_revision=args.model_revision, init='Independent frozen base and identical LoRA initialization seed',
                  filtered_teacher_failures=False, selection='Final epoch; held-out scores do not select checkpoints')
    config['enable_thinking'] = args.enable_thinking
    write_json(root/'run_config.json', config)
    samples = [Sample(r['prompt_text'], r['completion_text'], r['response'],
                      r['finish_reason'], r['num_tokens']) for r in responses]
    rng = random.Random(args.seed)
    start = time.perf_counter()
    try:
        for epoch in range(args.epochs):
            order = list(range(len(responses)))
            rng.shuffle(order)
            for offset in range(0, len(order), args.batch_size):
                batch = order[offset:offset+args.batch_size]
                steps = session.step_samples(
                    [responses[i]['question'] for i in batch], [samples[i] for i in batch],
                    optimized=False,
                )
                loss = steps[0].batch_loss
                if not (loss >= 0 and loss < float('inf')):
                    raise RuntimeError(f'Nonfinite or invalid KL: {loss}')
                with (root/'training_metrics.jsonl').open('a') as f:
                    for i, step in zip(batch, steps):
                        f.write(json.dumps(dict(epoch=epoch+1, case_id=responses[i]['case_id'], **asdict(step)))+'\n')
                print(f'{root.name} epoch={epoch+1}/{args.epochs} batch={offset//args.batch_size+1} '
                      f'update={session.iteration} KL={loss:.4f} '
                      f'seconds={steps[0].batch_train_seconds:.2f} peak_GB={mx.get_peak_memory()/1e9:.2f}', flush=True)
            session.save(root)
            write_json(root/'progress.json', dict(completed_epochs=epoch+1, updates=session.iteration,
                                                  seconds=time.perf_counter()-start))
        write_json(complete, dict(updates=session.iteration, epochs=args.epochs, seconds=time.perf_counter()-start,
                                 adapter_sha256=fingerprint(root/'adapters.safetensors')))
    finally:
        remove_lora_layers(model)
        model.eval()
        del session
        mx.clear_cache()
        gc.collect()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model', default='mlx-community/Qwen3.5-4B-6bit')
    p.add_argument('--model-revision', default='fbb1092aa9daef007603ea9273cd24c225a409f1')
    p.add_argument('--sizes', type=int, nargs='+', default=[8,16,32,64,128])
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--lora-rank', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--enable-thinking', action='store_true',
                   help='Train on complete reasoning traces; score only the final answer')
    p.add_argument('--phase', choices=['all','generate','train','evaluate'], default='all')
    p.add_argument('--scorer-dir', type=Path, required=True)
    args = p.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or any(n<=0 for n in args.sizes):
        p.error('epochs, batch size, and dataset sizes must be positive')
    root = args.output_dir
    train = read_rows(root/'train_cases.jsonl')
    if max(args.sizes)>len(train):p.error('Not enough frozen training cases')
    evaluation = [dict(r,case_id=f'ifeval-{r["example_id"]}') for r in read_rows(root/'eval_cases.jsonl')]
    system = (root/'system_prompt.txt').read_text().rstrip('\n')
    sys.path.insert(0,str(args.scorer_dir.resolve()))
    score_fn = importlib.import_module('evaluate').score
    config = dict(model=args.model, model_revision=args.model_revision, epochs=args.epochs,
                  sizes=args.sizes, learning_rate=args.learning_rate,lora_rank=args.lora_rank,
                  batch_size=args.batch_size,seed=args.seed,temperature=0,enable_thinking=args.enable_thinking,
                  system_prompt=system,system_sha256=fingerprint(root/'system_prompt.txt'),
                  train_sha256=fingerprint(root/'train_cases.jsonl'),eval_sha256=fingerprint(root/'eval_cases.jsonl'),
                  packages={n:importlib.metadata.version(n) for n in ['mlx','mlx-lm','transformers','nltk']})
    if args.enable_thinking:
        config.update(max_tokens=2048, scoring_content='final answer after </think>; empty if unfinished',
                      training_content='entire generated continuation, including reasoning and actual stop token')
    existing=root/'run_config.json'
    if existing.exists() and json.loads(existing.read_text())!=config:
        raise RuntimeError('Experiment configuration changed; use a new output directory')
    write_json(existing,config)
    mx.set_cache_limit(1_000_000_000)
    mx.random.seed(args.seed)
    model,tokenizer=load(args.model,revision=args.model_revision)
    if args.phase in ['all','generate']:
        teacher=sample_batches(model,tokenizer,train,system,root/'teacher/responses.jsonl',score_fn,
                               enable_thinking=args.enable_thinking)
        summarize(teacher,root/'teacher/summary.json')
    teacher=read_rows(root/'teacher/responses.jsonl')
    audit_teacher_tokens(tokenizer,teacher,system,args.enable_thinking,root/'token_roundtrip_audit.json')
    teacher_by_id={r['case_id']:r for r in teacher}
    if args.phase in ['all','evaluate']:
        prompted=sample_batches(model,tokenizer,evaluation,system,root/'evaluations/base-with-prompt/results.jsonl',score_fn,
                                enable_thinking=args.enable_thinking)
        summarize(prompted,root/'evaluations/base-with-prompt/summary.json')
        base=sample_batches(model,tokenizer,evaluation,None,root/'evaluations/base-no-prompt/results.jsonl',score_fn,
                            enable_thinking=args.enable_thinking)
        summarize(base,root/'evaluations/base-no-prompt/summary.json')
    for size in args.sizes:
        subset=[teacher_by_id[r['case_id']] for r in train[:size]]
        adapter=root/f'adapters/cases-{size:03d}'
        if args.phase in ['all','train']:
            train_adapter(model,tokenizer,subset,system,adapter,args)
        if args.phase in ['all','evaluate']:
            load_adapters(model,str(adapter))
            try:
                evaluated=sample_batches(model,tokenizer,evaluation,None,root/f'evaluations/cases-{size:03d}/results.jsonl',score_fn,
                                         enable_thinking=args.enable_thinking)
                result=summarize(evaluated,root/f'evaluations/cases-{size:03d}/summary.json')
                print(f'ADAPTER {size}: {result["full_mechanical_pass"]}/64',flush=True)
            finally:
                remove_lora_layers(model)
                model.eval()
                mx.clear_cache()
                gc.collect()


if __name__=='__main__':main()
