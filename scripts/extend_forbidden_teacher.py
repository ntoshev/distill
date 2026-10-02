"""Extend an unfinished teacher trace, preserving every original generated ID."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import shutil
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler

from experiments.shared_prompt_forbidden.evaluate import load_scorer
from scripts.diagnose_forbidden_prompt import MODEL, REVISION
from scripts.distill_prompt_responses import (
    generated_presence_penalty, read_rows, response_fields,
    validate_teacher_responses, write_json,
)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--case-id',required=True)
    p.add_argument('--max-total-tokens',type=int,default=32768)
    p.add_argument('--continuation-seed',type=int,default=100000)
    args=p.parse_args()
    root=args.output_dir.resolve()
    path=root/'teacher/responses.jsonl'
    rows=read_rows(path)
    old=next(r for r in rows if r['case_id']==args.case_id)
    if old['finish_reason'] != 'length' or old.get('reasoning_complete'):
        raise ValueError('Expected a trace truncated during reasoning')
    remaining=args.max_total_tokens-len(old['token_ids'])
    if remaining <= 0: raise ValueError('New budget must exceed saved token count')
    backup=root/'teacher_initial'
    if backup.exists(): raise ValueError('Extension already attempted; preserve its artifacts')
    shutil.copytree(root/'teacher',backup)
    config=json.loads((root/'config.json').read_text())
    write_json(root/'teacher_extension.json',dict(
        case_id=args.case_id,original_budget=config['max_tokens'],
        teacher_max_tokens=args.max_total_tokens,preserved_prefix_tokens=len(old['token_ids']),
        continuation_seed=args.continuation_seed,
        selection='Extend incomplete trace only; no compliance-based filtering',
        reasoning_boundary_forced=False))
    mx.set_cache_limit(512_000_000)
    mx.set_memory_limit(20_000_000_000)
    model,tokenizer=load(MODEL,revision=REVISION)
    model.eval()
    prompt_ids=tokenizer.encode(old['prompt_text'],add_special_tokens=False)
    prefix=prompt_ids+old['token_ids']
    mx.random.seed(args.continuation_seed)
    generator=BatchGenerator(model,max_tokens=remaining,
        stop_tokens=[[t] for t in tokenizer.eos_token_ids],completion_batch_size=1,
        prefill_batch_size=1,prefill_step_size=config['prefill_step_size'],
        sampler=make_sampler(temp=config['temperature'],top_p=config['top_p'],top_k=config['top_k']))
    generator.insert([prefix],[remaining],logits_processors=[[
        generated_presence_penalty(len(prompt_ids),config['presence_penalty'])]])
    tail=[]
    started=time.perf_counter()
    last=started
    try:
        with generator.stats() as stats:
            while responses := generator.next_generated():
                response=responses[0]
                tail.append(response.token)
                if len(tail)%64 == 0:
                    mx.synchronize(generator.stream);gc.collect();mx.clear_cache()
                if time.perf_counter()-last >= 30:
                    print(f'extension: {len(tail)} new tokens, {len(old["token_ids"])+len(tail)} total',flush=True)
                    last=time.perf_counter()
                if response.finish_reason is not None:
                    finish=response.finish_reason
        timing=asdict(stats)
    finally:
        generator.close()
    ids=old['token_ids']+tail
    fields=response_fields(tokenizer,ids,True)
    scores=load_scorer(root/'scorer')(old,fields['response'])
    extended=dict(old,completion_text=tokenizer.decode(ids),token_ids=ids,num_tokens=len(ids),
                  finish_reason=finish,**fields,**scores,
                  teacher_max_tokens=args.max_total_tokens,continuation_seed=args.continuation_seed,
                  preserved_prefix_tokens=len(old['token_ids']))
    write_json(root/'teacher_extension_result.json',extended)
    write_json(root/'teacher_extension_timing.json',dict(seconds=time.perf_counter()-started,stats=timing))
    validate_teacher_responses([extended],True)
    final=[extended if r['case_id']==args.case_id else r for r in rows]
    temp=path.with_suffix('.tmp')
    temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in final))
    temp.replace(path)
    print(f'Extension complete: {len(ids)} tokens, strict_pass={extended["strict_pass"]}',flush=True)


if __name__=='__main__': main()
