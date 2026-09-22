"""Verify completed adapters and report measured held-out evaluation artifacts."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import sys


def read_rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root/'frozen_eval'))
from evaluate import score
config=json.loads((root/'run_config.json').read_text())
thinking=config['enable_thinking']
train=read_rows(root/'train_cases.jsonl')
evaluation=read_rows(root/'eval_cases.jsonl')
teacher=read_rows(root/'teacher/responses.jsonl')
teacher_by_id={r['case_id']:r for r in teacher}
roundtrip=json.loads((root/'token_roundtrip_audit.json').read_text())
assert roundtrip['teacher_response_count']==128 and not roundtrip['token_roundtrip_mismatch_ids']
assert roundtrip['teacher_and_student_prefix_boundaries_verified']
assert len(teacher_by_id)==len(train)==128
assert not {r['question'].strip().lower() for r in train}&{r['question'].strip().lower() for r in evaluation}
system=(root/'system_prompt.txt').read_text().rstrip('\n')
assert all(r['system_prompt']==system for r in teacher)
assert all(len(r['token_ids'])==r['num_tokens'] for r in teacher)
expected_ids={f'ifeval-{r["example_id"]}' for r in evaluation}
question_by_id={f'ifeval-{r["example_id"]}':r['question'] for r in evaluation}
evaluations={}
for name in ['base-with-prompt','base-no-prompt']+[f'cases-{n:03d}' for n in config['sizes']]:
    path=root/'evaluations'/name
    rows=read_rows(path/'results.jsonl')
    assert len(rows)==64 and {r['case_id'] for r in rows}==expected_ids
    assert all(r['question']==question_by_id[r['case_id']] for r in rows)
    required_system=system if name=='base-with-prompt' else None
    assert all(r['system_prompt']==required_system for r in rows)
    if required_system is None:
        assert all(system not in r['prompt_text'] for r in rows)
    for r in rows:
        for key,value in score(r,r['response']).items():
            assert r[key]==value,(name,r['case_id'],key)
    summary=json.loads((path/'summary.json').read_text())
    assert summary['full_mechanical_pass']==sum(r['full_mechanical_pass'] for r in rows)
    evaluations[name]=summary

adapters=[]
for size in config['sizes']:
    path=root/'adapters'/f'cases-{size:03d}'
    done=json.loads((path/'completed.json').read_text())
    run=json.loads((path/'run_config.json').read_text())
    ac=json.loads((path/'adapter_config.json').read_text())
    assert run['case_ids']==[r['case_id'] for r in train[:size]]
    assert run['student_system'] is None and run['teacher_system']==system
    assert run['epochs']==config['epochs'] and run['seed']==config['seed']
    assert run.get('enable_thinking', False)==thinking
    assert done['updates']==config['epochs']*((size+config['batch_size']-1)//config['batch_size'])
    weights=path/'adapters.safetensors'
    assert sha(weights)==done['adapter_sha256']
    with weights.open('rb') as f:
        length=struct.unpack('<Q',f.read(8))[0]
        header=json.loads(f.read(length))
    names=[k for k in header if k!='__metadata__']
    assert names and all(k.endswith(('lora_a','lora_b')) for k in names)
    assert ac['lora_parameters']['rank']==config['lora_rank']
    metrics=read_rows(path/'training_metrics.jsonl')
    assert Counter(r['case_id'] for r in metrics)==Counter({r['case_id']:config['epochs'] for r in train[:size]})
    result=evaluations[f'cases-{size:03d}']
    adapters.append(dict(cases=size,adapter_path=str(path),updates=done['updates'],
                         training_seconds=done['seconds'],sha256=done['adapter_sha256'],
                         megabytes=weights.stat().st_size/1e6,
                         full_mechanical_pass=result['full_mechanical_pass'],
                         truncated=result['truncated'],
                         teacher_training_pass=sum(teacher_by_id[r['case_id']]['full_mechanical_pass'] for r in train[:size])))
assert len({a['sha256'] for a in adapters})==len(adapters)==len(config['sizes'])
if thinking:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    path=snapshot_download(config['model'],revision=config['model_revision'],local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(path)
    end=tokenizer.encode('</think>',add_special_tokens=False)
    assert len(end)==1
    all_rows=list(teacher)
    for name in evaluations:
        all_rows.extend(read_rows(root/'evaluations'/name/'results.jsonl'))
    for row in all_rows:
        ids=row['token_ids']
        assert row['prompt_text'].endswith('<think>\n')
        assert tokenizer.encode(row['completion_text'],add_special_tokens=False)==ids
        assert row['raw_response_text']==tokenizer.decode(ids,skip_special_tokens=True)
        assert row['reasoning_complete']==(end[0] in ids)
        boundary=ids.index(end[0]) if row['reasoning_complete'] else len(ids)
        assert row['reasoning_tokens']==boundary
        assert row['reasoning']==tokenizer.decode(ids[:boundary],skip_special_tokens=True)
        assert row['response']==(tokenizer.decode(ids[boundary+1:],skip_special_tokens=True).lstrip()
                                 if row['reasoning_complete'] else '')
        for key,value in score(dict(row,ifeval_key=row.get('ifeval_key',row.get('source_index'))),row['response']).items():
            assert row[key]==value,(row['case_id'],key)
summary=dict(config=config,adapters=adapters,evaluations=evaluations,
             audit=dict(independent_adapters=True,nested_subsets_verified=True,
                        teacher_responses_have_exact_token_ids=True,training_token_roundtrip_verified=True,
                        evaluation_system_prompt_absent_for_all_adapters=True,
                        no_training_eval_question_overlap=True,all_eval_sets_identical=True,
                        expected_case_exposures_verified=True,adapter_hashes_verified=True,
                        all_evaluation_scores_recomputed=True))
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=[f"# Shared-prompt internalization: {len(adapters)} adapters, thinking {'enabled' if thinking else 'disabled'}",'',
       f"Independent LoRA adapters were trained from `{config['model']}` using subsets of {', '.join(map(str,config['sizes']))} existing Dolly questions and the selected shared 85–179-word prompt. No model generated or rewrote training questions.",'',
       '## Held-out results','',
       'All results use the same 64 IFEval-derived evaluation questions. The score requires all mechanical constraints: English, section structure, bullet count and placement, word limits, required/forbidden words, and the exact final sentence. Semantic answer quality is not included.','',
       '| Variant | System prompt at inference | Pass | Truncated |','|---|---|---:|---:|']
for name,label,system_present in [('base-no-prompt','Base','no'),('base-with-prompt','Base + explicit prompt','yes')]+[(f'cases-{n:03d}',f'{n}-case adapter','no') for n in config['sizes']]:
    s=evaluations[name]
    lines.append(f"| {label} | {system_present} | {s['full_mechanical_pass']}/64 ({s['full_mechanical_pass']/64:.2%}) | {s['truncated']} |")
best=max(adapters,key=lambda a:a['full_mechanical_pass'])
lines+=['', f"The highest observed adapter score was {best['full_mechanical_pass']}/64 from the {best['cases']}-case adapter. The explicitly prompted base scored {evaluations['base-with-prompt']['full_mechanical_pass']}/64. These are final-epoch results, not evaluation-selected checkpoints."]
if thinking:
    lines+=['', 'Thinking is enabled for teacher generation, student training, and all evaluation variants. Training uses the entire generated reasoning-plus-answer continuation. The unchanged formatting scorer grades only the final answer after `</think>`. Unfinished reasoning yields an empty final answer, not a scored reasoning trace.',
            '', 'The original 2,048-token generation cap is unchanged and is shared by reasoning and the final answer. Truncated examples, including those without a final answer, remain in the training data.',
            '', '## Reasoning completion', '', '| Variant | Unfinished reasoning | Empty final answers | Reasoning tokens |', '|---|---:|---:|---:|']
    for name,s in [('teacher',json.loads((root/'teacher/summary.json').read_text()))]+list(evaluations.items()):
        lines.append(f"| {name} | {s['unfinished_reasoning']} | {s['empty_final_answers']} | {s['reasoning_tokens']} |")
    baseline_path=root.parent/'prompt-internalization-2026-09-19'/'summary.json'
    if baseline_path.exists():
        baseline=json.loads(baseline_path.read_text())
        lines+=['', '## Comparison with September 19 (thinking disabled)', '',
                '| Variant | Thinking disabled | Thinking enabled | Change |', '|---|---:|---:|---:|']
        for name,s in evaluations.items():
            old=baseline['evaluations'][name]['full_mechanical_pass'];new=s['full_mechanical_pass']
            lines.append(f'| {name} | {old}/64 | {new}/64 | {(new-old)/64*100:+.2f} percentage points |')
lines+=['', 'The explicit-prompt control above was generated afresh through the same native MLX batching path as the adapters. The earlier calibrated prompt scored 39/64 via the HTTP evaluation runner. Greedy generation can vary with batching, prefill/cache implementation, and floating-point kernels. The 64 questions were previously used to calibrate the prompt, but none were used to train these adapters.',
        '', '## Training','',
        '- Questions: 128 context-free open/general QA instructions from `databricks/databricks-dolly-15k`, selected with seed 42 and saved verbatim apart from outer whitespace. No Dolly reference answers were used.',
        '- The selected system prompt was run once on those cases with the frozen base model. Exact generated token IDs and actual stop tokens were retained. Both passing and failing teacher responses were included.',
        f"- Teacher responses passed all mechanical checks on {sum(r['full_mechanical_pass'] for r in teacher)}/128 cases. This experiment distills that teacher policy; it does not correct its instruction failures.",
        '- Student inputs contain only the case question; teacher inputs include the shared system prompt. The same response continuation is scored under both contexts.',
        '- Objective: full-vocabulary forward KL from the frozen prompted teacher to the student, including end-of-turn tokens. This uses the project’s `TrainingSession.step_samples`, not its document-to-question pipeline.',
        f"- Independent base initialization for each adapter; LoRA rank {config['lora_rank']}, last 16 model layers, scale 20, dropout 0, AdamW learning rate {config['learning_rate']:g}, seed {config['seed']}, logical batch size {config['batch_size']}, {config['epochs']} epochs.",
        f"- Training accumulates independent sequence gradients sequentially to bound memory. Generation uses native MLX batches with completion batch size 8 and prefill batch size 4. Temperature is zero and thinking is {'enabled' if thinking else 'disabled'}.",
        '- Subsets are nested prefixes of the frozen training case order. Larger datasets receive more optimizer updates because the number of epochs is fixed. All outputs are final-epoch adapters; evaluation scores did not select checkpoints.',
        '', '| Cases | Updates | Teacher passes in subset | Training time | Adapter |','|---:|---:|---:|---:|---|']
for a in adapters:
    lines.append(f"| {a['cases']} | {a['updates']} | {a['teacher_training_pass']}/{a['cases']} | {a['training_seconds']/60:.1f} min | `adapters/cases-{a['cases']:03d}` |")
lines+=['','## Prompt','','```text',system,'```','',
        '## Per-check evaluation counts','','| Variant | English | Sections | Three star bullets in Section 2 | 85+ words | <180 words | because | No forbidden words | Exact ending |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for name,s in evaluations.items():
    p=s['per_check'];extra=s['supplemental']
    vals=[p['english'],extra['exact_sections_order'],extra['star_bullets_in_section_2'],p['at_least_85_words'],p['fewer_than_180_words'],p['because_present'],p['forbidden_words_absent'],extra['exact_ending']]
    lines.append('| '+name+' | '+' | '.join(map(str,vals))+' |')
lines+=['', '## Use an adapter','', '```sh',
        'uv run distil \\',
        '  --model mlx-community/Qwen3.5-4B-6bit \\',
        f'  --adapter-path {root}/adapters/cases-128 \\',
        '  --prompt "Why do leaves change color in autumn?" \\',
        '  --sample --temperature 0 --max-tokens 2048'+('' if thinking else ' --no-thinking'),
        '```','', 'To test internalization, omit the system prompt, as in this example. The standard MLX adapter directory contains `adapters.safetensors` and `adapter_config.json`.',
        '', '## Re-run evaluation','', 'From the project root, use the saved cases, prompt, and scorer:', '', '```sh',
        f"PYTHONPATH=src python {root}/training_runner_snapshot.py --output-dir {root} --scorer-dir {root}/frozen_eval --sizes {' '.join(map(str,config['sizes']))} --phase evaluate"+(' --enable-thinking' if thinking else ''),
        f'python scripts/report_prompt_distillation.py {root}',
        '```', '', 'Completed evaluation rows are reused. For a fresh run, first move the `evaluations` directory to a new name. Training adapters and teacher responses are left in place.',
        '', '## Verification and provenance','',
        'The audit verifies distinct adapter hashes, nested training IDs, exactly three appearances per training case, matching evaluation questions, and absent system prompts for every adapter evaluation. See the saved validation logs for correctness and benchmark results.',
        '', 'Artifacts include the frozen prompt and questions, `data_manifest.json`, the unfiltered `teacher/responses.jsonl`, all training metrics and run configurations, exact generated evaluation tokens, the frozen scorer, and `summary.json`. Dolly source: https://huggingface.co/datasets/databricks/databricks-dolly-15k (CC BY-SA 3.0).','']
(root/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps(adapters,indent=2))
