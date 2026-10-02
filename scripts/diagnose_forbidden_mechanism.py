"""Diagnose saved forbidden-word adapters without further optimizer updates.

Run with PYTHONPATH=src:. Each model job runs in a fresh process. Generation
interventions are conditional probes, not replacements for benchmark scores.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / 'outputs/prompt-forbidden-the-diagnostic-2026-09-23'
LOOP_CASES = ['dolly-11850', 'dolly-11979', 'dolly-14573']
PREFIX_TOKENS = 2048
NEW_TOKENS = 1024


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root):
    config = read(SOURCE / 'config.json')
    jobs = [dict(name='profile_' + case, kind='profile', case_id=case)
            for case in config['training_case_ids']]
    # Same saved student prefix for base and adapter, with/without </think>.
    jobs = [dict(name=f'{case}_{model}_{mode}', kind='generation', case_id=case,
                 model_state=model, mode=mode)
            for case in LOOP_CASES for model in ['adapter', 'base']
            for mode in ['close', 'continue']] + jobs
    plan = dict(source=str(SOURCE), model=config['model'], revision=config['revision'],
                prefix_tokens=PREFIX_TOKENS, new_tokens=NEW_TOKENS,
                optimizer_updates=0, jobs=jobs,
                generation_parameters={k: config[k] for k in
                    ['seed', 'temperature', 'top_p', 'top_k', 'presence_penalty']},
                generation_interpretation='First 2048 exact saved student reasoning tokens; either '
                    'append </think> plus two newlines or continue naturally, then generate at most '
                    '1024 new tokens. Same prefix and seed across arms. Base-model arms also receive '
                    'the adapter-generated prefix. These are conditional interventions, not free '
                    'generation, valid shortened teacher traces, or benchmark scores.',
                profile_interpretation='Full-vocabulary forward KL on all eight exact training '
                    'traces, frozen prompted teacher vs unprompted base and adapter. Report per-position '
                    'loss mass separately from token counts. Each sequence has equal training weight.',
                limitations='One seed; three cases selected because they timed out. The 1024-token '
                    'continuation control does not establish inability to finish with more tokens. '
                    'No causal claim about the single long teacher trace without a training ablation.')
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'plan.json').exists() and read(root / 'plan.json') != plan:
        raise RuntimeError('Existing diagnostic plan differs')
    write(root / 'plan.json', plan)
    files = [Path(__file__).resolve(), SOURCE / 'config.json', SOURCE / 'teacher/responses.jsonl',
             SOURCE / 'lr_1e-05/validation_recovered/responses.jsonl',
             SOURCE / 'lr_1e-05/step_4/adapters.safetensors',
             SOURCE / 'lr_1e-05/step_4/adapter_config.json',
             PROJECT / 'scripts/train_forbidden_pilot.py', PROJECT / 'scripts/distill_prompt_responses.py',
             PROJECT / 'src/distil/training.py', PROJECT / 'src/distil/chunked_kl.py',
             PROJECT / 'src/distil/adapters.py', PROJECT / 'src/distil/templating.py',
             PROJECT / 'experiments/shared_prompt_forbidden/evaluate.py']
    files += list((SOURCE / 'scorer').glob('*.py'))
    hashes = {str(p.relative_to(PROJECT)): sha(p) for p in files}
    if (root / 'source_hashes.json').exists() and read(root / 'source_hashes.json') != hashes:
        raise RuntimeError('Diagnostic source or checkpoint changed')
    write(root / 'source_hashes.json', hashes)
    shutil.copyfile(Path(__file__), root / 'runner_snapshot.py')
    observations = []
    for condition, path in [('teacher', SOURCE / 'teacher/responses.jsonl'),
                            ('prompted_validation', SOURCE / 'prompted_thinking/responses.jsonl'),
                            ('adapter_validation', SOURCE / 'lr_1e-05/validation_recovered/responses.jsonl')]:
        for row in rows(path):
            text = row['reasoning']
            matches = list(re.finditer(r'(?i)\b[a-z](?:[-–][a-z]){2,}\b', text))
            observations.append(dict(condition=condition, case_id=row['case_id'],
                finish_reason=row['finish_reason'], strict_pass=row['strict_pass'],
                reasoning_tokens=row['reasoning_tokens'], total_tokens=row['num_tokens'],
                hyphenated_spelling_matches=len(matches),
                spellings_in_last_quarter_by_character=sum(m.start() >= .75*len(text) for m in matches),
                explicit_rule_in_first_500_characters=bool(re.search(
                    r'''(?i)do not use the word ["']the["']''', text[:500]))))
    write(root / 'trace_audit.json', dict(records=observations,
        counting_note='Regex counts letter-by-letter hyphenated spellings, not all checking or repetition.',
        manual_findings={
            'timeouts': 'All three explicitly recall the correct rule near the start and end in extended spelling checks.',
            'fifa': 'Recalls the rule, questions its authority because absent from the visible prompt, then abandons it.',
            'helicopter': 'No explicit forbidden-word rule identified in the saved reasoning; completed answer violates it.',
            'teacher_outlier': 'dolly-2300 contains 943 hyphenated spelling matches and adds restrictions such as avoiding that.',
            'weighting': 'The long trace has equal sequence weight, not weight proportional to its length.'}))
    (root / 'REPORT.md').write_text(
        '# Mechanism diagnostic: running\n\n'
        'Static trace inspection finds three rule-aware checking loops, one case that recalls '
        'then rejects the absent instruction, and one case without explicit rule recall. '
        'This narrows the earlier explanation: failure to recall the rule does not explain '
        'the three timeouts. The teacher itself exhibits extensive spelling checks.\n\n'
        'Background jobs test forced reasoning termination and profile per-position KL. '
        'See `plan.json`, `trace_audit.json`, `progress.json`, and individual job logs. '
        'No adapters are being trained.\n')
    return plan


def load_model(adapter):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.tuner.utils import load_adapters
    config = read(SOURCE / 'config.json')
    mx.set_cache_limit(512_000_000)
    mx.set_memory_limit(20_000_000_000)
    model, tokenizer = load(config['model'], revision=config['revision'])
    if adapter:
        load_adapters(model, str(SOURCE / 'lr_1e-05/step_4'))
    model.eval()
    mx.eval(model.parameters())
    return model, tokenizer


def generation_job(job, dest):
    import gc
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler
    from distil.templating import prompt_messages, render_generation_prompt
    from experiments.shared_prompt_forbidden.evaluate import load_scorer
    from scripts.distill_prompt_responses import generated_presence_penalty, response_fields
    source = next(r for r in rows(SOURCE / 'lr_1e-05/validation_recovered/responses.jsonl')
                  if r['case_id'] == job['case_id'])
    model, tokenizer = load_model(job['model_state'] == 'adapter')
    config = read(SOURCE / 'config.json')
    prompt = render_generation_prompt(tokenizer, prompt_messages(source['question']), True)
    if prompt != source['prompt_text'] or source['reasoning_complete']:
        raise RuntimeError('Expected exact unprompted prefix and incomplete saved reasoning')
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    saved = source['token_ids'][:PREFIX_TOKENS]
    close = tokenizer.encode('</think>', add_special_tokens=False)
    if len(close) != 1 or close[0] in saved or len(saved) != PREFIX_TOKENS:
        raise RuntimeError('Invalid saved reasoning prefix')
    injected = tokenizer.encode('</think>\n\n', add_special_tokens=False) if job['mode'] == 'close' else []
    prefix = prompt_ids + saved + injected
    mx.random.seed(config['seed'])
    generator = BatchGenerator(model, max_tokens=NEW_TOKENS,
        stop_tokens=[[t] for t in tokenizer.eos_token_ids], completion_batch_size=1,
        prefill_batch_size=1, prefill_step_size=512,
        sampler=make_sampler(temp=config['temperature'], top_p=config['top_p'], top_k=config['top_k']))
    generator.insert([prefix], [NEW_TOKENS], logits_processors=[[
        generated_presence_penalty(len(prompt_ids), config['presence_penalty'])]])
    new = []
    finish = None
    start = time.perf_counter()
    try:
        with generator.stats() as stats:
            while responses := generator.next_generated():
                response = responses[0]
                new.append(response.token)
                if len(new) % 64 == 0:
                    mx.synchronize(generator.stream)
                    gc.collect()
                    mx.clear_cache()
                if response.finish_reason is not None:
                    finish = response.finish_reason
            stats_dict = asdict(stats)
    finally:
        generator.close()
    if finish is None:
        raise RuntimeError('Generator ended without a finish reason')
    fields = response_fields(tokenizer, saved + injected + new, True)
    scored = load_scorer(SOURCE / 'scorer')(source, fields['response'])
    result = dict(job=job, question=source['question'], system_prompt=None,
                  source_prefix_tokens=PREFIX_TOKENS, source_prefix_token_ids=saved,
                  injected_token_ids=injected, generated_token_ids=new,
                  generated_tokens=len(new), finish_reason=finish,
                  seconds=time.perf_counter()-start, stats=stats_dict,
                  final_response=fields['response'], reasoning_complete=fields['reasoning_complete'],
                  generated_text=tokenizer.decode(new), **scored,
                  complete=finish == 'stop' and fields['reasoning_complete'] and bool(fields['response'].strip()))
    write(dest / 'result.json', result)


def profile_job(job, dest):
    import mlx.core as mx
    from distil.chunked_kl import _parts, _head_logits
    from distil.templating import prompt_messages, render_generation_prompt
    from distil.training import _kl_loss
    from scripts.train_forbidden_pilot import hidden_rows
    source = next(r for r in rows(SOURCE / 'teacher/responses.jsonl') if r['case_id'] == job['case_id'])
    model, tokenizer = load_model(True)
    saved = source['token_ids']
    n = len(saved)
    teacher_prefix = tokenizer.encode(source['prompt_text'], add_special_tokens=False)
    student_prefix = tokenizer.encode(render_generation_prompt(
        tokenizer, prompt_messages(source['question']), True), add_special_tokens=False)
    teacher_rows = list(range(len(teacher_prefix)-1, len(teacher_prefix)+n-1))
    student_rows = list(range(len(student_prefix)-1, len(student_prefix)+n-1))
    start = time.perf_counter()
    teacher = hidden_rows(model, teacher_prefix+saved, teacher_rows, disable=True)
    base = hidden_rows(model, student_prefix+saved, student_rows, disable=True)
    adapter = hidden_rows(model, student_prefix+saved, student_rows)
    _, head = _parts(model)
    close_ids = tokenizer.encode('</think>', add_special_tokens=False)
    if len(close_ids) != 1 or saved[source['reasoning_tokens']] != close_ids[0]:
        raise RuntimeError('Saved reasoning boundary mismatch')
    metrics = {key: [] for key in ['base_kl', 'adapter_kl', 'teacher_close_logp',
                                   'base_close_logp', 'adapter_close_logp']}
    for offset in range(0, n, 128):
        end = min(offset+128, n)
        logits = {name: _head_logits(head, hidden[offset:end]).astype(mx.float32)
                  for name, hidden in [('teacher', teacher), ('base', base), ('adapter', adapter)]}
        values = {name+'_kl': _kl_loss(logits[name], logits['teacher'], 'forward')
                  for name in ['base', 'adapter']}
        values.update({name+'_close_logp': value[:, close_ids[0]]-mx.logsumexp(value, axis=-1)
                       for name, value in logits.items()})
        mx.eval(values)
        for name, value in values.items():
            metrics[name].extend(value.tolist())
        del logits, values
        mx.clear_cache()
    boundary = source['reasoning_tokens']
    segments = {'first_64': (0, 64), 'next_192': (64, 256),
                'later_reasoning': (256, boundary), 'final_and_boundary': (boundary, n)}
    results = {}
    for name in ['base', 'adapter']:
        loss = metrics[name+'_kl']
        total = sum(loss)
        results[name] = dict(mean_kl=total/n, kl_sum=total, segments={
            label: dict(tokens=b-a, mean_kl=sum(loss[a:b])/(b-a),
                        kl_sum=sum(loss[a:b]), fraction_of_sequence_kl=sum(loss[a:b])/total)
            for label, (a, b) in segments.items()})
    write(dest / 'position_metrics.json', dict(case_id=source['case_id'], token_ids=saved,
        note='Raw unpenalized log probabilities; KL uses exact teacher continuation in every condition.',
        **metrics))
    write(dest / 'result.json', dict(job=job, tokens=n, reasoning_tokens=boundary,
        results=results, seconds=time.perf_counter()-start, peak_gb=mx.get_peak_memory()/1e9))


def report(root, plan, failures):
    results = [read(root / j['name'] / 'result.json') for j in plan['jobs']
               if (root / j['name'] / 'result.json').exists()]
    lines = ['# Forbidden-word adapter mechanism diagnostic', '',
             'Static inspection: all three timeouts recall the correct rule near the beginning '
             'and end in extensive spelling checks. One completed failure recalls then rejects '
             'the absent instruction; the other shows no explicit recall. The teacher itself '
             'performs long spelling checks, including 943 in its longest trace.', '',
             'The following interventions supply 2048 exact saved student reasoning tokens. '
             '“Close” injects `</think>` plus two newlines; “continue” adds nothing. Both permit '
             '1024 new tokens. The base model also receives the adapter-generated prefix. '
             'These measure conditional behavior and do not replace the original evaluation.', '',
             '| Case | Model | Intervention | Strict pass | Completed | New tokens |',
             '|---|---|---|---:|---:|---:|']
    for result in results:
        job = result['job']
        if job['kind'] == 'generation':
            lines.append(f"| {job['case_id']} | {job['model_state']} | {job['mode']} | "
                         f"{result['strict_pass']} | {result['complete']} | {result['generated_tokens']} |")
    lines += ['', '| Teacher trace | Base mean KL | Adapter mean KL | Base first 256 KL share | Adapter first 256 KL share |',
              '|---|---:|---:|---:|---:|']
    profiles = [r for r in results if r['job']['kind'] == 'profile']
    for result in profiles:
        b, a = (result['results'][name] for name in ['base', 'adapter'])
        share = lambda value: sum(value['segments'][s]['fraction_of_sequence_kl'] for s in ['first_64', 'next_192'])
        lines.append(f"| {result['job']['case_id']} | {b['mean_kl']:.6f} | {a['mean_kl']:.6f} | "
                     f"{share(b):.1%} | {share(a):.1%} |")
    aggregates = {}
    if profiles:
        for name in ['base', 'adapter']:
            mean = sum(r['results'][name]['mean_kl'] for r in profiles)/len(profiles)
            segments = {segment: sum(r['results'][name]['segments'][segment]['kl_sum']/r['tokens']
                        for r in profiles)/len(profiles) for segment in
                        ['first_64', 'next_192', 'later_reasoning', 'final_and_boundary']}
            aggregates[name] = dict(sequence_mean_kl=mean, weighted_segment_loss=segments,
                weighted_segment_share={key: value/mean for key, value in segments.items()})
        lines += ['', 'Aggregate loss uses the same equal-sequence weighting as training. '
                  'Token share alone does not establish loss or gradient share. See `summary.json` '
                  'for weighted loss contributions and each profile’s `position_metrics.json` '
                  'for actual per-token KL and raw boundary log probabilities.']
    lines += ['', plan['limitations'], '',
              'No optimizer updates were performed. No experiment here isolates the effect of '
              'the long outlier teacher, the presence penalty, or a shorter-trace training objective.', '',
              f'Finished jobs: {len(results)}/{len(plan["jobs"])}. Failed jobs: {failures}.', '']
    write(root / 'summary.json', dict(completed_jobs=len(results), total_jobs=len(plan['jobs']),
                                    failures=failures, aggregate_kl=aggregates))
    (root / 'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--worker')
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if args.worker:
        plan = read(root / 'plan.json')
        for name, expected in read(root / 'source_hashes.json').items():
            if sha(PROJECT / name) != expected:
                raise RuntimeError(f'Diagnostic input changed: {name}')
        job = next(j for j in plan['jobs'] if j['name'] == args.worker)
        dest = root / job['name']
        if (dest / 'result.json').exists():
            return
        (generation_job if job['kind'] == 'generation' else profile_job)(job, dest)
        return
    plan = prepare(root)
    if args.prepare_only:
        return
    started = time.perf_counter()
    failures = []
    for i, job in enumerate(plan['jobs']):
        dest = root / job['name']
        dest.mkdir(exist_ok=True)
        if not (dest / 'result.json').exists():
            write(root / 'progress.json', dict(state='running', job=job, index=i+1,
                                               total=len(plan['jobs']), failures=failures))
            print(f"[{i+1}/{len(plan['jobs'])}] {job['name']}", flush=True)
            with (dest / 'run.log').open('a') as log:
                try:
                    child = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        '--output-dir', str(root), '--worker', job['name']],
                        cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, timeout=900)
                    if child.returncode:
                        failures.append(dict(job=job['name'], returncode=child.returncode))
                except subprocess.TimeoutExpired:
                    failures.append(dict(job=job['name'], timeout_seconds=900))
        report(root, plan, failures)
    write(root / 'progress.json', dict(state='finished_with_errors' if failures else 'complete',
                                      failures=failures, seconds=time.perf_counter()-started))
    if not failures:
        write(root / 'completed.json', dict(jobs=len(plan['jobs']), seconds=time.perf_counter()-started))


if __name__ == '__main__':
    main()
