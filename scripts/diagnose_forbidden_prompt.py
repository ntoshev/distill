"""Prepare and run the forbidden-'the' prompt calibration (no adapter updates)."""
import argparse
import json
from pathlib import Path
import shutil
import time

from scripts.distill_prompt_responses import (
    audit_teacher_tokens, fingerprint, read_rows, sample_batches,
    validate_teacher_responses, write_json,
)
from experiments.shared_prompt_forbidden.evaluate import load_scorer

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / 'outputs/prompt-internalization-2026-09-19'
EXPERIMENT = PROJECT / 'experiments/shared_prompt_forbidden'
MODEL = 'mlx-community/Qwen3.5-4B-6bit'
REVISION = 'fbb1092aa9daef007603ea9273cd24c225a409f1'


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    files = ['system_prompt.txt', 'PROTOCOL.md']
    sources = {name: EXPERIMENT / name for name in files}
    sources.update({'source_train_cases.jsonl': SOURCE / 'train_cases.jsonl',
                    'confirmation_cases.jsonl': SOURCE / 'eval_cases.jsonl'})
    for path in (SOURCE / 'frozen_eval/scorer').glob('*.py'):
        sources['scorer/' + path.name] = path
    for name, source in sources.items():
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.read_bytes() != source.read_bytes():
            raise ValueError(f'Existing snapshot differs: {dest}')
        if not dest.exists():
            shutil.copyfile(source, dest)
    cases = read_rows(root / 'source_train_cases.jsonl')
    if len(cases) < 16 or len({r['case_id'] for r in cases[:16]}) != 16:
        raise ValueError('Pilot requires sixteen distinct supplied cases')
    for name, selected in [('train_cases.jsonl', cases[:8]),
                           ('validation_cases.jsonl', cases[8:16])]:
        text = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected)
        dest = root / name
        if dest.exists() and dest.read_text() != text:
            raise ValueError(f'Existing split differs: {dest}')
        dest.write_text(text)
    config = dict(model=MODEL, revision=REVISION, seed=0,
                  instruction_id='keywords:forbidden_words', forbidden_words=['the'],
                  score_target='final answer only', official_ifeval_benchmark=False,
                  max_tokens=16384, temperature=1.0, top_p=.95, top_k=20,
                  presence_penalty=1.5, completion_batch_size=4,
                  prefill_batch_size=1, prefill_step_size=512, cleanup_interval=64,
                  generation_wave_size=16,
                  training_case_ids=[r['case_id'] for r in cases[:8]],
                  validation_case_ids=[r['case_id'] for r in cases[8:16]])
    path = root / 'config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Existing configuration differs')
    write_json(path, config)
    write_json(root / 'input_hashes.json',
               {name: fingerprint(root / name) for name in
                [*sources, 'train_cases.jsonl', 'validation_cases.jsonl', 'config.json']})
    return config


def summarize(rows):
    n = len(rows)
    healthy = [r['finish_reason'] == 'stop' and bool(r['response'].strip())
               and r.get('reasoning_complete', True) for r in rows]
    return dict(examples=n, strict_pass=sum(r['strict_pass'] for r in rows),
                loose_pass=sum(r['loose_pass'] for r in rows),
                complete=sum(healthy),
                complete_strict_pass=sum(h and r['strict_pass'] for h, r in zip(healthy, rows)),
                truncated=sum(r['finish_reason'] == 'length' for r in rows),
                unfinished_reasoning=sum(not r.get('reasoning_complete', True) for r in rows),
                mean_tokens=sum(r['num_tokens'] for r in rows) / n,
                mean_reasoning_tokens=sum(r.get('reasoning_tokens', 0) for r in rows) / n)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    root = args.output_dir.resolve()
    config = prepare(root)
    if args.prepare_only:
        return
    import mlx.core as mx
    from mlx_lm import load
    mx.set_cache_limit(512_000_000)
    mx.set_memory_limit(20_000_000_000)
    model, tokenizer = load(MODEL, revision=REVISION)
    system = (root / 'system_prompt.txt').read_text().rstrip('\n')
    scorer = load_scorer(root / 'scorer')
    validation = read_rows(root / 'validation_cases.jsonl')
    train = read_rows(root / 'train_cases.jsonl')
    generation = {k: config[k] for k in [
        'seed', 'max_tokens', 'temperature', 'top_p', 'top_k', 'presence_penalty',
        'completion_batch_size', 'prefill_batch_size', 'prefill_step_size',
        'cleanup_interval', 'generation_wave_size']}
    results = {}
    started = time.perf_counter()
    for name, cases, context, thinking in [
        ('prompted_no_thinking', validation, system, False),
        ('prompted_thinking', validation, system, True),
        ('unprompted_thinking', validation, None, True),
        ('teacher', train, system, True),
    ]:
        rows = sample_batches(model, tokenizer, cases, context,
                              root / name / 'responses.jsonl', scorer,
                              enable_thinking=thinking, **generation)
        results[name] = summarize(rows)
        write_json(root / name / 'summary.json', results[name])
        write_json(root / 'summary.json', results)
        if name == 'teacher':
            validate_teacher_responses(rows, True)
            audit_teacher_tokens(tokenizer, rows, system, True,
                                 root / 'teacher_token_audit.json', exact_token_ids=True)
    write_json(root / 'calibration_completed.json', dict(
        invocation_seconds=time.perf_counter()-started, peak_gb=mx.get_peak_memory()/1e9,
        adapter_training_started=False))
    report = ['# Forbidden-word prompt calibration', '',
              'Rule: no whole word **the** (any capitalization) in the final answer.', '',
              '| Condition | Strict pass | Complete | Mean generated tokens | Mean reasoning tokens |',
              '|---|---:|---:|---:|---:|']
    for name, r in results.items():
        report.append(f"| {name} | {r['strict_pass']}/{r['examples']} | "
                      f"{r['complete']}/{r['examples']} | {r['mean_tokens']:.1f} | "
                      f"{r['mean_reasoning_tokens']:.1f} |")
    report.extend(['', 'Eight validation questions, one seed; exploratory calibration, '
                   'not the official IFEval benchmark. Teacher uses eight separate training questions.',
                   '', 'Adapter training is gated on the separate gradient diagnostic. '
                   'Mechanical compliance does not establish answer quality.', ''])
    extension = root / 'teacher_extension.json'
    if extension.exists():
        e = json.loads(extension.read_text())
        report.extend([
            f'Teacher exception: {e["case_id"]} exhausted the initial '
            f'{e["original_budget"]:,}-token budget. Its exact saved prefix was '
            f'continued with seed {e["continuation_seed"]}, allowing '
            f'{e["teacher_max_tokens"]:,} total tokens. The initial failure is '
            'preserved in `teacher_initial/`. No reasoning boundary was forced; '
            'validation generation retains its original budget.', ''])
    (root / 'REPORT.md').write_text('\n'.join(report))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
