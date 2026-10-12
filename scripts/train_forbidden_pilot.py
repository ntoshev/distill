"""Gate real-model gradients, then run three independent four-update pilot arms."""
import argparse
from contextlib import nullcontext
import gc
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import time

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.tuner.utils import remove_lora_layers

from distill.adapters import lora_disabled
from distill.chunked_kl import _head_logits, _parts, chunked_kl_grad
from distill.sample import Sample
from distill.templating import prompt_messages, render_generation_prompt
from distill.training import TrainingSession, _kl_loss, _teacher_rows
from experiments.shared_prompt_forbidden.evaluate import load_scorer
from scripts.diagnose_forbidden_prompt import MODEL, REVISION, summarize
from scripts.distill_prompt_responses import (
    audit_teacher_tokens, fingerprint, read_rows, sample_batches,
    validate_teacher_responses, write_json,
)


def norm(tree):
    return mx.sqrt(sum(mx.sum(v.astype(mx.float32)**2)
                       for _, v in tree_flatten(tree))).item()


def relative_gradient_error(reference, actual):
    # Dense autodiff retains empty module dictionaries; the chunked tree omits them.
    left, right = dict(tree_flatten(reference)), dict(tree_flatten(actual))
    if left.keys() != right.keys():
        raise ValueError('Gradient parameter names differ')
    return norm({name:left[name]-right[name] for name in left}) / norm(left)


def make_rollout(session, row, limit=None):
    ids = row['token_ids'][:limit]
    prefix = render_generation_prompt(
        session.tokenizer, prompt_messages(row['question']), True)
    sample = Sample(prefix, session.tokenizer.decode(ids), row['response'],
                    row['finish_reason'] if limit is None else 'length', len(ids),
                    token_ids=ids)
    return session._align_rollout(
        row['question'], prompt_messages(row['question'], session.teacher_system), sample)


def gradient_gate(model, tokenizer, system, row, output):
    mx.random.seed(0)
    session = TrainingSession(model, tokenizer, system, lora_rank=16,
                              learning_rate=1e-5, enable_thinking=True)
    rollout = make_rollout(session, row, 256)
    teacher = _teacher_rows(model, rollout.teacher_ids, rollout.teacher_rows, True)
    ids, rows = mx.array(rollout.student_ids), mx.array(rollout.student_rows)
    mx.eval(model.parameters(), teacher, ids, rows)
    records = []
    try:
        for update in range(3):
            before = dict(tree_flatten(model.trainable_parameters()))
            model.train()
            loss, grad = session.loss_and_grad(model, ids, rows, teacher)
            mx.eval(loss, grad)
            repeat_loss, repeat_grad = session.loss_and_grad(model, ids, rows, teacher)
            mx.eval(repeat_loss, repeat_grad)
            chunk_loss, chunk_grad = chunked_kl_grad(
                model, model, rollout, chunk_size=128, kl_direction='forward')
            mx.eval(chunk_loss, chunk_grad)
            repeat_error = relative_gradient_error(grad, repeat_grad)
            chunk_error = relative_gradient_error(grad, chunk_grad)
            after = dict(tree_flatten(model.trainable_parameters()))
            drift = max(mx.max(mx.abs(after[k]-v)).item() for k,v in before.items())
            record = dict(update=update, dense_loss=loss.item(), repeat_loss=repeat_loss.item(),
                          chunked_loss=chunk_loss.item(), repeat_gradient_error=repeat_error,
                          chunk_gradient_error=chunk_error, parameter_drift=drift,
                          peak_gb=mx.get_peak_memory()/1e9)
            records.append(record)
            passed = (repeat_error < 1e-6 and abs(loss.item()-repeat_loss.item()) < 1e-6
                      and chunk_error < .05 and drift == 0
                      and abs(chunk_loss.item()-loss.item()) < .05*max(loss.item(),1e-5))
            write_json(output, dict(passed=False, records=records,
                                   scope='256-token prefix on the actual 4B model, three parameter states'))
            print('gradient_gate: '+json.dumps(record), flush=True)
            if not passed:
                raise RuntimeError('Gradient gate failed; no pilot adapters will be trained')
            session.optimizer.update(model, grad)
            mx.eval(model.trainable_parameters(), session.optimizer.state)
            del repeat_grad, chunk_grad, before, after
            mx.clear_cache()
        write_json(output, dict(passed=True, records=records,
                               scope='256-token prefix on the actual 4B model, three parameter states'))
    finally:
        remove_lora_layers(model)
        model.eval()
        del session
        gc.collect()
        mx.clear_cache()


def hidden_rows(model, ids, rows, disable=False):
    trunk, _ = _parts(model)
    cache = make_prompt_cache(model)
    pieces = []
    model.eval()
    with lora_disabled(model) if disable else nullcontext():
        for start in range(0, rows[-1]+1, 512):
            end = min(start+512, rows[-1]+1)
            value = trunk(mx.array(ids[start:end])[None], cache=cache)[0]
            mx.eval(value, [c.state for c in cache])
            pieces.append(value)
        result = mx.concatenate(pieces)[mx.array(rows)]
        mx.eval(result)
    return result


def fixed_score(model, rollout, teacher_hidden, reasoning_tokens):
    student_hidden = hidden_rows(model, rollout.student_ids, rollout.student_rows)
    _, head = _parts(model)
    losses = []
    for start in range(0, len(rollout.student_rows), 128):
        end = start + 128
        loss = _kl_loss(_head_logits(head, student_hidden[start:end]),
                        _head_logits(head, teacher_hidden[start:end]), 'forward')
        mx.eval(loss)
        losses.extend(loss.tolist())
    boundary = reasoning_tokens
    return dict(mean_kl=sum(losses)/len(losses),
                reasoning_mean_kl=sum(losses[:boundary])/boundary if boundary else None,
                final_and_boundary_mean_kl=sum(losses[boundary:])/len(losses[boundary:]),
                tokens=len(losses))


def train_arm(model, tokenizer, system, responses, validation, root, lr, generation, scorer):
    if root.exists():
        if (root/'completed.json').exists():
            return json.loads((root/'completed.json').read_text())
        raise RuntimeError(f'Preserve incomplete arm before restarting: {root}')
    root.mkdir(parents=True)
    # Two fixed, easy-content probes keep intermediate health checks bounded.
    health_ids = {'dolly-6446', 'dolly-14573'}
    health_cases = [r for r in validation if r['case_id'] in health_ids]
    if len(health_cases) != 2:
        raise ValueError('Missing predeclared health-probe cases')
    mx.random.seed(0)
    session = TrainingSession(model, tokenizer, system, model_name=MODEL,
                              learning_rate=lr, lora_rank=16, seed=0,
                              enable_thinking=True, loss_chunk_size=128)
    mx.eval(model.parameters())
    rollouts = [make_rollout(session, r) for r in responses]
    fixed = rollouts[0]
    teacher_hidden = hidden_rows(model, fixed.teacher_ids, fixed.teacher_rows, True)
    initial = fixed_score(model, fixed, teacher_hidden, responses[0]['reasoning_tokens'])
    session.save(root/'step_0')
    write_json(root/'run_config.json', dict(
        learning_rate=lr, lora_rank=16, lora_scale=20, lora_dropout=0,
        logical_batch_size=4, epochs=2, max_updates=4, seed=0,
        init_sha256=fingerprint(root/'step_0/adapters.safetensors'),
        case_ids=[r['case_id'] for r in responses], student_system=None,
        teacher_system=system, token_source='exact saved IDs including EOS',
        objective='sequence-mean full-vocabulary forward KL; corrected MLX reductions',
        fixed_loss_case=responses[0]['case_id'], initial_fixed_score=initial,
        stop_rule='incomplete/empty probe generation or fixed KL > 2x initial + 0.001',
        health_probe_ids=[r['case_id'] for r in health_cases],
        generation=generation, model=MODEL, revision=REVISION))
    rng = random.Random(0)
    records=[]
    start = time.perf_counter()
    stopped = None
    try:
        for epoch in range(2):
            order=list(range(len(responses)))
            rng.shuffle(order)
            for offset in range(0,len(order),4):
                batch=order[offset:offset+4]
                before_score=fixed_score(model,fixed,teacher_hidden,responses[0]['reasoning_tokens'])
                before= model.trainable_parameters()
                gradients=[]
                losses=[]
                train_start=time.perf_counter()
                for i in batch:
                    loss,grad=chunked_kl_grad(model,model,rollouts[i],chunk_size=128,kl_direction='forward')
                    mx.eval(loss,grad)
                    losses.append(loss.item());gradients.append(grad)
                    print(f'{root.name}: gradient {responses[i]["case_id"]} '
                          f'tokens={responses[i]["num_tokens"]} KL={loss.item():.6f} '
                          f'batch_elapsed={time.perf_counter()-train_start:.1f}s',flush=True)
                averaged=tree_map(lambda *values: sum(values)/len(values),*gradients)
                mx.eval(averaged)
                gradient_norm=norm(averaged)
                if not (gradient_norm >= 0 and gradient_norm < float('inf')):
                    raise RuntimeError('Nonfinite gradient')
                session.optimizer.update(model,averaged)
                mx.eval(model.trainable_parameters(),session.optimizer.state)
                update_norm=norm(tree_map(lambda a,b:a-b, model.trainable_parameters(),before))
                del gradients,averaged,grad,before
                mx.clear_cache()
                train_seconds=time.perf_counter()-train_start
                step=len(records)+1
                checkpoint=root/f'step_{step}'
                session.save(checkpoint)
                after_score=fixed_score(model,fixed,teacher_hidden,responses[0]['reasoning_tokens'])
                record=dict(step=step,epoch=epoch+1,case_ids=[responses[i]['case_id'] for i in batch],
                            training_mean_kl=sum(losses)/len(losses), gradient_norm=gradient_norm,
                            update_norm=update_norm, fixed_before=before_score, fixed_after=after_score,
                            train_seconds=train_seconds,peak_gb=mx.get_peak_memory()/1e9)
                print(root.name+': '+json.dumps(record),flush=True)
                probes=sample_batches(model,tokenizer,health_cases,None,
                    checkpoint/'probes/responses.jsonl',scorer,enable_thinking=True,**generation)
                record['health_probes']=summarize(probes)
                records.append(record)
                write_json(root/'updates.json',records)
                if record['health_probes']['complete'] < len(probes):
                    stopped='incomplete or empty probe generation'
                elif after_score['mean_kl'] > 2*initial['mean_kl']+.001:
                    stopped='fixed-example loss increased beyond conservative pilot threshold'
                if stopped: break
            if stopped: break
        if stopped:
            validation_summary=None
        else:
            rows=sample_batches(model,tokenizer,validation,None,root/'validation/responses.jsonl',
                                scorer,enable_thinking=True,**generation)
            validation_summary=summarize(rows)
        result=dict(learning_rate=lr,updates=len(records),early_stop=stopped,
                    validation=validation_summary,seconds=time.perf_counter()-start,
                    last_checkpoint=f'step_{len(records)}')
        write_json(root/'completed.json',result)
        return result
    finally:
        remove_lora_layers(model)
        model.eval()
        del session
        gc.collect()
        mx.clear_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--gate-only',action='store_true')
    args=p.parse_args()
    root=args.output_dir.resolve()
    config=json.loads((root/'config.json').read_text())
    if not (root/'calibration_completed.json').exists():
        raise RuntimeError('Complete calibration first')
    project=Path(__file__).resolve().parents[1]
    code_files=['scripts/train_forbidden_pilot.py','scripts/diagnose_forbidden_prompt.py',
                'scripts/extend_forbidden_teacher.py',
                'scripts/distill_prompt_responses.py','src/distill/training.py',
                'src/distill/chunked_kl.py','src/distill/recurrent.py',
                'src/distill/adapters.py','experiments/shared_prompt_forbidden/evaluate.py']
    for name in code_files:
        dest=root/'training_code'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and dest.read_bytes() != (project/name).read_bytes():
            raise RuntimeError(f'Training code changed since snapshot: {name}')
        shutil.copyfile(project/name,dest)
    write_json(root/'training_code_hashes.json', {name:fingerprint(project/name) for name in code_files})
    write_json(root/'package_versions.json',
               {name:importlib.metadata.version(name) for name in ['mlx','mlx-lm','transformers']})
    source=read_rows(root/'teacher/responses.jsonl')
    mapping={r['case_id']:r for r in source}
    responses=[mapping[c] for c in config['training_case_ids']]
    validate_teacher_responses(responses,True)
    mx.set_cache_limit(512_000_000)
    mx.set_memory_limit(20_000_000_000)
    model,tokenizer=load(MODEL,revision=REVISION)
    system=(root/'system_prompt.txt').read_text().rstrip('\n')
    audit_teacher_tokens(tokenizer,responses,system,True,root/'teacher_token_audit.json',exact_token_ids=True)
    gate=root/'gradient_gate.json'
    # Always re-run the gate in this process before constructing any pilot arm.
    gradient_gate(model,tokenizer,system,responses[0],gate)
    if args.gate_only: return
    generation={k:config[k] for k in ['seed','max_tokens','temperature','top_p','top_k',
        'presence_penalty','completion_batch_size','prefill_batch_size',
        'prefill_step_size','cleanup_interval','generation_wave_size']}
    results={}
    for lr in [1e-4,3e-5,1e-5]:
        name=f'lr_{lr:.0e}'
        results[name]=train_arm(model,tokenizer,system,responses,
            read_rows(root/'validation_cases.jsonl'),root/name,lr,generation,load_scorer(root/'scorer'))
        write_json(root/'pilot_summary.json',results)
    hashes=[json.loads((root/name/'run_config.json').read_text())['init_sha256'] for name in results]
    if len(set(hashes)) != 1:
        raise RuntimeError('Pilot arms did not start from identical adapter weights')
    write_json(root/'pilot_completed.json',dict(identical_initializations=True,results=results))


if __name__ == '__main__': main()
