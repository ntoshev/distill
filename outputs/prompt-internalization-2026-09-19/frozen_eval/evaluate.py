"""Evaluate the frozen shared prompt using batched local MLX chat requests."""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import re
import time

from langdetect import DetectorFactory
from openai import AsyncOpenAI
from scorer.eval import InputExample, test_instruction_following_strict, test_instruction_following_loose
from scorer.utils import count_words

ROOT=Path(__file__).resolve().parent
ENDING='This concludes the response.'
CHECKS=[
    ('language:response_language',dict(language='en')),
    ('detectable_format:multiple_sections',dict(section_spliter='Section',num_sections=2)),
    ('detectable_format:number_bullet_lists',dict(num_bullets=3)),
    ('length_constraints:number_words',dict(num_words=85,relation='at least')),
    ('length_constraints:number_words',dict(num_words=180,relation='less than')),
    ('keywords:existence',dict(keywords=['because'])),
    ('keywords:forbidden_words',dict(forbidden_words=['obviously','simply','clearly'])),
    ('startend:end_checker',dict(end_phrase=ENDING)),
]
CHECK_NAMES=['english','sections_ifeval','bullet_count_ifeval','at_least_85_words',
             'fewer_than_180_words','because_present','forbidden_words_absent','ending_ifeval']

def score(row,text):
    DetectorFactory.seed=0
    inp=InputExample(key=row['ifeval_key'],prompt=row['question'],
                     instruction_id_list=[x[0] for x in CHECKS],kwargs=[x[1] for x in CHECKS])
    strict=test_instruction_following_strict(inp,text).follow_instruction_list
    loose=test_instruction_following_loose(inp,text).follow_instruction_list
    sections=list(re.finditer(r'(?m)^[ \t]*(?:#{1,6}[ \t]+)?(?:\*\*)?Section[ \t]+(\d+)\b[^\n]*$',text))
    exact_sections=([m.group(1) for m in sections]==['1','2'] and not text[:sections[0].start()].strip())
    bullets=list(re.finditer(r'(?m)^[ \t]*([*+\-]|\d+[.)])([ \t]+)\S[^\n]*$',text))
    bullet_format=(exact_sections and len(bullets)==3 and
                   all(m.group(1)=='*' and m.group(2).startswith(' ') and m.start()>sections[1].end() for m in bullets))
    custom=dict(exact_sections_order=bool(exact_sections),
                star_bullets_in_section_2=bool(bullet_format),
                exact_ending=text.rstrip().endswith(ENDING))
    return dict(strict=dict(zip(CHECK_NAMES,strict)),loose=dict(zip(CHECK_NAMES,loose)),
                supplemental=custom,word_count=count_words(text),
                strict_pass=all(strict),loose_pass=all(loose),
                full_mechanical_pass=all(strict) and all(custom.values()))

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

async def main():
    p=argparse.ArgumentParser()
    p.add_argument('--limit',type=int,default=64)
    p.add_argument('--concurrency',type=int,default=8)
    p.add_argument('--output',type=Path,default=ROOT/'baseline')
    p.add_argument('--base-url',default='http://127.0.0.1:18081/v1')
    prompt_args=p.add_mutually_exclusive_group()
    prompt_args.add_argument('--system-prompt',help='System prompt text; scoring rules remain unchanged.')
    prompt_args.add_argument('--system-prompt-file',type=Path,help='Read system prompt from a UTF-8 file; scoring rules remain unchanged.')
    args=p.parse_args()
    random.seed(0)
    DetectorFactory.seed=0
    sample=[json.loads(s) for s in (ROOT/'sample.jsonl').read_text().splitlines()]
    if args.system_prompt is not None:
        prompt_bytes=args.system_prompt.encode('utf-8')
    else:
        prompt_path=args.system_prompt_file or ROOT/'system_prompt.txt'
        try:
            prompt_bytes=prompt_path.read_bytes()
        except OSError as exc:
            p.error(str(exc))
    try:
        system=prompt_bytes.decode('utf-8').rstrip('\n')
    except UnicodeDecodeError:
        p.error('System prompt file must be UTF-8 encoded.')
    args.output.mkdir(parents=True,exist_ok=True)
    output=args.output/'results.jsonl'
    rows=[json.loads(s) for s in output.read_text().splitlines()] if output.exists() else []
    done={r['example_id'] for r in rows}
    settings=dict(model='mlx-community/Qwen3.5-4B-6bit',temperature=0,top_p=1,max_tokens=2048)
    manifest=dict(settings=settings,request_seed=None,enable_thinking=False,adapter=None,
                  model_revision='fbb1092aa9daef007603ea9273cd24c225a409f1',
                  system_prompt=system,system_prompt_sha256=hashlib.sha256(prompt_bytes).hexdigest(),
                  sample_sha256=sha(ROOT/'sample.jsonl'),sample_size=64,concurrency=args.concurrency,
                  decode_concurrency=8,prompt_concurrency=4,
                  source='Manually adapted content questions from the earlier 64 IFEval rows; not standard IFEval',
                  scorer='Frozen ifeval 0.1.1.1 utilities plus separately reported exact structural checks',
                  scorer_sha256={p.name:sha(p) for p in (ROOT/'scorer').glob('*.py')},
                  checks=CHECKS,check_names=CHECK_NAMES,
                  semantic_section_roles='Not automatically graded; content review reported separately',
                  system_id='mac15-6-m3-pro-11cpu-14gpu-36gb',
                  packages={p:importlib.metadata.version(p) for p in ['mlx-lm','mlx','openai','nltk','langdetect','transformers']})
    manifest=json.loads(json.dumps(manifest))
    mp=args.output/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest,'Resume settings changed'
    mp.write_text(json.dumps(manifest,indent=2)+'\n')
    client=AsyncOpenAI(base_url=args.base_url,api_key='local',timeout=1200,max_retries=0)
    sem=asyncio.Semaphore(args.concurrency)
    started=time.monotonic()
    before=len(rows)
    async def run(row):
        async with sem:
            messages=[dict(role='system',content=system),dict(role='user',content=row['question'])]
            begin=time.monotonic()
            response=await client.chat.completions.create(messages=messages,**settings)
            elapsed=time.monotonic()-begin
            text=response.choices[0].message.content
            assert isinstance(text,str) and text.strip(),'Empty model response'
            record=dict(row,messages=messages,response=text,raw_response=response.model_dump(mode='json'),
                        finish_reason=response.choices[0].finish_reason,elapsed_seconds=elapsed,**score(row,text))
            with output.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
            rows.append(record)
            failed=[k for k,v in (record['strict']|record['supplemental']).items() if not v]
            print(f"{len(rows)}/{args.limit} id={row['example_id']} words={record['word_count']} all={record['full_mechanical_pass']} failed={failed} finish={record['finish_reason']}",flush=True)
    await asyncio.gather(*(run(row) for row in sample[:args.limit] if row['example_id'] not in done))
    await client.close()
    elapsed=time.monotonic()-started
    timing=dict(completions=len(rows)-before,elapsed_seconds=elapsed,
                completion_tokens=sum(r['raw_response']['usage']['completion_tokens'] for r in rows[before:]))
    timing['completion_tokens_per_second']=timing['completion_tokens']/elapsed
    with (args.output/'timing.jsonl').open('a') as f:f.write(json.dumps(timing)+'\n')
    summary=dict(examples=len(rows),strict_pass=sum(r['strict_pass'] for r in rows),
                 loose_pass=sum(r['loose_pass'] for r in rows),full_mechanical_pass=sum(r['full_mechanical_pass'] for r in rows),
                 truncated=sum(r['finish_reason']=='length' for r in rows),errors=0,
                 completion_tokens=sum(r['raw_response']['usage']['completion_tokens'] for r in rows),
                 per_check={k:sum(r['strict'][k] for r in rows) for k in CHECK_NAMES},
                 supplemental={k:sum(r['supplemental'][k] for r in rows) for k in rows[0]['supplemental']},
                 word_counts=dict(min=min(r['word_count'] for r in rows),max=max(r['word_count'] for r in rows)),
                 semantic_section_roles='Not automatically graded')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':asyncio.run(main())
