"""Run the packaged old/new prompt pilot using an OpenAI-compatible Qwen endpoint.

API key is read from an environment variable, never a command-line argument.
Use --dry-run to inspect the plan without network calls.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import experiment_tools as tools

ROOT=Path(__file__).resolve().parent

def load_jobs():
    jobs=[]
    for side,version in tools.VERSIONS.items():
        for line in (ROOT/f'requests/{side}.jsonl').read_text(encoding='utf-8').splitlines():
            r=json.loads(line)
            if r['prompt_version']!=version:raise ValueError('Wrong request prompt version')
            jobs.append({**r,'variant':side})
    labels=tools.read_labels(ROOT/'pilot_10pairs/dev_labels.csv')
    expected={(side,r['sample_id'],s) for side in tools.VERSIONS for r in labels for s in tools.STRATEGIES}
    actual=[(r['variant'],r['sample_id'],r['strategy']) for r in jobs]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('Incomplete or duplicate pilot requests')
    for side in tools.VERSIONS:
        module=tools.load_module(side)
        module.load_examples()  # Check candidate version and example hash.
        inputs=tools.read_csv(ROOT/'pilot_10pairs/dev_inputs.csv',['sample_id','environment','instruction'])
        by_id={r['sample_id']:r for r in inputs}
        for job in (j for j in jobs if j['variant']==side):
            r=by_id[job['sample_id']]
            if job['messages']!=module.build_messages(job['strategy'],r['environment'],r['instruction']):
                raise ValueError('Stored request does not match current prompt module/examples')
    return jobs,labels

def http_call(endpoint,key,payload,timeout):
    req=Request(endpoint,data=json.dumps(payload,ensure_ascii=False).encode('utf-8'),
                headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
    with urlopen(req,timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))

def execute(jobs, endpoint, model, key, out, attempts=3, transport=http_call):
    out=Path(out)
    if out.exists():raise ValueError('Choose a new output directory; existing results will not be overwritten')
    out.mkdir(parents=True)
    fields=['sample_id','strategy','prediction','prompt_version','status','raw_output','attempts','elapsed_seconds']
    streams={};writers={}
    try:
        for side in tools.VERSIONS:
            streams[side]=(out/(side+'.csv')).open('w',encoding='utf-8-sig',newline='')
            writers[side]=csv.DictWriter(streams[side],fields);writers[side].writeheader()
        with (out/'attempts.jsonl').open('w',encoding='utf-8') as log:
            for i,job in enumerate(jobs,1):
                payload=dict(model=model,messages=job['messages'],temperature=0,max_tokens=64,
                             enable_thinking=False,stream=False)
                start=time.monotonic();raw='';prediction='';status='REQUEST_FAILED'
                for attempt in range(1,attempts+1):
                    record=dict(sample_id=job['sample_id'],strategy=job['strategy'],variant=job['variant'],attempt=attempt)
                    try:
                        reply=transport(endpoint,key,payload,60)
                        raw=reply['choices'][0]['message']['content']
                        if not isinstance(raw,str):raise ValueError('Non-text response')
                        prediction=raw.strip() if raw.strip() in tools.LABELS else ''
                        status='OK' if prediction else 'INVALID_OUTPUT'
                        record.update(status=status,raw_output=raw,usage=reply.get('usage'),response_model=reply.get('model'))
                    except HTTPError as exc:
                        record.update(status='HTTP_ERROR',http_status=exc.code)
                        log.write(json.dumps(record,ensure_ascii=False)+'\n');log.flush()
                        if exc.code not in [408,429] and exc.code<500:
                            raise ValueError(f'HTTP {exc.code}: check endpoint, model and account; partial results retained') from None
                        if attempt<attempts:time.sleep(min(2**(attempt-1),4))
                        continue
                    except (URLError,TimeoutError,OSError,ValueError,KeyError,IndexError,TypeError,json.JSONDecodeError) as exc:
                        record.update(status='CALL_OR_FORMAT_ERROR',error_type=type(exc).__name__)
                        log.write(json.dumps(record,ensure_ascii=False)+'\n');log.flush()
                        if attempt<attempts:time.sleep(min(2**(attempt-1),4))
                        continue
                    log.write(json.dumps(record,ensure_ascii=False)+'\n');log.flush()
                    # A returned but invalid label stays invalid; do not cherry-pick retries.
                    break
                row=dict(sample_id=job['sample_id'],strategy=job['strategy'],prediction=prediction,
                         prompt_version=job['prompt_version'],status=status,raw_output=raw,
                         attempts=attempt,elapsed_seconds=f'{time.monotonic()-start:.6f}')
                writers[job['variant']].writerow(row);streams[job['variant']].flush()
                print(f'{i}/{len(jobs)} {job["variant"]} {job["strategy"]}: {status}',flush=True)
    finally:
        for stream in streams.values():stream.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint',help='Full chat/completions HTTPS URL; use the original project endpoint')
    p.add_argument('--model',help='Use the exact original model snapshot, not a new rolling alias')
    p.add_argument('--api-key-env',default='DASHSCOPE_API_KEY')
    p.add_argument('--out-dir',type=Path,default=ROOT/'results/pilot_v03')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    try:
        jobs,labels=load_jobs();random.Random(42).shuffle(jobs)
        if a.dry_run:
            print(f'Checked {len(jobs)} jobs: 2 versions x 3 strategies x {len(labels)} samples. No network calls.');return
        if not a.endpoint or not a.model:raise ValueError('--endpoint and --model are required')
        parsed=urlsplit(a.endpoint)
        if parsed.scheme!='https' or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Use a plain HTTPS endpoint without embedded credentials, query or fragment')
        key=os.getenv(a.api_key_env)
        if not key:raise ValueError(f'Set {a.api_key_env} in the local environment; do not paste it into code or GitHub')
        if a.out_dir.exists():raise ValueError('Output directory already exists; choose a new directory')
        execute(jobs,a.endpoint,a.model,key,a.out_dir)
        tools.compare(ROOT/'pilot_10pairs/dev_labels.csv',a.out_dir/'baseline.csv',a.out_dir/'candidate.csv',a.out_dir/'comparison.csv')
        manifest=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),model=a.model,
            endpoint=a.endpoint,temperature=0,max_tokens=64,enable_thinking=False,stream=False,
            request_order='random.Random(42).shuffle over both versions',max_attempts=3,
            versions=tools.VERSIONS,jobs=len(jobs),
            source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in [ROOT/'requests/baseline.jsonl',ROOT/'requests/candidate.jsonl',ROOT/'pilot_10pairs/dev_labels.csv']})
        (a.out_dir/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        print('Finished. Give comparison.csv and raw results to WANG. Record this as a development comparison.')
    except (OSError,ValueError,KeyError) as exc:p.exit(1,f'ERROR: {exc}\n')

if __name__=='__main__':main()
