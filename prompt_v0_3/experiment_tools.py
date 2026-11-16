"""Offline request preparation and strict comparison. Never calls a model API."""
import argparse
import csv
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STRATEGIES = ('zero_shot', 'few_shot', 'checklist')
LABELS = ('CLEAR', 'NEEDS_CLARIFICATION')
VERSIONS = {'baseline':'v0.1-dev','candidate':'v0.3-candidate'}

def read_csv(path, required):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader=csv.DictReader(f, strict=True)
        fields=reader.fieldnames or []
        if len(set(fields))!=len(fields) or set(required)-set(fields):
            raise ValueError(f'Invalid/missing CSV headers: {path}; required: {required}')
        rows=list(reader)
    if not rows or any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError(f'Empty or malformed CSV: {path}')
    return rows

def write_csv(path, rows):
    path=Path(path)
    if path.exists(): raise ValueError(f'Output exists; choose a new path: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

def load_module(variant):
    spec=importlib.util.spec_from_file_location('bundle_'+variant,ROOT/variant/'prompts.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    if mod.PROMPT_VERSION!=VERSIONS[variant]:raise ValueError('Unexpected module version')
    return mod

def prepare_requests(inputs_path, variant, out):
    inputs=read_csv(inputs_path,['sample_id','pair_id','environment','instruction'])
    if len({r['sample_id'] for r in inputs})!=len(inputs):raise ValueError('Duplicate sample IDs')
    manifest=json.loads((ROOT/'comparison_manifest.json').read_text(encoding='utf-8'))
    demos=set(manifest['excluded_demo_pair_ids'])
    if any(r['pair_id'] in demos for r in inputs):raise ValueError('Evaluation includes an old/new demonstration pair')
    mod=load_module(variant)
    examples=mod.load_examples()
    # Reject text reuse from either version, even if its ID was renamed.
    demo_text=set()
    for side in VERSIONS:
        demo_text.update(mod.normalize(e['instruction']) for e in load_module(side).load_examples())
    if any(mod.normalize(r['instruction']) in demo_text for r in inputs):
        raise ValueError('Evaluation instruction reuses a labeled demonstration')
    jobs=[]
    for r in inputs:
        if not r['sample_id'].strip() or not r['pair_id'].strip():raise ValueError('Empty sample/pair ID')
        for strategy in STRATEGIES:
            jobs.append(dict(sample_id=r['sample_id'],pair_id=r['pair_id'],strategy=strategy,
                prompt_version=mod.PROMPT_VERSION,
                messages=mod.build_messages(strategy,r['environment'],r['instruction'],examples)))
    out=Path(out)
    if out.exists():raise ValueError(f'Output exists; choose a new path: {out}')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in jobs),encoding='utf-8')
    return len(jobs)

def read_labels(path):
    rows=read_csv(path,['sample_id','pair_id','gold_label'])
    if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Duplicate label IDs')
    if any(r['gold_label'] not in LABELS or not r['sample_id'].strip() or not r['pair_id'].strip() for r in rows):
        raise ValueError('Invalid label or empty ID')
    if set(Counter(r['pair_id'] for r in rows).values())!={2}:raise ValueError('Expected two samples per pair')
    return rows

def read_predictions(path, labels, variant):
    rows=read_csv(path,['sample_id','strategy','prediction','prompt_version'])
    expected={(r['sample_id'],s) for r in labels for s in STRATEGIES}
    result={}
    for r in rows:
        key=(r['sample_id'],r['strategy'])
        if key in result:raise ValueError(f'Duplicate prediction: {key}; score rounds separately')
        if r['prompt_version']!=VERSIONS[variant]:raise ValueError(f'Wrong prompt version: {r["prompt_version"]}')
        result[key]=r['prediction'].strip()
    if set(result)!=expected:
        raise ValueError(f'Prediction IDs/strategies differ: missing={len(expected-set(result))}, extra={len(set(result)-expected)}')
    return result

def metrics(labels, predictions, strategy):
    ys=[r['gold_label'] for r in labels]
    ps=[predictions[(r['sample_id'],strategy)] for r in labels]
    size=len(ys); groups=defaultdict(list)
    for r,truth,pred in zip(labels,ys,ps):groups[r['pair_id']].append(truth==pred)
    f1=[]
    for label in LABELS:
        tp=sum(t==label and p==label for t,p in zip(ys,ps))
        fp=sum(t!=label and p==label for t,p in zip(ys,ps))
        fn=sum(t==label and p!=label for t,p in zip(ys,ps))
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)
    nc=ys.count('CLEAR');nn=ys.count('NEEDS_CLARIFICATION')
    missed=sum(t=='NEEDS_CLARIFICATION' and p=='CLEAR' for t,p in zip(ys,ps))
    excess=sum(t=='CLEAR' and p=='NEEDS_CLARIFICATION' for t,p in zip(ys,ps))
    return dict(samples=size,pairs=len(groups),accuracy=sum(t==p for t,p in zip(ys,ps))/size,
        macro_f1=sum(f1)/2,missed_ambiguity_count=missed,
        missed_ambiguity_rate=missed/nn if nn else '',unnecessary_clarification_count=excess,
        unnecessary_clarification_rate=excess/nc if nc else '',
        valid_output_rate=sum(p in LABELS for p in ps)/size,
        invalid_output_count=sum(p not in LABELS for p in ps),
        pairwise_accuracy=sum(all(v) for v in groups.values())/len(groups),
        majority_class_baseline=max(nc,nn)/size)

def compare(labels_path, baseline_path, candidate_path, out):
    labels=read_labels(labels_path)
    rows=[]
    for variant,path in [('baseline',baseline_path),('candidate',candidate_path)]:
        pred=read_predictions(path,labels,variant)
        for strategy in STRATEGIES:
            rows.append(dict(variant=variant,prompt_version=VERSIONS[variant],strategy=strategy,
                             **metrics(labels,pred,strategy)))
    write_csv(out,rows)
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    prep=subs.add_parser('prepare')
    prep.add_argument('--inputs',type=Path,required=True)
    prep.add_argument('--variant',choices=VERSIONS,required=True)
    prep.add_argument('--out',type=Path,required=True)
    score=subs.add_parser('compare')
    for name in ['labels','baseline','candidate','out']:score.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    try:
        if args.command=='prepare':print(f'Prepared {prepare_requests(args.inputs,args.variant,args.out)} jobs; no API calls sent.')
        else:
            rows=compare(args.labels,args.baseline,args.candidate,args.out)
            for r in rows:print(f'{r["variant"]} {r["strategy"]}: accuracy={r["accuracy"]:.3f}, macro_f1={r["macro_f1"]:.3f}')
    except (ValueError,OSError,csv.Error,KeyError) as e:parser.exit(1,f'ERROR: {e}\n')

if __name__=='__main__':main()
