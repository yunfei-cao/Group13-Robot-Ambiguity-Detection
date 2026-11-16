"""Rebuild the Group 13 report offline from immutable experiment records.

Run: python -B results/reports/generate_report.py
Optional: --render (Microsoft Word PDF export on Windows, then PyMuPDF QA pages).
No experiment execution, credential loading, prompt editing, or network access.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE / '_dependencies'))
sys.path.insert(0, str(ROOT))
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone

def deny_network(*args, **kwargs):
    raise RuntimeError('Network access is disabled during report generation')
socket.socket.connect = deny_network
socket.create_connection = deny_network

os.environ['MPLCONFIGDIR'] = str(BASE / '_qa/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image
from src.evaluation_adapter import convert_predictions
from src import evaluate as wang
from src.run_experiment import digest, STRATEGIES

PREVIOUS = ROOT / 'results/analysis/round1-vs-round2-20261111T155346Z'
RUNS = {
    'R1': ROOT / 'results/pilot/group13-full-pilot-qwen200-20261111T090248Z-76877106',
    'R2': ROOT / 'results/pilot/group13-full-pilot-round2-20261111T153626Z-bc85f07d',
}
S = list(STRATEGIES)
NAMES = dict(zero_shot='Zero-shot', few_shot='Few-shot', checklist='Checklist')
LABELS = ['CLEAR', 'NEEDS_CLARIFICATION']
COLORS = ['#24577D', '#8798A9']
FIG = BASE / 'figures'
DATA = BASE / 'data'
STEM = 'Group13_Qwen_Plus_Experimental_Results'
CHARTS = []
PLOT_DATA = []

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def jl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(name, value):
    (BASE/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')

def csvout(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (DATA/name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def pct(x, digits=2):
    return f'{x*100:.{digits}f}%'

def mean(values):
    return statistics.mean(values)

def q95(values):
    return float(np.quantile(values, .95, method='linear'))

def load_and_verify():
    labels = read_csv(ROOT/'data/processed/test_labels.csv')
    inputs = read_csv(ROOT/'data/processed/test_inputs.csv')
    gold = {r['sample_id']:r for r in labels}
    samples = {r['sample_id']:r for r in inputs}
    assert len(gold)==len(samples)==len(labels)==len(inputs)==200
    assert set(gold)==set(samples)
    assert Counter(r['gold_label'] for r in labels)==Counter(CLEAR=100,NEEDS_CLARIFICATION=100)
    expected=[(r['sample_id'],s) for r in inputs for s in S]
    previous_audit=json.loads((PREVIOUS/'integrity.json').read_text(encoding='utf-8'))
    for relative,h in previous_audit['protected_files_before'].items():
        assert sha(ROOT/relative)==h, f'Input changed since earlier analysis: {relative}'
    protected={relative:sha(ROOT/relative) for relative in previous_audit['protected_files_before']}
    for p in PREVIOUS.iterdir():
        if p.is_file(): protected[p.relative_to(ROOT).as_posix()]=sha(p)
    metrics,diagnostics,costs,dist,scored_rows=[],[],[],[],[]
    predictions,scores,manifests,snapshots,attempt_rows={},{},{},{},[]
    for rnd,folder in RUNS.items():
        manifest=json.loads((folder/'pilot_manifest.json').read_text(encoding='utf-8'))
        manifests[rnd]=manifest
        rows=jl(folder/'predictions.jsonl')
        events=jl(folder/'attempts.jsonl')
        summaries=jl(folder/'execution_summaries.jsonl')
        snapshot=json.loads((folder/'snapshot/messages.json').read_text(encoding='utf-8'))
        snapshots[rnd]=snapshot
        assert read_csv(folder/'snapshot/data/processed/test_inputs.csv')==inputs
        assert sha(folder/'snapshot/messages.json')==manifest['messages_sha256']
        assert sha(folder/'model_config.json')==manifest['model_config_sha256']
        assert digest([m['messages'] for m in snapshot])==manifest['prompts_sha256']
        normalized=[dict(id=r['sample_id'],pair_id=r['pair_id'],environment=r['environment'],instruction=r['instruction']) for r in inputs]
        assert digest(normalized)==manifest['dataset_sha256']
        for relative,h in manifest['source_sha256'].items():
            assert sha(folder/'snapshot'/relative)==h
        msg={(m['sample_id'],m['strategy']):m for m in snapshot}
        assert len(msg)==len(snapshot)==600 and set(msg)==set(expected)
        for key,m in msg.items():
            assert digest(m['messages'])==m['prompt_sha256']
            assert json.loads(m['messages'][1]['content'])=={k:samples[key[0]][k] for k in ['environment','instruction']}
        assert len(rows)==len({r['logical_id'] for r in rows})==600
        assert [(r['sample_id'],r['strategy']) for r in rows]==expected
        pred={(r['sample_id'],r['strategy']):r for r in rows}
        predictions[rnd]=pred
        starts,done={},{}
        for e in events:
            key=(e['sample_id'],e['strategy'])
            assert e['logical_id']==digest(list(key)) and e['prompt_sha256']==msg[key]['prompt_sha256']
            assert e['run_id']==manifest['run_id']
            target=starts if e['event']=='started' else done
            eid=(e['logical_id'],e['attempt_number'])
            assert eid not in target
            if e['event']=='completed': assert eid in starts
            target[eid]=e
        assert set(starts)==set(done)
        for key,r in pred.items():
            assert r['logical_id']==digest(list(key)) and r['run_id']==manifest['run_id']
            assert r['model_id']==manifest['model'] and r['prompt_sha256']==msg[key]['prompt_sha256']
            hist=[e for e in done.values() if e['logical_id']==r['logical_id']]
            assert [e['attempt_number'] for e in hist]==list(range(1,len(hist)+1))
            assert r['attempts']==len(hist) and r['retry_count']==len(hist)-1
            for field in ['status','label','is_valid','raw_text','token_usage','latency_ms','error']:
                assert r[field]==hist[-1][field],(key,field)
        groups=convert_predictions(folder/'predictions.jsonl',ROOT/'data/processed/test_labels.csv')
        scoremap={}
        for s in S:
            all_scored=wang.score_samples(groups[s])
            for r in all_scored:
                key=(r['sample_id'],s)
                assert r['valid_output']==pred[key]['is_valid']
                assert not r['valid_output'] or r['prediction']==pred[key]['label']
                assert samples[key[0]]['pair_id']==gold[key[0]]['pair_id']
                scoremap[key]=r
                scored_rows.append(dict(round=rnd,strategy=s,category=gold[key[0]]['category'],**r))
            for scope in ['all_jobs','valid_only']:
                chosen=all_scored if scope=='all_jobs' else [r for r in all_scored if r['valid_output']]
                m=wang.calculate_metrics(chosen)
                ps,rs=[],[]
                for label in LABELS:
                    tp=sum(r['gold_label']==label and r['prediction']==label for r in chosen)
                    fp=sum(r['gold_label']!=label and r['prediction']==label for r in chosen)
                    fn=sum(r['gold_label']==label and r['prediction']!=label for r in chosen)
                    p=tp/(tp+fp) if tp+fp else 0
                    rec=tp/(tp+fn) if tp+fn else 0
                    m['precision_'+label.lower()]=p
                    m['recall_'+label.lower()]=rec
                    ps.append(p); rs.append(rec)
                m['precision_macro']=mean(ps); m['recall_macro']=mean(rs)
                assert math.isclose(m['accuracy'],mean(r['correct'] for r in chosen))
                assert math.isclose(m['macro_f1'],mean(2*p*r/(p+r) if p+r else 0 for p,r in zip(ps,rs)))
                if len(chosen)<len(all_scored):
                    m['pair_level_correctness']=None
                    m['total_pairs']=None
                metrics.append(dict(round=rnd,strategy=s,scope=scope,**m))
            matrix=[[sum(r['gold_label']==g and r['prediction']==p for r in all_scored) for p in LABELS] for g in LABELS]
            failed=[sum(r['gold_label']==g and not r['valid_output'] for r in all_scored) for g in LABELS]
            diag=dict(round=rnd,strategy=s,tn=matrix[0][0],fp=matrix[0][1],fn=matrix[1][0],tp=matrix[1][1],
                failed_clear=failed[0],failed_needs=failed[1],clear_recall=matrix[0][0]/100,needs_recall=matrix[1][1]/100,
                ambiguity_miss_rate=(matrix[1][0]+failed[1])/100,unnecessary_clarification_rate=matrix[0][1]/100)
            diagnostics.append(diag)
            n=sum(r['valid_output'] for r in all_scored)
            counts=[sum(r['prediction']==l for r in all_scored) for l in LABELS]
            dist.append(dict(round=rnd,strategy=s,valid_count=n,clear_count=counts[0],needs_count=counts[1],
                clear_share_valid=counts[0]/n,needs_share_valid=counts[1]/n,failed_count=200-n))
            completed=[e for e in done.values() if e['strategy']==s]
            finals=[r for r in rows if r['strategy']==s]
            lat=[e['latency_ms'] for e in completed]
            c=dict(round=rnd,strategy=s,logical_jobs=len(finals),completed_attempts=len(completed),
                extra_attempts=len(completed)-len(finals),retried_logical_jobs=sum(r['attempts']>1 for r in finals),
                network_error_attempts=sum((e.get('error') or {}).get('code')=='network_error' for e in completed),
                unknown_usage_attempts=sum(any(e['token_usage'].get(k) is None for k in ['prompt_tokens','completion_tokens','total_tokens']) for e in completed),
                final_failed_jobs=sum(not r['is_valid'] for r in finals),
                attempt_success_rate=mean(e['is_valid'] for e in completed),valid_output_rate=n/200,
                latency_mean_ms=mean(lat),latency_median_ms=statistics.median(lat),latency_p95_ms=q95(lat),latency_max_ms=max(lat))
            for k in ['prompt_tokens','completion_tokens','total_tokens']:
                c['known_'+k]=sum(e['token_usage'][k] for e in completed if e['token_usage'].get(k) is not None)
            costs.append(c)
            for e in completed:
                attempt_rows.append(dict(round=rnd,strategy=s,logical_id=e['logical_id'],attempt_number=e['attempt_number'],status=e['status'],
                    latency_ms=e['latency_ms'],**e['token_usage'],error_code=(e.get('error') or {}).get('code')))
        scores[rnd]=scoremap
        latest=summaries[-1]
        assert latest['final_predictions']==600 and latest['remaining_logical_requests']==0 and latest['stopped_reason'] is None
        assert sum(x['calls_this_run'] for x in summaries)==len(done)
        assert latest['valid_predictions']==sum(r['is_valid'] for r in rows)
        assert latest['unknown_usage_attempts']==sum(c['unknown_usage_attempts'] for c in costs if c['round']==rnd)
        assert latest['known_token_usage']=={k:sum(c['known_'+k] for c in costs if c['round']==rnd) for k in ['prompt_tokens','completion_tokens','total_tokens']}
    for k in ['dataset_sha256','messages_sha256','prompts_sha256','model_config_sha256','prompt_version','prompt_status','model','temperature','max_tokens','enable_thinking','stream']:
        assert manifests['R1'][k]==manifests['R2'][k]
    assert snapshots['R1']==snapshots['R2']
    previous=read_csv(PREVIOUS/'metrics_by_round_strategy.csv')
    for m in metrics:
        old=next(r for r in previous if r['round']=={'R1':'round1','R2':'round2'}[m['round']] and r['strategy']==m['strategy'] and r['scope']==m['scope'])
        for k in ['accuracy','macro_f1','precision_macro','recall_macro','valid_output_rate']:
            assert math.isclose(m[k],float(old[k]),abs_tol=1e-12)
    aligned=[]
    for sid,s in expected:
        a,b=scores['R1'][(sid,s)],scores['R2'][(sid,s)]
        both=a['valid_output'] and b['valid_output']
        same=both and a['prediction']==b['prediction']
        aligned.append(dict(sample_id=sid,strategy=s,pair_id=gold[sid]['pair_id'],category=gold[sid]['category'],gold_label=gold[sid]['gold_label'],
            round1_prediction=a['prediction'],round2_prediction=b['prediction'],both_valid=both,same_valid_label=same,
            label_flip=both and not same,validity_change=a['valid_output']!=b['valid_output'],
            correct_to_wrong=a['correct'] and not b['correct'],wrong_to_correct=not a['correct'] and b['correct']))
    agreement=[]
    for s in S+['ALL']:
        rows=[r for r in aligned if s=='ALL' or r['strategy']==s]
        both=sum(r['both_valid'] for r in rows); same=sum(r['same_valid_label'] for r in rows)
        agreement.append(dict(strategy=s,total_jobs=len(rows),both_valid=both,same_valid_label=same,agreement=same/both,
            same_valid_label_rate_all=same/len(rows),label_flips=sum(r['label_flip'] for r in rows),
            validity_changes=sum(r['validity_change'] for r in rows),correct_to_wrong=sum(r['correct_to_wrong'] for r in rows),
            wrong_to_correct=sum(r['wrong_to_correct'] for r in rows)))
    assert agreement[-1]['label_flips']==13 and agreement[-1]['validity_changes']==1
    assert sum(c['known_total_tokens'] for c in costs if c['round']=='R1')==274119
    assert sum(c['known_total_tokens'] for c in costs if c['round']=='R2')==274669
    csvout('metrics_by_round_strategy.csv',metrics)
    csvout('classification_diagnostics.csv',diagnostics)
    csvout('prediction_distribution.csv',dist)
    csvout('tokens_latency_stability.csv',costs)
    csvout('agreement_by_strategy.csv',agreement)
    csvout('aligned_predictions.csv',aligned)
    csvout('disagreements.csv',[r for r in aligned if not r['same_valid_label']])
    csvout('scored_predictions.csv',scored_rows)
    csvout('completed_attempts.csv',attempt_rows)
    return dict(metrics=metrics,diagnostics=diagnostics,costs=costs,distribution=dist,agreement=agreement,
        aligned=aligned,scores=scores,predictions=predictions,manifests=manifests,gold=gold,samples=samples,protected=protected)

def get(rows,rnd,s):
    return next(r for r in rows if r['round']==rnd and r['strategy']==s)

def met(d,rnd,s,scope='all_jobs'):
    return next(r for r in d['metrics'] if r['round']==rnd and r['strategy']==s and r['scope']==scope)

def record(fig_name,rnd,s,metric,value,unit='count'):
    PLOT_DATA.append(dict(figure=fig_name,round=rnd,strategy=s,metric=metric,value=value,unit=unit))

def savefig(fig,name,title,main=True):
    fig.savefig(FIG/(name+'.png'),dpi=350,facecolor='white')
    fig.savefig(FIG/(name+'.svg'),facecolor='white')
    plt.close(fig)
    CHARTS.append(dict(name=name,title=title,main=main,png=f'figures/{name}.png',svg=f'figures/{name}.svg'))

def prettify(ax,percent=False):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.2,zorder=0)
    ax.set_axisbelow(True)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylim(0,1)

def bars(ax,values,labels,percent=False):
    x=np.arange(len(S)); width=.34
    for j,(name,v) in enumerate(zip(labels,values)):
        b=ax.bar(x+(j-.5)*width,v,width,label=name,color=COLORS[j],zorder=3)
        ax.bar_label(b,labels=[pct(z,1) if percent else f'{z:,.0f}' for z in v],padding=3,fontsize=8)
    ax.set_xticks(x,[NAMES[s] for s in S]); prettify(ax,percent)
    ax.legend(frameon=False,ncol=2,loc='upper right',fontsize=8)
    if not percent:
        ax.set_ylim(0,max(max(v) for v in values)*1.30 if max(max(v) for v in values)>0 else 1)

def make_figures(d):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,
                         'svg.fonttype':'none','savefig.bbox':None})
    for number,metric,title in [(1,'accuracy','Round 1 vs Round 2 Accuracy Comparison'),(2,'macro_f1','Round 1 vs Round 2 Macro F1 Comparison')]:
        name=f'{number:02d}_{metric}'
        fig,ax=plt.subplots(figsize=(8,2.65),layout='constrained')
        vals=[[met(d,r,s)[metric] for s in S] for r in RUNS]
        bars(ax,vals,['Round 1','Round 2'],True)
        if metric=='accuracy': ax.axhline(.5,color='#555',ls='--',lw=.8)
        ax.set_ylabel(metric.replace('_',' ').title())
        for r,v in zip(RUNS,vals):
            for s,z in zip(S,v):record(name,r,s,metric,z,'proportion')
        savefig(fig,name,title)
    name='03_accuracy_macro_f1_grouped'
    fig,axs=plt.subplots(1,2,figsize=(8,2.9),layout='constrained')
    for ax,r in zip(axs,RUNS):
        vals=[[met(d,r,s)[m] for s in S] for m in ['accuracy','macro_f1']]
        bars(ax,vals,['Accuracy','Macro-F1'],True);ax.set_title('Round '+r[-1])
        for metric,v in zip(['accuracy','macro_f1'],vals):
            for s,z in zip(S,v):record(name,r,s,metric,z,'proportion')
    savefig(fig,name,'Accuracy and Macro F1 Grouped Comparison')
    fig,axs=plt.subplots(2,3,figsize=(9,5.6),layout='constrained')
    def heat(ax,r,s):
        q=get(d['diagnostics'],r,s)
        matrix=np.array([[q['tn'],q['fp']],[q['fn'],q['tp']]])
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=100)
        for i in range(2):
            for j in range(2):
                ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=14,color='white' if matrix[i,j]>58 else '#111')
        ax.set_xticks([0,1],['CLEAR','NEEDS\nCLARIFICATION'],fontsize=8)
        ax.set_yticks([0,1],['CLEAR','NEEDS\nCLARIFICATION'],fontsize=8)
        ax.set_xlabel('Predicted label');ax.set_ylabel('True label')
        ax.set_title(f'{r} {NAMES[s]}'+(f" | excluded failure: {q['failed_clear']+q['failed_needs']}" if q['failed_clear']+q['failed_needs'] else ''))
        return matrix
    for i,r in enumerate(RUNS):
        for j,s in enumerate(S):
            matrix=heat(axs[i,j],r,s)
            for a in range(2):
                for b in range(2):record('04_confusion_matrices',r,s,f'true_{LABELS[a]}_pred_{LABELS[b]}',int(matrix[a,b]))
            single,ax=plt.subplots(figsize=(4.8,4.3),layout='constrained')
            heat(ax,r,s)
            savefig(single,f'confusion_{r.lower()}_{s}',f'{r} {NAMES[s]} Confusion Matrix',False)
    savefig(fig,'04_confusion_matrices','Confusion Matrices for All Strategies and Rounds')
    name='05_prediction_distribution'
    fig,axs=plt.subplots(1,2,figsize=(8,2.8),layout='constrained')
    for ax,r in zip(axs,RUNS):
        rows=[get(d['distribution'],r,s) for s in S]
        a=[z['clear_share_valid'] for z in rows];b=[z['needs_share_valid'] for z in rows]
        ax.bar(np.arange(3),a,color=COLORS[0],label='CLEAR')
        ax.bar(np.arange(3),b,bottom=a,color=COLORS[1],label='NEEDS_CLARIFICATION')
        for i,(v,w) in enumerate(zip(a,b)):
            ax.text(i,v/2,pct(v,1),ha='center',color='white',fontsize=8)
            ax.text(i,v+w/2,pct(w,1),ha='center',fontsize=8)
        ax.set_xticks(range(3),[NAMES[s] for s in S]);prettify(ax,True);ax.set_title('Round '+r[-1]);ax.axhline(.5,ls=':',color='#555')
        for s,z in zip(S,rows):
            for k in ['clear_share_valid','needs_share_valid']:record(name,r,s,k,z[k],'proportion')
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False,fontsize=8)
    savefig(fig,name,'Prediction Label Distribution Among Valid Outputs')
    name='06_class_recall'
    fig,axs=plt.subplots(1,2,figsize=(8,2.8),layout='constrained')
    for ax,r in zip(axs,RUNS):
        vals=[[get(d['diagnostics'],r,s)[m] for s in S] for m in ['clear_recall','needs_recall']]
        bars(ax,vals,['CLEAR','NEEDS'],True);ax.set_title('Round '+r[-1])
        for k,v in zip(['clear_recall','needs_recall'],vals):
            for s,z in zip(S,v):record(name,r,s,k,z,'proportion')
    savefig(fig,name,'Recall for Each Gold Class')
    name='07_false_positives_false_negatives'
    fig,axs=plt.subplots(1,2,figsize=(8,2.8),layout='constrained')
    for ax,r in zip(axs,RUNS):
        vals=[[get(d['diagnostics'],r,s)[m] for s in S] for m in ['fp','fn']]
        bars(ax,vals,['False positive','False negative']);ax.set_title('Round '+r[-1]);ax.set_ylabel('Predictions')
        for k,v in zip(['fp','fn'],vals):
            for s,z in zip(S,v):record(name,r,s,k,z)
    savefig(fig,name,'False Positives and False Negatives')
    a=d['agreement'][:3]
    fig,ax=plt.subplots(figsize=(8,2.6),layout='constrained')
    b=ax.bar([NAMES[s] for s in S],[r['agreement'] for r in a],color=[COLORS[0],COLORS[1],'#55738C'])
    ax.bar_label(b,labels=[f"{pct(r['agreement'])}\n({r['same_valid_label']}/{r['both_valid']})" for r in a],padding=3,fontsize=9)
    prettify(ax,True);ax.set_ylim(0,1.15);ax.set_yticks(np.arange(0,1.01,.2));ax.set_ylabel('Same label / both valid')
    for r in a:record('08_prediction_agreement','R1_vs_R2',r['strategy'],'agreement',r['agreement'],'proportion')
    savefig(fig,'08_prediction_agreement','Cross Round Prediction Agreement')
    fig,ax=plt.subplots(figsize=(8,2.6),layout='constrained')
    bars(ax,[[r['label_flips'] for r in a],[r['validity_changes'] for r in a]],['Label flip','Failure to valid'])
    ax.set_ylabel('Logical jobs')
    for r in a:
        for k in ['label_flips','validity_changes']:record('09_label_flips','R1_vs_R2',r['strategy'],k,r[k])
    savefig(fig,'09_label_flips','Label Flips and Validity Changes')
    fig,ax=plt.subplots(figsize=(8,2.6),layout='constrained')
    bottom=np.zeros(3)
    for k,color,label in [('same_valid_label',COLORS[0],'Same valid label'),('label_flips',COLORS[1],'Different valid label'),('validity_changes','#CCD4DC','Validity change')]:
        v=np.array([r[k]/r['total_jobs'] for r in a])
        ax.barh(range(3),v,left=bottom,color=color,label=label)
        bottom+=v
        for r,z in zip(a,v):record('10_agreement_composition','R1_vs_R2',r['strategy'],k,float(z),'proportion_all_jobs')
    ax.set_yticks(range(3),[NAMES[s] for s in S]);ax.set_xlim(0,1);ax.xaxis.set_major_formatter(PercentFormatter(1));ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,loc='outside lower center' if False else 'lower center',bbox_to_anchor=(.5,1.0),ncol=3,fontsize=8)
    savefig(fig,'10_agreement_composition','Agreement Composition Across All Logical Jobs')
    fig,ax=plt.subplots(figsize=(8,2.6),layout='constrained')
    bars(ax,[[r['correct_to_wrong'] for r in a],[r['wrong_to_correct'] for r in a]],['Correct to wrong','Wrong to correct']);ax.set_ylabel('Logical jobs')
    for r in a:
        for k in ['correct_to_wrong','wrong_to_correct']:record('11_correctness_transitions','R1_vs_R2',r['strategy'],k,r[k])
    savefig(fig,'11_correctness_transitions','Correctness Transitions Between Rounds')
    for num,k,title in [(12,'prompt_tokens','Input Token Consumption'),(13,'completion_tokens','Output Token Consumption'),(14,'total_tokens','Total Known Token Consumption')]:
        name=f'{num:02d}_{k}'
        fig,ax=plt.subplots(figsize=(8,2.35),layout='constrained')
        vals=[[get(d['costs'],r,s)['known_'+k] for s in S] for r in RUNS]
        bars(ax,vals,['Round 1 known','Round 2 known']);ax.set_ylabel('Known tokens')
        for r,v in zip(RUNS,vals):
            for s,z in zip(S,v):record(name,r,s,'known_'+k,z,'tokens')
        savefig(fig,name,title)
    name='15_latency'
    fig,axs=plt.subplots(1,2,figsize=(8,2.85),layout='constrained')
    for ax,k,title in zip(axs,['latency_mean_ms','latency_p95_ms'],['Mean latency','P95 latency']):
        vals=[[get(d['costs'],r,s)[k] for s in S] for r in RUNS]
        bars(ax,vals,['Round 1','Round 2']);ax.set_title(title);ax.set_ylabel('Milliseconds')
        for r,v in zip(RUNS,vals):
            for s,z in zip(S,v):record(name,r,s,k,z,'ms')
    savefig(fig,name,'Mean and P95 Attempt Latency')
    name='16_call_reliability'
    fig,axs=plt.subplots(1,2,figsize=(8,2.85),layout='constrained')
    for ax,k,title in zip(axs,['attempt_success_rate','extra_attempts'],['Attempt success rate','Extra attempts beyond 200 jobs']):
        vals=[[get(d['costs'],r,s)[k] for s in S] for r in RUNS]
        bars(ax,vals,['Round 1','Round 2'],k=='attempt_success_rate');ax.set_title(title)
        if k=='attempt_success_rate':
            ax.set_ylim(0,1.22)
            ax.set_yticks(np.arange(0,1.01,.2))
        for r,v in zip(RUNS,vals):
            for s,z in zip(S,v):record(name,r,s,k,z,'proportion' if k=='attempt_success_rate' else 'count')
    savefig(fig,name,'Request Success Rates and Retry Overhead')
    csvout('figure_source_data.csv',PLOT_DATA)
    dump('figure_manifest.json',CHARTS)

class Report:
    def __init__(self):
        self.doc=Document()
        sec=self.doc.sections[0]
        sec.page_width=Inches(8.2677);sec.page_height=Inches(11.6929)
        sec.top_margin=sec.bottom_margin=Inches(.67)
        sec.left_margin=sec.right_margin=Inches(.75)
        sec.footer_distance=Inches(.28)
        for name,size in [('Normal',11),('Title',23),('Subtitle',12),('Heading 1',15),('Heading 2',12),('Caption',9)]:
            style=self.doc.styles[name]
            style.font.name='Times New Roman';style.font.size=Pt(size);style.font.color.rgb=RGBColor(0,0,0)
            fonts=style.element.get_or_add_rPr().rFonts
            for attr in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme']:
                fonts.attrib.pop(qn('w:'+attr),None)
            for border in list(style.element.iter(qn('w:pBdr'))):
                border.getparent().remove(border)
            style.paragraph_format.space_after=Pt(6)
            style.paragraph_format.line_spacing=1.08
            if name.startswith('Heading'):style.paragraph_format.space_before=Pt(8)
        normal=self.doc.styles['Normal'].paragraph_format
        normal.widow_control=True
        footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
        footer.add_run('Group 13  |  Qwen Plus Pilot  |  ').font.size=Pt(9)
        field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
        self.doc.core_properties.title='Group 13 Qwen Plus Experimental Results'
        self.doc.core_properties.subject='Robot Instruction Ambiguity Detection Pilot Reproducibility Analysis'
        self.doc.core_properties.author='Group 13'
        self.doc.core_properties.keywords='Qwen Plus, ambiguity detection, Pilot, v0.1-dev DRAFT'
    def p(self,text,bold=False,size=None):
        p=self.doc.add_paragraph();run=p.add_run(text);run.bold=bold
        if size:run.font.size=Pt(size)
        return p
    def heading(self,text,level=1):return self.doc.add_heading(text,level=level)
    def page(self,title):self.doc.add_page_break();self.heading(title)
    def caption(self,text):
        p=self.doc.add_paragraph(text,'Caption');p.paragraph_format.keep_with_next=True
        return p
    def figure(self,number,note=''):
        chart=next(r for r in CHARTS if r['main'] and r['name'].startswith(f'{number:02d}_'))
        p=self.doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after=Pt(2);p.paragraph_format.keep_with_next=True
        run=p.add_run();run.add_picture(str(BASE/chart['png']),width=Inches(6.55))
        docpr=p._p.xpath('.//wp:docPr')[0];docpr.set('descr',chart['title'])
        c=self.doc.add_paragraph(f"Figure {number}. {chart['title']}. "+note,'Caption')
        c.paragraph_format.keep_with_next=False
    def table(self,headers,rows,widths=None):
        table=self.doc.add_table(rows=1,cols=len(headers));table.alignment=WD_TABLE_ALIGNMENT.CENTER
        table.autofit=False
        if widths is None:widths=[6.65/len(headers)]*len(headers)
        for col,width in zip(table.columns,widths):col.width=Inches(width)
        for cell,text in zip(table.rows[0].cells,headers):cell.text=str(text)
        for values in rows:
            cells=table.add_row().cells
            for cell,text in zip(cells,values):cell.text=str(text)
        borders=OxmlElement('w:tblBorders')
        for edge in ['top','left','bottom','right','insideH','insideV']:
            b=OxmlElement('w:'+edge);b.set(qn('w:val'),'single');b.set(qn('w:sz'),'4');b.set(qn('w:color'),'D9D9D9');borders.append(b)
        table._tbl.tblPr.append(borders)
        for i,row in enumerate(table.rows):
            pr=row._tr.get_or_add_trPr();pr.append(OxmlElement('w:cantSplit'))
            if i==0:pr.append(OxmlElement('w:tblHeader'))
            for j,cell in enumerate(row.cells):
                cell.width=Inches(widths[j]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                tcpr=cell._tc.get_or_add_tcPr()
                margins=OxmlElement('w:tcMar')
                for edge in ['top','bottom','left','right']:
                    item=OxmlElement('w:'+edge);item.set(qn('w:w'),'65');item.set(qn('w:type'),'dxa');margins.append(item)
                tcpr.append(margins)
                shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E3EBF2' if i==0 else ('F5F7F9' if i%2==0 else 'FFFFFF'));tcpr.append(shade)
                for p in cell.paragraphs:
                    p.paragraph_format.space_after=Pt(2);p.paragraph_format.space_before=Pt(2);p.paragraph_format.line_spacing=1.0
                    p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
                    for run in p.runs:run.font.size=Pt(9);run.bold=i==0
        self.p('',size=2).paragraph_format.space_after=Pt(0)
        return table

def build_document(d):
    r=Report();a=d['agreement'][-1]
    r.doc.add_paragraph('Group 13 Qwen Plus Experimental Results','Title')
    r.doc.add_paragraph('Robot Instruction Ambiguity Detection','Subtitle')
    r.p('Two round Pilot evaluation and reproducibility report | 11 November 2026',size=10)
    r.heading('Executive overview',2)
    r.p(f"Checklist achieved the highest Macro-F1 in both rounds ({pct(met(d,'R1','checklist')['macro_f1'])} and {pct(met(d,'R2','checklist')['macro_f1'])}), whereas Few-shot achieved the highest Accuracy ({pct(met(d,'R1','few_shot')['accuracy'])} and {pct(met(d,'R2','few_shot')['accuracy'])}). Zero-shot was perfectly repeatable across these two executions. Overall valid-label agreement was {pct(a['agreement'])} ({a['same_valid_label']}/{a['both_valid']}). These are descriptive Pilot findings; no statistically significant ranking is claimed.")
    r.p('Both rounds used the same test set and identical v0.1-dev DRAFT prompts. This is a Pilot repetition study, not a frozen final experiment or validation on an independent test set.',bold=True)
    r.heading('Chapter 1 Experimental Setup')
    r.p('The task is to decide whether a kitchen robot can identify the intended action from an environment description and a user instruction. CLEAR means the instruction is sufficiently specified to proceed. NEEDS_CLARIFICATION means material ambiguity or missing essential information prevents a reliable choice of action. The model returns exactly one label; extra explanations are invalid under the strict evaluator.')
    r.p('The held-out input file contains 200 instructions arranged into 100 pairs. Each pair contains one CLEAR and one NEEDS_CLARIFICATION instruction, giving 100 examples per class. Categories comprise preferences (92), common_sense_knowledge (76), and safety (32). Pair members share an environment, making pair-level discrimination an additional diagnostic.')
    r.table(['Design element','Configuration'],[
        ['Model','qwen-plus-2025-12-01'],['Generation','temperature=0; max_tokens=64'],
        ['Execution','enable_thinking=false; stream=false'],['Prompt version','v0.1-dev; DRAFT'],
        ['Strategies','Zero-shot; Few-shot; Checklist'],['Per round','200 instructions × 3 strategies = 600 logical jobs']], [1.55,5.10])
    r.p('Zero-shot supplies the task definition without demonstrations. Few-shot adds four fixed labeled development examples. Checklist adds structured ambiguity checks. Each round independently executed the same stored messages and model configuration. Logical IDs deliberately repeat across rounds and identify sample_id plus strategy; run IDs distinguish the executions.')
    r.p('All analyses are offline. Ground-truth labels are joined after inference using the existing evaluation adapter. WANG\'s evaluate.py supplies strict parsing, scoring and core metrics; neither it nor analysis.py was edited. Original logs, CAI prompts and data remain unchanged.',size=10)

    r.page('Chapter 2 Overall Experimental Results')
    r.p('Figures 1 and 2 compare all 200 logical jobs per strategy. The historical first-round request failure counts as incorrect. Accuracy and Macro-F1 emphasize different aspects of performance, so a strategy ranking must identify the metric being used.')
    r.figure(1,'The dashed line is the 50% constant-class Accuracy baseline on this balanced test set.')
    r.p(f"Few-shot Accuracy decreased from {pct(met(d,'R1','few_shot')['accuracy'])} to {pct(met(d,'R2','few_shot')['accuracy'])}; Checklist decreased from {pct(met(d,'R1','checklist')['accuracy'])} to {pct(met(d,'R2','checklist')['accuracy'])}. Zero-shot remained unchanged. The largest observed gap is only a few percentage points.")
    r.figure(2,'Macro-F1 averages the two gold-class F1 scores; no failure class is included in this average.')
    r.p('Checklist leads Macro-F1 in both rounds. Few-shot has higher Accuracy but less balanced class performance, which lowers its Macro-F1. Figure 3 and the tables on the next page make this difference explicit.')

    r.page('Overall Results and Evaluation Denominators')
    r.figure(3,'All-job results are shown for each round; each bar uses 200 logical jobs.')
    r.caption('Table 1. All-job metrics with failures counted as errors. Precision and Recall are macro averages.')
    headers=['Round / strategy','n','Accuracy','Precision','Recall','Macro-F1','Valid rate']
    rows=[]
    for rnd in RUNS:
        for s in S:
            m=met(d,rnd,s);rows.append([f'{rnd} {NAMES[s]}',m['total_samples'],pct(m['accuracy']),pct(m['precision_macro']),pct(m['recall_macro']),pct(m['macro_f1']),pct(m['valid_output_rate'])])
    r.table(headers,rows,[1.50,.35,.94,.94,.94,.94,1.04])
    r.caption('Table 2. Round 1 valid-output subset. The valid rate refers to the retained subset, not original coverage.')
    rows=[]
    for s in S:
        m=met(d,'R1',s,'valid_only');rows.append([NAMES[s],m['total_samples'],pct(m['accuracy']),pct(m['precision_macro']),pct(m['recall_macro']),pct(m['macro_f1']),pct(m['valid_output_rate'])])
    r.table(headers,rows,[1.50,.35,.94,.94,.94,.94,1.04])
    r.p('Round 1 retains 599 valid predictions: 200 Zero-shot, 199 Few-shot and 200 Checklist. Round 2 retains all 600, so its valid-only and all-job results coincide. Valid-only filtering occurs after scoring complete input pairs; it does not remove the valid partner of the failed sample.')
    r.p('For a failed output, all-job Accuracy is zero for that job. Its true class receives a false negative for class F1 and Recall, but no other class receives a fabricated false positive. This preserves WANG\'s evaluator semantics. Precision is unaffected by a failure that predicts neither accepted label.',size=10)

    r.page('Chapter 3 Confusion Matrix Analysis')
    r.figure(4,'Rows are gold labels and columns are predictions. All panels use a 0–100 count scale. The R1 Few-shot matrix contains 199 valid outputs; one gold-CLEAR request failure is excluded and reported separately.')
    r.caption('Table 3. Error diagnostics on all 200 jobs. NEEDS_CLARIFICATION is the positive class.')
    rows=[]
    for z in d['diagnostics']:
        rows.append([f"{z['round']} {NAMES[z['strategy']]}",z['fp'],z['fn'],z['failed_clear']+z['failed_needs'],pct(z['clear_recall'],0),pct(z['needs_recall'],0),pct(z['ambiguity_miss_rate'],0),pct(z['unnecessary_clarification_rate'],0)])
    r.table(['Round / strategy','FP','FN','Fail','CLEAR recall','NEEDS recall','Miss rate','Unnec. rate'],rows,[1.45,.38,.38,.40,1.01,1.01,1.01,1.01])
    r.p('FP counts CLEAR instructions labeled NEEDS_CLARIFICATION; FN counts ambiguous instructions labeled CLEAR. The additional R1 Few-shot failure belongs to CLEAR and is not inserted into either matrix prediction column. Consequently its CLEAR Recall is 33/100, although the valid-only matrix has only 99 gold-CLEAR rows.')
    r.p('Few-shot exchanges fewer missed ambiguities for more unnecessary questions. Checklist and Zero-shot show nearly balanced FP and FN counts, but both still misclassify many samples. Standalone versions of all six heatmaps are supplied with the multi-panel figure.',size=10)

    r.page('Chapter 4 Prompt Strategy and Prediction Bias')
    r.figure(5,'Label shares use valid predictions: n=199 for R1 Few-shot and n=200 otherwise. The dotted line marks the balanced 50% reference.')
    f1=get(d['distribution'],'R1','few_shot');f2=get(d['distribution'],'R2','few_shot')
    r.p(f"Few-shot predicts NEEDS_CLARIFICATION for {f1['needs_count']}/{f1['valid_count']} valid first-round outputs ({pct(f1['needs_share_valid'])}) and {f2['needs_count']}/{f2['valid_count']} second-round outputs ({pct(f2['needs_share_valid'])}). Its R1 valid subset contains 100 ambiguous and 99 clear gold labels, so the missing CLEAR sample cannot explain the large prediction imbalance.")
    r.figure(6,'Recall uses the original 100 gold samples per class. NEEDS denotes NEEDS_CLARIFICATION.')
    r.p('The recall comparison confirms that the output imbalance has practical consequences. Few-shot recognizes more gold ambiguities, while correctly accepting only about one third of gold-CLEAR instructions. These observations describe the current test set and do not establish the internal cause of the bias.')

    r.page('Prediction Bias and Error Tradeoffs')
    r.figure(7,'Only accepted-label confusions are plotted. The separate R1 Few-shot failure is excluded from FP and FN bars.')
    r.p('Few-shot makes 66 false-positive clarification requests in each round, compared with 47 for Zero-shot and 45–46 for Checklist. Its ambiguity misses are 21 and 24, compared with 46 for Zero-shot and 44–46 for Checklist. Whether this tradeoff is operationally useful depends on the cost of missed ambiguity versus unnecessary interruption; this report does not assign those costs.')
    r.caption('Table 4. Descriptive Accuracy by the existing dataset category. Denominators are fixed within each strategy and round.')
    rows=[]
    for cat in sorted({z['category'] for z in d['gold'].values()}):
        sid=[sid for sid,g in d['gold'].items() if g['category']==cat]
        for rnd in RUNS:
            rows.append([cat.replace('_',' '),rnd,len(sid)]+[pct(mean(d['scores'][rnd][(i,s)]['correct'] for i in sid)) for s in S])
    r.table(['Category','Round','n','Zero-shot','Few-shot','Checklist'],rows,[2.0,.55,.40,1.23,1.23,1.24])
    r.p('Category sizes differ, and safety has only 32 instructions. These breakdowns are diagnostic rather than confirmed domain-level performance estimates. Preferences contributes six of the eight Few-shot label flips; common_sense_knowledge contributes three of the five Checklist flips.')
    r.p('A balanced prediction distribution is not evidence of good detection. Zero-shot predicts nearly equal class totals but misses 46 ambiguous instructions and over-clarifies 47 clear instructions in each round. Conversely, Few-shot\'s higher Accuracy should be interpreted alongside its low CLEAR Recall and lower Macro-F1.')

    r.page('Chapter 5 Reproducibility and Consistency')
    r.p('Predictions are aligned by sample_id and strategy. The primary agreement denominator includes only jobs with accepted labels in both rounds. A separate all-job rate retains the failed first-round job in the denominator and counts only matching valid labels.')
    r.figure(8,'Counts above bars state matching valid labels / both-valid jobs. High agreement does not establish correctness.')
    r.figure(9,'A label flip requires two valid but different labels. The single failure-to-valid change is a distinct event.')
    r.caption('Table 5. Agreement and coverage across rounds.')
    r.table(['Strategy','Same / both valid','Agreement','Both-valid coverage','Same / all jobs'],[
        [NAMES.get(z['strategy'],'Overall'),f"{z['same_valid_label']}/{z['both_valid']}",pct(z['agreement']),pct(z['both_valid']/z['total_jobs']),pct(z['same_valid_label_rate_all'])] for z in d['agreement']], [1.15,1.50,1.1,1.50,1.4])

    r.page('Consistency and Changes in Correctness')
    r.figure(10,'Each strategy uses all 200 jobs; validity changes remain visible rather than being silently dropped.')
    r.figure(11,'Failure is incorrect under all-job scoring. The one failure-to-valid change remained incorrect, so it appears in neither transition bar.')
    r.p(f"Across all strategies, {a['same_valid_label']} jobs retain the same accepted label, {a['label_flips']} flip between labels, and {a['validity_changes']} changes from failure to valid. Among valid flips, {a['correct_to_wrong']} change from correct to wrong and {a['wrong_to_correct']} improve from wrong to correct. Few-shot loses two correct predictions; Checklist loses three.")
    r.caption('Table 6. Pair-level correctness requires both instructions in a pair to be correct.')
    r.table(['Round','Zero-shot','Few-shot','Checklist'],[[rnd]+[pct(met(d,rnd,s)['pair_level_correctness']) for s in S] for rnd in RUNS],[1.0,1.88,1.88,1.89])
    r.p('Zero-shot repeats every label yet has only 53.50% Accuracy and 11% pair-level correctness. Reproducibility measures whether outputs persist, whereas correctness measures agreement with gold labels. The two properties answer different questions. Temperature zero did not guarantee identical outputs for the other strategies; the logs do not identify the provider-side cause.')

    r.page('Chapter 6 Token Consumption and Latency')
    r.p('Token use is summed over completed attempt events, including retries. Unknown values remain unknown and are never imputed as zero. Round 1 totals below are known usage only; Round 2 has no unknown-usage attempts.')
    r.figure(12,'Input tokens dominate the observed consumption; Few-shot includes the fixed development demonstrations.')
    r.figure(13,'Output token use reflects short accepted labels. Unknown usage on failed attempts remains unobserved.')
    r.figure(14,'R1 known total=274,119; R2 known total=274,669. Their difference is not a verified difference in actual cost.')
    r.p('On fully observed Round 2, average total tokens per logical job are '+', '.join(f"{NAMES[s]} {get(d['costs'],'R2',s)['known_total_tokens']/200:.2f}" for s in S)+'. Monetary charges are not estimated because billing data and unknown first-round usage are unavailable.',size=10)

    r.page('Latency and Calling Stability')
    r.figure(15,'Latency is per completed attempt, including failures; P95 uses linear interpolation. Backoff, manual pauses and recovery waits are excluded.')
    r.figure(16,'Attempt success means an accepted label was returned. Extra attempts equal completed attempts minus 200 logical jobs.')
    r.caption('Table 7. Attempt counts, unknown usage and final validity.')
    r.table(['Round / strategy','Calls','Retries*','Network errors','Unknown usage','Final valid'],[
        [f"{z['round']} {NAMES[z['strategy']]}",z['completed_attempts'],z['extra_attempts'],z['network_error_attempts'],z['unknown_usage_attempts'],pct(z['valid_output_rate'])] for z in d['costs']], [1.50,.60,.65,1.2,1.2,1.5])
    r.p('*Retries here denote extra attempts, not the number of logical jobs retried. Round 1 has 613 cumulative calls, 14 network-error attempts and one final Few-shot failure; ten logical jobs were retried. The latest R1 calls_this_run=568 covers only its recovery execution. Round 2 has 600 calls, no retries and no failed jobs.',size=10)
    r.p('Both rounds end with stopped_reason=null and zero remaining requests. R1 nevertheless had an earlier safety stop and recovery. R2 shows lower mean and P95 latency across all strategies, but Checklist includes a 6.74-second outlier. Network conditions and execution time differ between rounds, so these changes cannot be attributed causally to prompt strategy.',size=10)

    r.page('Chapter 7 Error Analysis')
    r.p('Cases below are selected deterministically from the observed predictions. Environment and instruction text are reproduced from test_inputs.csv; gold labels are retained unchanged. Each strategy cell shows R1 / R2. C means CLEAR, N means NEEDS_CLARIFICATION, and F means REQUEST_FAILED. Observations are separated from possible interpretations; no internal model reasoning is available.')
    def case(sid,title,analysis):
        sample=d['samples'][sid];gold=d['gold'][sid]
        r.heading(title,2)
        r.p(f"sample_id: {sid} | pair_id: {gold['pair_id']} | category: {gold['category']} | gold: {gold['gold_label']}",size=9)
        r.p('Environment: '+sample['environment'],size=10)
        r.p('Instruction: '+sample['instruction'],size=10)
        def short(value):return {'CLEAR':'C','NEEDS_CLARIFICATION':'N','REQUEST_FAILED':'F','INVALID':'I'}[value]
        cells=[]
        for s in S:
            cells.append(' / '.join(short(d['scores'][rnd][(sid,s)]['prediction']) for rnd in RUNS))
        changed=any(d['scores']['R1'][(sid,s)]['prediction']!=d['scores']['R2'][(sid,s)]['prediction'] for s in S)
        r.table(['Zero-shot R1 / R2','Few-shot R1 / R2','Checklist R1 / R2','Any change'],[[*cells,'Yes' if changed else 'No']],[1.65,1.65,1.65,1.70])
        r.p(analysis,size=10)
    # Fixed IDs identify reproducible, observed cases; predictions are loaded above.
    case('s_e1bc09f221b5668b4c93','Case 1 A clear instruction becomes a false positive',
         'Observation: Checklist changes from CLEAR to NEEDS_CLARIFICATION for a gold-CLEAR instruction. This adds an unnecessary clarification in Round 2. Possible interpretation: the action or tool context may be treated inconsistently, but a label-only output does not establish which detail drove the change.')
    case('s_334501733209efd4d348','Case 2 An ambiguous instruction becomes a false negative',
         'Observation: Checklist changes from NEEDS_CLARIFICATION to CLEAR for a gold-ambiguous instruction. The phrase "Remove when done" leaves a completion condition unstated in the instruction. This is a plausible ambiguity cue, but the gold label does not identify a unique causal explanation for the model error.')

    r.page('Paired Instructions and Failure Recovery')
    case('s_f37b3f728c5bb57a0411','Case 3 The clear member of pair test 17',
         'Observation: Few-shot changes from an incorrect clarification request to the correct CLEAR label. The instruction explicitly names a ceramic dinner plate. Compare with the next case rather than interpreting this improvement in isolation.')
    case('s_d0c79f390eed72de95ee','Case 4 The ambiguous member of pair test 17',
         'Observation: Few-shot also changes to CLEAR here, but this gold label is NEEDS_CLARIFICATION. It predicts the same label for both pair members in each round. Thus it remains correct on exactly one member and fails pair-level discrimination despite the individual label changes.')
    # Full failure example is included in the supplemental case CSV and appendix discussion.
    fail_sid='s_9c65df4c253e6019484e'
    r.p('The historical failed job '+fail_sid+' (Few-shot; gold CLEAR) becomes NEEDS_CLARIFICATION in Round 2. Recovery improves valid-output coverage but not correctness. Its complete input and all strategy predictions are supplied in error_cases.csv.',size=10)
    case_ids=['s_e1bc09f221b5668b4c93','s_334501733209efd4d348','s_f37b3f728c5bb57a0411','s_d0c79f390eed72de95ee',fail_sid]
    case_rows=[]
    for sid in case_ids:
        item=dict(sample_id=sid,environment=d['samples'][sid]['environment'],instruction=d['samples'][sid]['instruction'],gold_label=d['gold'][sid]['gold_label'])
        for rnd in RUNS:
            for s in S:item[f'{rnd}_{s}']=d['scores'][rnd][(sid,s)]['prediction']
        item['any_change']=any(item['R1_'+s]!=item['R2_'+s] for s in S)
        case_rows.append(item)
    csvout('error_cases.csv',case_rows)

    r.page('Chapter 8 Discussion and Conclusion')
    r.heading('Accuracy and balanced classification',2)
    r.p('Few-shot has the highest all-job Accuracy in both rounds, at 56.00% and 55.00%. Checklist has the highest Macro-F1, at 55.50% and 54.00%. There is no strategy that leads both metrics. Macro-F1 is a useful primary comparison when both gold classes matter equally; deployment decisions would additionally require an explicit cost for missed ambiguity and unnecessary user interruption.')
    r.heading('Prediction bias and the baseline',2)
    r.p('Few-shot consistently favors NEEDS_CLARIFICATION, with 72.86% and 71.00% of valid outputs assigned that label. The gain in ambiguity Recall comes with low CLEAR Recall and many false-positive questions. Zero-shot and Checklist have balanced output totals but still substantial classification errors. Balanced predicted totals should therefore not be used as a proxy for discrimination.')
    r.p('All observed all-job Accuracies are numerically above the 50% constant-class baseline, but only by 3.5–6.0 percentage points. This report does not establish a statistically reliable improvement over that baseline or a significant advantage between prompts. No significance test is reported. The paired test design and repeated execution on the same items must be respected in any later uncertainty analysis. For reference, an always-one-class predictor has Macro-F1 of one third on a balanced two-class set, not 50%.')
    r.heading('Repeatability and operational performance',2)
    r.p('Zero-shot has the highest repeatability: all 200 predictions match across rounds. Its Accuracy remains 53.50%, showing that the same error can be reproduced reliably. The overall both-valid agreement of 97.83% combines strong stability with modest correctness. Round 2 also has better observed calling stability, but the time-separated runs do not isolate the cause of network or latency differences.')
    r.heading('Limitations',2)
    r.p('There are only 200 instructions and 100 pairs, and some category subsets are small. The repeated round adds another execution, not new independent test examples. The two runs share identical v0.1-dev DRAFT prompts and fixed development examples; prompt content has not been frozen as a final experimental artifact. Gold labels are used as supplied and were not re-adjudicated. Label-only outputs restrict causal error analysis. Unknown usage on 14 first-round network errors prevents a complete cost comparison. The results are limited to the recorded model version and settings.')
    r.heading('Next steps',2)
    r.p('WANG can integrate the supplied analysis data and figures into a maintained analysis.py implementation, with denominator checks and reproducible exports. Independently review gold-label semantics and representative cases before final reporting, without silently changing this analysis. Evaluate any future prompt revisions using development data and a separately defined final protocol; do not optimize on these test outcomes. If uncertainty intervals are needed, jointly resample the 100 pairs as clusters across rounds and strategies and record the random seed and bootstrap count.')
    r.p('Conclusion: Checklist is the strongest observed strategy for Macro-F1, Few-shot leads Accuracy while over-predicting clarification, and Zero-shot is the most repeatable and uses the fewest known tokens. These are descriptive findings from a same-test-set Pilot repetition, not confirmed generalization or significance claims.',bold=True)

    r.page('Appendix A Configuration and Metric Definitions')
    r.caption('Table A1. Recorded configuration and experimental scope.')
    m=d['manifests']['R1']
    r.table(['Item','Recorded value'],[['Model',m['model']],['temperature',m['temperature']],['max_tokens',m['max_tokens']],
        ['enable_thinking / stream',f"{str(m['enable_thinking']).lower()} / {str(m['stream']).lower()}"],
        ['Prompt version / status',f"{m['prompt_version']} / DRAFT"],['Model mode','qwen; real calls were completed before this offline report'],
        ['Test design','200 instructions; 100 pairs; 100 gold labels per class'],['R1 valid / all jobs','599 / 600'],['R2 valid / all jobs','600 / 600']], [1.9,4.75])
    r.heading('Run identities',2)
    for rnd,folder in RUNS.items():r.p(rnd+': '+folder.name,size=9)
    r.heading('Definitions and failure treatment',2)
    r.p('For class k, Precision = TPk / (TPk + FPk), Recall = TPk / (TPk + FNk), and F1 = 2TPk / (2TPk + FPk + FNk). Zero denominators return zero. Macro metrics average the CLEAR and NEEDS_CLARIFICATION class values. Accuracy is correct jobs divided by the chosen denominator. Valid rate is accepted labels divided by jobs in that scope.')
    r.p('For binary diagnostics, NEEDS_CLARIFICATION is positive. FP is gold CLEAR predicted NEEDS_CLARIFICATION; FN is gold NEEDS_CLARIFICATION predicted CLEAR. A failure is neither predicted class, but contributes an FN to its gold class in class metrics. Ambiguity Miss Rate includes all gold ambiguities not returned as NEEDS_CLARIFICATION. Unnecessary Clarification Rate counts only gold CLEAR predicted NEEDS_CLARIFICATION; failures are reported separately.')
    r.p('Both-valid agreement = matching valid labels / jobs valid in both rounds. All-job same-label rate = matching valid labels / all aligned jobs. Pair-level correctness = pairs with both labels correct / complete input pairs. The valid-only subset with one missing pair member has no reported pair score.')
    r.p('Token totals are sums of known usage fields over completed attempts, not only final predictions. Latency mean and P95 use completed-attempt milliseconds; P95 is a linearly interpolated quantile. Attempt success means an accepted label; final validity measures the result after retries. Extra attempts and retried logical jobs are distinct quantities.',size=10)

    r.page('Appendix B Data Provenance and Reproducibility')
    r.p('Every source prediction is reconciled with its final completed attempt. Checks cover unique logical IDs, complete sample/strategy coverage and order, label and pair joins, prompt/message hashes, snapshots, model configuration, event start/completion pairing, usage sums and final execution summaries. Recomputed core metrics match the prior offline analysis within floating-point tolerance. All protected source files are hashed before and after generation.')
    paths=[ROOT/'data/processed/test_inputs.csv',ROOT/'data/processed/test_labels.csv',ROOT/'src/evaluate.py',ROOT/'src/evaluation_adapter.py']
    paths += [folder/file for folder in RUNS.values() for file in ['pilot_manifest.json','predictions.jsonl','attempts.jsonl','execution_summaries.jsonl']]
    r.caption('Table B1. SHA-256 prefixes for compact inspection. integrity.json contains every full 64-character hash.')
    r.table(['Source file','SHA-256 prefix'],[[str(p.relative_to(ROOT)).replace('results/pilot/','').replace(RUNS['R1'].name,'R1').replace(RUNS['R2'].name,'R2'),sha(p)[:24]] for p in paths],[3.7,2.95])
    r.heading('Rebuild command',2)
    r.p('From the repository root: python -B results/reports/generate_report.py --render',size=10)
    r.p('Required packages: python-docx, matplotlib, numpy, Pillow, PyMuPDF and pywin32 for Windows Word automation. README.md records installation and renderer details. Without --render the same raw-data checks, CSVs, figures and editable DOCX are rebuilt without launching Word.',size=10)
    r.heading('Delivered analysis files',2)
    r.p('The reports directory contains the DOCX and PDF, this generation script, README.md, integrity.json, figure_manifest.json and package versions. figures contains 16 numbered main figures plus six standalone confusion matrices, each in PNG and SVG. data contains metrics, classification diagnostics, prediction distributions, attempt costs and latency, aligned predictions, disagreements, error cases, per-attempt records and the exact figure source values.',size=10)
    r.p('All plots are produced from the loaded raw records. PNG exports use 350 DPI; SVG files retain vector shapes and text. Word paragraphs and tables remain editable. src/analysis.py remains unchanged and empty; these deliverables provide an independent handoff rather than replacing WANG\'s module. No Qwen API call, experiment rerun, test-based prompt revision, commit or push is performed.',size=10)
    target=BASE/(STEM+'.docx')
    r.doc.save(target)
    return target

def render_and_check(docx_path):
    # Canonical packaged renderer is attempted first and its diagnostic preserved.
    qa=BASE/'_qa';qa.mkdir(exist_ok=True)
    renderer=Path('C:/Users/49414/.codex/plugins/cache/openai-primary-runtime/documents/26.1030.10813/skills/documents/render_docx.py')
    if renderer.exists():
        env=os.environ.copy();env['PYTHONPATH']=str(BASE/'_dependencies')
        result=subprocess.run([sys.executable,str(renderer),str(docx_path),'--output_dir',str(qa/'canonical'),'--emit_pdf'],capture_output=True,text=True,env=env,timeout=60)
        (qa/'canonical_renderer.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    # Native Word is the available Windows pagination authority.
    import win32com.client
    import pythoncom
    pythoncom.CoInitialize()
    word=win32com.client.DispatchEx('Word.Application')
    word.Visible=False;word.DisplayAlerts=0
    doc=None
    try:
        doc=word.Documents.Open(str(docx_path.resolve()),ReadOnly=True,AddToRecentFiles=False)
        doc.Repaginate()
        count=int(doc.ComputeStatistics(2))
        pdf=BASE/(STEM+'.pdf')
        doc.ExportAsFixedFormat(str(pdf.resolve()),17,OpenAfterExport=False)
    finally:
        if doc is not None:doc.Close(False)
        word.Quit()
        pythoncom.CoUninitialize()
    import pymupdf as fitz
    pdfdoc=fitz.open(pdf)
    assert len(pdfdoc)==count
    pages=[]
    for i,page in enumerate(pdfdoc):
        pix=page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False)
        target=qa/f'page-{i+1:02d}.png';pix.save(target)
        text=page.get_text()
        assert text.strip(),f'Blank page {i+1}'
        pages.append(dict(page=i+1,words=len(page.get_text('words')),images=len(page.get_images()),text=text))
    dump('_qa/page_inventory.json',pages)
    # Contact sheets are for navigation; full pages remain available for review.
    for offset in range(0,len(pages),6):
        thumbs=[]
        for i in range(offset,min(offset+6,len(pages))):
            im=Image.open(qa/f'page-{i+1:02d}.png').convert('RGB');im.thumbnail((420,600));thumbs.append(im.copy())
        sheet=Image.new('RGB',(1260,1200),'#D6DADF')
        for j,im in enumerate(thumbs):sheet.paste(im,((j%3)*420,(j//3)*600))
        sheet.save(qa/f'contact-{offset//6+1}.png')
    return dict(renderer='Microsoft Word native PDF export; PyMuPDF rasterization',pdf=str(pdf),pages=count)

def write_readme():
    text='''# Group 13 Qwen Plus experimental report

This is an offline, same-test-set repetition of two completed Pilot executions.
Prompts remain v0.1-dev DRAFT. This report does not modify the experiment,
test labels, prompts, evaluate.py or analysis.py, and makes no model calls.

## Rebuild

Run from the repository root (Python 3.13 was used):

```powershell
python -m pip install --target results/reports/_dependencies python-docx pymupdf pypdf pdf2image
python -m pip install matplotlib numpy Pillow pywin32
python -B results/reports/generate_report.py
python -B results/reports/generate_report.py --render
```

The first command installs document dependencies only in this results directory.
Use the dependency packages recorded in package_versions.json for reproducibility.
The --render option requires Microsoft Word and pywin32 on Windows. It runs Word
invisibly, opens only the generated DOCX read-only, exports the same content to
PDF, closes its own Word instance, and renders every PDF page using PyMuPDF.
The packaged DOCX renderer is attempted first; its log documents the absence of
LibreOffice in this environment. Native Word is used as the pagination authority.

## Outputs

- Group13_Qwen_Plus_Experimental_Results.docx: editable paragraphs and tables.
- Group13_Qwen_Plus_Experimental_Results.pdf: native Word PDF export.
- figures/: 16 main figures plus six standalone confusion matrices, PNG/SVG.
- data/: recomputed metrics, per-class diagnostics, prediction distributions,
  agreements, aligned predictions, disagreements, attempts, error cases and
  figure_source_data.csv containing the numeric values used for every main plot.
- integrity.json: source hashes before/after, checks and document structure.
- figure_manifest.json: figure titles and file paths.
- artifact_sha256.json: output file hashes, excluding temporary dependencies/QA.
- _qa/: renderer diagnostics, per-page PNGs and text inventory for layout review.

## Statistical scope

All-job metrics retain all 200 jobs per strategy and count failures as errors.
Valid-only R1 has 200/199/200 outputs, or 599 overall; R2 has 600 valid outputs.
Macro-Precision, Macro-Recall and Macro-F1 average the two gold classes.
Failure contributes FN to its true class, with no invented predicted label.
Confusion heatmaps contain accepted labels only and disclose the R1 failure.
Primary agreement is matching valid labels divided by both-valid jobs.
All-job agreement composition separates 13 label flips from one validity change.
Token sums use every completed attempt and preserve unknown usage separately.
Latency includes completed failed attempts, excludes waits, and uses linear P95.
No significance or generalization advantage is asserted; pair clustering matters.

## Source inputs

- results/pilot/group13-full-pilot-qwen200-20261111T090248Z-76877106/
- results/pilot/group13-full-pilot-round2-20261111T153626Z-bc85f07d/
- data/processed/test_inputs.csv and test_labels.csv
- src/evaluate.py and src/evaluation_adapter.py, imported without edits
- results/analysis/round1-vs-round2-20261111T155346Z/, cross-checked read-only
'''
    (BASE/'README.md').write_text(text,encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--render',action='store_true');args=parser.parse_args()
    BASE.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True);DATA.mkdir(exist_ok=True)
    d=load_and_verify()
    make_figures(d)
    target=build_document(d)
    write_readme()
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        media=[n for n in z.namelist() if n.startswith('word/media/')]
        assert len(media)==16
    check=Document(target)
    assert len(check.inline_shapes)==16
    assert len(check.tables)>=12
    for p in FIG.glob('*.png'):
        with Image.open(p) as im:assert min(im.info['dpi'])>=300
    rendering=render_and_check(target) if args.render else {'renderer':'not requested'}
    after={p:sha(ROOT/p) for p in d['protected']}
    assert after==d['protected']
    versions={name:__import__(name).__version__ for name in ['docx','matplotlib','numpy','PIL']}
    versions.update(python=sys.version,platform=platform.platform())
    dump('package_versions.json',versions)
    dump('integrity.json',dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),all_checks_passed=True,
        network_disabled=True,source_files_unchanged=True,protected_files_before=d['protected'],protected_files_after=after,
        checks=['600 unique final IDs per round','200 jobs per strategy','sample and strategy alignment','message and source snapshot hashes',
                'same input set and model settings','gold label and pair integrity','attempt and final reconciliation','complete started/completed pairing',
                'execution summary reconciliation','core metrics match previous analysis','13 valid label flips and 1 validity change',
                'known token sums 274119 and 274669','DOCX ZIP valid','16 embedded figures','PNG DPI >= 300','protected source hashes unchanged'],
        document=dict(path=str(target),embedded_figures=len(check.inline_shapes),editable_tables=len(check.tables)),
        figures=dict(main=16,standalone_confusion=6,png=len(list(FIG.glob('*.png'))),svg=len(list(FIG.glob('*.svg')))),rendering=rendering))
    artifacts={p.relative_to(BASE).as_posix():sha(p) for p in BASE.rglob('*') if p.is_file() and
               not any(part in {'_dependencies','_qa','__pycache__'} for part in p.relative_to(BASE).parts) and p.name!='artifact_sha256.json'}
    dump('artifact_sha256.json',artifacts)
    print(json.dumps(dict(docx=str(target),rendering=rendering,figures=len(CHARTS),integrity='PASS'),indent=2))

if __name__=='__main__':main()
