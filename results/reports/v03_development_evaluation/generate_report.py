"""Rebuild v0.3 development report offline; never dispatches model requests.

Usage: python -B results/reports/v03_development_evaluation/generate_report.py --render
Dependencies: python-docx, matplotlib, numpy, Pillow, PyMuPDF; pywin32 + Word for PDF.
Existing report-local dependencies are supported as an environment fallback.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT / 'results/reports/_dependencies'))
sys.path.insert(0, str(ROOT))
import argparse, csv, hashlib, json, os, socket, statistics, subprocess, zipfile
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone

def deny_network(*args, **kwargs):
    raise RuntimeError('Network disabled: report generation is offline only')
socket.socket.connect = deny_network
socket.create_connection = deny_network
os.environ['MPLCONFIGDIR'] = str(BASE / '_qa/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from PIL import Image
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from src.evaluation_adapter import convert_predictions
from src import evaluate as wang
from src.run_experiment import digest

EVAL = ROOT / 'results/analysis/v03-development-evaluation-20261113'
EXP = ROOT / 'results/prompt_comparison/group13-dev-v01-v03-20261113T150521Z-b2a4a074'
CAND = EXP / 'candidate'
RUN = CAND / 'pilot/group13-dev-v01-v03-20261113T150521Z-b2a4a074-candidate'
BUNDLE = ROOT / 'prompt_v0_3/candidate'
REF = ROOT / 'results/reports/Group13_Qwen_Plus_Experimental_Results.docx'
LABEL = ROOT / 'prompt_v0_3/development_comparison_85pairs/dev_labels.csv'
STEM = 'Group13_Qwen_Plus_v0.3_Development_Evaluation'
TITLE = 'Group 13 – Qwen Plus Prompt v0.3 Development Evaluation Report'
S = ['zero_shot', 'few_shot', 'checklist']
NAMES = dict(zip(S, ['Zero-shot', 'Few-shot', 'Checklist']))
COLORS = ['#24577D', '#8798A9', '#54788F']
FIGS, PLOT_DATA = [], []

def read_csv(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f, strict=True))
def jl(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]
def js(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, x):
    (BASE / p).write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
def csvout(p, rows):
    with (BASE / p).open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def pct(x, n=2): return f'{100*x:.{n}f}%'
def get(d, s, subset='all_85_pairs'): return next(m for m in d['metrics'] if m['strategy']==s and m['subset']==subset)

def load():
    protected=[p for folder in [EVAL, CAND, BUNDLE, ROOT/'prompt_v0_3/development_comparison_85pairs']
               for p in folder.rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    protected += [REF, REF.with_suffix('.pdf'), ROOT/'results/reports/generate_report.py',
                  ROOT/'src/evaluate.py', ROOT/'src/evaluation_adapter.py',
                  ROOT/'prompt_v0_3/comparison_manifest.json', EXP/'development_overlap.json', EXP/'development_pairs.csv']
    before={p.relative_to(ROOT).as_posix():sha(p) for p in protected}
    audit=js(EVAL/'audit.json')
    for p,h in audit['source_sha256'].items(): assert sha(ROOT/p)==h, f'Prior audit source changed: {p}'
    labels=read_csv(LABEL); gold={r['sample_id']:r for r in labels}
    inputs=read_csv(CAND/'snapshot/dev_inputs.csv'); inp={r['sample_id']:r for r in inputs}
    assert len(gold)==len(inp)==len(labels)==len(inputs)==170 and gold.keys()==inp.keys()
    assert Counter(r['gold_label'] for r in labels)==Counter(CLEAR=85, NEEDS_CLARIFICATION=85)
    overlap=js(EXP/'development_overlap.json'); hist={p['pair_id'] for p in overlap['pairs'] if p['previously_run']}
    allpairs={r['pair_id'] for r in labels}; assert len(allpairs)==85 and len(hist)==9
    subsets={'all_85_pairs':allpairs,'historical_9_pairs':hist,'previously_unused_76_pairs':allpairs-hist}
    config=js(RUN/'run_config.json');plan=js(CAND/'plan.json');manifest=js(BUNDLE/'prompts/prompt_manifest.json')
    assert config['mode']=='qwen' and config['prompt_version']=='v0.3-candidate' and plan['dataset_split']=='dev'
    assert config['logical_requests']==510 and config['strategies']==S
    for p,h in plan['sealed_files_sha256'].items(): assert sha(CAND/p)==h, f'Sealed file changed: {p}'
    assert not (allpairs & set(js(ROOT/'prompt_v0_3/comparison_manifest.json')['excluded_demo_pair_ids']))
    for name,h in manifest['files_sha256'].items():
        content=(BUNDLE/'prompts'/name).read_bytes().replace(b'\r\n',b'\n').replace(b'\r',b'\n')
        assert hashlib.sha256(content).hexdigest()==h
    messages=js(CAND/'snapshot/messages.json')
    assert len(messages)==510
    msg={(r['sample_id'],r['strategy']):r for r in messages}
    assert len(msg)==510 and digest([r['messages'] for r in messages])==config['prompts_sha256']
    raw=jl(RUN/'predictions.jsonl');assert len(raw)==510
    pred={(r['sample_id'],r['strategy']):r for r in raw};assert len(pred)==510
    assert set(pred)=={(sid,s) for sid in gold for s in S}
    for key,r in pred.items():
        assert r['status']=='success' and r['is_valid'] and r['raw_text'].strip()==r['label']
        assert r['run_id']==config['run_id'] and r['model_id']==config['model']
        assert r['prompt_sha256']==digest(msg[key]['messages'])
        assert json.loads(msg[key]['messages'][-1]['content'])=={k:inp[key[0]][k] for k in ['environment','instruction']}
    groups=convert_predictions(RUN/'predictions.jsonl',LABEL)
    supplied=read_csv(EVAL/'metrics.csv'); metrics=[]; scored=[]; scores={}; pairs=[]
    prior_scored={(r['sample_id'],r['strategy']):r for r in read_csv(EVAL/'scored_predictions.csv')}
    for s in S:
        rows=wang.score_samples(groups[s]);assert len(rows)==170 and len({r['sample_id'] for r in rows})==170
        for r in rows:
            sid=r['sample_id'];scores[(sid,s)]=r
            assert r['prediction']==prior_scored[(sid,s)]['prediction']
            assert str(r['correct'])==prior_scored[(sid,s)]['correct']
            scored.append(dict(strategy=s,subset='historical_9_pairs' if r['pair_id'] in hist else 'previously_unused_76_pairs',
                               **r,environment=inp[sid]['environment'],instruction=inp[sid]['instruction']))
        for sub,pids in subsets.items():
            chosen=[r for r in rows if r['pair_id'] in pids];m=wang.calculate_metrics(chosen)
            cm=[[sum(r['gold_label']==g and r['prediction']==p for r in chosen)
                 for p in ['CLEAR','NEEDS_CLARIFICATION']] for g in ['CLEAR','NEEDS_CLARIFICATION']]
            tn,fp=cm[0];fn,tp=cm[1]
            precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn)
            pc=tn/(tn+fn) if tn+fn else 0;rc=tn/(tn+fp)
            result=dict(subset=sub,strategy=s,samples=len(chosen),pairs=len(pids),correct=tn+tp,
                        accuracy=m['accuracy'],macro_f1=m['macro_f1'],precision=precision,recall=recall,
                        precision_clear=pc,recall_clear=rc,f1_clear=m['f1_clear'],
                        f1_needs_clarification=m['f1_needs_clarification'],tn=tn,fp=fp,fn=fn,tp=tp,
                        pair_accuracy=m['pair_level_correctness'],correct_pairs=round(m['pair_level_correctness']*len(pids)))
            old=next(r for r in supplied if r['strategy']==s and r['subset']==sub)
            for k,v in result.items():
                if isinstance(v,(float,int)): assert abs(float(old[k])-v)<1e-12,(s,sub,k)
            metrics.append(result)
        for pid in sorted(allpairs):
            pr=[r for r in rows if r['pair_id']==pid]
            pairs.append(dict(strategy=s,pair_id=pid,correct_members=sum(r['correct'] for r in pr)))
    events=jl(RUN/'attempts.jsonl');starts={};done={}
    for e in events:
        key=(e['logical_id'],e['attempt_number']);target=starts if e['event']=='started' else done
        assert key not in target;target[key]=e
    assert starts.keys()==done.keys()
    completed=list(done.values());assert len(completed)==521
    for key,r in pred.items():
        history=sorted([e for e in completed if e['logical_id']==r['logical_id']],key=lambda e:e['attempt_number'])
        assert len(history)==r['attempts'] and r['retry_count']==len(history)-1
        assert [e['attempt_number'] for e in history]==list(range(1,len(history)+1))
        for k in ['raw_text','status','label','is_valid','token_usage','latency_ms']:
            assert r[k]==history[-1][k]
    tokens=[]
    for s in S:
        calls=[e for e in completed if e['strategy']==s]
        known=[e for e in calls if isinstance((e.get('token_usage') or {}).get('total_tokens'),int)]
        totals={k:sum(e['token_usage'][k] for e in known) for k in ['prompt_tokens','completion_tokens','total_tokens']}
        latency=[e['latency_ms'] for e in calls if isinstance(e.get('latency_ms'),(int,float))]
        tokens.append(dict(strategy=s,calls=len(calls),successful=sum(e['status']=='success' for e in calls),
                           failed=sum(e['status']!='success' for e in calls),unknown_usage=len(calls)-len(known),
                           **totals,mean_latency_ms=statistics.mean(latency),p95_latency_ms=float(np.quantile(latency,.95)),
                           calls_with_latency=len(latency)))
    last=jl(RUN/'execution_summaries.jsonl')[-1]
    assert last['final_predictions']==last['valid_predictions']==510 and last['remaining_logical_requests']==0
    assert sum(t['total_tokens'] for t in tokens)==last['known_token_usage']['total_tokens']==349077
    assert sum(t['unknown_usage'] for t in tokens)==last['unknown_usage_attempts']==11
    categories=Counter(r['category'] for r in labels)
    d=dict(metrics=metrics,scored=scored,scores=scores,pred=pred,inputs=inp,labels=gold,pairs=pairs,
           tokens=tokens,config=config,plan=plan,manifest=manifest,overlap=overlap,hist=hist,completed=completed,
           last=last,before=before,categories=categories)
    csvout('data/metrics_recomputed.csv',metrics);csvout('data/scored_predictions.csv',scored)
    csvout('data/token_usage_and_reliability.csv',tokens);csvout('data/pair_correctness.csv',pairs)
    dump('data/model_configuration.json',config)
    return d

def record(fig, strategy, metric, value, subgroup='all_85_pairs'):
    PLOT_DATA.append(dict(figure=fig,strategy=strategy,subgroup=subgroup,metric=metric,value=value))
def prettify(ax, percent=False):
    ax.spines[['top','right']].set_visible(False);ax.spines[['left','bottom']].set_color('#A0A6AD')
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    if percent:ax.yaxis.set_major_formatter(PercentFormatter(1))
def save(fig,name,title,main=True):
    for ext in ['png','svg']:fig.savefig(BASE/'charts'/f'{name}.{ext}',dpi=350,facecolor='white')
    if main:FIGS.append(dict(number=len(FIGS)+1,name=name,title=title))
    plt.close(fig)
def basic(d,metric,name,title,baseline=None):
    fig,ax=plt.subplots(figsize=(7.0,2.65),layout='constrained')
    vals=[get(d,s)[metric] for s in S]
    b=ax.bar(list(NAMES.values()),vals,color=COLORS,width=.53)
    ax.bar_label(b,labels=[pct(v) for v in vals],padding=4,fontsize=9)
    ax.set_ylim(0,1);ax.set_ylabel(metric.replace('_',' ').title());ax.set_title(title,pad=10);prettify(ax,True)
    if baseline is not None:ax.axhline(baseline,color='#555',ls='--',lw=.9,label=f'{pct(baseline,0)} reference');ax.legend(frameon=False,loc='upper right',fontsize=8)
    for s,v in zip(S,vals):record(name,s,metric,v)
    save(fig,name,title)

def charts(d):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,'svg.fonttype':'none'})
    basic(d,'accuracy','01_accuracy','Instruction-level accuracy',.5)
    basic(d,'macro_f1','02_macro_f1','Macro-F1 across the two gold classes',.5)
    fig,axes=plt.subplots(1,2,figsize=(7,3.15),layout='constrained')
    for ax,label in zip(axes,['CLEAR','NEEDS_CLARIFICATION']):
        for i,s in enumerate(S):
            m=get(d,s);keys=['precision_clear','recall_clear','f1_clear'] if label=='CLEAR' else ['precision','recall','f1_needs_clarification']
            vals=[m[k] for k in keys];b=ax.bar(np.arange(3)+(i-1)*.25,vals,.24,color=COLORS[i],label=NAMES[s])
            ax.bar_label(b,labels=[f'{v*100:.1f}' for v in vals],padding=2,fontsize=6.7)
            for k,v in zip(keys,vals):record('03_per_class',s,k,v,label)
        ax.set_title(label.replace('_',' '));ax.set_xticks(range(3),['Precision','Recall','F1']);ax.set_ylim(0,.85);prettify(ax,True)
    axes[0].set_ylabel('Class score');fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
    save(fig,'03_per_class','Precision, recall and F1 by class')
    def heat(ax,s):
        m=get(d,s);cm=np.array([[m['tn'],m['fp']],[m['fn'],m['tp']]])
        ax.imshow(cm,vmin=0,vmax=85,cmap='Blues')
        for i in range(2):
            for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=14,color='#111')
        ax.set_xticks([0,1],['CLEAR','NEEDS\nCLARIFICATION'],fontsize=7)
        ax.set_yticks([0,1],['CLEAR','NEEDS\nCLARIFICATION'],fontsize=7)
        ax.set_xlabel('Predicted label',fontsize=8);ax.set_ylabel('Gold label',fontsize=8);ax.set_title(NAMES[s])
        return cm
    fig,axes=plt.subplots(1,3,figsize=(7.05,2.8),layout='constrained')
    for ax,s in zip(axes,S):
        cm=heat(ax,s)
        for i in range(2):
            for j in range(2):record('04_confusion_matrices',s,f'cell_{i}_{j}',int(cm[i,j]))
    save(fig,'04_confusion_matrices','Confusion matrices for all 170 instructions')
    for s in S:
        fig,ax=plt.subplots(figsize=(4.2,3.8),layout='constrained');heat(ax,s)
        save(fig,'04_confusion_'+s,'Confusion matrix '+NAMES[s],False)
    basic(d,'pair_accuracy','05_pair_accuracy','Both instructions must be correct',.25)
    fig,ax=plt.subplots(figsize=(7,2.5),layout='constrained');left=np.zeros(3)
    for k,color,label in [(2,COLORS[0],'Both correct'),(1,'#A7B8C6','One correct'),(0,'#E2E7EB','Neither correct')]:
        vals=[sum(r['strategy']==s and r['correct_members']==k for r in d['pairs']) for s in S]
        ax.barh(list(NAMES.values()),vals,left=left,color=color,label=label)
        for i,v in enumerate(vals):
            ax.text(left[i]+v/2,i,str(v),ha='center',va='center',color='white' if k==2 else '#111',fontsize=9)
            record('06_pair_composition',S[i],label,v)
        left+=vals
    ax.set_xlim(0,85);ax.set_xlabel('Number of instruction pairs');ax.invert_yaxis();ax.set_title('How correct instructions are distributed across pairs')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.23),ncol=3,frameon=False,fontsize=8);ax.spines[['top','right']].set_visible(False)
    save(fig,'06_pair_composition','Pair correctness composition')
    fig,axes=plt.subplots(1,2,figsize=(7,3),layout='constrained')
    for ax,key,title in zip(axes,['accuracy','macro_f1'],['Accuracy','Macro-F1']):
        for i,(sub,label,color) in enumerate([('historical_9_pairs','Historical 9 pairs',COLORS[0]),('previously_unused_76_pairs','Previously unused 76 pairs',COLORS[1])]):
            vals=[get(d,s,sub)[key] for s in S];b=ax.bar(np.arange(3)+(i-.5)*.34,vals,.32,label=label,color=color)
            ax.bar_label(b,labels=[pct(v,1) for v in vals],padding=3,fontsize=7)
            for s,v in zip(S,vals):record('07_history',s,key,v,sub)
        ax.set_xticks(range(3),list(NAMES.values()),fontsize=8);ax.set_ylim(0,1);ax.set_title(title);prettify(ax,True)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False,fontsize=8)
    save(fig,'07_history','Historical use and previously unused development samples')
    fig,ax=plt.subplots(figsize=(7,2.6),layout='constrained');bottom=np.zeros(3)
    for label,color in [('CLEAR',COLORS[0]),('NEEDS_CLARIFICATION',COLORS[1])]:
        vals=[sum(r['strategy']==s and r['prediction']==label for r in d['scored']) for s in S]
        ax.bar(list(NAMES.values()),np.array(vals)/170,bottom=bottom,color=color,label=label,width=.5)
        for i,v in enumerate(vals):
            ax.text(i,bottom[i]+v/340,f'{v}\n({pct(v/170,1)})',ha='center',va='center',fontsize=8,color='white' if label=='CLEAR' else '#111')
            record('08_distribution',S[i],label,v)
        bottom+=np.array(vals)/170
    ax.set_ylim(0,1);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylabel('Share of 170 predictions');ax.set_title('Prediction distribution by strategy')
    ax.legend(loc='outside lower center' if False else 'upper center',bbox_to_anchor=(.5,-.2),ncol=2,frameon=False,fontsize=8);ax.spines[['top','right']].set_visible(False)
    save(fig,'08_distribution','Predicted label distribution')
    fig,ax=plt.subplots(figsize=(7,2.45),layout='constrained')
    pids=['dev_10','dev_26','dev_33','dev_53'];x=np.arange(4)
    for i,s in enumerate(S):
        vals=[next(r['correct_members'] for r in d['pairs'] if r['strategy']==s and r['pair_id']==p) for p in pids]
        b=ax.bar(x+(i-1)*.23,vals,.22,color=COLORS[i],label=NAMES[s]);ax.bar_label(b,padding=2,fontsize=8)
        for p,v in zip(pids,vals):record('09_case_outcomes',s,'correct_members',v,p)
    ax.set_xticks(x,pids);ax.set_yticks([0,1,2]);ax.set_ylim(0,2.5);ax.set_ylabel('Correct members per pair');ax.set_title('Selected observed error contrasts')
    ax.legend(loc='outside lower center' if False else 'upper center',bbox_to_anchor=(.5,-.19),ncol=3,frameon=False,fontsize=8);prettify(ax)
    save(fig,'09_case_outcomes','Representative pair outcomes')
    fig,axes=plt.subplots(1,2,figsize=(7,2.8),layout='constrained')
    for ax,key,title in zip(axes,['prompt_tokens','completion_tokens'],['Known input tokens','Known output tokens']):
        vals=[t[key] for t in d['tokens']];b=ax.bar(list(NAMES.values()),vals,color=COLORS,width=.5)
        ax.bar_label(b,labels=[f'{v:,}' for v in vals],padding=3,fontsize=8);ax.set_ylim(0,max(vals)*1.25);ax.set_title(title);ax.set_ylabel('Tokens');prettify(ax)
        ax.tick_params(axis='x',labelsize=8)
        for s,v in zip(S,vals):record('10_tokens',s,key,v)
    save(fig,'10_tokens','Known token usage by strategy and token type')
    fig,axes=plt.subplots(1,2,figsize=(7,2.55),layout='constrained')
    retries=[t['failed'] for t in d['tokens']];b=axes[0].bar(list(NAMES.values()),retries,color=COLORS,width=.5)
    axes[0].bar_label(b,padding=3);axes[0].set_ylim(0,5.5);axes[0].set_ylabel('Failed attempts');axes[0].set_title('Recovered request failures')
    vals=[t['mean_latency_ms'] for t in d['tokens']];b=axes[1].bar(list(NAMES.values()),vals,color=COLORS,width=.5)
    axes[1].bar_label(b,labels=[f'{v:,.0f}' for v in vals],padding=3,fontsize=8);axes[1].set_ylim(0,max(vals)*1.3);axes[1].set_ylabel('Milliseconds');axes[1].set_title('Mean recorded attempt latency')
    for ax in axes:prettify(ax);ax.tick_params(axis='x',labelsize=8)
    for i,s in enumerate(S):record('11_reliability',s,'failed_attempts',retries[i]);record('11_reliability',s,'mean_latency_ms',vals[i])
    save(fig,'11_reliability','Attempt reliability and recorded latency')
    csvout('data/figure_source_data.csv',PLOT_DATA);dump('figure_manifest.json',FIGS)

class Report:
    def __init__(self):
        # Retain reference styles, section geometry, theme and page furniture.
        self.doc=Document(REF)
        body=self.doc._element.body
        for e in list(body):
            if e.tag != qn('w:sectPr'):body.remove(e)
        footer=self.doc.sections[0].footer.paragraphs[0]
        for e in list(footer._p):
            if e.tag != qn('w:pPr'):footer._p.remove(e)
        footer.add_run('Group 13  |  Qwen Plus v0.3 Development  |  ').font.size=Pt(9)
        fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');footer._p.append(fld)
        self.doc.core_properties.title=TITLE;self.doc.core_properties.author='Group 13'
        self.doc.core_properties.subject='Robot Instruction Ambiguity Detection Development Evaluation'
        self.doc.core_properties.keywords='Qwen Plus, v0.3-candidate, development evaluation, 85 pairs'
        self.tnum=0
    def p(self,text,bold=False,size=None):
        p=self.doc.add_paragraph();r=p.add_run(text);r.bold=bold
        if size:r.font.size=Pt(size)
        return p
    def h(self,text,level=1):return self.doc.add_heading(text,level)
    def page(self,title):self.doc.add_page_break();self.h(title)
    def caption(self,text):
        p=self.doc.add_paragraph(text,'Caption');p.paragraph_format.keep_with_next=True
        return p
    def fig(self,n,note='',width=6.65):
        f=FIGS[n-1];p=self.doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next=True;p.add_run().add_picture(str(BASE/'charts'/f"{f['name']}.png"),width=Inches(width))
        c=self.doc.add_paragraph(f"Figure {n}. {f['title']}. {note}",'Caption');c.paragraph_format.keep_with_next=False
    def table(self,title,headers,rows,widths=None):
        self.tnum+=1;self.caption(f'Table {self.tnum}. {title}')
        table=self.doc.add_table(rows=1,cols=len(headers));table.alignment=WD_TABLE_ALIGNMENT.CENTER;table.autofit=False
        widths=widths or [6.65/len(headers)]*len(headers)
        for col,w in zip(table.columns,widths):col.width=Inches(w)
        for c,t in zip(table.rows[0].cells,headers):c.text=str(t)
        for values in rows:
            for c,t in zip(table.add_row().cells,values):c.text=str(t)
        borders=OxmlElement('w:tblBorders')
        for edge in ['top','left','bottom','right','insideH','insideV']:
            e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');borders.append(e)
        table._tbl.tblPr.append(borders)
        for i,row in enumerate(table.rows):
            trpr=row._tr.get_or_add_trPr();trpr.append(OxmlElement('w:cantSplit'))
            if i==0:trpr.append(OxmlElement('w:tblHeader'))
            for j,c in enumerate(row.cells):
                c.width=Inches(widths[j]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                pr=c._tc.get_or_add_tcPr();mar=OxmlElement('w:tcMar')
                for edge in ['top','bottom','left','right']:
                    e=OxmlElement('w:'+edge);e.set(qn('w:w'),'65');e.set(qn('w:type'),'dxa');mar.append(e)
                pr.append(mar);shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E3EBF2' if i==0 else 'F5F7F9' if i%2==0 else 'FFFFFF');pr.append(shade)
                for p in c.paragraphs:
                    p.paragraph_format.space_before=Pt(2);p.paragraph_format.space_after=Pt(2);p.paragraph_format.line_spacing=1
                    p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
                    for run in p.runs:run.font.size=Pt(9);run.bold=i==0
        self.p('',size=2).paragraph_format.space_after=Pt(0)
        return table

def document(d):
    r=Report();c=d['config'];tok=d['tokens']
    r.doc.add_paragraph(TITLE,'Title')
    r.doc.add_paragraph('Robot Instruction Ambiguity Detection','Subtitle')
    r.p('Development evaluation | 85 instruction pairs | 13 November 2026',size=11)
    r.p('Prepared for Group 13',size=11)
    r.h('1 Executive Summary')
    r.p(f"Zero-shot achieves the highest instruction-level Accuracy ({pct(get(d,'zero_shot')['accuracy'],4)}) and Macro-F1 ({pct(get(d,'zero_shot')['macro_f1'],4)}) on this development set. Checklist achieves the highest Pair Accuracy ({pct(get(d,'checklist')['pair_accuracy'],4)}), because it places more correct predictions together within the same pair. These rankings answer different questions and are compatible.")
    r.table('Principal results on all 170 development instructions', ['Strategy','Correct','Accuracy','Macro-F1','Pair Accuracy'],
            [[NAMES[s],f"{get(d,s)['correct']}/170",pct(get(d,s)['accuracy'],4),pct(get(d,s)['macro_f1'],4),pct(get(d,s)['pair_accuracy'],4)] for s in S],[1.22,.65,1.2,1.2,1.38])
    r.p('All 510 final predictions are valid exact labels. Each strategy covers the same 170 unique sample IDs, with no duplicate, missing or unmatched prediction. Ground Truth is joined after inference through WANG\'s existing evaluation adapter and evaluator.')
    r.p('This is a development evaluation, not a new independent blind test. Nine of the 85 pairs appeared in historical development runs; the remaining 76 pairs had not previously been run in the recorded development history. Prompt provenance also records prior exposure to original test aggregates and some cases. No claim of improvement over v0.1 is justified by comparing different datasets.',bold=True)
    r.p('Accuracy is approximately 57%, only modestly above the verified 50% constant-class baseline. The three strategies differ by only one or two correct instructions. The report makes no statistical significance claim and does not treat format compliance as evidence of semantic reliability.')
    r.p('The candidate log contains 521 completed attempts for 510 final predictions, including 11 recovered errors. Cumulative known usage is 349,077 tokens; usage for 11 failed attempts is unknown. All analysis and document generation are offline.')
    r.page('2 Project Background and Research Objectives')
    r.p('A kitchen robot must decide whether an instruction supplies enough information to proceed or whether it should ask the user a clarifying question. An unnecessary question interrupts an executable request, while a missed ambiguity permits the robot to select an ingredient, object or destination without sufficient authorization. The project studies this decision before execution; it does not evaluate a robot controller or the quality of a generated action plan.')
    r.p('The immediate objective is to describe the completed Qwen Plus v0.3-candidate run, compare three prompt strategies on identical development instructions, and expose the distinction between correctness on individual instructions and correctness on complete contrastive pairs. A further objective is reproducibility: every reported metric must be traceable to an immutable prediction, its sample ID, its Ground Truth label and the recorded prompt snapshot.')
    r.h('3 Binary Classification Task Definition')
    r.p('Each input consists of an environment description and one robot instruction. The only accepted outputs are CLEAR and NEEDS_CLARIFICATION. CLEAR denotes a sufficiently determined goal and material user choices; ordinary execution steps can still require planning. NEEDS_CLARIFICATION denotes an unresolved material choice or essential fact that the user must supply, where plausible alternatives would change the task.')
    r.table('Task contract and diagnostic interpretation',['Element','Definition'],[
        ['Input','Environment description and instruction, serialized in the user message.'],
        ['CLEAR','Proceeding does not require an unresolved material user choice.'],
        ['NEEDS_CLARIFICATION','A material choice or essential condition remains unresolved.'],
        ['Positive class','NEEDS_CLARIFICATION for binary Precision and Recall.'],
        ['False positive','Gold CLEAR predicted NEEDS_CLARIFICATION: an unnecessary question.'],
        ['False negative','Gold NEEDS_CLARIFICATION predicted CLEAR: a missed ambiguity.']],[1.9,4.75])
    r.p('The evaluation retains the supplied dataset labels as Ground Truth. A plausible alternative human reading of an individual instruction is a limitation to discuss, not a reason to rewrite its label after seeing the model output. The binary task measures agreement with this annotation contract, not physical task safety or successful execution.')
    r.h('Research questions',2)
    r.p('Which strategy achieves the highest Accuracy and Macro-F1 on the shared development set? How do false positives and false negatives differ? How many pairs have both members classified correctly? Are observed outcomes different for previously used and previously unused development pairs? What request and token evidence supports the reliability of the recorded run?')
    r.page('4 Qwen Plus Model and Experimental Configuration')
    r.table('Recorded model and execution configuration',['Parameter','Recorded value'],[
        ['Model',c['model']],['Mode / temperature',f"{c['mode']} / {c['temperature']}"],
        ['max_tokens / timeout',f"{c['max_tokens']} / {c['timeout_seconds']} seconds"],
        ['enable_thinking / stream',f"{c['enable_thinking']} / {c['stream']}"],
        ['Prompt version',c['prompt_version']],['Strategies','Zero-shot; Few-shot; Checklist'],
        ['Logical requests',str(c['logical_requests'])],['Dataset split / runner split','dev / pilot'],
        ['Initial token stop / final token stop','300,000 / no total stop after authorized resume']],[2.25,4.4])
    r.p('The recorded model ID is reported exactly rather than replaced with an unverified current product name. Temperature 0.0 does not by itself establish repeatability: the present report contains one completed candidate execution per logical sample-strategy job. The runner uses a pilot routing mode, while the experiment plan explicitly identifies the data as development samples.')
    r.h('5 Development Dataset Composition')
    r.p('The evaluated subset contains 85 complete pairs and 170 instructions. Counting the actual labels yields 85 CLEAR and 85 NEEDS_CLARIFICATION instructions. Each pair contributes one member of each class, so balance is verified rather than assumed. All three strategies evaluate precisely the same sample IDs.')
    r.table('Verified development composition',['Partition','Pairs','Instructions','CLEAR','NEEDS'],[
        ['All development samples',85,170,85,85],['Historical development use',9,18,9,9],
        ['Previously unused development',76,152,76,76]],[2.55,.8,1.1,1.1,1.1])
    r.p('The 9/76 partition is read from development_overlap.json and cross-checked against development_pairs.csv and the historical development prediction IDs. The seven demonstration pair IDs excluded from the common comparison are dev_0, dev_8, dev_12, dev_36, dev_63, dev_77 and dev_98. None is among the 85 evaluated pairs.')
    r.p('The supplied category field contains '+', '.join(f'{k.replace("_"," ")}: {v} instructions' for k,v in sorted(d['categories'].items()))+'. Categories are retained as annotation metadata; the primary comparison uses complete shared pairs rather than post-hoc category selection.')
    r.page('6 Prompt v0.3 Design')
    r.p('All three strategies share a task definition that separates material user choices from routine robot planning. They resolve references using the whole instruction and environment, prohibit invented objects or preferences, and recognize explicit permission to choose any suitable item or all matching items. The environment is not treated as a necessarily exhaustive inventory of ordinary kitchen resources.')
    r.table('Strategy-specific additions to the shared task rules',['Strategy','Observed prompt design'],[
        ['Zero-shot','Shared definitions and decision rules only; no labeled demonstrations.'],
        ['Few-shot','The shared rules followed by four fixed labeled development examples, alternating CLEAR and NEEDS_CLARIFICATION.'],
        ['Checklist','The shared rules followed by silent checks of goal, references, materiality and the final decision. Only the binary label is emitted.']],[1.2,5.45])
    r.p('The Few-shot examples illustrate a specified cheddar-cheese omelette, an unresolved bread choice, locating named breakfast ingredients, and a missing specified destination. Their exact IDs and order come from the candidate manifest. Demonstration pairs are excluded from evaluation, including their unshown partners.')
    r.table('Candidate demonstration identities in manifest order',['Order','Sample ID','Pair','Label'],
            [[i+1,sid,pid,label] for i,(sid,pid,label) in enumerate(zip(d['manifest']['example_ids_in_order'],d['manifest']['example_pair_ids_in_order'],d['manifest']['example_labels_in_order']))],[.5,2.65,.8,2.7])
    r.p('The Checklist is an instruction to perform internal checks, not an observed reasoning trace. With enable_thinking=false and label-only output, the records do not reveal whether or how each internal step was followed. Explanations of why a strategy behaved differently therefore remain hypotheses unless the labels themselves establish the claim.')
    r.p('The manifest identifies prompt version v0.3-candidate and code version '+d['manifest']['code_version']+'. Its UNFROZEN and not-yet-tested language reflects the export-time status. Completed prediction and execution logs establish that the run subsequently finished; they do not retroactively certify team approval or a frozen final prompt.')
    r.p('Candidate provenance explicitly states that original test aggregates and some cases had already been seen. This exposure reinforces the development interpretation. The report describes the executed candidate and does not alter CAI\'s prompt assets or infer that revised wording improved generalization.')
    r.page('7 Experimental Procedure and Reproducibility')
    r.p('The experiment prepares the shared 170-input snapshot, constructs all three message sets, and records one logical job per sample ID and strategy. The journal preserves started and completed attempts, retries, response text, status, prompt digest, model ID and available usage. After the final resume, all 510 logical jobs have valid final predictions and no unresolved job remains.')
    r.p('Offline evaluation reads predictions.jsonl and joins labels strictly by sample_id using evaluation_adapter.convert_predictions. WANG\'s score_samples validates complete pairs and strict response parsing. calculate_metrics supplies Accuracy, both class F1 scores, Macro-F1 and Pair Accuracy. Additional confusion counts and per-class Precision/Recall are calculated from the same scored rows. Independent counts reproduce every numeric field in the earlier metrics.csv to within 1e-12.')
    r.p('Coverage checks require exactly 170 unique IDs for each strategy, 510 unique sample-strategy identities overall, identical strategy coverage, one run ID, valid labels, and no unmatched Ground Truth row. Snapshot message hashes, sealed plan hashes and each final prediction\'s last completed attempt are reconciled. Input, label, prompt and evaluator hashes are checked before and after report generation.')
    r.h('8 Evaluation Metrics')
    r.table('Metric definitions and denominators',['Metric','Definition'],[
        ['Accuracy','(TP + TN) / 170 on the full set; denominator changes to 18 or 152 in the two partitions.'],
        ['Precision / Recall','For NEEDS_CLARIFICATION: TP/(TP+FP) and TP/(TP+FN). Reported separately for CLEAR by exchanging classes.'],
        ['Class F1 / Macro-F1','Class F1 = 2TP/(2TP+FP+FN); Macro-F1 is the arithmetic mean of the two class F1 scores.'],
        ['Pair Accuracy','Pairs with both members correct / complete pairs: denominator 85, 9 or 76.'],
        ['Valid-output rate','Accepted exact labels after strip() / predictions; 100% for each strategy.']],[1.45,5.2])
    r.p('Macro-F1 assigns equal weight to the two classes. It penalizes a strategy that gains ambiguity Recall by over-requesting clarification, or gains CLEAR Recall by missing ambiguity. Even with balanced labels, unequal predicted class totals can make Macro-F1 lower than Accuracy.')
    r.h('Verified reference baselines',2)
    r.p('Because the observed classes are 85/85, always-CLEAR and always-NEEDS both achieve 50% Accuracy, 33.3333% Macro-F1 and 0% Pair Accuracy. Independent fair random labels have expected Accuracy 50% and expected Pair Accuracy 25%. A 50% population Macro-F1 is a fair-coin reference, not an exact expected finite-sample Macro-F1. No random predictions were generated and no significance test is implied.')
    r.page('9 Overall Classification Results')
    r.fig(1,'Dashed line: verified 50% constant-class Accuracy baseline. All strategies use n=170.')
    r.fig(2,'The 50% line is a population fair-coin reference; constant-class Macro-F1 is 33.3333%.')
    r.table('Overall instruction-level metrics',['Strategy','Accuracy','Macro-F1','Precision','Recall','Errors'],
            [[NAMES[s],pct(get(d,s)['accuracy'],4),pct(get(d,s)['macro_f1'],4),pct(get(d,s)['precision']),pct(get(d,s)['recall']),170-get(d,s)['correct']] for s in S],[1.15,1.15,1.15,1.05,1.05,1.1])
    r.p('Zero-shot classifies 98 instructions correctly, compared with 97 for Checklist and 96 for Few-shot. Its advantage is 0.5882 percentage points over Checklist and 1.1765 points over Few-shot in Accuracy. These are descriptive within-set differences of one and two instructions, not evidence of a stable superiority in the wider task population.')
    r.p('All outputs are valid, so valid-only and all-prediction classification metrics coincide. The 100% valid-output rate measures output contract compliance; it does not change the 72, 74 and 73 semantic errors in Zero-shot, Few-shot and Checklist respectively.')
    r.page('10 Per-Class Precision Recall and F1')
    r.fig(3,'Each gold class has support 85. Bar labels show percentages to one decimal place.')
    r.table('Editable per-class results with support 85 in every row',['Strategy','Gold class','Precision','Recall','F1'],
            [[NAMES[s],label,pct(get(d,s)[p],4),pct(get(d,s)[rec],4),pct(get(d,s)[f],4)]
             for s in S for label,p,rec,f in [('CLEAR','precision_clear','recall_clear','f1_clear'),('NEEDS','precision','recall','f1_needs_clarification')]],
            [1.25,1.2,1.4,1.4,1.4])
    r.p('Few-shot has the highest NEEDS_CLARIFICATION Recall, 49/85 (57.6471%), but also makes the most unnecessary clarification requests, 38. Its CLEAR Recall is only 47/85 (55.2941%). Detecting two more ambiguous instructions than Zero-shot is offset by four fewer correct CLEAR decisions, producing two fewer correct predictions overall.')
    r.p('Checklist has the highest CLEAR Recall, 53/85 (62.3529%), and the lowest NEEDS_CLARIFICATION Recall, 44/85 (51.7647%). Relative to Zero-shot it gains two correct CLEAR decisions but loses three correct ambiguity decisions. Its class F1 scores are consequently less balanced: 59.2179% for CLEAR and 54.6584% for NEEDS_CLARIFICATION.')
    r.p('Zero-shot\'s class F1 scores are 58.6207% and 56.6265%, yielding the highest average. This numerical balance explains the Macro-F1 ranking. It does not identify a causal mechanism in the model or establish that shorter prompts generally perform better.')
    r.page('11 Confusion Matrix Analysis')
    r.fig(4,'Rows: Ground Truth; columns: predictions. Shared color scale 0–85. Every matrix totals 170.')
    r.table('Confusion counts and class-specific error rates',['Strategy','TN','FP','FN','TP','Miss rate','Unnecessary rate'],
            [[NAMES[s],get(d,s)['tn'],get(d,s)['fp'],get(d,s)['fn'],get(d,s)['tp'],pct(get(d,s)['fn']/85),pct(get(d,s)['fp']/85)] for s in S],
            [1.25,.65,.65,.65,.65,1.4,1.4])
    r.p('With NEEDS_CLARIFICATION as the positive class, false negatives are gold ambiguities incorrectly accepted as CLEAR. Zero-shot misses 38/85 ambiguities; Few-shot misses 36/85; Checklist misses 41/85. False positives are gold CLEAR instructions that trigger clarification: 34, 38 and 32 respectively. No request failures are hidden in these matrices because every final prediction is valid.')
    r.h('Why Zero-shot leads in instruction-level metrics',2)
    r.p('Zero-shot has the smallest total error count: 34 + 38 = 72, compared with Few-shot\'s 38 + 36 = 74 and Checklist\'s 32 + 41 = 73. Against Checklist, the gain of three ambiguity detections outweighs the loss of two correct CLEAR decisions. Against Few-shot, four fewer unnecessary questions outweigh two additional missed ambiguities. These arithmetic tradeoffs fully account for the Accuracy ranking.')
    r.p('Macro-F1 also considers false-positive and false-negative balance within each class. Zero-shot\'s predicted distribution is closer to balanced than Checklist\'s, while its lower overall error count gives it a higher Macro-F1 than Few-shot. All three matrices nevertheless show substantial overlap between the classes; roughly four in ten instructions remain misclassified.')
    r.h('Interpretation boundary',2)
    r.p('The matrices support statements about error direction and frequency. They cannot establish that examples caused over-clarification or that the silent checklist was faithfully executed. A plausible hypothesis is that the checklist shifts attention toward routine planning permissions in some contexts. Testing that hypothesis requires controlled ablations and additional data, not retrospective inference from label counts alone.')
    r.page('12 Pair-Level Accuracy Analysis')
    r.fig(5,'Pair denominator: 85. Dashed line: 25% expected under independent fair random labels.')
    r.fig(6,'Every row totals 85 pairs. Counts distinguish both-correct, one-correct and neither-correct pairs.')
    r.p('A pair is successful only when its CLEAR and NEEDS_CLARIFICATION members are both correct. Zero-shot succeeds on 16 pairs, Few-shot on 15, and Checklist on 17. Thus a higher number of correct individual instructions need not imply a higher number of fully discriminated pairs.')
    r.p('The arithmetic resolves the apparent ranking conflict. Zero-shot has 16 both-correct, 66 one-correct and 3 neither-correct pairs: 2 × 16 + 66 = 98 correct instructions. Checklist has 17 both-correct, 63 one-correct and 5 neither-correct pairs: 2 × 17 + 63 = 97. Checklist concentrates slightly more successes within the same pairs, while also having more completely failed pairs.')
    both={s:{r['pair_id'] for r in d['pairs'] if r['strategy']==s and r['correct_members']==2} for s in S}
    r.p(f"Zero-shot and Checklist share {len(both['zero_shot']&both['checklist'])} fully correct pairs. Checklist uniquely succeeds on {len(both['checklist']-both['zero_shot'])} pairs that Zero-shot does not fully solve, while Zero-shot uniquely succeeds on {len(both['zero_shot']-both['checklist'])}. The net one-pair advantage is therefore local, not uniform dominance. All Pair Accuracy values remain below the 25% independent fair-coin expectation; no significance conclusion is drawn from this comparison.")
    r.page('13 Previously Used and Previously Unused Samples')
    r.fig(7,'Historical partition: n=18, 9 pairs. Previously unused partition: n=152, 76 pairs. Both remain development data.')
    r.table('Partition results and complete-pair success',['Partition','Strategy','Correct','Accuracy','Macro-F1','Pairs correct'],
            [[label,NAMES[s],f"{get(d,s,sub)['correct']}/{get(d,s,sub)['samples']}",pct(get(d,s,sub)['accuracy'],4),pct(get(d,s,sub)['macro_f1'],4),f"{get(d,s,sub)['correct_pairs']}/{get(d,s,sub)['pairs']}"]
             for sub,label in [('historical_9_pairs','Historical'),('previously_unused_76_pairs','Unused')] for s in S],
            [1,1.05,.8,1.3,1.3,1.1])
    r.p('Checklist leads on the small historical partition at 13/18 correct (72.2222%) and 4/9 fully correct pairs (44.4444%). Zero-shot leads on the larger previously unused partition at 86/152 correct (56.5789%) and 56.5112% Macro-F1. All three strategies correctly classify both members of exactly 13/76 previously unused pairs (17.1053%).')
    r.p('Historical results are higher for every strategy, but the partitions are neither randomized nor equal in size. Their difference cannot establish memorization, a prompt benefit or a causal exposure effect. In the 18-instruction partition, a single changed prediction moves Accuracy by 5.5556 percentage points; one pair moves Pair Accuracy by 11.1111 points.')
    r.p('Previously unused means absent from the recorded historical development run, not unseen by all people or independent of prompt development. The 76-pair result is a useful disclosed development slice, but it does not convert this exercise into a blind test.')
    r.page('14 Prompt Strategy Comparison')
    r.fig(8,'Gold distribution is 85 CLEAR / 85 NEEDS_CLARIFICATION; predicted totals may differ.')
    r.p('Zero-shot emits 89 CLEAR and 81 NEEDS_CLARIFICATION labels. Few-shot emits 83 and 87, while Checklist emits 94 and 76. These are relatively small aggregate shifts, yet each strategy makes many individual mistakes. A nearly balanced output distribution is therefore insufficient evidence of correct decisions.')
    transitions=[]
    for s in ['few_shot','checklist']:
        gain=sum(not d['scores'][(sid,'zero_shot')]['correct'] and d['scores'][(sid,s)]['correct'] for sid in d['labels'])
        loss=sum(d['scores'][(sid,'zero_shot')]['correct'] and not d['scores'][(sid,s)]['correct'] for sid in d['labels'])
        transitions.append([NAMES[s],gain,loss,gain-loss])
    r.table('Sample correctness changes relative to Zero-shot',['Compared strategy','Wrong to correct','Correct to wrong','Net correct change'],transitions,[2.05,1.5,1.5,1.6])
    r.fig(9,'Four real pairs selected to show gains and losses. This is an illustrative selection, not a random or exhaustive error taxonomy.')
    r.p('The selected pairs show that strategy gains are context dependent. Checklist recovers a clear chip-specific request in dev_10 and a missing destination in dev_53, while Zero-shot alone among these two strategies detects the pie choice in dev_26 and the bowl choice in dev_33. The following section preserves complete instruction text and the supplied environment for each case.')
    r.page('15 Error Case Analysis')
    r.p('Cases are selected deterministically by pair ID (dev_10, dev_26, dev_33, dev_53) to illustrate opposing strategy outcomes. Full records, sample IDs, predictions and supplied labels are exported to data/error_cases.csv. Interpretations concern the visible wording; the model did not output rationales. C denotes CLEAR and N denotes NEEDS_CLARIFICATION in the case tables.',size=10)
    def case(pid,title,analysis):
        r.h(title,2)
        rows=sorted([x for x in d['labels'].values() if x['pair_id']==pid],key=lambda x:x['gold_label'])
        r.p('Environment: '+d['inputs'][rows[0]['sample_id']]['environment'],size=9.5)
        for row in rows:
            sid=row['sample_id'];r.p(f"Gold {row['gold_label']}: "+d['inputs'][sid]['instruction'],size=10)
        r.table(f'Observed predictions for {pid}',['Gold','Zero-shot','Few-shot','Checklist'],
                [[row['gold_label'].replace('NEEDS_CLARIFICATION','NEEDS')]+['C' if d['scores'][(row['sample_id'],s)]['prediction']=='CLEAR' else 'N' for s in S] for row in rows],
                [2.3,1.45,1.45,1.45])
        r.p(analysis,size=10)
    case('dev_10','Case A Explicit chips and an unresolved chip choice',
         'Checklist correctly distinguishes the explicit jalapeno chips from the generic chips among three chip types. Zero-shot and Few-shot request clarification for both members, creating a false positive on the specified instruction. This is one concrete contributor to Checklist\'s pair advantage. The records do not establish which clause triggered the other strategies\' rejection.')
    case('dev_26','Case B A missing pie choice',
         'Zero-shot identifies the unresolved pie reference in the ambiguous member while accepting the explicit apple-pie request. Few-shot and Checklist predict CLEAR for both, missing the dataset\'s ambiguity. The five pie varieties in the supplied environment make the unresolved object choice visible. The added safety wording does not name a pie; an attention effect is only a hypothesis.')
    r.page('15 Error Case Analysis Continued')
    case('dev_33','Case C A ceramic bowl versus an unspecified bowl',
         'The gold CLEAR member names the ceramic bowl; the ambiguous member asks for a bowl among metal, plastic and ceramic alternatives. Zero-shot and Few-shot classify the pair correctly. Checklist accepts both and misses the ambiguous member. Treating bowl selection as a routine tool choice could explain the result, but no internal reasoning is available to confirm that hypothesis.')
    case('dev_53','Case D An explicit glass versus somewhere',
         'All strategies accept the explicit glass request. Few-shot and Checklist flag the ambiguous somewhere destination, while Zero-shot accepts it and misses the ambiguity. The environment contains a glass, but the Ground Truth does not treat its presence as resolving the unspecified destination. This case contributes one full pair to Checklist\'s advantage over Zero-shot.')
    r.h('What the cases establish',2)
    r.p('Checklist\'s successes are not universal: the same strategy that improves dev_10 and dev_53 fails to discriminate dev_26 and dev_33. These local outcomes support the pair-composition explanation, while also showing why a single broad claim such as “checklists resolve ambiguity better” would exceed the evidence. Error types are not manually assigned to the full dataset, so the report does not present a fabricated frequency taxonomy.')
    r.p('The supplied labels remain unchanged even when a reader could debate whether a missing utensil or destination is operationally material. An independent annotation review would be a separate future activity with a written adjudication protocol. It should not silently replace labels used in this completed evaluation.')
    r.page('16 API Token Usage and Experiment Reliability')
    r.fig(10,'Known usage summed over completed attempts; missing usage remains unknown. Input tokens dominate.')
    r.table('Attempt accounting and known total tokens',['Strategy','Calls','Errors','Unknown usage','Known tokens','Tokens / final'],
            [[NAMES[t['strategy']],t['calls'],t['failed'],t['unknown_usage'],f"{t['total_tokens']:,}",f"{t['total_tokens']/170:,.2f}"] for t in tok],
            [1.15,.6,.6,1.2,1.45,1.65])
    r.fig(11,'Each strategy ends with 170 valid predictions. Recorded latency includes failed attempts and is an operational diagnostic.')
    r.p('The journal has 521 started and 521 completed attempts: 510 successes and 11 errors (6 timeouts, 5 network errors). The final predictions have no invalid output and no unresolved sample. Failures are retained in the journal and accounted for in retry totals rather than erased by the successful final labels.',size=10)
    r.p('Known usage is 347,347 input tokens + 1,730 output tokens = 349,077 total. All 11 failed attempts lack complete token usage, so 349,077 is a known subtotal, not an exact billing total. The execution first stopped at 300,568 known tokens and resumed under the recorded authorization to remove the total-token stop. Cumulative summaries are not added together, which would double-count usage.',size=10)
    r.p('Few-shot uses 146,024 known tokens versus 89,554 for Zero-shot, an increase of '+pct(146024/89554-1)+'. Checklist uses 113,499 ('+pct(113499/89554-1)+' more than Zero-shot). These measured token differences are consistent with longer prompt inputs, but no monetary cost is estimated because billing rates and unknown failed-attempt usage are not established.',size=10)
    r.page('17 Discussion and Limitations')
    r.h('Observed ranking and practical meaning',2)
    r.p('Zero-shot has the best Accuracy and Macro-F1 because it minimizes the observed total classification errors while keeping the two class F1 scores relatively balanced. Checklist has the best Pair Accuracy because 17 of its correct CLEAR decisions coincide with correct ambiguity decisions within the same pair. It also has five completely failed pairs, compared with three for Zero-shot. These are measurable outcome structures, not claims about unobserved model reasoning.')
    r.p('The absolute performance remains limited. Zero-shot correctly labels 98/170 instructions and fully solves only 16/85 pairs. Checklist misses 41/85 gold ambiguities despite the highest pair count. None of these results supports deploying unreviewed binary decisions as a reliable execution gate. This report evaluates annotation agreement, with no robot execution, severity weighting or user-interaction experiment.')
    r.h('Limits of generalization and comparison',2)
    r.p('The set is small, paired and used for development. Nine pairs have historical exposure, and the candidate was designed after exposure to original test results and some cases. Few-shot demonstrations are development examples, although their pairs are excluded here. Strategy differences of one or two instructions and one or two pairs are small descriptive effects. There are no confidence intervals, pre-registered superiority criteria or statistical significance claims.')
    r.p('The earlier v0.1 results around 53–56% were obtained under a different evaluation context. Comparing them directly with the present approximately 57% development Accuracy confounds prompt version with sample selection and development exposure. A version difference can be estimated rigorously only after both versions are evaluated on the same samples under a compatible protocol. Even that would remain a development comparison unless an independent held-out protocol is established.')
    r.p('The single recorded execution does not measure repeated-run variance. API retries recover transport errors but do not supply independent semantic replicates. Known tokens exclude usage that failed calls did not report. Ground Truth is accepted as supplied; its consistency and ambiguity severity were not independently adjudicated for this report.')
    r.h('18 Conclusions and Next Steps')
    r.p('Use Zero-shot as the descriptive leader for instruction-level Accuracy and Macro-F1 in this completed candidate evaluation; retain Checklist\'s slightly higher complete-pair score as a separate result. Neither ranking establishes a general improvement over v0.1 or a statistically reliable advantage.')
    r.p('Next steps are to finish the already planned same-sample version comparison, examine paired gains and losses, and record any future prompt changes as new versions. Define the final held-out evaluation before using it, keep demonstration and development examples separate, and evaluate repeated executions if stability is a research question. If uncertainty is quantified, resample complete pairs jointly across strategies rather than treating the two members as independent observations.')
    r.page('19 Appendix Experiment Metadata and Reproducibility')
    r.p('Run ID: '+c['run_id'],size=9)
    r.table('Traceable experiment identities',['Item','Recorded value'],[
        ['Prompt version / code version',c['prompt_version']+' / '+d['manifest']['code_version']],
        ['Dataset / routing split','Development / pilot runner mode'],
        ['Creation time UTC',c['created_at']],['Final completion UTC',d['last']['timestamp']],
        ['Dataset digest prefix',c['dataset_sha256'][:24]],['Prompt collection digest prefix',c['prompts_sha256'][:24]],
        ['Historical pair IDs',', '.join(sorted(d['hist']))]], [1.9,4.75])
    r.h('Source map',2)
    r.p('Paths below are relative to the repository root. Full SHA-256 values for every protected source are supplied in integrity.json. The prior report controls the visual design only; its numeric results are not merged into this evaluation.',size=10)
    r.table('Primary evidence and its role',['Source','Role'],[
        ['results/analysis/v03-development-evaluation-20261113/','Earlier metrics.csv, report.md, scored predictions, pair results, coverage and audit; cross-checked numerically.'],
        ['results/prompt_comparison/.../candidate/','plan.json, model_config.json, snapshots, raw predictions, attempt journal, summaries and resume authorization. The exact run ID above identifies the directory.'],
        ['prompt_v0_3/candidate/prompts/','Prompt definitions, four demonstration records and version manifest.'],
        ['prompt_v0_3/development_comparison_85pairs/','170 instruction inputs and sample_id-linked Ground Truth labels.'],
        ['src/evaluate.py; src/evaluation_adapter.py','Strict parsing, pair validation, label join and scoring semantics.']],[2.8,3.85])
    r.h('Rebuild and audit outputs',2)
    r.p('From the repository root: python -B results/reports/v03_development_evaluation/generate_report.py --render',size=9)
    r.p('The script blocks network calls, validates the completed records, recomputes metrics, exports editable Word tables and PNG/SVG charts, and uses Microsoft Word to export PDF on Windows. README.md documents dependencies and the renderer fallback. Reruns replace only outputs in this new report directory; historical reports and experiment sources remain unchanged.',size=10)
    r.p('data/ contains recomputed metrics, scored predictions, per-pair counts, token and reliability summaries, full selected error cases, model configuration and numeric figure source data. figure_manifest.json numbers the 11 main figures; three additional standalone confusion matrices are supplied for reuse. quality_assurance.json records PDF pagination, editable structure and numeric checks.',size=10)
    r.p('No external references are added. All claims derive from the identified project artifacts; proposed explanations are marked as hypotheses. No real Qwen API call, experiment rerun, source-code modification, original-label edit, commit or push is part of report generation.',size=10)
    errorrows=[r for r in d['scored'] if r['pair_id'] in {'dev_10','dev_26','dev_33','dev_53'}]
    csvout('data/error_cases.csv',errorrows)
    path=BASE/(STEM+'.docx');r.doc.save(path)
    return path,r.tnum

def render(docx):
    qa=BASE/'_qa'; renderer=Path('C:/Users/49414/.codex/plugins/cache/openai-primary-runtime/documents/26.1111.10930/skills/documents/render_docx.py')
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'results/reports/_dependencies')
    attempt=subprocess.run([sys.executable,str(renderer),str(docx),'--output_dir',str(qa/'canonical'),'--emit_pdf'],capture_output=True,text=True,env=env,timeout=60)
    (qa/'canonical_renderer.log').write_text(attempt.stdout+'\n'+attempt.stderr,encoding='utf-8')
    import pythoncom, win32com.client
    pythoncom.CoInitialize();word=win32com.client.DispatchEx('Word.Application');word.Visible=False;word.DisplayAlerts=0
    doc=None
    try:
        doc=word.Documents.Open(str(docx.resolve()),ReadOnly=False,AddToRecentFiles=False)
        doc.Fields.Update()
        for sec in doc.Sections:
            for foot in sec.Footers:foot.Range.Fields.Update()
        doc.Repaginate();count=int(doc.ComputeStatistics(2));doc.Save()
        pdf=BASE/(STEM+'.pdf');doc.ExportAsFixedFormat(str(pdf.resolve()),17,OpenAfterExport=False)
    finally:
        if doc is not None:doc.Close(False)
        word.Quit();pythoncom.CoUninitialize()
    import pymupdf as fitz
    pdfdoc=fitz.open(pdf);assert len(pdfdoc)==count
    inventory=[]
    for i,p in enumerate(pdfdoc):
        text=p.get_text();assert len(text.strip())>80,f'Blank or nearly empty page {i+1}'
        assert abs(p.rect.width-595.28)<1 and abs(p.rect.height-841.89)<1
        words=p.get_text('words')
        assert all(w[0]>=0 and w[1]>=0 and w[2]<=p.rect.width+1 and w[3]<=p.rect.height+1 for w in words)
        p.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).save(qa/f'page-{i+1:02}.png')
        inventory.append(dict(page=i+1,words=len(words),images=len(p.get_images()),text=text))
    for start in range(0,count,4):
        sheet=Image.new('RGB',(1000,1420),'#D6DADF')
        for j in range(start,min(start+4,count)):
            im=Image.open(qa/f'page-{j+1:02}.png');im.thumbnail((500,705));sheet.paste(im,((j-start)%2*500,(j-start)//2*710))
        sheet.save(qa/f'contact-{start//4+1}.png')
    dump('_qa/page_inventory.json',inventory)
    return dict(renderer='Microsoft Word native PDF export; PyMuPDF page rasterization',pages=count,
                word_open_and_export=True,canonical_renderer_returncode=attempt.returncode)

def readme():
    (BASE/'README.md').write_text('''# Group 13 Qwen Plus v0.3 Development Evaluation

This directory contains an offline English research report on 510 completed
Qwen Plus v0.3-candidate predictions: 85 development pairs, 170 instructions,
three prompt strategies. This is development evaluation, not a blind test.

## Sources

- results/analysis/v03-development-evaluation-20261113/ (all metrics and audit)
- results/prompt_comparison/group13-dev-v01-v03-20261113T150521Z-b2a4a074/candidate/
  (sealed inputs/messages, plans, model configuration, predictions and journals)
- prompt_v0_3/candidate/ and prompt_v0_3/development_comparison_85pairs/
- src/evaluate.py and src/evaluation_adapter.py (reused without modification)
- Previous DOCX/PDF and generation script in results/reports/ (design reference)

## Rebuild

From the repository root, with Python 3.13 and python-docx, matplotlib, numpy,
Pillow and PyMuPDF available:

```powershell
python -B results/reports/v03_development_evaluation/generate_report.py
python -B results/reports/v03_development_evaluation/generate_report.py --render
```

PDF export additionally requires Microsoft Word and pywin32 on Windows.
The current host has no exposed workspace dependency loader or bundled Python;
the existing report-local document dependencies are reused as a fallback, with
the existing Python environment supplying plotting and Word automation packages.
No dependency is installed or changed. The packaged render_docx.py is attempted
and its diagnostic retained under _qa/canonical_renderer.log; native Word is the
pagination authority when the packaged LibreOffice route is unavailable. Word
runs invisibly, opens only the new report, refreshes page fields and exports PDF.
Each final PDF page is rasterized at 108 DPI for visual review; charts use 350 DPI.
Reruns replace only this directory's generated outputs. No API or experiment runs.

## Outputs

- Group13_Qwen_Plus_v0.3_Development_Evaluation.docx: editable body and native tables.
- Group13_Qwen_Plus_v0.3_Development_Evaluation.pdf: Word-exported equivalent.
- charts/: 11 main figures and three standalone confusion matrices, PNG and SVG.
- data/: recomputed metrics, full sample scoring, pair composition, token usage,
  selected real cases, model configuration and numeric figure source records.
- generate_report.py: reproducible offline report builder with network blocking.
- figure_manifest.json, integrity.json, quality_assurance.json: traceability and checks.
- _qa/: renderer logs, source design evidence and final page images for internal QA.

Known usage is 349,077 tokens. Eleven failed attempts lack usage; no missing
tokens are imputed as zero and no monetary billing total is asserted. Historical
and previously unused partitions remain development data. No v0.1 improvement
or statistical significance claim is made. Inputs, labels, CAI/WANG source code
and historical reports are never rewritten. No commit or push is performed.
''',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--render',action='store_true');args=parser.parse_args()
    for p in ['charts','data','_qa']:(BASE/p).mkdir(parents=True,exist_ok=True)
    d=load();charts(d);docx,tables=document(d);readme()
    rendering=render(docx) if args.render else dict(renderer='not requested')
    after={p:sha(ROOT/p) for p in d['before']};assert after==d['before'],'Protected source changed'
    doc=Document(docx);assert len(doc.inline_shapes)==11
    assert len(doc.tables)==tables and not doc.settings.element.findall(qn('w:documentProtection'))
    assert doc.styles['Normal'].font.name=='Times New Roman'
    for s in S:
        m=get(d,s)
        for v in [m['accuracy'],m['macro_f1'],m['recall'],m['pair_accuracy']]:assert 0<=v<=1
    dump('integrity.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256_before=d['before'],source_sha256_after=after,
                                unchanged=True,api_calls=0,model_experiments_rerun=0,metrics_tolerance=1e-12))
    dump('quality_assurance.json',dict(numeric_reconciliation='PASS',sample_coverage='170 unique per strategy; 510 valid; no missing or duplicate',
                                       editable_word_tables=tables,inline_chart_figures=len(doc.inline_shapes),
                                       chart_png=len(list((BASE/'charts').glob('*.png'))),chart_svg=len(list((BASE/'charts').glob('*.svg'))),
                                       source_integrity='PASS',rendering=rendering,visual_review='Pending manual page inspection'))
    dump('package_versions.json',dict(python=sys.version,python_docx=__import__('docx').__version__,matplotlib=matplotlib.__version__,numpy=np.__version__))
    print(json.dumps(dict(docx=str(docx),figures=len(FIGS),tables=tables,rendering=rendering,integrity='PASS'),indent=2))

if __name__=='__main__':main()
