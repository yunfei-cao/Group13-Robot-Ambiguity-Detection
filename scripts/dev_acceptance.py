"""Read-only dev audit and formal preparation; never invokes a real client.

Run from project root: python -B -m scripts.dev_acceptance
Requires matplotlib for exported scientific charts. Existing outputs fail closed.
"""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import socket
import statistics
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src import prompts
from src.data_prepare import normalize
from src.evaluation_adapter import convert_predictions
from src.evaluate import calculate_f1, calculate_metrics, export_scored_csv, score_samples
from src.model_client import ModelClient
from src.prompt_adapter import official_prompt_builder
from src.run_experiment import STRATEGIES, digest, load_samples, run_experiment, safety_stop

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / 'results/dev/group13-dev-qwen20-20261111T060727Z-d9300025'
OUT = ROOT / 'results/analysis/group13-dev-acceptance-20261111-verified'
FORMAL_ID = 'group13-formal-qwen200-prepared-20261111-v1'
FORMAL = ROOT / 'results/formal' / FORMAL_ID
TARGET = '9f2c7b2d7194a7947eba7803725a407eaa776a167b9e5f81a28a4d36d71d63ff'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream, strict=True))


def jsonl(path):
    # Deliberately avoid runner.read_journal, which may truncate a partial tail.
    data = path.read_bytes()
    assert data.endswith(b'\n'), f'Unterminated source journal: {path.name}'
    return [json.loads(line) for line in data.splitlines()]


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def write_text(path, value):
    with path.open('x', encoding='utf-8-sig', newline='') as stream:
        stream.write(value)


def sums(rows):
    return {k: sum(r.get('token_usage', {}).get(k) or 0 for r in rows)
            for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}


def main():
    assert not OUT.exists() and not FORMAL.exists(), 'Choose fresh output paths; no overwrite allowed'
    protected = [*ROOT.glob('src/*.py'), *ROOT.glob('tests/*.py'),
                 *ROOT.glob('prompts/*'), *ROOT.glob('data/raw/*'),
                 *ROOT.glob('data/processed/*'), *DEV.iterdir()]
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected if p.is_file()}
    OUT.mkdir(parents=True)
    FORMAL.mkdir(parents=True)
    attempts, predictions = jsonl(DEV / 'attempts.jsonl'), jsonl(DEV / 'predictions.jsonl')
    config = read_json(DEV / 'run_config.json')
    assert len(predictions) == len({r['logical_id'] for r in predictions}) == 60
    selected = load_samples(DEV / 'selected_inputs.csv')
    expected = {(s['id'], strategy) for s in selected for strategy in STRATEGIES}
    assert {(r['sample_id'], r['strategy']) for r in predictions} == expected
    pending, count = {}, Counter()
    inputs_by_id = {s['id']: s for s in selected}
    for row in attempts + predictions:
        assert row['logical_id'] == digest([row['sample_id'], row['strategy']])
        assert row['run_id'] == DEV.name
        sample = inputs_by_id[row['sample_id']]
        prompt = official_prompt_builder({k: sample[k] for k in ('environment', 'instruction')}, row['strategy'])
        assert row['prompt_sha256'] == digest(prompt)
    for row in attempts:
        lid = row['logical_id']
        if row['event'] == 'started':
            assert lid not in pending and row['attempt_number'] == count[lid] + 1
            count[lid] += 1
            pending[lid] = row['attempt_number']
        else:
            assert row['event'] == 'completed' and pending.pop(lid) == row['attempt_number']
    assert pending == {TARGET: 1}
    for row in predictions:
        history = [a for a in attempts if a['logical_id'] == row['logical_id']]
        assert row['attempts'] == count[row['logical_id']]
        if row['logical_id'] != TARGET:
            assert history[-1]['event'] == 'completed'
            for key in ('status', 'label', 'raw_text', 'is_valid', 'token_usage', 'latency_ms', 'error'):
                assert row[key] == history[-1][key]
    groups = convert_predictions(DEV / 'predictions.jsonl', ROOT / 'data/processed/dev_labels.csv')
    stats, scored_groups = {}, {}
    for strategy in STRATEGIES:
        final = [r for r in predictions if r['strategy'] == strategy]
        events = [r for r in attempts if r['strategy'] == strategy]
        completed = [r for r in events if r['event'] == 'completed']
        scored = score_samples(groups[strategy])
        scored_groups[strategy] = scored
        valid = [r for r in scored if r['valid_output']]
        latency = [r['latency_ms'] for r in completed]
        successes = [r['latency_ms'] for r in completed if r['status'] == 'success']
        stats[strategy] = dict(logical_requests=len(final), final_statuses=dict(Counter(r['status'] for r in final)),
            valid_predictions=len(valid), coverage=len(valid)/len(final),
            started_intents=sum(r['event'] == 'started' for r in events), completed_attempts=len(completed),
            attempt_statuses=dict(Counter(r['status'] for r in completed)),
            error_codes=dict(Counter(r['error']['code'] for r in completed if r.get('error'))),
            known_token_usage=sums(completed), unknown_usage_completed_attempts=sum(
                any(type(r.get('token_usage', {}).get(k)) is not int for k in
                    ('prompt_tokens', 'completion_tokens', 'total_tokens')) for r in completed),
            unknown_outcome_attempts=sum(lid in pending for lid in {r['logical_id'] for r in events}),
            known_attempt_latency_seconds=sum(latency)/1000,
            completed_attempt_latency_mean_seconds=statistics.mean(latency)/1000,
            successful_attempt_latency_mean_seconds=statistics.mean(successes)/1000,
            successful_attempt_latency_median_seconds=statistics.median(successes)/1000,
            successful_attempt_latency_max_seconds=max(successes)/1000,
            valid_only=dict(samples=len(valid), correct=sum(r['correct'] for r in valid),
                gold_distribution=dict(Counter(r['gold_label'] for r in valid)),
                accuracy=sum(r['correct'] for r in valid)/len(valid),
                macro_f1=sum(calculate_f1(valid, label) for label in prompts.LABELS)/2),
            all_records_metrics=calculate_metrics(scored),
            confusion={gold: dict(Counter(r['prediction'] for r in scored if r['gold_label']==gold))
                       for gold in prompts.LABELS})
        export_scored_csv(scored, OUT / f'scored_{strategy}.csv')

    # Verify data and prompt manifests without regenerating collaborator artifacts.
    dm = read_json(ROOT / 'data/processed/data_manifest.json')
    pm = read_json(ROOT / 'prompts/prompt_manifest.json')
    hash_checks = []
    def check_manifest_hash(path, expected):
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        lf = hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()
        assert actual == expected or lf == expected, f'Content hash mismatch: {path.name}'
        hash_checks.append(dict(path=str(path.relative_to(ROOT)), expected_sha256=expected,
            actual_sha256=actual, lf_normalized_sha256=lf, byte_match=actual==expected,
            lf_normalized_match=lf==expected))
    for name, value in dm['output_sha256'].items():
        check_manifest_hash(ROOT / 'data/processed' / name, value)
    for split, entry in dm['sources'].items():
        check_manifest_hash(ROOT / 'data/raw' / entry['filename'], entry['sha256'])
    for name, value in pm['files_sha256'].items():
        check_manifest_hash(ROOT / 'prompts' / name, value)
    for name, value in pm['development_source_sha256'].items():
        check_manifest_hash(ROOT / 'data/processed' / name, value)
    write_json(OUT / 'manifest_hash_checks.json', hash_checks)
    examples = prompts.load_examples()
    assert examples == prompts.choose_examples(ROOT / 'data/processed')[0]
    assert [e['sample_id'] for e in examples] == pm['example_ids_in_order']
    assert [e['pair_id'] for e in examples] == pm['example_pair_ids_in_order']
    assert [e['gold_label'] for e in examples] == pm['example_labels_in_order']
    for strategy in STRATEGIES:
        assert (ROOT / 'prompts' / f'{strategy}.txt').read_text(encoding='utf-8') == prompts.system_prompt(strategy) + '\n'
    test = load_samples(ROOT / 'data/processed/test_inputs.csv')
    dev = load_samples(ROOT / 'data/processed/dev_inputs.csv')
    test_labels = csv_rows(ROOT / 'data/processed/test_labels.csv')
    label_by_id = {r['sample_id']: r for r in test_labels}
    pairs = Counter(s['pair_id'] for s in test)
    assert len(test) == 200 and len(pairs) == 100 and set(pairs.values()) == {2}
    assert len(label_by_id) == len(test_labels) == 200 and set(label_by_id) == {s['id'] for s in test}
    for sample in test:
        assert sample['pair_id'] == label_by_id[sample['id']]['pair_id']
    for pid in pairs:
        assert {r['gold_label'] for r in test_labels if r['pair_id'] == pid} == set(prompts.LABELS)
    assert set(pairs) == set(dm['test_pair_ids']) == set((ROOT / 'data/processed/test_pair_ids.txt').read_text().splitlines())
    assert not {s['id'] for s in test} & {s['id'] for s in dev}
    assert not set(pairs) & {s['pair_id'] for s in dev}
    raw_dev = csv_rows(ROOT / 'data/raw/ambik_calib_100.csv')
    raw_dev_text = {normalize(r[k]) for r in raw_dev for k in
                    ('unambiguous_direct', 'unambiguous_indirect', 'ambiguous_task') if k in r and r[k].strip()}
    assert not {normalize(s['instruction']) for s in test} & raw_dev_text
    assert not {s['pair_id'] for s in selected} & {e['pair_id'] for e in examples}
    assert all(s in dev for s in selected)
    formal_prompts = []
    for sample in test:
        for strategy in STRATEGIES:
            messages = official_prompt_builder(dict(sample, gold_label='FORBIDDEN_GOLD',
                category='FORBIDDEN_CATEGORY', paired_instruction='FORBIDDEN_PAIR'), strategy)
            assert json.loads(messages[1]['content']) == {k: sample[k] for k in ('environment','instruction')}
            assert 'FORBIDDEN_' not in json.dumps(messages)
            formal_prompts.append(messages)
    model = read_json(DEV / 'model_config.json')
    assert set(model) == {'mode','model','base_url','allow_real_requests','temperature','max_tokens','timeout_seconds'}
    assert model['model'] == config['model'] == 'qwen-plus-2025-12-01'
    assert model['temperature'] == config['temperature'] == 0 and model['max_tokens'] == config['max_tokens'] == 64
    assert config['enable_thinking'] is False and config['stream'] is False
    assert config['dataset_sha256'] == digest(selected)
    assert config['prompts_sha256'] == digest([official_prompt_builder(s, t) for s in selected for t in STRATEGIES])

    # Exercise the exact journals in a disposable directory with no client transport.
    class NoCalls:
        config = SimpleNamespace(**{k: model[k] for k in
            ('mode','model','base_url','temperature','max_tokens','timeout_seconds')}, api_key='', mock_response='CLEAR')

        def classify(self, messages):
            raise AssertionError('Client call forbidden in audit')

    with tempfile.TemporaryDirectory(prefix='group13-audit-') as tmp:
        dest = Path(tmp) / 'dev' / DEV.name
        dest.mkdir(parents=True)
        for name in ('attempts.jsonl', 'predictions.jsonl', 'run_config.json'):
            (dest / name).write_bytes((DEV / name).read_bytes())
        with patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden')):
            resumed = run_experiment(selected, results_root=tmp, run_id=DEV.name, split='dev',
                client=NoCalls(), builder=official_prompt_builder, prompt_version=config['prompt_version'],
                stop_on_anomaly=True, max_logical_requests=60)
        resumed.pop('output_dir')
        assert resumed['calls_this_run'] == 0 and resumed['valid_predictions'] == 59

    # Run all existing offline tests. No real client is constructed by this audit.
    buffer = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
    with patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden')):
        result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(suite)
    write_text(OUT / 'offline_tests.txt', buffer.getvalue())
    tests = dict(run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                 skipped=len(result.skipped), successful=result.wasSuccessful())
    assert result.wasSuccessful(), 'Offline tests failed; inspect offline_tests.txt'
    assert safety_stop([r for r in attempts if r['event']=='completed']) is None
    wall = (datetime.fromisoformat(attempts[-1]['timestamp']) - datetime.fromisoformat(attempts[0]['timestamp'])).total_seconds()
    target_events = [dict(line_number=i, **r) for i,r in enumerate(attempts,1) if r['logical_id']==TARGET]
    write_json(OUT / 'interrupted_diagnosis.json', dict(logical_id=TARGET, events=target_events,
        prediction=next(r for r in predictions if r['logical_id']==TARGET), recoverable_from_attempts=False,
        provider_delivery_confirmed=False, provider_response_confirmed=False, resume_verification=resumed,
        note='Provider usage confirmed by user globally; no per-request acknowledgement for this logical ID.'))
    checks = dict(test_instructions=200, complete_test_pairs=100, test_gold_counts=dict(Counter(r['gold_label'] for r in test_labels)),
        strategies=list(STRATEGIES), logical_requests=600,
        all_manifest_byte_hashes_match=all(r['byte_match'] for r in hash_checks),
        all_manifest_content_hashes_match_after_lf_normalization=True,
        manifest_hash_note='CRLF vs LF only; originals unchanged; preparation pins actual byte hashes.',
        prompt_templates_match_runtime=True, fixed_examples_match_dev_sources=True,
        dev_test_sample_and_pair_overlap=0, test_instruction_overlap_with_original_dev=0,
        pilot_example_pair_overlap=0, all_600_messages_pass_input_whitelist=True,
        prompt_manifest_status=pm['status'], prompt_version=pm['prompt_version'],
        thinking=False, stream=False, model=model['model'], temperature=0, max_tokens=64,
        data_semantic_review=dm['manual_semantic_review'], offline_tests=tests,
        technical_interfaces_pass=True, formal_launch_ready=False,
        blockers=['Prompt manifest is DRAFT; owner review and explicit freeze required.'],
        dev_acceptance='59/60 valid; one unresolved request retained; not complete classification coverage')
    write_json(OUT / 'readiness_checks.json', checks)
    summary = dict(source_run=DEV.name, generated_at=datetime.now(timezone.utc).isoformat(),
        strategies=stats, total_final_records=60, valid_predictions=59, coverage=59/60,
        started_intents=62, completed_attempts=61, known_token_usage=sums([r for r in attempts if r['event']=='completed']),
        observed_journal_span_seconds=wall, known_attempt_latency_seconds=sum(
            r.get('latency_ms') or 0 for r in attempts if r['event']=='completed')/1000,
        notes=['Started counts are persisted intents, not confirmed provider deliveries.',
            'Token sums are lower bounds: two timeout attempts and one interrupted outcome have unknown usage.',
            'Known latency excludes the interrupted attempt, retry waits and restart downtime.',
            'Valid-only metrics exclude failed outputs; all-record metrics use unchanged WANG evaluator.',
            'Valid-only sample metrics may include incomplete pairs; no valid-only pair accuracy is reported.',
            '20 instructions/10 pairs selected by first eligible CSV order; preliminary, not randomized population estimate.'])
    write_json(OUT / 'dev_summary.json', summary)
    with (OUT / 'strategy_summary.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        fields = ['strategy','logical_requests','valid_predictions','coverage','started_intents','completed_attempts',
                  'accuracy_valid','macro_f1_valid','accuracy_all','macro_f1_all','prompt_tokens','completion_tokens',
                  'total_tokens','known_attempt_latency_seconds','successful_attempt_latency_mean_seconds']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for strategy,r in stats.items():
            writer.writerow(dict(strategy=strategy, **{k:r[k] for k in fields[1:6]},
                accuracy_valid=r['valid_only']['accuracy'], macro_f1_valid=r['valid_only']['macro_f1'],
                accuracy_all=r['all_records_metrics']['accuracy'], macro_f1_all=r['all_records_metrics']['macro_f1'],
                **r['known_token_usage'], known_attempt_latency_seconds=r['known_attempt_latency_seconds'],
                successful_attempt_latency_mean_seconds=r['successful_attempt_latency_mean_seconds']))
    make_charts(stats, scored_groups)
    prepare_formal(model, pm, formal_prompts, checks)
    write_report(summary, checks)
    assert all(sha(ROOT / name) == value for name,value in before.items()), 'Protected file changed'
    write_json(OUT / 'source_integrity.json', dict(all_protected_files_unchanged=True, sha256=before))
    print(json.dumps(dict(output=str(OUT.relative_to(ROOT)), formal_preparation=str(FORMAL.relative_to(ROOT)),
        statistics=stats, tests=tests, formal_launch_ready=False), ensure_ascii=True, indent=2))


def make_charts(stats, groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    names = ['Zero-shot', 'Few-shot', 'Checklist']
    x = np.arange(3)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for offset,key,title,color in [(-.18,'accuracy','Accuracy','#2377a8'),(.18,'macro_f1','Macro-F1','#c16a26')]:
        vals=[stats[s]['valid_only'][key]*100 for s in STRATEGIES]
        bars=axes[0,0].bar(x+offset,vals,.32,label=title,color=color)
        axes[0,0].bar_label(bars,fmt='%.1f%%',padding=3)
    axes[0,0].set(title='Valid predictions only (n=19 / 20 / 20)',ylim=(0,110),ylabel='Percent',xticks=x,xticklabels=names)
    axes[0,0].legend(loc='upper left',fontsize=8)
    for offset,key,title,color in [(-.18,'accuracy','Accuracy','#2377a8'),(.18,'macro_f1','Macro-F1','#c16a26')]:
        vals=[stats[s]['all_records_metrics'][key]*100 for s in STRATEGIES]
        bars=axes[0,1].bar(x+offset,vals,.32,label=title,color=color)
        axes[0,1].bar_label(bars,fmt='%.1f%%',padding=3)
    axes[0,1].set(title='All 20 records per strategy (failures penalized)',ylim=(0,110),xticks=x,xticklabels=names)
    axes[0,1].legend(loc='upper left',fontsize=8)
    valid=[stats[s]['valid_predictions'] for s in STRATEGIES]
    bars=axes[1,0].bar(names,valid,color='#438d79',label='Valid')
    axes[1,0].bar(names,[20-v for v in valid],bottom=valid,color='#bc5353',label='Interrupted')
    for index,v in enumerate(valid):
        axes[1,0].text(index,20.5,f'{v}/20 ({v/20:.0%})',ha='center',va='bottom')
    axes[1,0].set(title='Prediction coverage',ylim=(0,24),ylabel='Records')
    axes[1,0].legend(loc='lower right')
    bars=axes[1,1].bar(names,[stats[s]['known_token_usage']['total_tokens'] for s in STRATEGIES],color='#8273a4')
    axes[1,1].bar_label(bars,padding=3)
    axes[1,1].set(title='Known total tokens (unknown usage excluded)',ylabel='Tokens')
    axes[1,1].margins(y=.18)
    fig.suptitle('Group 13 | Qwen Plus development pilot | 10 pairs',fontsize=16)
    fig.savefig(OUT/'dev_overview.png',dpi=180)
    fig.savefig(OUT/'dev_overview.svg')
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,strategy,name in zip(axes,STRATEGIES,names):
        matrix=np.array([[sum(r['gold_label']==g and r['prediction']==p for r in groups[strategy])
                          for p in (*prompts.LABELS,'REQUEST_FAILED')] for g in prompts.LABELS])
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=10)
        for (i,j),v in np.ndenumerate(matrix): ax.text(j,i,str(v),ha='center',va='center',color='white' if v>5 else '#172b4d')
        ax.set(title=name,xticks=[0,1,2],xticklabels=['CLEAR','NEEDS\nCLARIFICATION','FAILED'],
               yticks=[0,1],yticklabels=['CLEAR','NEEDS\nCLARIFICATION'],xlabel='Prediction',ylabel='Gold')
    fig.suptitle('Development confusion matrices | all 20 records per strategy',fontsize=14)
    fig.savefig(OUT/'dev_confusion.png',dpi=180)
    fig.savefig(OUT/'dev_confusion.svg')
    plt.close(fig)


def prepare_formal(model, manifest, formal_prompts, checks):
    write_json(FORMAL/'model_config.json',model)
    pinned = {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in
        [*ROOT.glob('src/*.py'), *ROOT.glob('prompts/*'), *ROOT.glob('data/processed/*')] if p.is_file()}
    # The manifest's status must change on owner review; pin the actual templates,
    # examples and code, while separately verifying version/status at execution.
    pinned.pop('prompts/prompt_manifest.json')
    pinned[str((FORMAL/'model_config.json').relative_to(ROOT)).replace('\\','/')]=sha(FORMAL/'model_config.json')
    write_json(FORMAL/'preparation_manifest.json',dict(status='BLOCKED_DRAFT_PROMPTS_NOT_EXECUTED',
        run_id=FORMAL_ID, split='formal', input='data/processed/test_inputs.csv',
        model_config=str((FORMAL/'model_config.json').relative_to(ROOT)).replace('\\','/'),
        prompt_version=manifest['prompt_version'], prompt_status_at_preparation=manifest['status'],
        logical_requests=600, strategies=list(STRATEGIES), prompts_sha256=digest(formal_prompts),
        enable_thinking=False, stream=False, temperature=0,max_tokens=64,
        real_requests_this_task=0, expected_sha256=pinned, readiness=checks))
    command = f'''# PREPARED ONLY. Do not execute before owner prompt review/freeze and authorization.
# Run from the repository root. Same command resumes this run without resending final rows.
$ErrorActionPreference = 'Stop'
$formalDir = 'results/formal/{FORMAL_ID}'
$plan = Get-Content -LiteralPath "$formalDir/preparation_manifest.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$promptManifest = Get-Content -LiteralPath 'prompts/prompt_manifest.json' -Raw -Encoding UTF8 | ConvertFrom-Json
if ($promptManifest.status -notmatch '^(FROZEN|APPROVED)(:|$)') {{
    throw 'BLOCKED: prompt owner must review and explicitly freeze/approve the prompt manifest.'
}}
if ($promptManifest.prompt_version -ne $plan.prompt_version) {{
    throw 'Prompt version changed. Regenerate and review formal preparation under a fresh run ID.'
}}
foreach ($entry in $plan.expected_sha256.PSObject.Properties) {{
    if ((Get-FileHash -LiteralPath $entry.Name -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) {{
        throw 'Prepared input, prompt, code or model configuration changed. Regenerate preparation.'
    }}
}}
if ([string]::IsNullOrWhiteSpace($env:DASHSCOPE_API_KEY)) {{
    throw 'DASHSCOPE_API_KEY is absent; no API request attempted.'
}}
python -B -m src.run_experiment --input data/processed/test_inputs.csv --model-config "$formalDir/model_config.json" --real-api --split formal --run-id {FORMAL_ID} --results-root results --builder src.prompt_adapter:official_prompt_builder --prompt-version $plan.prompt_version --stop-on-anomaly --max-logical-requests 600
if ($LASTEXITCODE -ne 0) {{ throw 'Experiment process failed. Inspect persisted journals before resuming.' }}
'''
    write_text(FORMAL/'run_command.ps1',command)


def write_report(summary, checks):
    lines=['# Group 13 开发集验收与正式实验准备', '',
        '开发集日志与评估接口验收通过，但分类覆盖率为59/60，尚有1条结果未知；正式实验当前禁止启动：Prompt仍为DRAFT。', '',
        f"来源：`{DEV.relative_to(ROOT).as_posix()}`。标准答案：`data/processed/dev_labels.csv`。仅评估已选20条指令/10个完整配对。", '',
        '|策略|逻辑数|有效/覆盖率|启动/完成事件|超时尝试|已知输入/输出/总Token|已知尝试耗时(s)|成功平均耗时(s)|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s,r in summary['strategies'].items():
        u=r['known_token_usage']
        lines.append(f"|{s}|20|{r['valid_predictions']}/20 ({r['coverage']:.0%})|{r['started_intents']}/{r['completed_attempts']}|{r['error_codes'].get('timeout',0)}|{u['prompt_tokens']}/{u['completion_tokens']}/{u['total_tokens']}|{r['known_attempt_latency_seconds']:.3f}|{r['successful_attempt_latency_mean_seconds']:.3f}|")
    lines += ['', '最终状态：zero_shot为19 success + 1 interrupted；其余各20 success。没有最终error或invalid_output。两次timeout均在重试后成功，全部属于zero_shot，且不是目标interrupted请求。', '',
        f"已知总用量{summary['known_token_usage']['total_tokens']} Token。两次timeout和一次interrupted用量未知，不能按零计费；统计为下界，无法仅凭本地日志确认账单总额。已完成尝试耗时合计{summary['known_attempt_latency_seconds']:.3f}秒；首末事件跨度{summary['observed_journal_span_seconds']:.3f}秒，包含重启/等待，不是纯模型推理时间。", '',
        '|策略|有效样本正确数|有效样本Accuracy|有效样本Macro-F1|全部20条Accuracy|全部20条Macro-F1|完整配对正确率|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s,r in summary['strategies'].items():
        v,m=r['valid_only'],r['all_records_metrics']
        lines.append(f"|{s}|{v['correct']}/{v['samples']}|{v['accuracy']:.2%}|{v['macro_f1']:.2%}|{m['accuracy']:.2%}|{m['macro_f1']:.2%}|{m['pair_level_correctness']:.2%}|")
    lines += ['', '全量指标调用未修改的WANG score_samples/calculate_metrics：失败计为错误，并计入真实类别FN。有效样本指标仅过滤已经评分的有效行，以原calculate_f1计算两类平均，不重新用配对校验器处理19条不完整子集，也不报告有效子集配对指标。zero_shot有效子集的类别分布详见JSON；不同策略有效分母不同。初步效果基于首10个合格开发配对，不能当作正式测试集效果或随机抽样估计。', '',
        '异常请求：`'+TARGET+'`，sample `s_f07e8be2c8b536f86e35` / zero_shot。attempts第33行只有started（2026-11-11T06:22:23.006679+00:00，attempt_number=1）；没有completed、原文、标签、Token或服务端确认，不能从已有日志恢复。predictions第16行为interrupted/outcome_unknown。model_id为配置回填，不证明服务端响应。用户确认后台存在真实调用，但无法关联证明此请求已到达或返回。中断具体原因无异常栈可证；不能将其归因为已记录timeout。', '',
        '第1–33行：17次启动=15个成功逻辑请求+1次超时重试+1次未知请求；第34–123行：45次启动=44个成功逻辑请求+1次超时重试。累计60条终态=15此前成功+44后续成功+1 interrupted。当前续跑会跳过所有最终记录，包括interrupted；临时副本禁网/禁止客户端调用复验证实新增调用0。', '',
        '单条补跑提案（未实施）：给运行器增加可选strategies参数及CLI --strategies zero_shot，默认保持三策略，预算与run_config按所选策略计算；独立新run只输入目标sample，并使用原模型/提示词哈希。先离线验证仅1个逻辑请求及重复运行0调用。原重试规则仍可能产生最多3次物理尝试。独立reconciliation manifest记录原/新run、logical_id、未知结果及潜在重复收费，评估时明确映射这一条替代结果；保留原日志。当前仅选sample仍会跑三策略，故不建议直接运行。', '',
        '正式集实测为200条指令、100个完整配对，CLEAR/NEEDS_CLARIFICATION各100；三策略共600个逻辑请求。数据/Prompt manifest的原始字节哈希存在CRLF/LF差异，全部在仅归一化CRLF为LF后匹配；没有修改源文件或manifest。manifest_hash_checks.json逐项保留预期/实际/归一化哈希，正式准备绑定当前实际字节。正式集与开发集sample_id/pair_id零交集，正式指令与原始开发集direct/indirect/ambiguous指令归一化零重复。开发pilot排除固定示例配对。', '',
        '固定Few-shot为4个开发示例，来自dev_0/dev_8/dev_63/dev_36，两类各2，顺序和内容与manifest/开发源一致。三份模板与运行时生成文本一致。全部600个计划消息检查通过：当前输入只有environment/instruction，未带gold/category/pair元数据或配对另一条指令；Few-shot演示标签为设计内的开发标签。WANG标签仅在离线评估连接。', '',
        f"模型配置：qwen-plus-2025-12-01，temperature=0，max_tokens=64，timeout=30秒；客户端请求固定enable_thinking=false、stream=false。全部离线测试{checks['offline_tests']['run']}项通过，失败0，错误0。", '',
        '安全机制核对：401/403立即停止；累计3次429或连续3次不成功停止；成功/无效输出缺失或非法用量停止；单次prompt>16000、completion>64、total>16064及累计已知Token>=300000停止。累计阈值在一次尝试完成后检查，属于软停止，可越过阈值；未知请求用量不在累计内。600为逻辑请求上限，不是物理调用硬上限（每条最多3次）。无重试600次时completion上限38400，prompt部分须另计；开发已知用量按10倍缩放得到275570，仅为不含未知用量的粗略外推，不能作为正式用量上下界或预算保证。300000停止阈值可能导致正式集未跑完。', '',
        '续跑校验输入/配置/Prompt哈希与身份，已有终态跳过；完成事件无最终行可恢复；孤立started仅标记unknown不重发；重复预测/事件及中间损坏拒绝，未换行尾片段会在实际续跑时截断；OS锁阻止并发写入。只保留已确认用量，不能实现跨网络恰好一次调用。客户端没有隐藏自动重试。', '',
        f"正式准备目录：`{FORMAL.relative_to(ROOT).as_posix()}`，独立run_id `{FORMAL_ID}`；含model_config.json、preparation_manifest.json、run_command.ps1。没有创建真实run_config或attempts/predictions。准备脚本校验FROZEN/APPROVED状态、版本与输入/模板/示例/代码/模型哈希，当前DRAFT会阻止执行。若冻结涉及版本或内容变化，必须重新准备并复测，不可绕过哈希。正式运行及其后续恢复均须由用户另行授权执行。", '',
        '现有runner本身不会校验DRAFT状态，正式命令包装增加了显式检查。manifest状态文字必须由Prompt负责人根据实际审核更新；本次不改动或自行冻结。data manifest的manual_semantic_review仍标PENDING，作为已有元数据待办披露，不重新否定用户已接受的200条正式集。', '',
        '验收判断：离线技术接口条件通过，正式启动条件尚未满足。明确阻塞为Prompt仍DRAFT；开发集完整验收另有1条未知记录需获准补跑或书面接受缺失。单策略补跑能力尚未实现。300000 Token软停止可能提前结束正式批次，届时先审查原因与预算，不可通过更换run_id或删除日志绕过停止。', '',
        '图表：dev_overview.png/.svg与dev_confusion.png/.svg；逐条评分scored_*.csv；结构化dev_summary.json、strategy_summary.csv、interrupted_diagnosis.json、readiness_checks.json；完整测试输出offline_tests.txt；保护文件哈希source_integrity.json。图表标签用英文确保跨平台渲染。', '',
        '本次只新增审计脚本、派生报告/图表及准备文件；原始开发日志、CAI数据/Prompt、WANG评估及所有已有src/tests均以哈希验证未改动。未调用真实API、未补跑、未启动正式实验、未Commit/Push、未输出或记录真实API Key。']
    write_text(OUT/'acceptance_report.md','\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
