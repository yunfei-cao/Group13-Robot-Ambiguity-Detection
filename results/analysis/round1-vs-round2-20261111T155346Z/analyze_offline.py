"""One-off offline Pilot handoff; does not implement or replace src/analysis.py.
Run from the repository root: python -B <path-to-this-file>
All outputs remain beside this script. No inference/client calls are made.
"""
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import json
import math
import socket
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def no_network(*args, **kwargs):
    raise RuntimeError('Network disabled for offline analysis')

socket.socket.connect = no_network
socket.create_connection = no_network
from src import evaluate as wang
from src.evaluation_adapter import convert_predictions
from src.run_experiment import digest, STRATEGIES

RUNS = {
    'round1': ROOT / 'results/pilot/group13-full-pilot-qwen200-20261111T090248Z-76877106',
    'round2': ROOT / 'results/pilot/group13-full-pilot-round2-20261111T153626Z-bc85f07d',
}
LABELS = ['CLEAR', 'NEEDS_CLARIFICATION']
PRED_COLUMNS = LABELS + ['INVALID', 'REQUEST_FAILED']
DISPLAY = dict(zero_shot='Zero-shot', few_shot='Few-shot', checklist='Checklist', ALL='ALL')

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def write_csv(name, rows):
    assert rows, name
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def pct(value):
    return f'{value:.2%}'

def table(rows, fields):
    return '\n'.join(['| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join(['---'] * len(fields)) + ' |'] +
                     ['| ' + ' | '.join(str(r.get(k, '')) for k in fields) + ' |' for r in rows])

def quantile(values, p):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)

def metrics(scored):
    # Already scored on complete pairs by WANG; filtering after scoring preserves
    # the requested valid-only denominator without bypassing input validation.
    result = wang.calculate_metrics(scored)
    precisions, recalls = [], []
    for label in LABELS:
        tp = sum(r['gold_label'] == label and r['prediction'] == label for r in scored)
        fp = sum(r['gold_label'] != label and r['prediction'] == label for r in scored)
        fn = sum(r['gold_label'] == label and r['prediction'] != label for r in scored)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        precisions.append(p)
        recalls.append(r)
        result.update({f'precision_{label.lower()}': p, f'recall_{label.lower()}': r})
    result.update(precision_macro=statistics.mean(precisions), recall_macro=statistics.mean(recalls))
    # Independent count-based verification of WANG's Accuracy and Macro-F1.
    assert math.isclose(result['accuracy'], sum(r['correct'] for r in scored) / len(scored))
    f1 = [2*p*r/(p+r) if p+r else 0.0 for p,r in zip(precisions, recalls)]
    assert math.isclose(result['macro_f1'], statistics.mean(f1))
    return result

def main():
    protected = list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'data').rglob('*')) + list((ROOT / 'prompts').rglob('*'))
    for run in RUNS.values():
        protected.extend(run.rglob('*'))
    protected = sorted(set(p for p in protected if p.is_file()))
    before = {p.relative_to(ROOT).as_posix(): sha(p) for p in protected}
    labels = read_csv(ROOT / 'data/processed/test_labels.csv')
    by_id = {r['sample_id']: r for r in labels}
    assert len(labels) == len(by_id) == 200
    assert Counter(r['gold_label'] for r in labels) == Counter(CLEAR=100, NEEDS_CLARIFICATION=100)
    matrices, score_rows, metric_rows, distribution, cost_rows = {}, [], [], [], []
    integrity = {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'network_disabled': True,
                 'experiment_kind': 'same-test-set v0.1-dev DRAFT Pilot repetition', 'rounds': {},
                 'label_counts': dict(Counter(r['gold_label'] for r in labels)), 'checks': {}}
    all_predictions, all_scores, manifests, message_lists = {}, {}, {}, {}
    for rnd, run in RUNS.items():
        manifest = read_json(run / 'pilot_manifest.json')
        manifests[rnd] = manifest
        predictions = jsonl(run / 'predictions.jsonl')
        attempts = jsonl(run / 'attempts.jsonl')
        summaries = jsonl(run / 'execution_summaries.jsonl')
        messages = read_json(run / 'snapshot/messages.json')
        message_lists[rnd] = messages
        inputs = read_csv(run / 'snapshot/data/processed/test_inputs.csv')
        input_map = {r['sample_id']: r for r in inputs}
        normalized_inputs = [dict(id=r['sample_id'], pair_id=r['pair_id'],
                                 environment=r['environment'], instruction=r['instruction']) for r in inputs]
        assert digest(normalized_inputs) == manifest['dataset_sha256']
        input_ids = [r['sample_id'] for r in inputs]
        assert len(input_ids) == len(set(input_ids)) == 200 and set(input_ids) == set(by_id)
        assert all(r['pair_id'] == by_id[r['sample_id']]['pair_id'] for r in inputs)
        expected_order = [(sid, strategy) for sid in input_ids for strategy in STRATEGIES]
        expected = set(expected_order)
        msg_map = {(m['sample_id'], m['strategy']): m for m in messages}
        assert len(msg_map) == len(messages) == 600 and set(msg_map) == expected
        assert sha(run / 'snapshot/messages.json') == manifest['messages_sha256']
        assert all(digest(m['messages']) == m['prompt_sha256'] for m in messages)
        assert digest([m['messages'] for m in messages]) == manifest['prompts_sha256']
        for m in messages:
            supplied = json.loads(m['messages'][1]['content'])
            original = input_map[m['sample_id']]
            assert supplied == {k: original[k] for k in ['environment','instruction']}
        assert sha(run / 'model_config.json') == manifest['model_config_sha256']
        assert sha(run / 'snapshot/data/processed/test_inputs.csv') == manifest['source_sha256']['data/processed/test_inputs.csv']
        for relative, expected_hash in manifest['source_sha256'].items():
            snapshot = run / 'snapshot' / relative
            assert sha(snapshot) == expected_hash, relative
        assert len(predictions) == 600 and len({r['logical_id'] for r in predictions}) == 600
        assert [(r['sample_id'], r['strategy']) for r in predictions] == expected_order
        pred_map = {(r['sample_id'], r['strategy']): r for r in predictions}
        assert set(pred_map) == expected
        starts, completed = {}, {}
        for event in attempts:
            key = event['sample_id'], event['strategy']
            assert key in expected
            assert event['logical_id'] == digest(list(key))
            assert event['run_id'] == manifest['run_id']
            assert event['prompt_sha256'] == msg_map[key]['prompt_sha256']
            attempt_key = event['logical_id'], event['attempt_number']
            assert event['event'] in {'started', 'completed'}
            target = starts if event['event'] == 'started' else completed
            assert attempt_key not in target
            if event['event'] == 'completed':
                assert attempt_key in starts, 'Completion precedes start'
            target[attempt_key] = event
        assert set(completed) <= set(starts)
        by_logical = defaultdict(list)
        for event in completed.values():
            by_logical[event['logical_id']].append(event)
        for key, p in pred_map.items():
            assert p['logical_id'] == digest(list(key))
            assert p['run_id'] == manifest['run_id'] and p['model_id'] == manifest['model']
            assert p['prompt_sha256'] == msg_map[key]['prompt_sha256']
            history = by_logical[p['logical_id']]
            assert [e['attempt_number'] for e in history] == list(range(1, len(history)+1))
            assert p['attempts'] == len(history) and p['retry_count'] == len(history)-1
            last = history[-1]
            for field in ['status', 'label', 'is_valid', 'raw_text', 'token_usage', 'latency_ms', 'error']:
                assert p[field] == last[field], (key, field)
        groups = convert_predictions(run / 'predictions.jsonl', ROOT / 'data/processed/test_labels.csv')
        scored_by_strategy = {s: wang.score_samples(groups[s]) for s in STRATEGIES}
        scored_map = {(r['sample_id'], s): r for s, rows in scored_by_strategy.items() for r in rows}
        for key, scored in scored_map.items():
            p = pred_map[key]
            assert scored['valid_output'] == p['is_valid']
            if scored['valid_output']:
                assert scored['prediction'] == p['label']
            score_rows.append(dict(round=rnd, strategy=key[1], **scored, category=by_id[key[0]]['category'],
                                   logical_id=p['logical_id'], prompt_sha256=p['prompt_sha256']))
        all_predictions[rnd], all_scores[rnd] = pred_map, scored_map
        for s in [*STRATEGIES, 'ALL']:
            scored = scored_by_strategy[s] if s != 'ALL' else list(scored_map.values())
            for scope in ['all_jobs', 'valid_only']:
                rows = scored if scope == 'all_jobs' else [r for r in scored if r['valid_output']]
                values = metrics(rows)
                # Pooling duplicates pair IDs across strategies has no meaningful pair metric.
                # Filtering a single failed member also leaves one incomplete pair.
                if s == 'ALL' or (scope == 'valid_only' and len(rows) != len(scored)):
                    values['pair_level_correctness'] = None
                    values['total_pairs'] = None
                metric_rows.append(dict(round=rnd, strategy=s, scope=scope, **values))
                matrix = [[sum(r['gold_label'] == g and r['prediction'] == p for r in rows)
                           for p in PRED_COLUMNS] for g in LABELS]
                normalized = [[v / sum(row) if sum(row) else 0 for v in row] for row in matrix]
                matrices[f'{rnd}/{s}/{scope}'] = dict(gold_rows=LABELS, prediction_columns=PRED_COLUMNS,
                                                     counts=matrix, row_normalized=normalized)
                assert sum(map(sum, matrix)) == len(rows)
            valid = [r for r in scored if r['valid_output']]
            dist = dict(round=rnd, strategy=s, total_jobs=len(scored), valid_count=len(valid),
                        failures=len(scored)-len(valid))
            for label in LABELS:
                n = sum(r['prediction'] == label for r in valid)
                dist[f'{label.lower()}_count'] = n
                dist[f'{label.lower()}_share_valid'] = n / len(valid)
                dist[f'{label.lower()}_share_all'] = n / len(scored)
                dist[f'gold_{label.lower()}_share_valid'] = sum(r['gold_label']==label for r in valid)/len(valid)
            distribution.append(dist)
            events = [e for e in completed.values() if s == 'ALL' or e['strategy'] == s]
            started = [e for e in starts.values() if s == 'ALL' or e['strategy'] == s]
            finals = [p for p in predictions if s == 'ALL' or p['strategy'] == s]
            durations = [e['latency_ms'] for e in events]
            successes = [e['latency_ms'] for e in events if e['is_valid']]
            cost = dict(round=rnd, strategy=s, logical_jobs=len(finals), started_attempts=len(started),
                        completed_attempts=len(events), unfinished_starts=len(started)-len(events),
                        extra_attempts=len(started)-len(finals), retried_logical_jobs=sum(p['attempts']>1 for p in finals),
                        final_failed_jobs=sum(not p['is_valid'] for p in finals),
                        valid_output_rate=sum(p['is_valid'] for p in finals)/len(finals),
                        successful_attempts=sum(e['is_valid'] for e in events),
                        attempt_success_rate=sum(e['is_valid'] for e in events)/len(events),
                        unknown_usage_attempts=sum(any(e['token_usage'].get(k) is None for k in ['prompt_tokens','completion_tokens','total_tokens']) for e in events),
                        network_error_attempts=sum((e.get('error') or {}).get('code')=='network_error' for e in events),
                        timeout_attempts=sum((e.get('error') or {}).get('code') in {'timeout','request_timeout'} for e in events),
                        latency_attempt_mean_ms=statistics.mean(durations), latency_attempt_median_ms=statistics.median(durations),
                        latency_attempt_p95_ms=quantile(durations,.95), latency_attempt_max_ms=max(durations),
                        latency_valid_attempt_mean_ms=statistics.mean(successes),
                        latency_attempt_sum_ms=sum(durations),
                        latency_logical_attempts_mean_ms=sum(durations)/len(finals),
                        stopped_reason=summaries[-1]['stopped_reason'])
            for k in ['prompt_tokens','completion_tokens','total_tokens']:
                known = [e['token_usage'].get(k) for e in events if e['token_usage'].get(k) is not None]
                cost[f'known_{k}'] = sum(known)
                cost[f'mean_{k}_per_known_attempt'] = statistics.mean(known) if known else None
            cost['known_total_tokens_per_logical_job'] = cost['known_total_tokens']/len(finals)
            cost_rows.append(cost)
        overall = cost_rows[-1]
        summary = summaries[-1]
        assert summary['final_predictions'] == len(predictions) == 600
        assert summary['valid_predictions'] == sum(p['is_valid'] for p in predictions)
        assert summary['remaining_logical_requests'] == 0 and summary['stopped_reason'] is None
        assert sum(s['calls_this_run'] for s in summaries) == len(starts)
        assert summary['unknown_usage_attempts'] == overall['unknown_usage_attempts']
        assert summary['known_token_usage'] == {k: overall[f'known_{k}'] for k in ['prompt_tokens','completion_tokens','total_tokens']}
        assert set(summary['unresolved_logical_ids']) == {p['logical_id'] for p in predictions if not p['is_valid']}
        integrity['rounds'][rnd] = dict(run_id=manifest['run_id'], final_records=len(predictions),
            unique_logical_ids=len({p['logical_id'] for p in predictions}), valid_predictions=sum(p['is_valid'] for p in predictions),
            strategy_counts=dict(Counter(p['strategy'] for p in predictions)), order_matches_snapshot=True,
            prompt_hashes_match=True, attempt_final_reconciliation=True, latest_summary=summary,
            historical_stop_reasons=[s['stopped_reason'] for s in summaries],
            cumulative_calls=sum(s['calls_this_run'] for s in summaries),
            attempt_error_codes=dict(Counter((e.get('error') or {}).get('code') for e in completed.values() if e.get('error'))))
    for k in ['messages_sha256','prompts_sha256','dataset_sha256','model_config_sha256','prompt_version','prompt_status','model','temperature','max_tokens','enable_thinking','stream']:
        assert manifests['round1'][k] == manifests['round2'][k], k
    assert message_lists['round1'] == message_lists['round2']
    assert sha(ROOT/'data/processed/test_inputs.csv') == sha(RUNS['round1']/'snapshot/data/processed/test_inputs.csv')
    integrity['checks'].update(same_test_set=True, identical_messages=True, same_model_settings=True,
        same_v01_dev_draft=True, current_test_inputs_match_snapshot=True, unique_complete_identities=True)
    agreements, differences, aligned = [], [], []
    for key, p1 in all_predictions['round1'].items():
        p2 = all_predictions['round2'][key]
        a, b = all_scores['round1'][key], all_scores['round2'][key]
        assert p1['logical_id'] == p2['logical_id'] and p1['prompt_sha256'] == p2['prompt_sha256']
        valid = a['valid_output'] and b['valid_output']
        same = valid and a['prediction'] == b['prediction']
        change = ('same_label' if same else 'label_flip' if valid else 'validity_change')
        transition = ('correct_to_wrong' if a['correct'] and not b['correct'] else
                      'wrong_to_correct' if not a['correct'] and b['correct'] else
                      'both_correct' if a['correct'] and b['correct'] else 'both_wrong')
        row = dict(sample_id=key[0], strategy=key[1], pair_id=by_id[key[0]]['pair_id'],
                   category=by_id[key[0]]['category'], gold_label=a['gold_label'], logical_id=p1['logical_id'],
                   prompt_sha256=p1['prompt_sha256'], round1_prediction=a['prediction'], round2_prediction=b['prediction'],
                   round1_status=p1['status'], round2_status=p2['status'],
                   round1_correct=a['correct'], round2_correct=b['correct'], both_valid=valid,
                   same_valid_label=same, change_type=change, correctness_transition=transition)
        aligned.append(row)
        if not same:
            current_input = next(r for r in inputs if r['sample_id'] == key[0])
            differences.append(dict(**row, environment=current_input['environment'], instruction=current_input['instruction']))
    for s in [*STRATEGIES,'ALL']:
        rows = [r for r in aligned if s=='ALL' or r['strategy']==s]
        both = sum(r['both_valid'] for r in rows)
        same = sum(r['same_valid_label'] for r in rows)
        agreements.append(dict(strategy=s, total_jobs=len(rows), both_valid=both, same_valid_label=same,
            agreement_both_valid=same/both, both_valid_coverage=both/len(rows), same_valid_label_rate_all=same/len(rows),
            label_flips=sum(r['change_type']=='label_flip' for r in rows),
            validity_changes=sum(r['change_type']=='validity_change' for r in rows),
            correct_to_wrong=sum(r['correctness_transition']=='correct_to_wrong' for r in rows),
            wrong_to_correct=sum(r['correctness_transition']=='wrong_to_correct' for r in rows)))
    delta_rows = []
    for s in [*STRATEGIES,'ALL']:
        x = next(r for r in cost_rows if r['round']=='round1' and r['strategy']==s)
        y = next(r for r in cost_rows if r['round']=='round2' and r['strategy']==s)
        delta = dict(strategy=s)
        for k in x:
            if isinstance(x[k], (int,float)) and isinstance(y[k], (int,float)):
                delta[f'{k}_round2_minus_round1'] = y[k]-x[k]
        for k in ['accuracy','macro_f1','precision_macro','recall_macro']:
            m1 = next(r for r in metric_rows if r['round']=='round1' and r['strategy']==s and r['scope']=='all_jobs')
            m2 = next(r for r in metric_rows if r['round']=='round2' and r['strategy']==s and r['scope']=='all_jobs')
            delta[f'{k}_round2_minus_round1'] = m2[k]-m1[k]
        delta_rows.append(delta)
    category_rows=[]
    for rnd in RUNS:
        for s in STRATEGIES:
            for cat in sorted({r['category'] for r in labels}):
                scored=[row for key,row in all_scores[rnd].items() if key[1]==s and by_id[key[0]]['category']==cat]
                category_rows.append(dict(round=rnd,strategy=s,category=cat,n=len(scored),
                    correct=sum(r['correct'] for r in scored),accuracy=sum(r['correct'] for r in scored)/len(scored)))
    write_csv('metrics_by_round_strategy.csv', metric_rows)
    write_csv('agreement_by_strategy.csv', agreements)
    write_csv('prediction_distribution.csv', distribution)
    write_csv('tokens_and_failures.csv', cost_rows)
    write_csv('round2_minus_round1.csv', delta_rows)
    write_csv('scored_predictions.csv', score_rows)
    write_csv('aligned_predictions.csv', aligned)
    write_csv('disagreements.csv', differences)
    write_csv('category_accuracy.csv', category_rows)
    write_json('confusion_matrices.json', matrices)
    plot(metric_rows, matrices)
    report(metric_rows, agreements, distribution, cost_rows, differences, category_rows, matrices)
    after = {p.relative_to(ROOT).as_posix(): sha(p) for p in protected}
    assert before == after, 'Protected source data changed during analysis'
    integrity.update(protected_files_before=before, protected_files_after=after,
                     protected_files_unchanged=True, analysis_module_bytes=(ROOT/'src/analysis.py').stat().st_size)
    integrity['checks']['all_passed'] = True
    write_json('integrity.json', integrity)
    artifacts = {p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'artifact_sha256.json'}
    write_json('artifact_sha256.json', artifacts)
    print(json.dumps(dict(output=str(OUT), all_integrity_checks_passed=True,
                         metrics=[r for r in metric_rows if r['strategy']!='ALL'], agreement=agreements,
                         disagreement_count=len(differences), costs=cost_rows), ensure_ascii=False, indent=2))

def plot(metric_rows, matrices):
    # Standard library SVG: portable preliminary figures, no plotting dependency.
    colors=['#3977b9','#dc7936']
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="470" viewBox="0 0 1000 470">',
         '<rect width="1000" height="470" fill="white"/>',
         '<g font-family="Arial,sans-serif" fill="#222">',
         '<text x="30" y="30" font-size="21">Group 13 Pilot: same test set / v0.1-dev DRAFT</text>',
         '<text x="30" y="56" font-size="14">All 200 jobs per strategy; failed requests count as errors</text>']
    for panel,metric in enumerate(['accuracy','macro_f1']):
        xbase=70+panel*490
        svg.append(f'<text x="{xbase}" y="90" font-size="18">{metric}</text>')
        for t in range(0,101,20):
            y=360-t*2.3
            svg.append(f'<line x1="{xbase}" y1="{y}" x2="{xbase+390}" y2="{y}" stroke="#ddd"/>')
            svg.append(f'<text x="{xbase-38}" y="{y+5}" font-size="12">{t}%</text>')
        for i,s in enumerate(STRATEGIES):
            for j,rnd in enumerate(RUNS):
                value=next(r[metric] for r in metric_rows if r['round']==rnd and r['strategy']==s and r['scope']=='all_jobs')
                x=xbase+15+i*130+j*48
                y=360-value*230
                svg.append(f'<rect x="{x}" y="{y}" width="42" height="{value*230}" fill="{colors[j]}"/>')
                svg.append(f'<text x="{x+21}" y="{y-8}" text-anchor="middle" font-size="12">{value:.1%}</text>')
            svg.append(f'<text x="{xbase+62+i*130}" y="386" text-anchor="middle" font-size="14">{DISPLAY[s]}</text>')
    for j,rnd in enumerate(RUNS):
        svg.append(f'<rect x="{330+j*160}" y="418" width="18" height="18" fill="{colors[j]}"/><text x="{356+j*160}" y="432" font-size="14">{rnd}</text>')
    svg.append('</g></svg>')
    (OUT/'accuracy_macro_f1.svg').write_text('\n'.join(svg),encoding='utf-8')

def report(metric_rows, agreements, distribution, costs, differences, category_rows, matrices):
    lines=['# Group 13：两轮 Qwen Plus Pilot 离线评估与重复性分析', '',
        '两轮使用**相同的 200 条测试指令（100 对、CLEAR/NEEDS_CLARIFICATION 各 100 条）**、相同的三种策略和 **v0.1-dev DRAFT Prompt**。这是 Pilot 重复性实验，不是最终冻结实验，也不是独立测试集复验。未依据测试结果修改 Prompt。', '',
        f"Round 1：`{RUNS['round1'].name}`；Round 2：`{RUNS['round2'].name}`。", '',
        '## 主要指标', '',
        '下表为每策略分别评估；Precision/Recall 使用两个金标准类别的宏平均，CSV 同时提供每类 Precision/Recall/F1。所有数值为比例，表中转为百分比。', '']
    rows=[]
    for m in metric_rows:
        if m['strategy']=='ALL': continue
        rows.append({'轮次':m['round'],'策略':DISPLAY[m['strategy']],'口径':m['scope'],'n':m['total_samples'],
                     'Accuracy':pct(m['accuracy']),'Macro-F1':pct(m['macro_f1']),
                     'Precision(macro)':pct(m['precision_macro']),'Recall(macro)':pct(m['recall_macro'])})
    lines += [table(rows,list(rows[0])), '',
        '使用 `src.evaluation_adapter.convert_predictions` 连接 `data/processed/test_labels.csv`，在每策略完整配对上调用 WANG 的 `score_samples`，再用 `calculate_metrics` 计算。valid_only 在完成评分后筛选，不会因一条失败丢弃其有效配对样本。独立计数公式交叉核验 Accuracy/Macro-F1。', '',
        'all_jobs：失败/无效输出计错，且只给真实类增加 FN，不虚构另一类 FP；Macro-F1 始终只平均两个真实类别。第一轮 600 条最终记录中 599 条有效，Few-shot 的一条历史 network_error 保留。valid_only：第一轮三策略分别为 200/199/200 条（合计 599），第二轮均为 200（合计 600）。', '',
        'CSV 还保留 WANG 的漏检率、不必要澄清率、完整配对正确率及有效率。第一轮 Few-shot valid_only 有一个不完整配对，其配对指标留空；ALL 为跨策略池化描述，配对指标也留空。', '',
        '## 两轮预测一致率', '']
    rows=[{'策略':DISPLAY[a['strategy']],'相同/双方有效':f"{a['same_valid_label']}/{a['both_valid']}",
           '一致率':pct(a['agreement_both_valid']),'双方有效覆盖率':pct(a['both_valid_coverage']),
           '全量相同有效标签率':pct(a['same_valid_label_rate_all']),'标签翻转':a['label_flips'],'有效性变化':a['validity_changes'],
           '对→错':a['correct_to_wrong'],'错→对':a['wrong_to_correct']} for a in agreements]
    lines += [table(rows,list(rows[0])), '',
        '主一致率 = 相同有效标签数 / 双方均有效记录数。另报全量相同有效标签数 / 全部逻辑请求，失败不作为标签一致。temperature=0 仍观察到的标签翻转属于此次实际 API 重复执行的输出差异；日志无法判定服务端随机性或后端具体原因。', '',
        '## 分类偏差', '']
    rows=[{'轮次':d['round'],'策略':DISPLAY[d['strategy']],'CLEAR':d['clear_count'],'NEEDS_CLARIFICATION':d['needs_clarification_count'],
           '失败':d['failures'],'预测需澄清/有效':pct(d['needs_clarification_share_valid']),
           '有效子集真实需澄清':pct(d['gold_needs_clarification_share_valid'])} for d in distribution if d['strategy']!='ALL']
    lines += [table(rows,list(rows[0])), '',
        '完整测试集真实类别为 50%/50%。Few-shot 两轮都明显偏向 NEEDS_CLARIFICATION：有效预测占比 72.86%/71.00%，真实歧义类 Recall 为 79%/76%，但 CLEAR Recall 仅 33%/34%，两轮均误报 66 条 CLEAR。它降低漏检的同时增加不必要澄清，因而 Accuracy 较高并未转化为最高 Macro-F1。第一轮有效子集的真实歧义比例为 50.25%，不足以解释此偏向。', '',
        'Zero-shot 两轮预测需澄清均为 50.50%；Checklist 为 50.50%/50.00%，类别占比接近平衡。不过占比平衡不代表识别准确：Zero-shot 每轮仍有 47 次误报、46 次漏检，Checklist 为 45/44 和 46/46。以上是本测试集的描述性偏向，不是对 Prompt 机制的因果解释。', '',
        '## 混淆矩阵', '',
        '行是真实标签，列是预测标签；顺序 CLEAR、NEEDS_CLARIFICATION。以下同时显式保留失败列；`confusion_matrices.json` 提供 all_jobs/valid_only 两种口径的原始计数和按真实行归一化值，valid_only 的左上 2×2 即通常的混淆矩阵。', '']
    for rnd in RUNS:
        for s in STRATEGIES:
            matrix=matrices[f'{rnd}/{s}/all_jobs']['counts']
            rows=[dict(zip(['真实标签',*PRED_COLUMNS],[label,*values])) for label,values in zip(LABELS,matrix)]
            lines += [f'### {rnd} / {DISPLAY[s]}', '',table(rows,list(rows[0])), '']
    lines += ['## Token、延迟和调用稳定性', '',
        'Token 从 `attempts.jsonl` 的 completed 事件逐次累加（包含重试），不是只加最终预测。未知 usage 单独计数，known Token 是已知小计，不能视为完整实际用量或费用。延迟来自每次 completed 事件的 latency_ms，均值/中位数/P95 包含失败，另给有效调用均值；不含重试退避、暂停及恢复等待，不能视为全流程耗时。P95 使用线性插值。', '']
    rows=[{'轮次':c['round'],'策略':DISPLAY[c['strategy']],'尝试':c['completed_attempts'],'额外尝试':c['extra_attempts'],
           '未知usage':c['unknown_usage_attempts'],'Known tokens':c['known_total_tokens'],
           '调用成功率':pct(c['attempt_success_rate']),'最终失败':c['final_failed_jobs'],
           '平均ms':f"{c['latency_attempt_mean_ms']:.1f}",'P95 ms':f"{c['latency_attempt_p95_ms']:.1f}"} for c in costs]
    lines += [table(rows,list(rows[0])), '',
        '第一轮最新 summary 的 calls_this_run=568 仅表示恢复后的那次执行；累计调用数应加上首次 45 次和一次 0 调用检查，共 613 次。第二轮累计 600 次。两轮 starts/completed 均一一匹配，无未结束的尝试。第一轮 14 次 network_error 的 usage 未知，最终留下 1 条失败；第二轮 0 次未知 usage、0 次最终失败。第一轮早期发生停止和恢复，最终 stopped_reason=null；第二轮最终也为 null，均 remaining_logical_requests=0。', '',
        '已知总 Token：第一轮 274119，第二轮 274669，差值 +550；第一轮的 14 次未知 usage 使此差值无法代表真实总成本变化。`tokens_and_failures.csv` 提供 prompt/completion/total 分项、重试逻辑请求数、timeout/network_error 数及各延迟统计；`round2_minus_round1.csv` 提供各策略指标和调用统计差值。', '',
        '## 差异样本与案例', '',
        f'共 {len(differences)} 条记录在标签或有效性上不同，涉及 {len({r["sample_id"] for r in differences})} 个 sample_id；完整原文、环境、类别、gold、状态、预测和消息哈希见 `disagreements.csv`。这些是模型标签变化的观测；输出仅有标签，无法从日志恢复模型内部推理，以下案例仅作现象描述。', '']
    counts=Counter((r['strategy'],r['category'],r['change_type']) for r in differences)
    rows=[{'策略':DISPLAY[s],'类别':cat,'变化':kind,'数量':n} for (s,cat,kind),n in sorted(counts.items())]
    lines += [table(rows,list(rows[0])), ''] if rows else []
    lines += ['13 次有效标签翻转中，Few-shot 为 5 次对→错、3 次错→对，净少 2 个正确；Checklist 为 4 次对→错、1 次错→对，净少 3 个正确。唯一有效性变化的样本 gold=CLEAR，第二轮预测 NEEDS_CLARIFICATION，仍然错误，因此恢复有效输出没有增加正确数。', '',
        'Few-shot 的 8 次翻转中 6 次来自 preferences（该类每策略 92 条），2 次来自 common_sense_knowledge（76 条）；Checklist 的 5 次翻转分布为 common_sense_knowledge 3 次、preferences 1 次、safety 1 次（32 条）。类别规模不同，不能仅据次数认定某类更不稳定。', '',
        '例如 Few-shot 的同一对洗盘子指令 s_f37b3f728c5bb57a0411（CLEAR）和 s_d0c79f390eed72de95ee（NEEDS_CLARIFICATION）第一轮都预测需澄清、第二轮都预测 CLEAR：两轮各正确一条，均未同时区分这对指令。配对正确率因此能补充单条 Accuracy。Checklist 对“Retrieve the spatula from the wall-mounted hook.”由误报变为正确，而对“Slice the croissant using the paring knife on the kitchen table.”由正确变为误报；不应将所有变化笼统归为性能改善。', '']
    for r in differences:
        lines += [f"- `{r['sample_id']}` / {DISPLAY[r['strategy']]} / {r['category']} / gold={r['gold_label']}：{r['round1_prediction']} → {r['round2_prediction']}（{r['correctness_transition']}）。指令：{r['instruction']}"]
    lines += ['', 'category_accuracy.csv 按数据现有 category 给出各轮各策略的正确数和分母，用于描述性核查；不把类别差异解释为已确认的错误原因。', '',
        '## 表现判断与局限', '']
    for rnd in RUNS:
        values=[m for m in metric_rows if m['round']==rnd and m['strategy']!='ALL' and m['scope']=='all_jobs']
        best=max(values,key=lambda m:(m['macro_f1'],m['accuracy']))
        lines.append(f"- {rnd}：按全量 Macro-F1、再参考 Accuracy，{DISPLAY[best['strategy']]} 最高（Accuracy {pct(best['accuracy'])}，Macro-F1 {pct(best['macro_f1'])}）。")
    lines += ['', '按 Accuracy，Few-shot 两轮分别为 56.00%/55.00%，均最高；按 Macro-F1，Checklist 两轮均最高。不存在同时在这两个指标上领先的策略。若重视两类的均衡识别，Checklist 是本次 Pilot 的较优结果；若只看 Accuracy，则 Few-shot 领先。Zero-shot 的重复性最好且已知 Token 最少。']
    lines += ['', '同一测试集的两轮结果是配对重复测量，不能当作 400 条独立测试指令；未做显著性或总体泛化优势断言。未生成独立样本假设下的置信区间。Prompt 仍为 DRAFT，结论限定于本次 Pilot，不使用测试集结果优化 Prompt。', '',
        '## WANG analysis.py 当前状态与待办', '',
        '`src/analysis.py` 当前为 **0 字节空文件**，没有任何函数或入口；README 约定 WANG 负责 evaluation、visualization、case analysis，但该模块尚未实现。`src/evaluate.py` 已实现严格解析、验证、Accuracy、Macro-F1、每类 F1、漏检/不必要澄清率、配对正确率及有效率；其 main 仍只运行四条模拟示例。', '',
        '待完成工作：', '',
        '- 在本人模块中实现真实结果读取/评估入口与结果导出，复用现有 evaluator 和 adapter。',
        '- 实现 Precision/Recall、混淆矩阵及归一化展示，明确失败/无效输出口径。',
        '- 实现按 sample_id/strategy 的两轮对齐、一致率、覆盖率及差异样本导出。',
        '- 实现策略类别分布、按类别错误分析、成对样本分析和人工案例解释。',
        '- 实现逐尝试 Token/延迟/重试/失败统计及未知 usage 处理。',
        '- 实现正式可视化：策略指标对比、混淆矩阵、类别分布、延迟/Token 对比与差异案例展示。',
        '- 补充分析模块的有效性校验、可复现运行说明和适当测试；若添加区间，按 100 个 pair 联合聚类重采样。', '',
        '本目录是独立的一次性分析交接材料，不修改或替代 WANG 模块。初步 SVG 可供参考，正式模块及图表仍由 WANG 完成。', '',
        '## 完整性与复现', '',
        '两轮均有 600 条唯一 logical_id 和唯一 (sample_id,strategy)，各策略 200；logical_id 按 [sample_id,strategy] 重算一致，跨轮 ID 相同是设计行为。sample/pair 与标签对应、预测顺序、策略/消息哈希、快照文件、模型配置及每条最终结果与最后一次 completed 事件均已核验。原始日志、源码、Prompt 和数据的分析前/后 SHA-256 见 `integrity.json`；所有受保护文件完全未变。', '',
        '复现：在仓库根目录运行以下命令，仅离线读取、写入本目录，脚本禁止 socket 网络连接。', '',
        f'```powershell\npython -B {OUT.relative_to(ROOT).as_posix()}/analyze_offline.py\n```', '',
        '主要交接文件：comparison.md、metrics_by_round_strategy.csv、agreement_by_strategy.csv、confusion_matrices.json、prediction_distribution.csv、tokens_and_failures.csv、round2_minus_round1.csv、disagreements.csv、aligned_predictions.csv、scored_predictions.csv、category_accuracy.csv、accuracy_macro_f1.svg、integrity.json、artifact_sha256.json。CSV 为 UTF-8 BOM，便于 Excel 中文查看。未调用真实 API，未修改 Prompt、原始数据、原始日志、evaluate.py 或 analysis.py，未 Commit/Push。', '']
    (OUT/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__ == '__main__':
    main()
