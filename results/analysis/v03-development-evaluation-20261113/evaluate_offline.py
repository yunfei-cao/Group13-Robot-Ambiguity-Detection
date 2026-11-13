"""Offline-only v0.3 development evaluation; exclusive-create outputs, no API calls."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from src.evaluation_adapter import convert_predictions
from src.evaluate import score_samples, calculate_metrics

OUT = Path(__file__).resolve().parent
EXP = ROOT / 'results/prompt_comparison/group13-dev-v01-v03-20261113T150521Z-b2a4a074'
CAND = EXP / 'candidate'
RUN = CAND / 'pilot/group13-dev-v01-v03-20261113T150521Z-b2a4a074-candidate'
PRED = RUN / 'predictions.jsonl'
LABEL = ROOT / 'prompt_v0_3/development_comparison_85pairs/dev_labels.csv'
INPUT = CAND / 'snapshot/dev_inputs.csv'
HISTORY = ROOT / 'results/dev/group13-dev-qwen20-20261111T060727Z-d9300025/predictions.jsonl'
NAMES = {'zero_shot': 'Zero-shot', 'few_shot': 'Few-shot', 'checklist': 'Checklist'}
SUBSETS = {'all_85_pairs': '全部85对', 'historical_9_pairs': '历史使用9对',
           'previously_unused_76_pairs': '此前未运行76对'}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, strict=True))


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(name, rows):
    with (OUT / name).open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def by_id(rows):
    result = {r['sample_id']: r for r in rows}
    assert len(result) == len(rows), 'Duplicate source IDs'
    return result


def main():
    protected = [PRED, LABEL, INPUT, HISTORY, EXP / 'development_overlap.json',
                 EXP / 'development_pairs.csv', CAND / 'plan.json', RUN / 'run_config.json',
                 ROOT / 'data/processed/dev_inputs.csv', ROOT / 'data/processed/dev_labels.csv',
                 ROOT / 'prompt_v0_3/development_comparison_85pairs/dev_inputs.csv',
                 ROOT / 'src/evaluate.py', ROOT / 'src/evaluation_adapter.py']
    protected += [p for p in (CAND / 'snapshot/verified_lf').rglob('*') if p.is_file()]
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    plan, config = read_json(CAND / 'plan.json'), read_json(RUN / 'run_config.json')
    assert plan['dataset_split'] == 'dev' and config['prompt_version'] == 'v0.3-candidate'
    assert config['mode'] == 'qwen' and config['logical_requests'] == 510
    labels, inputs = by_id(read_csv(LABEL)), by_id(read_csv(INPUT))
    assert len(labels) == len(inputs) == 170 and labels.keys() == inputs.keys()
    original_labels = by_id(read_csv(ROOT / 'data/processed/dev_labels.csv'))
    original_inputs = by_id(read_csv(ROOT / 'data/processed/dev_inputs.csv'))
    bundle_inputs = by_id(read_csv(ROOT / 'prompt_v0_3/development_comparison_85pairs/dev_inputs.csv'))
    assert bundle_inputs == inputs
    for sid in labels:
        assert labels[sid] == original_labels[sid], f'GT changed: {sid}'
        assert inputs[sid] == original_inputs[sid], f'Instruction changed: {sid}'
        assert labels[sid]['pair_id'] == inputs[sid]['pair_id']
    overlap = read_json(EXP / 'development_overlap.json')
    pairs = {r['pair_id']: r for r in overlap['pairs']}
    assert len(pairs) == len(overlap['pairs']) == 85
    hist_ids = {r['sample_id'] for r in jsonl(HISTORY)}
    historical = {pid for pid, r in pairs.items() if r['previously_run']}
    unused = set(pairs) - historical
    assert len(historical) == 9 and len(unused) == 76
    pair_csv = {r['pair_id']: r for r in read_csv(EXP / 'development_pairs.csv')}
    assert pair_csv.keys() == pairs.keys()
    for pid, meta in pairs.items():
        ids = {sid for sid, r in labels.items() if r['pair_id'] == pid}
        assert ids == set(meta['sample_ids']) and len(ids) == 2
        assert (ids <= hist_ids) == meta['previously_run']
        assert not (ids & hist_ids) if not meta['previously_run'] else ids <= hist_ids
        assert (pair_csv[pid]['previously_run'] == 'true') == meta['previously_run']
        assert set(pair_csv[pid]['sample_ids'].split(';')) == ids
    raw = jsonl(PRED)
    assert len(raw) == 510
    assert len({(r['sample_id'], r['strategy']) for r in raw}) == 510
    for r in raw:
        assert r['run_id'] == config['run_id'] and r['model_id'] == config['model']
        assert r['status'] == 'success' and r['is_valid'] is True
        assert r['raw_text'].strip() == r['label']
    groups = convert_predictions(PRED, LABEL)
    metrics, scored_all, pair_rows, audits = [], [], [], []
    subset_pairs = {'all_85_pairs': set(pairs), 'historical_9_pairs': historical,
                    'previously_unused_76_pairs': unused}
    for strategy, rows in groups.items():
        counts = Counter(r['sample_id'] for r in rows)
        assert len(rows) == 170 and set(counts) == set(labels) and all(n == 1 for n in counts.values())
        scored = score_samples(rows)
        assert all(r['valid_output'] for r in scored)
        audits.append(dict(strategy=strategy, predictions=len(rows), unique_sample_ids=len(counts),
                           duplicates=0, missing_ids=0, extra_ids=0, valid_outputs=len(scored), pairs=85))
        for r in scored:
            scored_all.append(dict(strategy=strategy, subset='historical_9_pairs' if r['pair_id'] in historical
                                   else 'previously_unused_76_pairs', **r,
                                   environment=inputs[r['sample_id']]['environment'],
                                   instruction=inputs[r['sample_id']]['instruction']))
        for subset, pids in subset_pairs.items():
            selected = [r for r in scored if r['pair_id'] in pids]
            m = calculate_metrics(selected)
            tn = sum(r['gold_label'] == 'CLEAR' and r['prediction'] == 'CLEAR' for r in selected)
            fp = sum(r['gold_label'] == 'CLEAR' and r['prediction'] == 'NEEDS_CLARIFICATION' for r in selected)
            fn = sum(r['gold_label'] == 'NEEDS_CLARIFICATION' and r['prediction'] == 'CLEAR' for r in selected)
            tp = sum(r['gold_label'] == 'NEEDS_CLARIFICATION' and r['prediction'] == 'NEEDS_CLARIFICATION' for r in selected)
            assert tn + fp == fn + tp == len(pids)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            pc = tn / (tn + fn) if tn + fn else 0.0
            rc = tn / (tn + fp) if tn + fp else 0.0
            # Independent arithmetic cross-check of WANG's sample metrics.
            assert abs(m['accuracy'] - (tp + tn) / len(selected)) < 1e-12
            f1n = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            f1c = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
            assert abs(m['macro_f1'] - (f1n + f1c) / 2) < 1e-12
            grouped = defaultdict(list)
            for r in selected:
                grouped[r['pair_id']].append(r)
            correct_pairs = sum(all(r['correct'] for r in pr) for pr in grouped.values())
            assert abs(m['pair_level_correctness'] - correct_pairs / len(pids)) < 1e-12
            metrics.append(dict(subset=subset, strategy=strategy, samples=len(selected), pairs=len(pids),
                                correct=sum(r['correct'] for r in selected), accuracy=m['accuracy'],
                                macro_f1=m['macro_f1'], precision=precision, recall=recall,
                                precision_clear=pc, recall_clear=rc, macro_precision=(precision + pc) / 2,
                                macro_recall=(recall + rc) / 2, f1_clear=m['f1_clear'],
                                f1_needs_clarification=m['f1_needs_clarification'],
                                tn=tn, fp=fp, fn=fn, tp=tp, correct_pairs=correct_pairs,
                                pair_accuracy=m['pair_level_correctness'], valid_output_rate=m['valid_output_rate']))
        for pid in sorted(pairs):
            pr = [r for r in scored if r['pair_id'] == pid]
            clear = next(r for r in pr if r['gold_label'] == 'CLEAR')
            needs = next(r for r in pr if r['gold_label'] == 'NEEDS_CLARIFICATION')
            pair_rows.append(dict(strategy=strategy, pair_id=pid, previously_run=pid in historical,
                                  clear_sample_id=clear['sample_id'], clear_prediction=clear['prediction'],
                                  clear_correct=clear['correct'], needs_sample_id=needs['sample_id'],
                                  needs_prediction=needs['prediction'], needs_correct=needs['correct'],
                                  pair_correct=all(r['correct'] for r in pr)))
    after = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    assert before == after, 'Protected source changed during evaluation'
    write_csv('metrics.csv', metrics)
    write_csv('scored_predictions.csv', scored_all)
    write_csv('pair_results.csv', pair_rows)
    write_csv('coverage_validation.csv', audits)
    audit = dict(created_utc=datetime.now(timezone.utc).isoformat(), run_id=config['run_id'],
                 model=config['model'], prompt_version=config['prompt_version'], dataset_split='dev',
                 independent_blind_test=False, api_calls=0, source_hashes_unchanged=True,
                 source_sha256=before, coverage=audits, historical_pair_ids=sorted(historical),
                 previously_unused_pair_ids=sorted(unused),
                 evaluation_functions=['src.evaluation_adapter.convert_predictions',
                                       'src.evaluate.score_samples', 'src.evaluate.calculate_metrics'])
    with (OUT / 'audit.json').open('x', encoding='utf-8') as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    report = ['# v0.3-candidate 开发集评估', '',
              '**本次是开发集评估，不是新的独立盲测。** 76 对此前未运行样本仍来自开发集，不能称为独立测试集。', '',
              f"- Run: `{config['run_id']}`", f"- 模型: `{config['model']}`；Prompt: `{config['prompt_version']}`。",
              '- 只读取 candidate 的510条真实预测；不依赖、不等待 v0.1 实验。',
              '- 每种 Prompt 均严格按 sample_id 匹配170条指令和 Ground Truth，覆盖85对，每对包含两个不同标签。',
              '- 三种策略均为170条唯一预测，重复0、漏配0、额外ID 0，有效输出170/170；总计510/510有效。',
              '- 标签与 data/processed/dev_labels.csv 对应行完全一致；推理输入快照与开发集原始对应行及v0.3输入文件完全一致。',
              '- 9/76分组来自 development_overlap.json，并与 development_pairs.csv、历史真实预测ID交集交叉核验。', '',
              '## 计算口径', '',
              '复用 WANG 的 evaluation_adapter.convert_predictions、evaluate.score_samples 和 calculate_metrics。',
              '仅去除响应首尾空白，严格接受 CLEAR 或 NEEDS_CLARIFICATION；不修改预测或真实标签。',
              'Precision、Recall 默认以 NEEDS_CLARIFICATION 为正类；CSV另含 CLEAR 类及宏平均 Precision/Recall。',
              'Macro-F1 为两个类别F1的算术平均；Pair Accuracy 要求一对中两条预测均正确。',
              '混淆矩阵行是真实标签、列是预测标签，顺序均为 [CLEAR, NEEDS_CLARIFICATION]；矩阵为 [[TN, FP], [FN, TP]]。',
              'Accuracy、Macro-F1、Pair Accuracy 已用独立计数公式复核。CSV指标为0–1原始数值。', '']
    for subset, title in SUBSETS.items():
        report += [f'## {title}', '',
                   '| Prompt | 正确/样本 | Accuracy | Macro-F1 | Precision | Recall | 正确对/总对 | Pair Accuracy | 混淆矩阵 |',
                   '|---|---:|---:|---:|---:|---:|---:|---:|---|']
        for m in metrics:
            if m['subset'] != subset:
                continue
            report.append(f"| {NAMES[m['strategy']]} | {m['correct']}/{m['samples']} | {m['accuracy']:.4%} | {m['macro_f1']:.4%} | {m['precision']:.4%} | {m['recall']:.4%} | {m['correct_pairs']}/{m['pairs']} | {m['pair_accuracy']:.4%} | [[{m['tn']}, {m['fp']}], [{m['fn']}, {m['tp']}]] |")
        report.append('')
    full = [m for m in metrics if m['subset'] == 'all_85_pairs']
    best = max(full, key=lambda m: (m['accuracy'], m['macro_f1']))
    report += ['## 结论与产物', '',
               f"全85对上，{NAMES[best['strategy']]} 表现最好：Accuracy {best['accuracy']:.4%}，Macro-F1 {best['macro_f1']:.4%}。该结论限于本次开发集结果。", '',
               '- metrics.csv：3个分组 × 3种策略的完整指标和混淆矩阵计数。',
               '- scored_predictions.csv：510条逐样本匹配、原始响应、预测、GT、环境与指令。',
               '- pair_results.csv：85对 × 3种策略的逐对完整结果，包含历史使用标记。',
               '- coverage_validation.csv：每种Prompt的重复、漏配及有效性检查。',
               '- audit.json：输入来源、SHA-256、分组清单与评估函数。',
               '- evaluate_offline.py：离线复现脚本；使用排他创建，拒绝覆盖已有输出。', '',
               '本次未调用Qwen API，未修改预测、Ground Truth或CAI Prompt，未覆盖历史实验结果，未Commit、未Push。',
               '受保护源文件在评估前后的SHA-256完全一致。']
    with (OUT / 'report.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(report) + '\n')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
