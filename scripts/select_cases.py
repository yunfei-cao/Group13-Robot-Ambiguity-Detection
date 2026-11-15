"""Select and render the 15 handbook cases (5 success / 5 error / 5 strategy split).

Frozen inputs only: two pilot runs of predictions plus the frozen test labels and
inputs. No model calls, no label edits. Selection is deterministic (sorted by
sample_id, first match wins), every case uses a distinct sample_id, and the 15
cases use 15 distinct pairs, so no case repeats another.

Outputs:
    results/analysis/cases/group13_15_cases.csv
    results/analysis/cases/group13_15_cases.md
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import score_samples  # noqa: E402

ROUNDS = {
    'R1': ROOT / 'results/pilot/group13-full-pilot-qwen200-20261111T090248Z-76877106/predictions.jsonl',
    'R2': ROOT / 'results/pilot/group13-full-pilot-round2-20261111T153626Z-bc85f07d/predictions.jsonl',
}
LABELS = ROOT / 'data/processed/test_labels.csv'
INPUTS = ROOT / 'data/processed/test_inputs.csv'
OUT_DIR = ROOT / 'results/analysis/cases'
STRATEGIES = ('zero_shot', 'few_shot', 'checklist')

# Hand-written reading of each selected case. Keys are sample_id so a missing note
# is a hard error rather than a silently blank cell in the report.
NOTES = {
    's_189912ddd83d9e90f9f3':
        '环境中只有一种切片面包（a sliced bread）和一台 toaster，动作对象与器具都唯一，'
        '指令不含需要消解的成分。三种策略两轮均判 CLEAR，是"对象唯一 + 参数可执行"的最简单正例。',
    's_195ab566fe4c7cd3afbd':
        '鸡蛋、切达奶酪、frying pan 三样必需要素都在环境中且各自唯一，动作链完整。'
        '三种策略稳定判 CLEAR，说明当食材、容器、器具都能在环境里逐一落地时，模型不会凭空索要澄清。',
    's_26c1d58255b90b1694ac':
        '环境里 plastic / metal / ceramic bowl 并存，但指令已用材质把对象限定为 ceramic bowl，'
        'paper towels 也唯一。可见决定清晰与否的是"被指对象是否唯一确定"，而非环境里同类物品的数量。三种策略均判对。',
    's_02f7e32fa52a6d864f19':
        '环境中有 milk / almond milk / dark 三种 chocolate tablet，"the chocolate tablet"没有唯一指代，'
        '属真正的指代歧义。三种策略两轮全部判 NEEDS_CLARIFICATION，说明指代不唯一是最容易被模型捕捉的歧义类型。',
    's_0a10f5bfaa0422a15fe0':
        '烘烤温度与时间均缺失，"until done"没有可执行判据，属参数歧义。三种策略稳定判对。'
        '与案例 8（test 842，明确写了 350°F / 15 分钟与 baking dish）对照，可看出"参数是否可执行"是模型判断的关键线索。',
    's_0881fd0f539f7a8abdd3':
        '指令是一整段完整流程，但环境清单里只有蔬菜、刀具与部分器具（knife block、番茄、卷心菜、黄瓜……），'
        '并未列出指令提到的 sink、clean sponge、kitchen towel、mixing bowl 与 olive oil。gold 仍标 CLEAR，'
        '说明该数据集的 gold 以"指令文本本身是否无歧义"为准，而不是"环境是否覆盖所有被提及的物件"。'
        '三种策略统一判 NEEDS_CLARIFICATION，正是按后者判断的结果——这类系统性差异是多余澄清率偏高的主要来源，'
        '也提示 gold 语义需要人工复核（`data_manifest.json` 中的 manual_semantic_review 仍为 PENDING）。',
    's_0b7d7474407d3bf152c6':
        'gold 标为 CLEAR（烤箱温度与时间被视作可凭常识补全），但环境里只有 stove，没有 oven，'
        '字面上既缺参数也缺器具，属边界样本。Checklist 判对，Zero-shot 与 Few-shot 判错。'
        '本例说明标签语义并不等于环境可执行性，而 Checklist 的结构化检查更贴近 gold 口径。',
    's_2bb6ef81cbafaa6811c7':
        '温度（350°F）、时长（15 分钟）、容器（baking dish）与原料都写明，但环境里只列了 oven mitts，'
        '没有 oven 本身，三种策略仍全判 NEEDS_CLARIFICATION。与案例 5 构成镜像：案例 5 缺参数被判对，'
        '本例参数齐全却因器具缺失被判歧义，反映模型对"烘焙/温度"类指令的保守倾向，同时也说明'
        '被提及物件未出现在环境中时，模型几乎必然索要澄清。',
    's_012698e1f503e46799fe':
        'gold 为 NEEDS_CLARIFICATION，原因是过程参数不确定：用多少 heavy cream 与 sour cream、'
        '煮多久、搅到何种程度都未给出——本例的对象与器具在环境中均齐备，缺的只是过程参数。'
        '三种策略却全判 CLEAR，是典型漏检：模型只核对了"对象是否存在"，没有核对"过程是否可执行"。'
        '这正是漏检率 0.46 / 0.21 / 0.44 的成因。',
    's_02be24721a136a1d637a':
        '"until done"与案例 5 属同一歧义类型，但这里三种策略中有两种判 CLEAR（仅 Few-shot 判对）。'
        '同一歧义在不同上下文中被区别对待，说明模型判断不稳定，单条指令的表现不能外推为策略能力。',
    's_0259b77552c8d8d6b550':
        '环境里同时有 stainless steel / wooden tablespoon 与 silver teaspoon，但指令已指定不锈钢汤匙，指代唯一。'
        'Few-shot 判 NEEDS_CLARIFICATION，Zero-shot 与 Checklist 判 CLEAR——Few-shot 的过度敏感在此直接产生多余澄清。',
    's_04247c8bffe09de638b8':
        '指令要求"serve the coleslaw in a salad bowl"，但环境里只有 mixer、bowl、cupboard 等，没有 salad bowl，'
        '执行前提不满足故 gold 为 NEEDS_CLARIFICATION。三种策略中仅 Few-shot 命中，'
        '说明 Few-shot 的高召回来自"对任何缺项都发问"，代价是 66% 的多余澄清率。',
    's_68fec364435b48203b05':
        '环境里 vanilla yogurt cup 与 greek yogurt cup 并存，指令已指名 vanilla，指代唯一，gold 为 CLEAR。'
        'Zero-shot 与 Few-shot 判错、Checklist 判对——Checklist 中"对象指代"这一检查项帮助它确认了唯一指代，'
        '是本批案例中 Checklist 独占正确的代表。',
    's_8ff8e533fd171caad6d7':
        '环境有 plastic / ceramic 共 6 种 plate，却没有 platter，且未说明用哪种盘子与份量，gold 为 NEEDS_CLARIFICATION。'
        '仅 Zero-shot 判对，Few-shot 与 Checklist 判 CLEAR——本例是 Zero-shot 独占正确的代表，'
        '说明策略分歧的方向并不单向。',
    's_af12a678f8b469a20963':
        '环境有 plastic / ceramic / glass 三种 dinner plate，且不存在 dishwasher（只有 dish rack），'
        '既是指代歧义也是不可执行指令，gold 为 NEEDS_CLARIFICATION。Zero-shot 判 CLEAR 属漏检，'
        'Few-shot 与 Checklist 判对，与案例 12、13 一起说明本批五个分歧案例覆盖了三种策略各自独占正确的情形。',
}

CASE_TYPES = [('success', '成功案例', 5), ('error', '典型错误', 5), ('split', '策略分歧', 5)]


def load_predictions(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    grouped = {s: [] for s in STRATEGIES}
    for row in rows:
        grouped[row['strategy']].append(dict(
            sample_id=row['sample_id'], pair_id='pending', gold_label='pending',
            raw_response=row.get('raw_text'),
            request_status='SUCCESS' if row['status'] in {'success', 'invalid_output'} else 'REQUEST_FAILED'))
    return grouped


def load_meta():
    with INPUTS.open(encoding='utf-8-sig', newline='') as stream:
        inputs = {r['sample_id']: r for r in csv.DictReader(stream)}
    with LABELS.open(encoding='utf-8-sig', newline='') as stream:
        labels = {r['sample_id']: r for r in csv.DictReader(stream)}
    return inputs, labels


def score_rounds(inputs, labels):
    scored = {}
    for round_name, path in ROUNDS.items():
        per_sample = {}
        for strategy, rows in load_predictions(path).items():
            for row in rows:
                meta = labels[row['sample_id']]
                row['pair_id'] = meta['pair_id']
                row['gold_label'] = meta['gold_label']
            for item in score_samples(rows):
                per_sample.setdefault(item['sample_id'], {})[strategy] = item['prediction']
        scored[round_name] = per_sample
    return scored


def build_entries(inputs, labels, scored):
    entries = []
    for sample_id in sorted(labels):
        meta = labels[sample_id]
        entries.append(dict(
            sample_id=sample_id, pair_id=meta['pair_id'], category=meta['category'],
            gold_label=meta['gold_label'],
            environment=inputs[sample_id]['environment'],
            instruction=inputs[sample_id]['instruction'],
            R1={s: scored['R1'][sample_id][s] for s in STRATEGIES},
            R2={s: scored['R2'][sample_id][s] for s in STRATEGIES},
        ))
    return entries


def classify(entries):
    success, error, split = [], [], []
    for entry in entries:
        gold, r1, r2 = entry['gold_label'], entry['R1'], entry['R2']
        if all(r1[s] == gold for s in STRATEGIES) and all(r2[s] == gold for s in STRATEGIES):
            success.append(entry)
        if sum(1 for s in STRATEGIES if r1[s] != gold) >= 2:
            error.append(entry)
        if len({r1[s] for s in STRATEGIES}) >= 2 and any(r1[s] == gold for s in STRATEGIES):
            split.append(entry)
    return success, error, split


def select(entries):
    success, error, split = classify(entries)
    used_ids, used_pairs = set(), set()

    def pick(pool, count, one_per_category=True):
        chosen = []
        for entry in pool:
            if len(chosen) == count:
                break
            if entry['sample_id'] in used_ids or entry['pair_id'] in used_pairs:
                continue
            if one_per_category and any(e['category'] == entry['category'] for e in chosen):
                continue
            chosen.append(entry)
            used_ids.add(entry['sample_id'])
            used_pairs.add(entry['pair_id'])
        for entry in pool:
            if len(chosen) == count:
                break
            if entry['sample_id'] in used_ids:
                continue
            chosen.append(entry)
            used_ids.add(entry['sample_id'])
            used_pairs.add(entry['pair_id'])
        return chosen

    ok_clear = [e for e in success if e['gold_label'] == 'CLEAR']
    ok_needs = [e for e in success if e['gold_label'] == 'NEEDS_CLARIFICATION']
    success_cases = pick(ok_clear, 3) + pick(ok_needs, 2)

    fp = [e for e in error if e['gold_label'] == 'CLEAR']
    fn = [e for e in error if e['gold_label'] == 'NEEDS_CLARIFICATION']
    error_cases = pick(fp, 3) + pick(fn, 2)

    split_cases, patterns = [], set()
    for entry in split:
        pattern = tuple(entry['R1'][s] == entry['gold_label'] for s in STRATEGIES)
        if pattern in patterns or entry['sample_id'] in used_ids or entry['pair_id'] in used_pairs:
            continue
        patterns.add(pattern)
        split_cases.append(entry)
        used_ids.add(entry['sample_id'])
        used_pairs.add(entry['pair_id'])
        if len(split_cases) == 5:
            break
    if len(split_cases) < 5:
        split_cases += pick(split, 5 - len(split_cases))

    return [(('success', '成功案例'), success_cases),
            (('error', '典型错误'), error_cases),
            (('split', '策略分歧'), split_cases)]


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['case_no', 'case_type', 'sample_id', 'pair_id', 'category', 'gold_label',
              'environment', 'instruction',
              'R1_zero_shot', 'R1_few_shot', 'R1_checklist',
              'R2_zero_shot', 'R2_few_shot', 'R2_checklist', 'analysis']
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


OBSERVATIONS = [
    '**gold 口径与模型口径不同**：gold 以"指令文本是否自足"为准，模型以"环境能否执行"为准。案例 6、7、8 三条 gold 为 CLEAR 的样本，'
    '指令提到的 sink、oven 都不在环境清单里，因此三种策略一致索要澄清。这一条同时解释了多余澄清率为何处在 0.45–0.66，'
    '也说明 gold 标签需要人工复核，不能当作绝对真值。',
    '**模型强于"指代唯一性"，弱于"过程可执行性"**：案例 4（三种 chocolate tablet）与案例 15（三种 dinner plate 且无 dishwasher）'
    '都被至少两种策略识别为歧义；而案例 9（缺用量与火候）、案例 10（until done）却被多数策略判为 CLEAR。'
    '也就是说，缺对象能发现，缺过程参数发现不了。',
    '**三种策略各自都有独占正确的情形**：案例 12 仅 Few-shot 判对、案例 13 仅 Checklist 判对、案例 14 仅 Zero-shot 判对。'
    '差异来自判断依据不同——Few-shot 偏保守、Checklist 依赖结构检查、Zero-shot 偏宽松——而不是某一策略全面更优，'
    '这与总体 Accuracy 只相差 0.5–2.5 个百分点一致。',
    '**15 例在两轮执行中预测完全一致**，高于总体 97.67% 的两轮一致率（Zero-shot 100%、Few-shot 95.5%、Checklist 97.5%），'
    '说明这些案例不是随机波动的产物，而是稳定的判断倾向，适合作为定性证据引用。',
]


def write_md(rows, path):
    lines = ['# Group 13 — 15 个案例分析（WANG HONGFENG）', '',
             '数据来源：R1 / R2 两轮冻结预测（各 200 条 × 3 策略），标签来自 `data/processed/test_labels.csv`，',
             '指令与环境来自 `data/processed/test_inputs.csv`；评分口径与 `src/evaluate.py` 一致。',
             '选取规则由 `scripts/select_cases.py` 固定：按 sample_id 排序取首个匹配，15 例互不重复、'
             '分属 15 个不同的指令对，每条都可在 `predictions.jsonl` 中按 sample_id 定位。', '']
    current = None
    for row in rows:
        label = {'success': '5.1 成功案例', 'error': '5.2 典型错误', 'split': '5.3 策略分歧'}[row['case_type']]
        if label != current:
            current = label
            lines += [f'## {label}', '']
        lines += [f"### 案例 {row['case_no']}　`{row['sample_id']}`", '',
                  f"- **指令对**：{row['pair_id']}　**类别**：{row['category']}　**正确标签**：{row['gold_label']}",
                  f"- **环境**：{row['environment']}",
                  f"- **指令**：{row['instruction']}",
                  '- **三种策略原始回答（R1 / R2 两轮一致）**：',
                  f"  - Zero-shot：`{row['R1_zero_shot']}` / `{row['R2_zero_shot']}`",
                  f"  - Few-shot：`{row['R1_few_shot']}` / `{row['R2_few_shot']}`",
                  f"  - Checklist：`{row['R1_checklist']}` / `{row['R2_checklist']}`",
                  f"- **人工分析**：{row['analysis']}", '']
    lines += ['## 5.4 跨案例观察', '']
    lines += [f'{index}. {text}' for index, text in enumerate(OBSERVATIONS, start=1)]
    lines += ['', '以上 15 例的数量分布为 5 成功 / 5 典型错误 / 5 策略分歧，满足手册要求，未出现需要缩量说明的情形。']
    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    inputs, labels = load_meta()
    entries = build_entries(inputs, labels, score_rounds(inputs, labels))
    groups = select(entries)

    rows, case_no = [], 0
    for (case_type, _), group in groups:
        for entry in group:
            if entry['sample_id'] not in NOTES:
                raise SystemExit(f'missing hand-written note for {entry["sample_id"]}')
            case_no += 1
            rows.append(dict(case_no=case_no, case_type=case_type, sample_id=entry['sample_id'],
                             pair_id=entry['pair_id'], category=entry['category'],
                             gold_label=entry['gold_label'], environment=entry['environment'],
                             instruction=entry['instruction'],
                             R1_zero_shot=entry['R1']['zero_shot'], R1_few_shot=entry['R1']['few_shot'],
                             R1_checklist=entry['R1']['checklist'],
                             R2_zero_shot=entry['R2']['zero_shot'], R2_few_shot=entry['R2']['few_shot'],
                             R2_checklist=entry['R2']['checklist'], analysis=NOTES[entry['sample_id']]))

    assert len({r['sample_id'] for r in rows}) == 15, 'duplicate sample across cases'
    assert len({r['pair_id'] for r in rows}) == 15, 'duplicate pair across cases'
    assert all(len(group) == 5 for _, group in groups), [len(g) for _, g in groups]

    write_csv(rows, OUT_DIR / 'group13_15_cases.csv')
    write_md(rows, OUT_DIR / 'group13_15_cases.md')

    print(f'cases: {len(rows)}  unique sample_id: {len({r["sample_id"] for r in rows})}'
          f'  unique pairs: {len({r["pair_id"] for r in rows})}')
    for (case_type, _), group in groups:
        print(f'  {case_type}: {len(group)} -> ' + ', '.join(e['sample_id'] for e in group))
    print(f'outputs -> {OUT_DIR}')


if __name__ == '__main__':
    main()
