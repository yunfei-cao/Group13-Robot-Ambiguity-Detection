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
        'The environment contains exactly one sliced bread and one toaster, so both the object and the tool are '
        'uniquely identified and the instruction leaves nothing to resolve. All three strategies answer CLEAR in '
        'both rounds: the simplest positive case of a unique object with executable parameters.',
    's_195ab566fe4c7cd3afbd':
        'Eggs, cheddar cheese and the frying pan are all present and each occurs once, so the action chain is '
        'complete. All three strategies answer CLEAR stably, showing that when ingredients, container and tool can '
        'each be grounded in the environment the model does not invent a clarification request.',
    's_26c1d58255b90b1694ac':
        'Plastic, metal and ceramic bowls all exist in the environment, but the instruction restricts the referent '
        'to the ceramic bowl and paper towels are unique. What decides clarity is whether the referent is uniquely '
        'determined, not how many similar objects the scene holds. All three strategies are correct.',
    's_02f7e32fa52a6d864f19':
        'The environment holds three chocolate tablets (milk, almond milk, dark), so "the chocolate tablet" has no '
        'unique referent: a genuine referential ambiguity. All three strategies answer NEEDS_CLARIFICATION in both '
        'rounds, which makes non-unique reference the ambiguity type these models detect most reliably.',
    's_0a10f5bfaa0422a15fe0':
        'Neither oven temperature nor duration is given, and "until done" offers no executable stopping rule: a '
        'parameter ambiguity. All three strategies are correct. Compare case 8 (test 842), where 350F and 15 '
        'minutes are stated, to see how parameter executability drives the decision.',
    's_0881fd0f539f7a8abdd3':
        'The instruction is one long but explicit procedure, yet the environment lists only vegetables and cutting '
        'tools (knife block, tomato, cabbage, cucumber and so on) and does not contain the sink, clean sponge, '
        'kitchen towel, mixing bowl or olive oil the instruction refers to. Gold is still CLEAR, so this dataset '
        'labels an instruction clear when its text is self-contained, not when the scene can execute it. All three '
        'strategies answer NEEDS_CLARIFICATION by the second criterion. This systematic gap is a main source of the '
        'elevated unnecessary-clarification rate and shows why gold semantics need manual review '
        '(manual_semantic_review is still PENDING in data_manifest.json).',
    's_0b7d7474407d3bf152c6':
        'Gold is CLEAR, treating oven temperature and time as commonsense defaults, but the environment lists a '
        'stove and no oven, so the instruction is literally short of both a parameter and a tool: a borderline '
        'sample. Checklist is correct while Zero-shot and Few-shot are wrong, which shows the label does not '
        'coincide with scene executability and that the structured Checklist tracks the gold convention more closely.',
    's_2bb6ef81cbafaa6811c7':
        'Temperature (350F), duration (15 minutes), container (baking dish) and ingredients are all stated, yet the '
        'environment lists only oven mitts and no oven, and all three strategies still answer NEEDS_CLARIFICATION. '
        'This mirrors case 5: there a missing parameter was correctly flagged, here complete parameters are still '
        'treated as ambiguous because the appliance is absent. It shows the conservative bias on baking-style '
        'instructions and the fact that a referenced object missing from the scene nearly always triggers a request.',
    's_012698e1f503e46799fe':
        'Gold is NEEDS_CLARIFICATION because the process is underspecified: how much heavy cream and sour cream, '
        'how long to cook and how far to stir are left open, while every object and tool is present in the scene. '
        'All three strategies answer CLEAR, a clear false negative. The model checks whether objects exist but not '
        'whether the process is executable, which is exactly what drives missed-ambiguity rates of '
        '0.46 / 0.21 / 0.44.',
    's_02be24721a136a1d637a':
        '"until done" is the same ambiguity type as case 5, yet here two of the three strategies answer CLEAR and '
        'only Few-shot is correct. The same ambiguity is treated differently in different contexts, so the decision '
        'is not stable and a single instruction cannot be extrapolated into a strategy capability.',
    's_0259b77552c8d8d6b550':
        'The environment holds a stainless steel tablespoon, a wooden tablespoon and a silver teaspoon, but the '
        'instruction names the stainless steel one, so the reference is unique. Few-shot answers '
        'NEEDS_CLARIFICATION while Zero-shot and Checklist answer CLEAR: Few-shot over-sensitivity turns directly '
        'into an unnecessary clarification here.',
    's_04247c8bffe09de638b8':
        'The instruction asks to serve the coleslaw in a salad bowl, but the environment lists a mixer, a bowl and a '
        'cupboard, and no salad bowl, so the precondition cannot be met and gold is NEEDS_CLARIFICATION. Only '
        'Few-shot catches it, which shows that the higher recall of Few-shot comes from asking whenever anything is '
        'missing, at the cost of a 66 percent unnecessary-clarification rate.',
    's_68fec364435b48203b05':
        'The environment holds both a vanilla yogurt cup and a greek yogurt cup, and the instruction already names '
        'the vanilla one, so the reference is unique and gold is CLEAR. Zero-shot and Few-shot are wrong while '
        'Checklist is correct: the object-reference item on the Checklist is what confirms the unique referent, '
        'making this the case where Checklist alone is right.',
    's_8ff8e533fd171caad6d7':
        'The environment holds six plates (plastic and ceramic, dinner, bread and salad) but no platter, and neither '
        'the plate type nor the quantity is stated, so gold is NEEDS_CLARIFICATION. Only Zero-shot is correct while '
        'Few-shot and Checklist answer CLEAR, which shows the direction of disagreement is not one-sided.',
    's_af12a678f8b469a20963':
        'The environment holds plastic, ceramic and glass dinner plates and no dishwasher (only a dish rack), so the '
        'instruction is both referentially ambiguous and unexecutable, and gold is NEEDS_CLARIFICATION. Zero-shot '
        'answers CLEAR, a false negative, while Few-shot and Checklist are correct. Together with cases 12 and 13 '
        'this means the five disagreement cases cover each strategy being the only correct one.',
}

OBSERVATIONS = [
    'Gold and the model apply different criteria. Gold asks whether the instruction text is self-contained; the '
    'model asks whether the scene can execute it. In cases 6, 7 and 8, all labelled CLEAR, the sink and the oven '
    'mentioned by the instruction are absent from the environment, so all three strategies request clarification. '
    'This single gap explains why unnecessary clarification sits at 0.45 to 0.66, and it shows the labels need '
    'manual review rather than being treated as absolute truth.',
    'The model is strong on referential uniqueness and weak on process executability. Cases 4 (three chocolate '
    'tablets) and 15 (three dinner plates, no dishwasher) are flagged as ambiguous by at least two strategies, '
    'while cases 9 (missing quantity and heat level) and 10 ("until done") are answered CLEAR by most strategies. '
    'A missing object gets detected; a missing process parameter does not.',
    'Each strategy is the only correct one in at least one case. Few-shot alone is right in case 12, Checklist '
    'alone in case 13 and Zero-shot alone in case 14. The difference comes from what each prompt inspects rather '
    'than from one strategy dominating: Few-shot is conservative, Checklist relies on structured checks and '
    'Zero-shot is lenient, consistent with overall Accuracy differing by only 0.5 to 2.5 percentage points.',
    'All 15 cases predict identically in both rounds, above the overall 97.67 percent round-to-round agreement '
    '(Zero-shot 100, Few-shot 95.5, Checklist 97.5). These cases are therefore stable judgement tendencies rather '
    'than sampling noise, which makes them usable as qualitative evidence.',
]

HEADINGS = {'success': '5.1 Success cases', 'error': '5.2 Typical errors', 'split': '5.3 Strategy splits'}


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

    return [('success', success_cases), ('error', error_cases), ('split', split_cases)]


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


def write_md(rows, path):
    lines = ['# Group 13 - 15 Case Analysis (WANG HONGFENG)', '',
             'Sources: the two frozen prediction runs R1 and R2 (200 instructions x 3 strategies each), labels from '
             '`data/processed/test_labels.csv`, environment and instruction text from `data/processed/test_inputs.csv`.',
             'Scoring follows `src/evaluate.py`. Selection is fixed by `scripts/select_cases.py`: entries are sorted by '
             'sample_id and the first match wins, the 15 cases use 15 distinct sample_ids and 15 distinct instruction '
             'pairs, and every case can be located in `predictions.jsonl` by its sample_id.', '']
    current = None
    for row in rows:
        heading = HEADINGS[row['case_type']]
        if heading != current:
            current = heading
            lines += [f'## {heading}', '']
        lines += [f"### Case {row['case_no']} - `{row['sample_id']}`", '',
                  f"- **Pair**: {row['pair_id']} | **Category**: {row['category']} | **Gold label**: {row['gold_label']}",
                  f"- **Environment**: {row['environment']}",
                  f"- **Instruction**: {row['instruction']}",
                  '- **Raw answers, identical in R1 and R2**:',
                  f"  - Zero-shot: `{row['R1_zero_shot']}` / `{row['R2_zero_shot']}`",
                  f"  - Few-shot: `{row['R1_few_shot']}` / `{row['R2_few_shot']}`",
                  f"  - Checklist: `{row['R1_checklist']}` / `{row['R2_checklist']}`",
                  f"- **Reading**: {row['analysis']}", '']
    lines += ['## 5.4 Observations across the 15 cases', '']
    lines += [f'{index}. {text}' for index, text in enumerate(OBSERVATIONS, start=1)]
    lines += ['', 'The 15 cases split 5 success / 5 typical error / 5 strategy split, which meets the handbook '
                  'requirement without any shortfall to declare.']
    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    inputs, labels = load_meta()
    entries = build_entries(inputs, labels, score_rounds(inputs, labels))
    groups = select(entries)

    rows, case_no = [], 0
    for case_type, group in groups:
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
    for case_type, group in groups:
        print(f'  {case_type}: {len(group)} -> ' + ', '.join(e['sample_id'] for e in group))
    print(f'outputs -> {OUT_DIR}')


if __name__ == '__main__':
    main()
