#!/usr/bin/env python3
"""Prepare paired AmbiK data for CLEAR / NEEDS_CLARIFICATION classification.

Python 3.9+, standard library only. Never modifies source CSVs.
See README_使用说明.md for the cleaning policy and output contract.
"""
import argparse
import csv
import hashlib
import io
import json
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

VERSION = '1.0.0'
VARIANTS = [('clear', 'CLEAR'), ('ambiguous', 'NEEDS_CLARIFICATION')]
REQUIRED = ['environment_full', 'unambiguous_direct', 'ambiguous_task', 'ambiguity_type']
INPUT_FIELDS = ['sample_id', 'pair_id', 'environment', 'instruction']
LABEL_FIELDS = ['sample_id', 'pair_id', 'gold_label', 'category', 'source_record',
                'source_index', 'source_id', 'instruction_column']
CHECK_FIELDS = ['split', 'pair_id', 'source_record', 'source_index', 'source_id',
                'action', 'reason', 'related_pair_ids', 'environment',
                'clear_instruction', 'ambiguous_instruction']


def normalize(value):
    """Only for comparisons. Original strings are kept in exported model inputs."""
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_source(path, split):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, strict=True)
        fields = reader.fieldnames or []
        missing = sorted(set(REQUIRED) - set(fields))
        if missing:
            raise ValueError(f'{path}: missing required columns: {missing}')
        if len(fields) != len(set(fields)):
            raise ValueError(f'{path}: duplicate column names')
        rows = list(reader)
    if not rows:
        raise ValueError(f'{path}: empty CSV')
    pairs, seen = [], set()
    for record, row in enumerate(rows, 1):
        if None in row or any(v is None for v in row.values()):
            raise ValueError(f'{path}: malformed CSV record {record}')
        # The unnamed index is local to a split, NOT a global AmbiK identifier.
        index = row.get('', '')
        source_id = row.get('id', '')
        identity = index.strip() or source_id.strip() or f'record_{record}'
        if identity in seen:
            raise ValueError(f'{path}: duplicate source index {identity!r}')
        seen.add(identity)
        pair_id = f'{split}_{identity}'
        pairs.append(dict(split=split, pair_id=pair_id, source_record=record,
                          source_index=index, source_id=source_id,
                          environment=row['environment_full'],
                          clear=row['unambiguous_direct'], ambiguous=row['ambiguous_task'],
                          category=row['ambiguity_type']))
    return rows, pairs


def input_key(pair, variant):
    return normalize(pair['environment']), normalize(pair[variant])


def check_record(pair, reason, related=(), action='exclude'):
    return dict(split=pair['split'], pair_id=pair['pair_id'],
                source_record=pair['source_record'], source_index=pair['source_index'],
                source_id=pair['source_id'], action=action, reason=reason,
                related_pair_ids=';'.join(sorted(set(related))),
                environment=pair['environment'], clear_instruction=pair['clear'],
                ambiguous_instruction=pair['ambiguous'])


def clean_split(pairs, checks):
    """Exclude whole pairs; never retain one side of an excluded pair."""
    candidates = []
    for pair in sorted(pairs, key=lambda p: p['pair_id']):
        missing = [k for k in ['environment', 'clear', 'ambiguous', 'category']
                   if not normalize(pair[k])]
        if missing:
            checks.append(check_record(pair, 'missing_required_value:' + ','.join(missing)))
        elif normalize(pair['clear']) == normalize(pair['ambiguous']):
            checks.append(check_record(pair, 'identical_paired_instructions'))
        else:
            candidates.append(pair)

    # Detect contradictions across DIFFERENT pairs with the same actual input.
    index = defaultdict(list)
    for pair in candidates:
        for variant, label in VARIANTS:
            index[input_key(pair, variant)].append((pair['pair_id'], label))
    conflicts = defaultdict(set)
    for entries in index.values():
        if len({label for _, label in entries}) > 1:
            ids = {pid for pid, _ in entries}
            for pid in ids:
                conflicts[pid].update(ids - {pid})
    kept, seen_inputs = [], {}
    for pair in candidates:
        pid = pair['pair_id']
        if pid in conflicts:
            checks.append(check_record(pair, 'cross_pair_label_conflict', conflicts[pid]))
            continue
        keys = [input_key(pair, variant) for variant, _ in VARIANTS]
        duplicates = {seen_inputs[k] for k in keys if k in seen_inputs}
        if duplicates:
            checks.append(check_record(pair, 'duplicate_environment_instruction', duplicates))
            continue
        kept.append(pair)
        for key in keys:
            seen_inputs[key] = pid
    return kept


def expand(pairs):
    inputs, labels = [], []
    for pair in pairs:
        for variant, label in VARIANTS:
            # Opaque IDs avoid embedding the gold label in sample_id.
            sample_id = 's_' + hashlib.sha256(
                f"{pair['pair_id']}:{variant}".encode('utf-8')).hexdigest()[:20]
            inputs.append(dict(sample_id=sample_id, pair_id=pair['pair_id'],
                               environment=pair['environment'], instruction=pair[variant]))
            labels.append(dict(sample_id=sample_id, pair_id=pair['pair_id'],
                               gold_label=label, category=pair['category'],
                               source_record=pair['source_record'],
                               source_index=pair['source_index'], source_id=pair['source_id'],
                               instruction_column=('unambiguous_direct' if variant == 'clear'
                                                   else 'ambiguous_task')))
    return inputs, labels


def validate(inputs, labels, expected_pairs):
    if len(inputs) != 2 * expected_pairs or len(labels) != len(inputs):
        raise ValueError('Unexpected output sample count')
    ids = [r['sample_id'] for r in inputs]
    if len(set(ids)) != len(ids) or ids != [r['sample_id'] for r in labels]:
        raise ValueError('Duplicate or misaligned sample IDs')
    groups = defaultdict(list)
    for r in labels:
        groups[r['pair_id']].append(r['gold_label'])
    if len(groups) != expected_pairs or any(
            sorted(v) != ['CLEAR', 'NEEDS_CLARIFICATION'] for v in groups.values()):
        raise ValueError('Broken pairs or unbalanced labels')
    keys = [(normalize(r['environment']), normalize(r['instruction'])) for r in inputs]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate model inputs survived cleaning')


def csv_text(rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def counts(pairs):
    return dict(sorted(Counter(p['category'] for p in pairs).items()))


def prepare(calib_path, test_path, out_dir, n_pairs=100, seed=42, overwrite=False):
    calib_path, test_path, out_dir = map(Path, (calib_path, test_path, out_dir))
    if n_pairs < 1:
        raise ValueError('--n-pairs must be positive')
    if calib_path.resolve() == test_path.resolve():
        raise ValueError('Development and test source paths must be different')
    raw_dev, original_dev = read_source(calib_path, 'dev')
    raw_test, original_test = read_source(test_path, 'test')
    checks = []
    dev = clean_split(original_dev, checks)
    test = clean_split(original_test, checks)
    if not dev:
        raise ValueError('No development pairs survived cleaning')

    # Conservative leakage check: compare against ALL original development
    # instructions, including indirect variants and excluded development pairs.
    dev_texts = defaultdict(set)
    for raw, pair in zip(raw_dev, original_dev):
        for col in ['unambiguous_direct', 'unambiguous_indirect', 'ambiguous_task']:
            value = normalize(raw.get(col, ''))
            if value:
                dev_texts[value].add(pair['pair_id'])
    eligible = []
    for pair in test:
        related = dev_texts.get(normalize(pair['clear']), set()) | dev_texts.get(
            normalize(pair['ambiguous']), set())
        if related:
            checks.append(check_record(pair, 'development_instruction_overlap', related))
        else:
            eligible.append(pair)
    if len(eligible) < n_pairs:
        raise ValueError(f'Only {len(eligible)} eligible test pairs; requested {n_pairs}')

    selected = sorted(random.Random(seed).sample(eligible, n_pairs), key=lambda p: p['pair_id'])
    dev_inputs, dev_labels = expand(dev)
    test_inputs, test_labels = expand(selected)
    validate(dev_inputs, dev_labels, len(dev))
    validate(test_inputs, test_labels, n_pairs)
    assert not any(normalize(r['instruction']) in dev_texts for r in test_inputs)
    selected_ids = [p['pair_id'] for p in selected]
    reason_counts = dict(sorted(Counter((r['split'] + ':' + r['reason']) for r in checks).items()))
    manifest = dict(
        script_version=VERSION, seed=seed, requested_test_pairs=n_pairs,
        sampling='random.Random(seed).sample over lexically sorted eligible pair IDs',
        normalization='Unicode NFKC + casefold + collapse whitespace; comparison only',
        environment_column='environment_full',
        label_mapping={'unambiguous_direct': 'CLEAR', 'ambiguous_task': 'NEEDS_CLARIFICATION'},
        manual_semantic_review='PENDING: automatic checks do not establish label validity',
        sources={
            'dev': {'filename': calib_path.name, 'sha256': sha256(calib_path), 'rows': len(raw_dev)},
            'test': {'filename': test_path.name, 'sha256': sha256(test_path), 'rows': len(raw_test)}},
        counts={'dev_pairs': len(dev), 'dev_instructions': len(dev_inputs),
                'eligible_test_pairs': len(eligible), 'selected_test_pairs': len(selected),
                'test_instructions': len(test_inputs),
                'expected_logical_requests_for_three_strategies': len(test_inputs) * 3},
        category_pairs={'dev_raw': counts(original_dev), 'dev_clean': counts(dev),
                        'test_raw': counts(original_test), 'test_eligible': counts(eligible),
                        'test_selected': counts(selected)},
        excluded_by_reason=reason_counts, test_pair_ids=selected_ids,
        source_id_policy='split-prefixed unnamed source index, else id, else record number; '
                         'source_record is 1-based data-record ordinal, not a physical CSV line',
        leakage_policy='test direct/ambiguous text cannot match any original development '
                       'direct/indirect/ambiguous text after normalization',
    )
    files = {
        'dev_inputs.csv': csv_text(dev_inputs, INPUT_FIELDS),
        'dev_labels.csv': csv_text(dev_labels, LABEL_FIELDS),
        'test_inputs.csv': csv_text(test_inputs, INPUT_FIELDS),
        'test_labels.csv': csv_text(test_labels, LABEL_FIELDS),
        'excluded_pairs.csv': csv_text(checks, CHECK_FIELDS),
        'test_pair_ids.txt': '\n'.join(selected_ids) + '\n',
    }
    manifest['output_sha256'] = {
        name: hashlib.sha256(value.encode('utf-8-sig' if name.endswith('.csv') else 'utf-8')).hexdigest()
        for name, value in files.items()}
    files['data_manifest.json'] = json.dumps(manifest, ensure_ascii=False, indent=2) + '\n'
    for name in files:
        target = out_dir / name
        if target.resolve() in {calib_path.resolve(), test_path.resolve()}:
            raise ValueError('Output would overwrite a source file')
        if target.exists() and not overwrite:
            raise ValueError(f'{target} already exists; use another --out-dir or --overwrite')
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, value in files.items():
        encoding = 'utf-8-sig' if name.endswith('.csv') else 'utf-8'
        with (out_dir / name).open('w', encoding=encoding, newline='') as stream:
            stream.write(value)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calib', required=True, type=Path, help='Original development CSV')
    parser.add_argument('--test', required=True, type=Path, help='Original test CSV')
    parser.add_argument('--out-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--n-pairs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true', help='Replace generated files only')
    args = parser.parse_args()
    try:
        manifest = prepare(args.calib, args.test, args.out_dir, args.n_pairs, args.seed, args.overwrite)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f'ERROR: {exc}\n')
    print(json.dumps({'output_directory': str(args.out_dir.resolve()),
                      **manifest['counts'], 'excluded_by_reason': manifest['excluded_by_reason']},
                     ensure_ascii=True, indent=2))
    print('Automatic checks passed. Manual review of label semantics is still required.')


if __name__ == '__main__':
    main()
