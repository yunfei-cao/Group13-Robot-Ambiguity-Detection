"""Prepare/check isolated DEV comparisons; real dispatch requires explicit flags."""
import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid

from src.model_client import ModelClient, ModelConfig
from src.prompt_v03_adapter import VerifiedBundle, VERSIONS, normalize_text
from src.run_experiment import STRATEGIES, Sanitizer, append, digest, load_samples, run_experiment

ROOT = Path(__file__).resolve().parents[1]
MODEL = dict(mode='qwen', model='qwen-plus-2025-12-01',
             base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
             allow_real_requests=True, temperature=0.0, max_tokens=64, timeout_seconds=30)
CODE = ['src/prompt_v03_adapter.py', 'scripts/prompt_v03_comparison.py',
        'src/model_client.py', 'src/run_experiment.py', 'src/prompt_adapter.py', 'src/prompts.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream, strict=True))


def write_csv(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def current(sample):
    return {k: sample[k] for k in ('environment', 'instruction')}


def dataset_audit(bundles):
    """Dev provenance and prior actual dispatch only; no test labels or raw test CSV."""
    inputs_path = ROOT / 'data/processed/dev_inputs.csv'
    labels_path = ROOT / 'data/processed/dev_labels.csv'
    all_inputs = load_samples(inputs_path)
    labels = {r['sample_id']: r for r in csv_rows(labels_path)}
    by_id = {r['id']: r for r in all_inputs}
    excluded = set()
    demo_text = set()
    sources = {str(p.relative_to(ROOT)): sha(p) for p in (inputs_path, labels_path)}
    for bundle in bundles.values():
        for name, expected in bundle.manifest['development_source_sha256'].items():
            raw = (ROOT / 'data/processed' / name).read_bytes()
            if hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest() != expected:
                raise ValueError('Development provenance mismatch: ' + name)
        for e in bundle.examples:
            if (e['sample_id'] not in by_id or any(e[k] != by_id[e['sample_id']][k]
                    for k in ('pair_id', 'environment', 'instruction'))
                    or e['gold_label'] != labels[e['sample_id']]['gold_label']):
                raise ValueError('Demonstration differs from original dev source')
            excluded.add(e['pair_id'])
            demo_text.add(normalize_text(e['instruction']))
    samples = load_samples(ROOT / 'prompt_v0_3/development_comparison_85pairs/dev_inputs.csv')
    sources['prompt_v0_3/development_comparison_85pairs/dev_inputs.csv'] = sha(
        ROOT / 'prompt_v0_3/development_comparison_85pairs/dev_inputs.csv')
    expected = [s for s in all_inputs if s['pair_id'] not in excluded]
    if {s['id']: s for s in samples} != {s['id']: s for s in expected}:
        raise ValueError('Common comparison is not exactly dev minus both demo-pair sets')
    counts = Counter(s['pair_id'] for s in samples)
    if len(samples) != 170 or len(counts) != 85 or set(counts.values()) != {2}:
        raise ValueError('Expected 85 complete development pairs')
    if any(normalize_text(s['instruction']) in demo_text for s in samples):
        raise ValueError('Evaluation instruction repeats old/new demonstration')
    prior = {}
    # A prepared but unexecuted selection is not counted as an earlier run.
    for folder in sorted((ROOT / 'results/dev').glob('*')):
        selection = folder / 'selected_inputs.csv'
        if not selection.exists():
            continue
        selected = {r['id']: r for r in load_samples(selection)}
        observed = set()
        journals = [p for p in (folder / 'predictions.jsonl', folder / 'attempts.jsonl') if p.exists()]
        for path in journals:
            for line in path.read_bytes().splitlines():
                row = json.loads(line)
                if row.get('event') in (None, 'started', 'completed'):
                    observed.add(row['sample_id'])
            sources[str(path.relative_to(ROOT))] = sha(path)
        if observed - selected.keys():
            raise ValueError('Historical dev journal contains unknown sample')
        sources[str(selection.relative_to(ROOT))] = sha(selection)
        for sid in observed:
            if sid not in by_id or selected[sid] != by_id[sid]:
                raise ValueError('Historical selected dev source differs')
            prior.setdefault(selected[sid]['pair_id'], set()).add(folder.name)
    metadata = [dict(pair_id=pid, previously_run=pid in prior,
                     historical_run_ids=sorted(prior.get(pid, set())),
                     sample_ids=sorted(s['id'] for s in samples if s['pair_id'] == pid))
                for pid in sorted(counts)]
    return samples, dict(excluded_demo_pair_ids=sorted(excluded), pairs=metadata,
                        historical_pairs=sum(r['previously_run'] for r in metadata),
                        previously_unused_pairs=sum(not r['previously_run'] for r in metadata),
                        source_raw_sha256=sources)


class FrozenBuilder:
    def __init__(self, rows):
        self.messages = {}
        for row in rows:
            sample = json.loads(row['messages'][1]['content'])
            if set(sample) != {'environment', 'instruction'}:
                raise ValueError('Forbidden snapshot input metadata')
            key = digest([sample, row['strategy']])
            if key in self.messages:
                raise ValueError('Duplicate frozen input/strategy')
            self.messages[key] = row['messages']

    def __call__(self, sample, strategy):
        if set(sample) != {'environment', 'instruction'}:
            raise ValueError('Only environment/instruction allowed')
        return deepcopy(self.messages[digest([sample, strategy])])


def launcher(variant):
    return '''[CmdletBinding()]
param([switch]$CheckOnly, [int]$TokenBudget, [switch]$ExtendTokenBudget)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path
$plan = Join-Path $PSScriptRoot '__VARIANT__/plan.json'
Push-Location -LiteralPath $repoRoot
try {
    python -B -m scripts.prompt_v03_comparison check --plan $plan
    if ($LASTEXITCODE -ne 0) { throw 'Preflight failed; no API request sent.' }
    if ($CheckOnly) { return }
    if ([string]::IsNullOrWhiteSpace($env:DASHSCOPE_API_KEY)) { throw 'DASHSCOPE_API_KEY is absent.' }
    $runArgs = @('-B', '-m', 'scripts.prompt_v03_comparison', 'run', '--plan', $plan, '--real-api')
    if ($PSBoundParameters.ContainsKey('TokenBudget')) { $runArgs += @('--token-budget', [string]$TokenBudget) }
    if ($ExtendTokenBudget) { $runArgs += '--extend-token-budget' }
    python @runArgs
    if ($LASTEXITCODE -ne 0) { throw 'Run stopped or contains invalid results; inspect this run and retain its ID.' }
}
finally { Pop-Location }
'''.replace('__VARIANT__', variant)


def prepare(parent=None):
    with ExitStack() as stack:
        bundles = {v: stack.enter_context(VerifiedBundle(v)) for v in VERSIONS}
        samples, audit = dataset_audit(bundles)
        comparison_id = 'group13-dev-v01-v03-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
        folder = Path(parent or ROOT / 'results/prompt_comparison') / comparison_id
        folder.mkdir(parents=True, exist_ok=False)
        write_json(folder / 'development_overlap.json', audit)
        write_csv(folder / 'development_pairs.csv', [dict(pair_id=r['pair_id'],
                  previously_run=str(r['previously_run']).lower(),
                  historical_run_ids=';'.join(r['historical_run_ids']), sample_ids=';'.join(r['sample_ids']))
                  for r in audit['pairs']])
        code = {name: sha(ROOT / name) for name in CODE}
        for variant, bundle in bundles.items():
            side = folder / variant
            side.mkdir()
            rid = comparison_id + '-' + variant
            output = side / 'pilot' / rid
            output.mkdir(parents=True)
            # This is a NEW snapshot, never an old experiment snapshot.
            snapshot = side / 'snapshot'
            snapshot.mkdir()
            for relative in bundle.audit:
                dest = snapshot / 'verified_lf' / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes((bundle.folder / relative).read_bytes())
            write_json(snapshot / 'hash_audit.json', bundle.audit)
            write_csv(snapshot / 'dev_inputs.csv', [dict(sample_id=s['id'], pair_id=s['pair_id'], **current(s)) for s in samples])
            messages = [dict(sample_id=s['id'], strategy=t, messages=bundle(current(s), t))
                        for s in samples for t in STRATEGIES]
            write_json(snapshot / 'messages.json', messages)
            write_json(side / 'model_config.json', MODEL)
            entry = folder / ('run_' + variant + '.ps1')
            with entry.open('x', encoding='utf-8-sig', newline='') as stream:
                stream.write(launcher(variant))
            sealed = {p.relative_to(side).as_posix(): sha(p) for p in snapshot.rglob('*') if p.is_file()}
            sealed['model_config.json'] = sha(side / 'model_config.json')
            plan = dict(schema_version=1, comparison_id=comparison_id, variant=variant,
                        run_id=rid, prompt_version=VERSIONS[variant], dataset_split='dev',
                        runner_split='pilot', experiment_kind='development_comparison_not_final',
                        instructions=170, pairs=85, logical_requests=510,
                        initial_token_stop=300000, dataset_sha256=digest(samples),
                        prompt_messages_sha256=digest([r['messages'] for r in messages]),
                        sealed_files_sha256=sealed, code_raw_sha256=code,
                        overlap_sha256=sha(folder / 'development_overlap.json'),
                        pair_table_sha256=sha(folder / 'development_pairs.csv'),
                        launcher_sha256=sha(entry),
                        authorization='Preparation only; user manually launches real API commands.',
                        historical_test_results_seen=True, real_calls_during_preparation=0)
            write_json(side / 'plan.json', plan)
        for variant in VERSIONS:
            write_json(folder / variant / 'preflight.json', check(folder / variant / 'plan.json'))
        return folder


def check(plan_path):
    """Read-only, no credential lookup, no journal repairs, no real client."""
    plan_path = Path(plan_path).resolve()
    side = plan_path.parent
    folder = side.parent
    plan = read_json(plan_path)
    variant = plan['variant']
    if (variant not in VERSIONS or side.name != variant
            or not re.fullmatch(r'group13-dev-v01-v03-[A-Za-z0-9-]+', folder.name)
            or plan['comparison_id'] != folder.name or plan['run_id'] != folder.name + '-' + variant
            or plan['prompt_version'] != VERSIONS[variant]
            or plan['dataset_split'] != 'dev' or plan['runner_split'] != 'pilot'
            or plan['experiment_kind'] != 'development_comparison_not_final'
            or (plan['instructions'], plan['pairs'], plan['logical_requests'], plan['initial_token_stop']) != (170,85,510,300000)):
        raise ValueError('Comparison identity/policy mismatch')
    if set(plan['code_raw_sha256']) != set(CODE):
        raise ValueError('Missing code seals')
    for name, expected in plan['code_raw_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Experiment code changed: ' + name)
    for name, expected in plan['sealed_files_sha256'].items():
        path = (side / name).resolve()
        if not path.is_relative_to(side) or sha(path) != expected:
            raise ValueError('Sealed snapshot changed: ' + name)
    for name, field in [('development_overlap.json','overlap_sha256'),
                        ('development_pairs.csv','pair_table_sha256'),
                        ('run_' + variant + '.ps1','launcher_sha256')]:
        if sha(folder / name) != plan[field]:
            raise ValueError('Comparison evidence changed: ' + name)
    if read_json(side / 'model_config.json') != MODEL:
        raise ValueError('Model configuration differs from fixed comparison settings')
    with ExitStack() as stack:
        bundles = {v: stack.enter_context(VerifiedBundle(v)) for v in VERSIONS}
        samples, audit = dataset_audit(bundles)
        if audit != read_json(folder / 'development_overlap.json'):
            raise ValueError('Historical/source overlap evidence changed; prepare a new comparison')
        bundle = bundles[variant]
        if bundle.audit != read_json(side / 'snapshot/hash_audit.json'):
            raise ValueError('Checkout raw/LF hashes differ from prepared audit')
        for name, info in bundle.audit.items():
            if sha(side / 'snapshot/verified_lf' / name) != info['lf_sha256']:
                raise ValueError('LF snapshot mismatch')
        if load_samples(side / 'snapshot/dev_inputs.csv') != samples or digest(samples) != plan['dataset_sha256']:
            raise ValueError('Snapshot development inputs differ')
        rows = read_json(side / 'snapshot/messages.json')
        expected = [dict(sample_id=s['id'], strategy=t, messages=bundle(current(s), t))
                    for s in samples for t in STRATEGIES]
        if rows != expected or digest([r['messages'] for r in rows]) != plan['prompt_messages_sha256']:
            raise ValueError('Frozen messages differ from verified CAI module')
    identities = {digest([r['sample_id'],r['strategy']]):
                  (r['sample_id'],r['strategy'],digest(r['messages'])) for r in rows}
    output = side / 'pilot' / plan['run_id']
    for name in ('attempts.jsonl','predictions.jsonl'):
        path = output / name
        if not path.exists(): continue
        # Do not truncate crash tails here; the runner alone handles tail repair.
        data = path.read_bytes()
        lines = data.splitlines(keepends=True)
        if lines and not lines[-1].endswith(b'\n'):
            lines = lines[:-1]
        for line in lines:
            r = json.loads(line)
            if (r.get('run_id') != plan['run_id'] or identities.get(r.get('logical_id')) !=
                    (r.get('sample_id'),r.get('strategy'),r.get('prompt_sha256'))):
                raise ValueError('Foreign experiment journal; refusing resume')
    config_path = output / 'run_config.json'
    if config_path.exists():
        config = read_json(config_path)
        if (config['run_id'] != plan['run_id'] or config['snapshot_sha256'] != sha(plan_path)
                or config['prompt_version'] != plan['prompt_version']
                or config['dataset_sha256'] != plan['dataset_sha256']
                or config['prompts_sha256'] != plan['prompt_messages_sha256']):
            raise ValueError('Foreign or changed run configuration')
    return dict(run_id=plan['run_id'], prompt_version=plan['prompt_version'],
                instructions=170, pairs=85, logical_requests=510,
                historical_pairs=audit['historical_pairs'], previously_unused_pairs=audit['previously_unused_pairs'],
                all_assets_verified=True, windows_raw_and_lf_hashes_recorded=True,
                output_dir=str(output), real_api_calls=0)


def execute(plan_path, *, real_api=False, token_budget=None, extend_token_budget=False):
    if not real_api:
        raise ValueError('Real dispatch requires explicit --real-api')
    check(plan_path)
    plan_path = Path(plan_path).resolve()
    side = plan_path.parent
    plan = read_json(plan_path)
    config_path = side / 'pilot' / plan['run_id'] / 'run_config.json'
    budget = token_budget if token_budget is not None else (
        read_json(config_path)['safety_policy']['total_token_stop'] if config_path.exists() else plan['initial_token_stop'])
    # Only this explicitly authorized path reads credentials or constructs a real client.
    config = ModelConfig.from_file(side / 'model_config.json')
    summary = run_experiment(load_samples(side / 'snapshot/dev_inputs.csv'),
        results_root=side, run_id=plan['run_id'], split='pilot', client=ModelClient(config),
        builder=FrozenBuilder(read_json(side / 'snapshot/messages.json')),
        prompt_version=plan['prompt_version'], stop_on_anomaly=True,
        max_logical_requests=510, allow_draft_pilot=True, snapshot_sha256=sha(plan_path),
        total_token_stop=budget, extend_token_budget=extend_token_budget)
    append(side / 'pilot' / plan['run_id'] / 'execution_summaries.jsonl',
           dict(summary,timestamp=datetime.now(timezone.utc).isoformat()),Sanitizer(config))
    print(json.dumps(summary,indent=2))
    return 3 if summary.get('stopped_reason') else (0 if summary['valid_predictions'] == 510 else 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    commands.add_parser('prepare')
    for name in ('check','run'):
        child = commands.add_parser(name)
        child.add_argument('--plan',type=Path,required=True)
        if name == 'run':
            child.add_argument('--real-api',action='store_true')
            child.add_argument('--token-budget',type=int)
            child.add_argument('--extend-token-budget',action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare': print(prepare())
    elif args.command == 'check': print(json.dumps(check(args.plan),indent=2))
    else: return execute(args.plan,real_api=args.real_api,token_budget=args.token_budget,
                         extend_token_budget=args.extend_token_budget)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
