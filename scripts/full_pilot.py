"""Prepare/check an immutable 600-request pilot; real execution is user-only.

Preparation/check never constructs a real client or reads API credentials.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import uuid

from src import prompts
from src.model_client import ModelClient, ModelConfig
from src.prompt_adapter import official_prompt_builder
from src.run_experiment import STRATEGIES, Sanitizer, append, digest, load_samples, run_experiment

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'data/processed/test_inputs.csv'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def validate_inputs(samples):
    if len(samples) != 200:
        raise ValueError('Full pilot requires exactly 200 instructions')
    pairs = Counter(s['pair_id'] for s in samples)
    if len(pairs) != 100 or set(pairs.values()) != {2}:
        raise ValueError('Full pilot requires 100 complete input pairs')
    for pid in pairs:
        rows = [s for s in samples if s['pair_id'] == pid]
        if rows[0]['environment'] != rows[1]['environment'] or rows[0]['instruction'] == rows[1]['instruction']:
            raise ValueError('Input pairs require shared environment and distinct instructions')


class SnapshotBuilder:
    """Return only stored messages keyed by the permitted current input."""
    def __init__(self, rows):
        self.by_input = {}
        for row in rows:
            user = json.loads(row['messages'][1]['content'])
            if set(user) != {'environment', 'instruction'}:
                raise ValueError('Snapshot user message contains forbidden metadata')
            key = digest([user, row['strategy']])
            if key in self.by_input and self.by_input[key] != row['messages']:
                raise ValueError('Conflicting frozen messages for the same input')
            self.by_input[key] = deepcopy(row['messages'])

    def __call__(self, sample, strategy):
        if set(sample) != {'environment', 'instruction'}:
            raise ValueError('Only environment and instruction may enter the builder')
        return deepcopy(self.by_input[digest([sample, strategy])])


def prepare():
    samples = load_samples(INPUT)
    validate_inputs(samples)
    manifest = read_json(ROOT / 'prompts/prompt_manifest.json')
    if prompts.PROMPT_VERSION != 'v0.1-dev' or manifest['prompt_version'] != 'v0.1-dev':
        raise ValueError('This pilot is explicitly authorized for v0.1-dev only')
    examples = prompts.load_examples()
    if [e['sample_id'] for e in examples] != manifest['example_ids_in_order']:
        raise ValueError('Fixed example order mismatch')
    if {e['pair_id'] for e in examples} & {s['pair_id'] for s in samples}:
        raise ValueError('Test/few-shot pair overlap')
    rows = []
    for sample in samples:
        for strategy in STRATEGIES:
            current = {k: sample[k] for k in ('environment','instruction')}
            messages = official_prompt_builder(current, strategy)
            if json.loads(messages[1]['content']) != current:
                raise ValueError('Current input whitelist mismatch')
            if messages != prompts.build_messages(strategy, **current, examples=examples):
                raise ValueError('Messages differ from CAI official prompts')
            rows.append(dict(sample_id=sample['id'], strategy=strategy, prompt_sha256=digest(messages), messages=messages))
    if Counter(r['strategy'] for r in rows) != Counter({s:200 for s in STRATEGIES}):
        raise ValueError('Expected 200 jobs per strategy')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = f'group13-full-pilot-qwen200-{stamp}-{uuid.uuid4().hex[:8]}'
    folder = ROOT / 'results/pilot' / run_id
    folder.mkdir(parents=True, exist_ok=False)
    snapshot = folder / 'snapshot'
    snapshot.mkdir()
    source_paths = [INPUT, ROOT/'src/prompts.py', ROOT/'src/prompt_adapter.py',
        ROOT/'src/run_experiment.py', ROOT/'src/model_client.py', Path(__file__).resolve(),
        *sorted((ROOT/'prompts').glob('*'))]
    source_paths = [p for p in source_paths if p.is_file()]
    files = {}
    for source in source_paths:
        relative = source.relative_to(ROOT)
        dest = snapshot / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        files[str(relative).replace('\\','/')] = sha(source)
    write_json(snapshot/'messages.json', rows)
    model = dict(mode='qwen', model='qwen-plus-2025-12-01',
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', allow_real_requests=True,
        temperature=0.0, max_tokens=64, timeout_seconds=30)
    write_json(folder/'model_config.json', model)
    plan = dict(run_id=run_id, experiment_kind='full_pilot_not_final', split='pilot',
        input='data/processed/test_inputs.csv', output=str(folder.relative_to(ROOT)).replace('\\','/'),
        prompt_version='v0.1-dev', prompt_status=manifest['status'],
        authorized_draft_pilot=True,
        authorization='User explicitly authorized 200 test instructions x 3 strategies, paid real API pilot, retries allowed; user executes PowerShell.',
        instructions=200, complete_pairs=100, logical_requests=600,
        strategy_counts={s:200 for s in STRATEGIES}, initial_token_stop=300000,
        budget_note='Keep 300000 initially. Dev known usage x10 = 275570 is only a rough extrapolation, not a bound. Unknown usage and retries may exceed it. Explicit upward-only budget extension can resume this same run.',
        enable_thinking=False, stream=False, model=model['model'], temperature=0.0, max_tokens=64,
        source_sha256=files, messages_sha256=sha(snapshot/'messages.json'),
        prompts_sha256=digest([r['messages'] for r in rows]),
        dataset_sha256=digest(samples), model_config_sha256=sha(folder/'model_config.json'),
        created_at=datetime.now(timezone.utc).isoformat(), real_calls_during_preparation=0)
    write_json(folder/'pilot_manifest.json', plan)
    script = '''# Full Pilot, explicitly authorized DRAFT v0.1-dev; NOT final paper experiment.
# Actual input: data/processed/test_inputs.csv (all 200 instructions / 100 pairs).
# Logical jobs: Zero-shot=200, Few-shot=200, Checklist=200; retries max 2 per job.
# Run ID: __RUN_ID__
# Results: results/pilot/__RUN_ID__/
# Fixed model qwen-plus-2025-12-01; temperature=0; max_tokens=64; thinking=false.
# Effective runner: --split pilot --allow-draft-pilot --stop-on-anomaly
# --max-logical-requests 600 --prompt-version v0.1-dev --total-token-stop 300000
# Rerun this same script to resume. It skips all persisted final logical records.
# After budget-only stop: review usage, then use -TokenBudget 400000 -ExtendTokenBudget.
# Permission/failure/malformed-usage stops remain latched; budget extension cannot clear them.
[CmdletBinding()]
param(
    [ValidateRange(1, 2147483647)][int]$TokenBudget,
    [switch]$ExtendTokenBudget,
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path
$pilotManifest = Join-Path $PSScriptRoot 'pilot_manifest.json'
Push-Location -LiteralPath $repoRoot
try {
    python -B -m scripts.full_pilot check --manifest $pilotManifest
    if ($LASTEXITCODE -ne 0) { throw 'Pilot preflight failed. No API request attempted.' }
    if ($CheckOnly) { return }
    if ([string]::IsNullOrWhiteSpace($env:DASHSCOPE_API_KEY)) {
        throw 'DASHSCOPE_API_KEY is absent in this PowerShell process; no API request attempted.'
    }
    $pilotArgs = @('-B', '-m', 'scripts.full_pilot', 'run', '--manifest', $pilotManifest,
        '--real-api', '--allow-draft-pilot', '--stop-on-anomaly')
    if ($PSBoundParameters.ContainsKey('TokenBudget')) {
        $pilotArgs += @('--total-token-stop', [string]$TokenBudget)
    }
    if ($ExtendTokenBudget) { $pilotArgs += '--extend-token-budget' }
    python @pilotArgs
    if ($LASTEXITCODE -ne 0) {
        throw 'Pilot stopped or process failed. Inspect execution_summaries.jsonl and journals; keep this run ID.'
    }
}
finally { Pop-Location }
'''.replace('__RUN_ID__',run_id)
    with (folder/'run_full_pilot.ps1').open('x',encoding='utf-8-sig',newline='') as stream:
        stream.write(script)
    write_json(folder/'preparation_checks.json', check(folder/'pilot_manifest.json'))
    print(str(folder/'run_full_pilot.ps1'))


def check(manifest_path):
    folder = manifest_path.resolve().parent
    plan = read_json(manifest_path)
    if not re.fullmatch(r'group13-full-pilot-qwen200-[A-Za-z0-9-]+', plan['run_id']):
        raise ValueError('Invalid pilot run ID')
    if folder != ROOT/'results/pilot'/plan['run_id'] or plan['split'] != 'pilot' or not plan['authorized_draft_pilot']:
        raise ValueError('Pilot identity/authorization mismatch')
    if plan['experiment_kind'] != 'full_pilot_not_final' or plan['prompt_version'] != 'v0.1-dev':
        raise ValueError('Unexpected pilot purpose/version')
    for relative,expected in plan['source_sha256'].items():
        if sha(ROOT/relative) != expected or sha(folder/'snapshot'/relative) != expected:
            raise ValueError('Source or frozen snapshot changed; do not mix configurations within this run')
    if sha(folder/'snapshot/messages.json') != plan['messages_sha256'] or sha(folder/'model_config.json') != plan['model_config_sha256']:
        raise ValueError('Pilot messages/model snapshot changed')
    model = read_json(folder/'model_config.json')
    if model != dict(mode='qwen',model='qwen-plus-2025-12-01',
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',allow_real_requests=True,
        temperature=0.0,max_tokens=64,timeout_seconds=30):
        raise ValueError('Unexpected pilot model configuration')
    samples = load_samples(ROOT/plan['input'])
    validate_inputs(samples)
    if digest(samples) != plan['dataset_sha256']:
        raise ValueError('Pilot dataset mismatch')
    rows = read_json(folder/'snapshot/messages.json')
    expected_jobs = [(s['id'],t) for s in samples for t in STRATEGIES]
    if [(r['sample_id'],r['strategy']) for r in rows] != expected_jobs or len(rows)!=600:
        raise ValueError('Pilot coverage/order mismatch')
    builder = SnapshotBuilder(rows)
    for row,(sample,strategy) in zip(rows,((s,t) for s in samples for t in STRATEGIES)):
        current = {k:sample[k] for k in ('environment','instruction')}
        if row['messages'] != official_prompt_builder(current,strategy) or digest(row['messages']) != row['prompt_sha256']:
            raise ValueError('Snapshot differs from authorized CAI messages')
        if builder(current,strategy) != row['messages']:
            raise ValueError('Frozen builder routing mismatch')
    if digest([r['messages'] for r in rows]) != plan['prompts_sha256']:
        raise ValueError('Aggregate prompt hash mismatch')
    return dict(checked_at=datetime.now(timezone.utc).isoformat(),run_id=plan['run_id'],
        instructions=200,complete_pairs=100,strategy_counts={s:200 for s in STRATEGIES},
        logical_requests=600,prompt_version=plan['prompt_version'],prompt_status=plan['prompt_status'],
        hashes_match=True,current_input_fields=['environment','instruction'],real_api_calls=0)


def execute(args):
    if not args.real_api or not args.allow_draft_pilot or not args.stop_on_anomaly:
        raise ValueError('Real execution requires --real-api --allow-draft-pilot --stop-on-anomaly')
    check(args.manifest)
    folder = args.manifest.resolve().parent
    plan = read_json(args.manifest)
    budget = args.total_token_stop
    if budget is None:
        config_path = folder/'run_config.json'
        budget = read_json(config_path)['safety_policy']['total_token_stop'] if config_path.exists() else plan['initial_token_stop']
    # Credentials are first accessed here, only in the user-launched run command.
    config = ModelConfig.from_file(folder/'model_config.json')
    summary = run_experiment(load_samples(ROOT/plan['input']),results_root=ROOT/'results',
        run_id=plan['run_id'],split='pilot',client=ModelClient(config),
        builder=SnapshotBuilder(read_json(folder/'snapshot/messages.json')),
        prompt_version=plan['prompt_version'],stop_on_anomaly=True,max_logical_requests=600,
        allow_draft_pilot=True,snapshot_sha256=sha(args.manifest),
        total_token_stop=budget,extend_token_budget=args.extend_token_budget)
    append(folder/'execution_summaries.jsonl',dict(summary,timestamp=datetime.now(timezone.utc).isoformat()),Sanitizer(config))
    print(json.dumps(summary,ensure_ascii=True,indent=2))
    if summary.get('stopped_reason'):
        return 3
    if summary['final_predictions']!=600 or summary['valid_predictions']!=600:
        return 4
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('prepare')
    for name in ('check','run'):
        child=commands.add_parser(name)
        child.add_argument('--manifest',type=Path,required=True)
        if name=='run':
            child.add_argument('--real-api',action='store_true')
            child.add_argument('--allow-draft-pilot',action='store_true')
            child.add_argument('--stop-on-anomaly',action='store_true')
            child.add_argument('--total-token-stop',type=int)
            child.add_argument('--extend-token-budget',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare': prepare()
    elif args.command=='check': print(json.dumps(check(args.manifest),indent=2))
    else: return execute(args)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
