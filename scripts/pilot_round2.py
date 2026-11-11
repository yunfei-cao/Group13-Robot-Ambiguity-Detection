"""Independent DRAFT repeat pilot. Prepare/check never access credentials or API."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import uuid

from scripts.full_pilot import (ROOT, SnapshotBuilder, check as check_source,
                                read_json, sha, validate_inputs, write_json)
from src.model_client import ModelClient, ModelConfig
from src.run_experiment import (STRATEGIES, Sanitizer, append, digest,
                                load_samples, run_experiment)

SOURCE_ID = 'group13-full-pilot-qwen200-20261111T090248Z-76877106'
JOURNALS = ('predictions.jsonl', 'attempts.jsonl', 'execution_summaries.jsonl')


def seals(folder):
    return {p.relative_to(folder).as_posix(): sha(p)
            for p in sorted(folder.rglob('*')) if p.is_file()}


def prepare():
    source = ROOT/'results/pilot'/SOURCE_ID
    before = seals(source)
    check_source(source/'pilot_manifest.json')
    old = read_json(source/'pilot_manifest.json')
    run_id = ('group13-full-pilot-round2-' +
              datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    folder = ROOT/'results/pilot'/run_id
    folder.mkdir(exist_ok=False)
    # Copy immutable inputs only. Never copy journals, run_config, or recovery state.
    shutil.copytree(source/'snapshot', folder/'snapshot')
    shutil.copyfile(source/'model_config.json', folder/'model_config.json')
    plan = dict(old, run_id=run_id, output=folder.relative_to(ROOT).as_posix(),
                input=(folder/'snapshot/data/processed/test_inputs.csv').relative_to(ROOT).as_posix(),
                round=2, source_run_id=SOURCE_ID,
                source_manifest_sha256=sha(source/'pilot_manifest.json'),
                source_file_sha256=before, round2_code_sha256=sha(Path(__file__)),
                authorization='Preparation only by agent. Real API execution only by user manually launching PowerShell.',
                created_at=datetime.now(timezone.utc).isoformat(),
                budget_note='Initial and launch limit 300000; this entry point cannot raise it.',
                independent_state=True, real_calls_during_preparation=0)
    script = '''# Independent Round 2; DRAFT v0.1-dev, pilot only.
# 600 logical jobs (200 per strategy); retries can add paid HTTP attempts.
# Rerun this same script to resume ONLY Round 2. Token soft stop: 300000.
[CmdletBinding()]
param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path
$manifest = Join-Path $PSScriptRoot 'pilot_manifest.json'
Push-Location -LiteralPath $repoRoot
try {
    python -B -m scripts.pilot_round2 check --manifest $manifest
    if ($LASTEXITCODE -ne 0) { throw 'Round 2 preflight failed; no API request attempted.' }
    if ($CheckOnly) { return }
    if ([string]::IsNullOrWhiteSpace($env:DASHSCOPE_API_KEY)) {
        throw 'DASHSCOPE_API_KEY is absent; no API request attempted.'
    }
    python -B -m scripts.pilot_round2 run --manifest $manifest --real-api --allow-draft-pilot --stop-on-anomaly
    if ($LASTEXITCODE -ne 0) { throw 'Round 2 stopped or has invalid results; inspect Round 2 journals. Keep this run ID.' }
}
finally { Pop-Location }
'''
    with (folder/'run_round2.ps1').open('x', encoding='utf-8-sig', newline='') as stream:
        stream.write(script)
    plan['launcher_sha256'] = sha(folder/'run_round2.ps1')
    write_json(folder/'pilot_manifest.json', plan)
    write_json(folder/'preparation_checks.json', check(folder/'pilot_manifest.json'))
    if seals(source) != before:
        raise ValueError('Round 1 changed during preparation')
    print(folder)
    return folder


def check(manifest):
    manifest = manifest.resolve()
    folder = manifest.parent
    plan = read_json(manifest)
    rid = plan['run_id']
    if not re.fullmatch(r'group13-full-pilot-round2-[A-Za-z0-9-]+', rid):
        raise ValueError('Invalid Round 2 run ID')
    if folder != ROOT/'results/pilot'/rid or manifest.name != 'pilot_manifest.json':
        raise ValueError('Round 2 output identity mismatch')
    source = ROOT/'results/pilot'/SOURCE_ID
    if plan['source_run_id'] != SOURCE_ID or seals(source) != plan['source_file_sha256']:
        raise ValueError('Round 1 original files changed')
    if sha(source/'pilot_manifest.json') != plan['source_manifest_sha256']:
        raise ValueError('Source manifest changed')
    check_source(source/'pilot_manifest.json')
    old = read_json(source/'pilot_manifest.json')
    for key in ('source_sha256', 'messages_sha256', 'prompts_sha256', 'dataset_sha256',
                'model_config_sha256', 'model', 'temperature', 'max_tokens',
                'enable_thinking', 'stream', 'prompt_version', 'prompt_status'):
        if plan[key] != old[key]:
            raise ValueError('Round 1 configuration mismatch: '+key)
    if (plan['split'] != 'pilot' or plan['experiment_kind'] != 'full_pilot_not_final'
            or plan['round'] != 2 or not plan['authorized_draft_pilot']
            or not plan['independent_state'] or plan['initial_token_stop'] != 300000
            or not plan['prompt_status'].startswith('DRAFT')
            or plan['instructions'] != 200 or plan['logical_requests'] != 600
            or plan['strategy_counts'] != {s: 200 for s in STRATEGIES}):
        raise ValueError('Round 2 pilot policy mismatch')
    if sha(Path(__file__)) != plan['round2_code_sha256'] or sha(folder/'run_round2.ps1') != plan['launcher_sha256']:
        raise ValueError('Round 2 entry point changed')
    if seals(folder/'snapshot') != seals(source/'snapshot'):
        raise ValueError('Round 2 immutable snapshot differs from Round 1')
    if sha(folder/'model_config.json') != plan['model_config_sha256']:
        raise ValueError('Model configuration hash mismatch')
    expected_input = (folder/'snapshot/data/processed/test_inputs.csv').relative_to(ROOT).as_posix()
    if plan['input'] != expected_input or plan['output'] != folder.relative_to(ROOT).as_posix():
        raise ValueError('Round 2 paths must be independent')
    samples = load_samples(ROOT/plan['input'])
    validate_inputs(samples)
    if digest(samples) != plan['dataset_sha256']:
        raise ValueError('Dataset hash mismatch')
    rows = read_json(folder/'snapshot/messages.json')
    old_rows = read_json(source/'snapshot/messages.json')
    jobs = [(s['id'], t) for s in samples for t in STRATEGIES]
    if rows != old_rows or [(r['sample_id'], r['strategy']) for r in rows] != jobs:
        raise ValueError('600 prompt messages/order mismatch')
    builder = SnapshotBuilder(rows)
    for row, (sample, strategy) in zip(rows, ((s, t) for s in samples for t in STRATEGIES)):
        current = {k: sample[k] for k in ('environment', 'instruction')}
        if row['messages'] != builder(current, strategy) or digest(row['messages']) != row['prompt_sha256']:
            raise ValueError('Prompt hash or input whitelist mismatch')
    examples = read_json(folder/'snapshot/prompts/few_shot_examples.json')
    if {e['pair_id'] for e in examples} & {s['pair_id'] for s in samples}:
        raise ValueError('Test/few-shot pair overlap')
    # Read-only validation: never repair/truncate a journal during CheckOnly.
    expected = {digest([sid, strategy]): (sid, strategy, r['prompt_sha256'])
                for (sid, strategy), r in zip(jobs, rows)}
    for name in JOURNALS:
        path = folder/name
        if not path.exists():
            continue
        for line in path.read_bytes().splitlines():
            import json
            row = json.loads(line)
            if name == 'execution_summaries.jsonl':
                if Path(row['output_dir']).resolve() != folder:
                    raise ValueError('Foreign execution summary')
            elif (row.get('run_id') != rid or
                  expected.get(row.get('logical_id')) !=
                  (row.get('sample_id'), row.get('strategy'), row.get('prompt_sha256'))):
                raise ValueError('Foreign journal state; cannot resume Round 1')
    config_path = folder/'run_config.json'
    if config_path.exists():
        config = read_json(config_path)
        if (config['run_id'] != rid or config['snapshot_sha256'] != sha(manifest)
                or config['safety_policy']['total_token_stop'] != 300000):
            raise ValueError('Foreign or changed run state')
    return dict(run_id=rid, instructions=200, logical_requests=len(rows),
                strategy_counts=dict(Counter(r['strategy'] for r in rows)),
                matching_prompt_hashes=len(rows), messages_sha256=plan['messages_sha256'],
                prompts_sha256=plan['prompts_sha256'], model_config_sha256=plan['model_config_sha256'],
                prompt_status=plan['prompt_status'], total_token_stop=300000,
                round1_all_files_unchanged=True, test_gold_read=False, real_api_calls=0)


def execute(args):
    if not (args.real_api and args.allow_draft_pilot and args.stop_on_anomaly):
        raise ValueError('Manual real execution requires all three authorization flags')
    check(args.manifest)
    folder = args.manifest.resolve().parent
    plan = read_json(args.manifest)
    config = ModelConfig.from_file(folder/'model_config.json')
    summary = run_experiment(load_samples(ROOT/plan['input']), results_root=ROOT/'results',
        run_id=plan['run_id'], split='pilot', client=ModelClient(config),
        builder=SnapshotBuilder(read_json(folder/'snapshot/messages.json')),
        prompt_version=plan['prompt_version'], stop_on_anomaly=True,
        max_logical_requests=600, allow_draft_pilot=True,
        snapshot_sha256=sha(args.manifest), total_token_stop=300000)
    append(folder/'execution_summaries.jsonl',
           dict(summary, timestamp=datetime.now(timezone.utc).isoformat()), Sanitizer(config))
    import json
    print(json.dumps(summary, indent=2))
    return 3 if summary.get('stopped_reason') else (0 if summary['valid_predictions'] == 600 else 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare')
    for command in ('check', 'run'):
        child = commands.add_parser(command)
        child.add_argument('--manifest', type=Path, required=True)
        if command == 'run':
            for flag in ('real-api', 'allow-draft-pilot', 'stop-on-anomaly'):
                child.add_argument('--'+flag, action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'check':
        import json
        print(json.dumps(check(args.manifest), indent=2))
    else:
        return execute(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
