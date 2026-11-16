"""Explicit, boundary-bound recovery. Check mode never constructs an API client."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

from scripts.full_pilot import ROOT, SnapshotBuilder, check, read_json, sha
from src.model_client import ModelClient, ModelConfig
from src.run_experiment import append, digest, load_samples, Sanitizer

REASON = 'three_consecutive_unsuccessful_attempts'
JOURNALS = ('attempts.jsonl', 'predictions.jsonl', 'execution_summaries.jsonl')


def strict_rows(path):
    data = path.read_bytes() if path.exists() else b''
    if data and not data.endswith(b'\n'):
        raise ValueError('Incomplete journal; recovery will not truncate history')
    rows = [json.loads(line) for line in data.splitlines()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError('Invalid journal')
    return rows


def recovery_stop(rows, boundary, original, budget=300000):
    """Only consecutive failures reset; all other guards scan the entire history."""
    if budget != 300000:
        raise ValueError('Recovery budget is fixed at 300000')
    consecutive = rates = tokens = 0
    for index, row in enumerate(rows):
        if index == boundary:
            consecutive = 0
        reason = original([row], budget)
        if reason:
            return reason
        usage = row.get('token_usage') or {}
        total = usage.get('total_tokens')
        if type(total) is int:
            tokens += total
        error = row.get('error') or {}
        rates += int(error.get('http_status') == 429 or error.get('code') == 'rate_limit')
        consecutive = consecutive + 1 if row.get('status') != 'success' else 0
        if tokens >= budget:
            return 'token_budget_reached'
        if rates >= 3:
            return 'three_rate_limit_errors'
        if index >= boundary and consecutive >= 3:
            return REASON
    return None


def load_runner():
    # Isolated module globals: no monkeypatch of the shared/original runner.
    spec = importlib.util.spec_from_file_location('src._pilot_recovery_runner', ROOT/'src/run_experiment.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(folder, receipt, runner):
    if not isinstance(receipt.get('authorization'), str) or not receipt['authorization'].strip():
        raise ValueError('Missing reviewed recovery authorization')
    audit = read_json(folder/'reviewed_stop_audit.json')
    if (audit.get('authorization_sha256') != digest(receipt)
            or audit.get('reason') != REASON
            or audit.get('run_id') != receipt['run_id']
            or audit.get('journal_prefixes') != receipt['journal_prefixes']
            or audit.get('reviewed_completions') != receipt['reviewed_completions']
            or audit.get('authorization') != receipt['authorization']):
        raise ValueError('Reviewed stop audit/authorization mismatch')
    plan = read_json(folder/'pilot_manifest.json')
    if receipt['run_id'] != plan['run_id'] or receipt['manifest_sha256'] != sha(folder/'pilot_manifest.json'):
        raise ValueError('Recovery authorization identity mismatch')
    for name, seal in receipt['journal_prefixes'].items():
        data = (folder/name).read_bytes()
        if len(data) < seal['bytes'] or hashlib.sha256(data[:seal['bytes']]).hexdigest() != seal['sha256']:
            raise ValueError('Historical journal prefix changed')
    if set(receipt['journal_prefixes']) != set(JOURNALS):
        raise ValueError('Missing historical journal seals')
    attempts = strict_rows(folder/'attempts.jsonl')
    predictions = strict_rows(folder/'predictions.jsonl')
    strict_rows(folder/'execution_summaries.jsonl')
    config = read_json(folder/'run_config.json')
    if sha(folder/'run_config.json') != receipt['run_config_sha256'] or config['safety_policy']['total_token_stop'] != 300000:
        raise ValueError('Recovery cannot change configuration or budget')
    completions = [r for r in attempts if r['event'] == 'completed']
    boundary = receipt['reviewed_completions']
    if type(boundary) is not int or boundary <= 0 or boundary > len(completions):
        raise ValueError('Invalid recovery boundary')
    seal = receipt['journal_prefixes']['attempts.jsonl']
    sealed_attempts = [json.loads(line) for line in (folder/'attempts.jsonl').read_bytes()[:seal['bytes']].splitlines()]
    if sum(r['event'] == 'completed' for r in sealed_attempts) != boundary:
        raise ValueError('Boundary must equal reviewed journal prefix')
    if runner.safety_stop(completions[:boundary], 300000) != REASON:
        raise ValueError('Only reviewed consecutive-failure stops can be recovered')
    if audit.get('historical_stop') != completions[boundary-1]:
        raise ValueError('Historical stop position changed')
    summary_seal = receipt['journal_prefixes']['execution_summaries.jsonl']
    sealed_summaries = [json.loads(line) for line in (folder/'execution_summaries.jsonl').read_bytes()[:summary_seal['bytes']].splitlines()]
    if not sealed_summaries or sealed_summaries[-1].get('stopped_reason') != REASON:
        raise ValueError('Missing historical stop summary')
    # Require a final record for every started historical job: no old failed or
    # interrupted job is implicitly retried by this recovery entry point.
    done = {r['logical_id'] for r in predictions}
    if len(done) != len(predictions) or any(r['logical_id'] not in done for r in attempts):
        raise ValueError('Duplicate finals or unfinished historical job')
    event = recovery_event(folder, receipt)
    events = strict_rows(folder/'recovery_events.jsonl')
    if events and (len(events) != 1 or any(events[0].get(k) != v for k,v in event.items())):
        raise ValueError('Recovery audit mismatch; refusing a new reset')
    if len(completions) > boundary and not events:
        raise ValueError('New attempts without recovery audit')
    reason = recovery_stop(completions, boundary, runner.safety_stop)
    return dict(run_id=plan['run_id'], final_predictions=len(predictions),
                valid_predictions=sum(bool(r['is_valid']) for r in predictions),
                remaining_logical_requests=600-len(done), reviewed_completions=boundary,
                known_total_tokens=sum((r.get('token_usage') or {}).get('total_tokens') or 0 for r in completions),
                total_token_stop=300000, stopped_reason=reason,
                execution_requires='--resume-after-reviewed-stop', real_api_calls=0)


def recovery_event(folder, receipt):
    return dict(event='reviewed_failure_recovery', run_id=receipt['run_id'],
                authorization_sha256=digest(receipt),
                review_audit_sha256=sha(folder/'reviewed_stop_audit.json'),
                authorization_flag='--resume-after-reviewed-stop',
                reviewed_completions=receipt['reviewed_completions'])


def run_recovery(folder, receipt, runner, client, *, authorized=False, sleep=None):
    if not authorized:
        raise ValueError('Explicit --resume-after-reviewed-stop is required')
    original_stop, original_lock, original_read = runner.safety_stop, runner.run_lock, runner.read_journal
    boundary = receipt['reviewed_completions']
    runner.safety_stop = lambda rows, budget=300000: recovery_stop(rows, boundary, original_stop, budget)

    @contextmanager
    def locked(path):
        with original_lock(path):
            check(folder/'pilot_manifest.json')
            # Validation must use the unmodified circuit breaker.
            from types import SimpleNamespace
            report = validate(folder, receipt, SimpleNamespace(safety_stop=original_stop))
            if report['stopped_reason']:
                raise ValueError('Safety stop remains active: '+report['stopped_reason'])
            event = recovery_event(folder, receipt)
            events = strict_rows(folder/'recovery_events.jsonl')
            if events and (len(events) != 1 or any(events[0].get(k) != v for k,v in event.items())):
                raise ValueError('Recovery audit mismatch; refusing a new reset')
            if not events:
                append(folder/'recovery_events.jsonl', dict(event, timestamp=datetime.now(timezone.utc).isoformat()), lambda x:x)
            yield
    runner.run_lock = locked
    # Recovery must never use the original runner's torn-tail truncation repair.
    runner.read_journal = strict_rows
    plan = read_json(folder/'pilot_manifest.json')
    options = dict(results_root=folder.parent.parent, run_id=plan['run_id'], split='pilot',
        client=client, builder=SnapshotBuilder(read_json(folder/'snapshot/messages.json')),
        prompt_version=plan['prompt_version'], stop_on_anomaly=True, max_logical_requests=600,
        allow_draft_pilot=True, snapshot_sha256=sha(folder/'pilot_manifest.json'), total_token_stop=300000)
    if sleep is not None:
        options['sleep'] = sleep
    try:
        summary = runner.run_experiment(load_samples(ROOT/plan['input']), **options)
        append(folder/'execution_summaries.jsonl',dict(summary,timestamp=datetime.now(timezone.utc).isoformat()),Sanitizer(client.config))
        return summary
    finally:
        runner.safety_stop, runner.run_lock, runner.read_journal = original_stop, original_lock, original_read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--resume-after-reviewed-stop', '--authorize-reviewed-failures',
                        dest='resume_after_reviewed_stop', action='store_true')
    args = parser.parse_args()
    check(args.manifest)
    folder = args.manifest.resolve().parent
    receipt = read_json(folder/'reviewed_failure_authorization.json')
    runner = load_runner()
    report = validate(folder, receipt, runner)
    if args.check_only:
        print(json.dumps(report, indent=2))
        return 3 if report['stopped_reason'] else 0
    if not args.resume_after_reviewed_stop:
        raise ValueError('Explicit --resume-after-reviewed-stop is required; no API request attempted')
    if report['stopped_reason']:
        raise ValueError('Safety stop remains active: '+report['stopped_reason'])
    config = ModelConfig.from_file(folder/'model_config.json')
    summary = run_recovery(folder,receipt,runner,ModelClient(config),authorized=True)
    print(json.dumps(summary,indent=2))
    return 3 if summary.get('stopped_reason') else 0


if __name__ == '__main__':
    raise SystemExit(main())
