"""Full offline acceptance. Run from root: python -m scripts.mock_acceptance."""
from collections import Counter
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
from unittest.mock import patch

from src import prompts
from src.evaluation_adapter import convert_predictions
from src.evaluate import score_samples
from src.model_client import ModelClient, ModelConfig
from src.run_experiment import STRATEGIES, digest, load_samples, run_experiment

ROOT = Path(__file__).resolve().parents[1]


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='group13-full-mock-acceptance',
                        help='Choose a fresh ID for each new 600-call acceptance execution')
    args = parser.parse_args()
    protected = [*ROOT.glob('src/*.py'), *ROOT.glob('tests/*.py'),
                 *ROOT.glob('prompts/*'), *ROOT.glob('data/processed/*')]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    samples = load_samples(ROOT / 'data/processed/test_inputs.csv')
    check(len(samples) == 200, 'Expected exactly 200 instructions')
    pairs = Counter(s['pair_id'] for s in samples)
    check(len(pairs) == 100 and set(pairs.values()) == {2}, 'Expected 100 complete input pairs')
    expected = [(s, strategy) for s in samples for strategy in STRATEGIES]
    client = ModelClient(ModelConfig(mode='mock'))
    classify = client.classify
    calls = []

    def inspected_classify(messages):
        sample, strategy = expected[len(calls)]
        check(messages == prompts.build_messages(strategy, sample['environment'], sample['instruction']),
              'Model messages differ from official prompt')
        check(json.loads(messages[1]['content']) == {
            k: sample[k] for k in ('environment', 'instruction')}, 'Current input leakage')
        calls.append((sample['id'], strategy, digest(messages)))
        return classify(messages)

    client.classify = inspected_classify
    with patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden')), \
         patch.object(ModelClient, '_request', side_effect=AssertionError('Real API forbidden')) as transport:
        options = dict(results_root=ROOT / 'results', run_id=args.run_id, client=client)
        first = run_experiment(samples, **options)
        check(first['calls_this_run'] == 600, 'Use a fresh run ID for first acceptance execution')
        folder = Path(first['output_dir'])
        journals = [folder / n for n in ('attempts.jsonl', 'predictions.jsonl', 'run_config.json')]
        journal_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in journals}
        repeat = run_experiment(samples, **options)
        check(repeat['calls_this_run'] == 0 and len(calls) == 600, 'Resume made new calls')
        check(all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in journal_hashes.items()),
              'Resume changed journals')
        transport.assert_not_called()
    rows = lambda name: [json.loads(line) for line in (folder / name).read_text(encoding='utf-8').splitlines()]
    predictions, attempts = rows('predictions.jsonl'), rows('attempts.jsonl')
    check(len(predictions) == len({r['logical_id'] for r in predictions}) == 600, 'Prediction uniqueness')
    check({(r['sample_id'], r['strategy']) for r in predictions} ==
          {(s['id'], strategy) for s, strategy in expected}, 'Prediction coverage')
    check(not first['unresolved_logical_ids'] and first['valid_predictions'] == 600, 'Unresolved requests')
    check(len(attempts) == 1200, 'Expected 1200 attempt events')
    for index, (sid, strategy, prompt_hash) in enumerate(calls):
        start, end = attempts[index * 2:index * 2 + 2]
        prediction = predictions[index]
        check(start['event'] == 'started' and end['event'] == 'completed', 'Event order')
        for row in (start, end, prediction):
            check((row['sample_id'], row['strategy'], row['prompt_sha256']) ==
                  (sid, strategy, prompt_hash), 'Journal identity')
            check(row['logical_id'] == digest([sid, strategy]), 'Logical ID mismatch')
        check(start['attempt_number'] == end['attempt_number'] == prediction['attempts'] == 1,
              'Unexpected retries')
        for row in (end, prediction):
            check(row['status'] == 'success' and row['is_valid'] and row['error'] is None, 'Failed outcome')
            check(row['token_usage'] == dict(prompt_tokens=None, completion_tokens=None, total_tokens=None),
                  'Mock usage must be unknown')
            check(row['model_id'] == 'mock' and row['latency_ms'] >= 0, 'Model/latency metadata')
    config = json.loads((folder / 'run_config.json').read_text())
    check(config['mode'] == config['split'] == 'mock' and config['logical_requests'] == 600,
          'Incorrect run configuration')
    check(config['dataset_sha256'] == digest(samples) and config['prompt_version'] == prompts.PROMPT_VERSION,
          'Dataset/version mismatch')
    # Gold labels are first opened here, after inference and resume have finished.
    converted = convert_predictions(folder / 'predictions.jsonl', ROOT / 'data/processed/test_labels.csv')
    coverage = {}
    for strategy, group in converted.items():
        check(len(group) == 200, 'Evaluation strategy coverage')
        check(all(r['valid_output'] for r in score_samples(group)), 'Evaluation format failure')
        check(Counter(r['pair_id'] for r in group) == pairs, 'Evaluation/input pair mismatch')
        coverage[strategy] = dict(samples=len(group), complete_pairs=100)
        (folder / f'evaluation_{strategy}.json').write_text(json.dumps(group, ensure_ascii=False, indent=2), encoding='utf-8')
    secrets = [v for k, v in os.environ.items() if v and re.search(
        r'key|token|secret|password|credential|authorization', k, re.I)]
    for artifact in folder.glob('*.json*'):
        content = artifact.read_text(encoding='utf-8')
        check(not any(secret in content for secret in secrets), 'Credential exposure in output')
        if artifact.name in {'attempts.jsonl', 'predictions.jsonl', 'run_config.json'}:
            check(not any(s['instruction'] in content or s['environment'] in content for s in samples),
                  'Full input text persisted in journal')
    check(all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in before.items()),
          'Existing source, tests, data or prompts changed')
    summary = dict(mock_only=True, first_run=first, repeat_run=repeat, unique_predictions=600,
                   attempt_events=1200, coverage=coverage, network_calls=0,
                   journals_unchanged_on_repeat=True, protected_files_unchanged=True,
                   prompt_manifest_status=json.loads((ROOT / 'prompts/prompt_manifest.json').read_text())['status'])
    (folder / 'acceptance_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
