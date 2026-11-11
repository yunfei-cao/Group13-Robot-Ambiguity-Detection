"""Independent repeat checks. All outputs are temporary; no live transport."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import pilot_round2 as repeat
from src.model_client import ModelClient, ModelConfig, ModelError
from src.run_experiment import STRATEGIES, load_samples, run_experiment


class Round2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for guard in (patch.object(socket.socket, 'connect', side_effect=AssertionError('No network')),
                      patch.object(ModelClient, '_request', side_effect=AssertionError('No real API'))):
            guard.start()
            self.addCleanup(guard.stop)
        self.folder = sorted((repeat.ROOT/'results/pilot').glob('group13-full-pilot-round2-*'))[-1]
        self.manifest = self.folder/'pilot_manifest.json'
        self.plan = repeat.read_json(self.manifest)
        self.rows = repeat.read_json(self.folder/'snapshot/messages.json')
        self.samples = load_samples(repeat.ROOT/self.plan['input'])
        fixture = replace(ModelClient().classify('fixture'), model_id=self.plan['model'],
                          token_usage=dict(prompt_tokens=99, completion_tokens=1, total_tokens=100))
        self.client = SimpleNamespace(config=ModelConfig(mode='qwen', model=self.plan['model'],
            base_url='https://example.invalid/v1', api_key='offline-fixture', allow_real_requests=True),
            classify=Mock(return_value=fixture))
        self.options = dict(results_root=self.tmp.name, run_id=self.plan['run_id'], split='pilot',
            client=self.client, builder=repeat.SnapshotBuilder(self.rows), prompt_version='v0.1-dev',
            stop_on_anomaly=True, max_logical_requests=600, allow_draft_pilot=True,
            snapshot_sha256=repeat.sha(self.manifest), total_token_stop=300000, sleep=lambda _: None)

    def test_check_read_only_no_credentials_or_gold_access(self):
        before = repeat.seals(repeat.ROOT/'results/pilot'/repeat.SOURCE_ID)
        second_before = repeat.seals(self.folder)
        original = Path.open
        def guarded(path, *args, **kwargs):
            if path.name in ('test_labels.csv', 'ambik_test_900.csv'):
                raise AssertionError('Test gold must not be opened')
            return original(path, *args, **kwargs)
        with patch.object(ModelConfig, 'from_file', side_effect=AssertionError('No credentials')), \
                patch.object(Path, 'open', guarded):
            report = repeat.check(self.manifest)
        self.assertEqual(report['matching_prompt_hashes'], 600)
        self.assertEqual(report['strategy_counts'], {s: 200 for s in STRATEGIES})
        self.assertEqual(before, repeat.seals(repeat.ROOT/'results/pilot'/repeat.SOURCE_ID))
        self.assertEqual(second_before, repeat.seals(self.folder))

    def test_600_fresh_dispatches_exact_messages_and_resume_zero(self):
        source_before = repeat.seals(repeat.ROOT/'results/pilot'/repeat.SOURCE_ID)
        summary = run_experiment(self.samples, **self.options)
        self.assertEqual((summary['calls_this_run'], summary['final_predictions'], summary['valid_predictions']),
                         (600, 600, 600))
        for call, row in zip(self.client.classify.call_args_list, self.rows):
            self.assertEqual(call.args[0], row['messages'])
        self.assertEqual(run_experiment(self.samples, **self.options)['calls_this_run'], 0)
        output = Path(summary['output_dir'])
        predictions = [json.loads(s) for s in (output/'predictions.jsonl').read_text().splitlines()]
        self.assertEqual(len({r['logical_id'] for r in predictions}), 600)
        for strategy in STRATEGIES:
            self.assertEqual(sum(r['strategy'] == strategy for r in predictions), 200)
        self.assertEqual({r['run_id'] for r in predictions}, {self.plan['run_id']})
        self.assertEqual(source_before, repeat.seals(repeat.ROOT/'results/pilot'/repeat.SOURCE_ID))

    def test_network_retry_is_extra_attempt_not_extra_logical_job(self):
        success = self.client.classify.return_value
        error = replace(success, status='error', label=None, is_valid=False, token_usage={},
                        error=ModelError('network_error', 'offline', True))
        self.client.classify.side_effect = [error] + [success]*600
        summary = run_experiment(self.samples, **self.options)
        self.assertEqual((summary['calls_this_run'], summary['final_predictions']), (601, 600))
        self.assertEqual(summary['unknown_usage_attempts'], 1)

    def test_foreign_round1_journal_and_tampered_messages_rejected(self):
        root = Path(self.tmp.name)
        source = repeat.ROOT/'results/pilot'/repeat.SOURCE_ID
        shutil.copytree(source, root/'results/pilot'/repeat.SOURCE_ID)
        cloned = root/'results/pilot'/self.plan['run_id']
        shutil.copytree(self.folder, cloned)
        # Completed real runs contain absolute output paths. Rebase only this
        # temporary fixture so the existing foreign-state checks remain active.
        summary_path = cloned/'execution_summaries.jsonl'
        if summary_path.exists():
            summaries = [json.loads(line) for line in summary_path.read_text().splitlines()]
            for summary in summaries:
                summary['output_dir'] = str(cloned.resolve())
            summary_path.write_text(''.join(json.dumps(row)+'\n' for row in summaries), encoding='utf-8')
        with patch.object(repeat, 'ROOT', root), patch.object(repeat, 'check_source'):
            repeat.check(cloned/'pilot_manifest.json')
            shutil.copyfile(source/'predictions.jsonl', cloned/'predictions.jsonl')
            with self.assertRaisesRegex(ValueError, 'Foreign journal'):
                repeat.check(cloned/'pilot_manifest.json')
            (cloned/'predictions.jsonl').unlink()
            with (cloned/'snapshot/messages.json').open('ab') as stream:
                stream.write(b' ')
            with self.assertRaisesRegex(ValueError, 'snapshot differs'):
                repeat.check(cloned/'pilot_manifest.json')

    def test_run_requires_manual_flags_before_client_creation(self):
        with patch.object(ModelConfig, 'from_file', side_effect=AssertionError('No client')):
            with self.assertRaisesRegex(ValueError, 'authorization flags'):
                repeat.execute(SimpleNamespace(real_api=False, allow_draft_pilot=True, stop_on_anomaly=True))


if __name__ == '__main__':
    unittest.main()
