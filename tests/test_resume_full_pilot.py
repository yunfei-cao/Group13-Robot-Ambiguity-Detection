"""Recovery simulations use temporary copies, fake responses and blocked networking."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import full_pilot, resume_full_pilot as recovery
from src.model_client import ModelClient, ModelConfig, ModelError
from src.run_experiment import digest, load_samples, safety_stop


RUN_ID = 'group13-full-pilot-qwen200-20261111T090248Z-76877106'
SOURCE = full_pilot.ROOT


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.folder = self.root/'results/pilot'/RUN_ID
        shutil.copytree(SOURCE/'results/pilot'/RUN_ID, self.folder)
        # Allow rerunning this suite after a user execution without carrying its
        # appended records into the original 32-final offline fixture.
        self.receipt = full_pilot.read_json(self.folder/'reviewed_failure_authorization.json')
        for name, seal in self.receipt['journal_prefixes'].items():
            path = self.folder/name
            path.write_bytes(path.read_bytes()[:seal['bytes']])
        (self.folder/'recovery_events.jsonl').unlink(missing_ok=True)
        plan = full_pilot.read_json(self.folder/'pilot_manifest.json')
        for relative in plan['source_sha256']:
            target = self.root/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            # Historical recovery uses its sealed runner, independently of
            # later workspace runner updates.
            shutil.copyfile(self.folder/'snapshot'/relative, target)
        for context in (patch.object(full_pilot, 'ROOT', self.root),
                        patch.object(recovery, 'ROOT', self.root),
                        patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden')),
                        patch.object(ModelClient, '_request', side_effect=AssertionError('API forbidden'))):
            context.start()
            self.addCleanup(context.stop)
        self.runner = recovery.load_runner()
        fixture = replace(ModelClient().classify('offline'), model_id='qwen-plus-2025-12-01',
                          token_usage=dict(prompt_tokens=99, completion_tokens=1, total_tokens=100))
        config = ModelConfig.from_file(self.folder/'model_config.json',
                                      environ={'DASHSCOPE_API_KEY': 'offline-fixture'})
        self.client = SimpleNamespace(config=config, classify=Mock(return_value=fixture))
        self.before = {name: (self.folder/name).read_bytes() for name in recovery.JOURNALS}

    def validate(self):
        return recovery.validate(self.folder, self.receipt, self.runner)

    def run_recovery(self, authorized=True):
        return recovery.run_recovery(self.folder, self.receipt, self.runner,
                                     self.client, authorized=authorized, sleep=lambda _: None)

    def assert_history(self):
        for name, before in self.before.items():
            self.assertTrue((self.folder/name).read_bytes().startswith(before), name)

    def error_response(self, code='network_error', status=None):
        return replace(self.client.classify.return_value, status='error', label=None,
                       is_valid=False, token_usage={}, error=ModelError(code, 'offline', True, status))

    def test_check_only_has_no_client_credentials_or_writes(self):
        with patch.object(sys, 'argv', ['resume', '--manifest', str(self.folder/'pilot_manifest.json'), '--check-only']), \
                patch.object(recovery, 'ModelClient') as client, \
                patch.object(ModelConfig, 'from_file') as config:
            self.assertEqual(recovery.main(), 0)
        client.assert_not_called()
        config.assert_not_called()
        self.assertEqual(self.validate()['remaining_logical_requests'], 568)
        for name, before in self.before.items():
            self.assertEqual((self.folder/name).read_bytes(), before)
        self.assertFalse((self.folder/'recovery_events.jsonl').exists())

    def test_unauthorized_cli_never_constructs_client(self):
        with patch.object(sys, 'argv', ['resume', '--manifest', str(self.folder/'pilot_manifest.json')]), \
                patch.object(recovery, 'ModelClient') as client, \
                patch.object(ModelConfig, 'from_file') as config:
            with self.assertRaisesRegex(ValueError, 'Explicit'):
                recovery.main()
        client.assert_not_called()
        config.assert_not_called()
        self.assert_history()

    def test_unauthorized_function_never_calls_api_or_writes(self):
        with self.assertRaisesRegex(ValueError, 'Explicit'):
            self.run_recovery(authorized=False)
        self.client.classify.assert_not_called()
        self.assertFalse((self.folder/'recovery_events.jsonl').exists())
        self.assert_history()

    def test_original_entry_stays_latched(self):
        plan = full_pilot.read_json(self.folder/'pilot_manifest.json')
        summary = self.runner.run_experiment(load_samples(self.root/plan['input']),
            results_root=self.root/'results', run_id=RUN_ID, split='pilot', client=self.client,
            builder=full_pilot.SnapshotBuilder(full_pilot.read_json(self.folder/'snapshot/messages.json')),
            prompt_version=plan['prompt_version'], stop_on_anomaly=True,
            max_logical_requests=600, allow_draft_pilot=True,
            snapshot_sha256=full_pilot.sha(self.folder/'pilot_manifest.json'), total_token_stop=300000)
        self.assertEqual(summary['calls_this_run'], 0)
        self.assertEqual(summary['stopped_reason'], recovery.REASON)
        self.client.classify.assert_not_called()

    def test_authorized_568_only_preserves_all_history_and_one_failed_few_shot(self):
        historical = recovery.strict_rows(self.folder/'predictions.jsonl')
        failed = [r for r in historical if not r['is_valid']]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]['strategy'], 'few_shot')
        summary = self.run_recovery()
        self.assertEqual((summary['calls_this_run'], summary['final_predictions'],
                          summary['valid_predictions'], summary['remaining_logical_requests']), (568, 600, 599, 0))
        self.assertIsNone(summary['stopped_reason'])
        self.assertEqual(summary['known_token_usage']['total_tokens'], 12934+56800)
        finals = recovery.strict_rows(self.folder/'predictions.jsonl')
        self.assertEqual(len({r['logical_id'] for r in finals}), 600)
        self.assertEqual(finals[:32], historical)
        new = recovery.strict_rows(self.folder/'attempts.jsonl')[90:]
        self.assertTrue({r['logical_id'] for r in new}.isdisjoint({r['logical_id'] for r in historical}))
        events = recovery.strict_rows(self.folder/'recovery_events.jsonl')
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['authorization_sha256'], digest(self.receipt))
        self.assertEqual(self.run_recovery()['calls_this_run'], 0)
        self.assertEqual(len(recovery.strict_rows(self.folder/'recovery_events.jsonl')), 1)
        self.assert_history()

    def test_new_three_network_failures_stop_and_cannot_reset_on_rerun(self):
        self.client.classify.return_value = self.error_response()
        summary = self.run_recovery()
        self.assertEqual((summary['calls_this_run'], summary['final_predictions'], summary['stopped_reason']),
                         (3, 33, recovery.REASON))
        with self.assertRaisesRegex(ValueError, 'Safety stop remains active'):
            self.run_recovery()
        self.assertEqual(self.client.classify.call_count, 3)
        self.assert_history()

    def test_new_permission_error_stops_immediately(self):
        self.client.classify.return_value = self.error_response('permission_denied', 403)
        summary = self.run_recovery()
        self.assertEqual((summary['calls_this_run'], summary['stopped_reason']),
                         (1, 'authentication_or_permission_error'))
        with self.assertRaisesRegex(ValueError, 'Safety stop remains active'):
            self.run_recovery()
        self.assertEqual(self.client.classify.call_count, 1)

    def test_new_missing_usage_stops_immediately(self):
        self.client.classify.return_value = replace(self.client.classify.return_value, token_usage={})
        summary = self.run_recovery()
        self.assertEqual((summary['calls_this_run'], summary['stopped_reason']),
                         (1, 'missing_or_invalid_token_usage'))

    def test_new_abnormal_usage_stops_immediately(self):
        self.client.classify.return_value = replace(self.client.classify.return_value,
            token_usage=dict(prompt_tokens=99, completion_tokens=65, total_tokens=164))
        summary = self.run_recovery()
        self.assertEqual((summary['calls_this_run'], summary['stopped_reason']), (1, 'abnormal_token_usage'))

    def test_cumulative_budget_includes_history_and_cannot_resume(self):
        self.client.classify.return_value = replace(self.client.classify.return_value,
            token_usage=dict(prompt_tokens=16000, completion_tokens=1, total_tokens=16001))
        summary = self.run_recovery()
        self.assertEqual((summary['calls_this_run'], summary['stopped_reason']), (18, 'token_budget_reached'))
        self.assertEqual(summary['known_token_usage']['total_tokens'], 12934+18*16001)
        with self.assertRaisesRegex(ValueError, 'Safety stop remains active'):
            self.run_recovery()
        self.assertEqual(self.client.classify.call_count, 18)

    def test_historical_guards_even_after_reviewed_failure_stop(self):
        rows = [dict(status='error', error={'code': 'network_error'}) for _ in range(3)]
        guards = [dict(status='error', error={'http_status': 401}),
                  dict(status='success', token_usage={}),
                  dict(status='success', token_usage=dict(prompt_tokens=16001, completion_tokens=1, total_tokens=16002))]
        for row, reason in zip(guards, ('authentication_or_permission_error',
                                      'missing_or_invalid_token_usage', 'abnormal_token_usage')):
            with self.subTest(reason=reason):
                self.assertEqual(recovery.recovery_stop(rows+[row], 4, safety_stop), reason)
        rates = [dict(status='error', error={'http_status': 429}) for _ in range(3)]
        self.assertEqual(recovery.recovery_stop(rates, 3, safety_stop), 'three_rate_limit_errors')
        tokens = [dict(status='success', token_usage=dict(prompt_tokens=16000, completion_tokens=1, total_tokens=16001))]*19
        self.assertEqual(recovery.recovery_stop(tokens, 19, safety_stop), 'token_budget_reached')

    def test_post_boundary_failures_accumulate_and_success_resets(self):
        bad = dict(status='error', token_usage={}, error={'code': 'network_error'})
        good = dict(status='success', token_usage=dict(prompt_tokens=99, completion_tokens=1, total_tokens=100))
        old = [bad]*3
        for tail in ([bad], [bad]*2, [bad]*2+[good]+[bad]*2):
            self.assertIsNone(recovery.recovery_stop(old+tail, 3, safety_stop))
        self.assertEqual(recovery.recovery_stop(old+[bad]*3+[good], 3, safety_stop), recovery.REASON)

    def test_tampered_journal_prefixes_rejected_without_modification(self):
        for name in recovery.JOURNALS:
            with self.subTest(name=name):
                path = self.folder/name
                original = path.read_bytes()
                tampered = b' '+original[1:]
                path.write_bytes(tampered)
                with self.assertRaisesRegex(ValueError, 'prefix changed'):
                    self.run_recovery()
                self.assertEqual(path.read_bytes(), tampered)
                path.write_bytes(original)
        self.client.classify.assert_not_called()

    def test_torn_tail_never_truncated(self):
        for name in recovery.JOURNALS:
            with self.subTest(name=name):
                path = self.folder/name
                original = path.read_bytes()
                path.write_bytes(original+b'{')
                with self.assertRaisesRegex(ValueError, 'Incomplete journal'):
                    self.run_recovery()
                self.assertEqual(path.read_bytes(), original+b'{')
                path.write_bytes(original)
        self.client.classify.assert_not_called()

    def test_duplicate_finals_rejected(self):
        path = self.folder/'predictions.jsonl'
        path.write_bytes(path.read_bytes()+self.before['predictions.jsonl'].splitlines(keepends=True)[0])
        with self.assertRaisesRegex(ValueError, 'Duplicate finals'):
            self.run_recovery()
        self.client.classify.assert_not_called()

    def test_authorization_or_boundary_changes_rejected(self):
        original = dict(self.receipt)
        for change in ({'authorization': ''}, {'reviewed_completions': 44}, {'run_id': 'other'},
                       {'journal_prefixes': {}}, {'authorization': 'changed'}):
            self.receipt = dict(original, **change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_recovery()
        self.client.classify.assert_not_called()

    def test_budget_change_or_config_change_rejected(self):
        path = self.folder/'run_config.json'
        config = full_pilot.read_json(path)
        config['safety_policy']['total_token_stop'] = 400000
        path.write_text(json.dumps(config), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'configuration or budget'):
            self.run_recovery()
        with self.assertRaisesRegex(ValueError, 'fixed at 300000'):
            recovery.recovery_stop([], 45, safety_stop, 400000)
        self.client.classify.assert_not_called()

    def test_snapshot_change_rejected_under_lock(self):
        path = self.folder/'snapshot/messages.json'
        path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError, 'snapshot changed'):
            self.run_recovery()
        self.client.classify.assert_not_called()

    def test_recovery_event_tampering_or_torn_tail_blocks_check(self):
        path = self.folder/'recovery_events.jsonl'
        path.write_text('{"event":"different"}\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'audit mismatch'):
            self.validate()
        path.write_text('{', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Incomplete journal'):
            self.validate()


if __name__ == '__main__':
    unittest.main()
