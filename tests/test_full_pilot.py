"""Pilot authorization/budget tests; only offline response fixtures are used."""
from dataclasses import replace
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.full_pilot import SnapshotBuilder
from src.model_client import ModelClient, ModelConfig, ModelError
from src.run_experiment import digest, run_experiment


class FullPilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        guard = patch.object(socket.socket, 'connect', side_effect=AssertionError('No network'))
        guard.start()
        self.addCleanup(guard.stop)
        transport = patch.object(ModelClient, '_request', side_effect=AssertionError('No API'))
        transport.start()
        self.addCleanup(transport.stop)
        fixture = replace(ModelClient().classify('offline fixture'),model_id='qwen-plus-2025-12-01',
            token_usage=dict(prompt_tokens=99,completion_tokens=1,total_tokens=100))
        self.client = SimpleNamespace(config=ModelConfig(mode='qwen',model='qwen-plus-2025-12-01',
            base_url='https://example.invalid/v1',api_key='offline-fixture',allow_real_requests=True),
            classify=Mock(return_value=fixture))
        self.samples = [dict(id='offline',environment='fixture room',instruction='fixture instruction')]
        self.options = dict(results_root=self.tmp.name,run_id='pilot-test',split='pilot',client=self.client,
            builder=lambda sample,strategy: 'offline '+strategy,prompt_version='v0.1-dev',
            stop_on_anomaly=True,max_logical_requests=3,allow_draft_pilot=True,snapshot_sha256='a'*64)

    def run_job(self, **kw):
        return run_experiment(self.samples,**dict(self.options,**kw))

    def test_pilot_requires_authorization_snapshot_and_stops(self):
        for overrides in (dict(allow_draft_pilot=False),dict(snapshot_sha256=None),dict(stop_on_anomaly=False)):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.run_job(**overrides)
        self.client.classify.assert_not_called()

    def test_pilot_flag_cannot_bypass_formal_or_dev(self):
        for split in ('formal','dev'):
            with self.assertRaisesRegex(ValueError,'cannot bypass'):
                self.run_job(split=split)
        self.client.classify.assert_not_called()

    def test_formal_draft_rejected_without_pilot_options(self):
        with patch('src.run_experiment.Path.read_text',return_value='{"status":"DRAFT"}'):
            with self.assertRaisesRegex(ValueError,'frozen prompts'):
                self.run_job(split='formal',allow_draft_pilot=False,snapshot_sha256=None)
        self.client.classify.assert_not_called()

    def test_budget_stop_resume_extension_does_not_repeat_completed_jobs(self):
        first = self.run_job(total_token_stop=200)
        self.assertEqual((first['calls_this_run'],first['final_predictions'],first['stopped_reason']),
                         (2,2,'token_budget_reached'))
        self.assertEqual(self.run_job(total_token_stop=200)['calls_this_run'],0)
        with self.assertRaises(ValueError):
            self.run_job(total_token_stop=400)
        resumed = self.run_job(total_token_stop=400,extend_token_budget=True)
        self.assertEqual((resumed['calls_this_run'],resumed['final_predictions'],resumed['stopped_reason']),(1,3,None))
        self.assertEqual(resumed['known_token_usage']['total_tokens'],300)
        self.assertEqual(self.run_job(total_token_stop=400)['calls_this_run'],0)
        folder = Path(self.tmp.name)/'pilot/pilot-test'
        rows = [json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines()]
        self.assertEqual(len({r['logical_id'] for r in rows}),3)
        events = [json.loads(line) for line in (folder/'budget_events.jsonl').read_text().splitlines()]
        self.assertEqual([(r['old_limit'],r['new_limit']) for r in events],[(200,400)])
        self.assertEqual(json.loads((folder/'run_config.json').read_text())['safety_policy']['total_token_stop'],400)

    def test_budget_extension_cannot_change_other_settings(self):
        self.run_job(total_token_stop=200)
        path=Path(self.tmp.name)/'pilot/pilot-test/run_config.json'
        before=path.read_bytes()
        with self.assertRaises(ValueError):
            self.run_job(total_token_stop=400,extend_token_budget=True,prompt_version='changed')
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(self.client.classify.call_count,2)

    def test_budget_extension_rejects_first_run_equal_or_lower_limit(self):
        with self.assertRaises(ValueError):
            self.run_job(total_token_stop=400,extend_token_budget=True)
        self.run_job(total_token_stop=200)
        for budget in (200,100):
            with self.assertRaises(ValueError):
                self.run_job(total_token_stop=budget,extend_token_budget=True)
        self.assertEqual(self.client.classify.call_count,2)

    def test_budget_extension_cannot_clear_permission_stop(self):
        self.client.classify.return_value = replace(self.client.classify.return_value,status='error',label=None,
            is_valid=False,token_usage={},error=ModelError('permission_denied','fixture',False,403))
        self.assertEqual(self.run_job(total_token_stop=200)['stopped_reason'],'authentication_or_permission_error')
        resumed=self.run_job(total_token_stop=400,extend_token_budget=True)
        self.assertEqual(resumed['stopped_reason'],'authentication_or_permission_error')
        self.assertEqual(resumed['calls_this_run'],0)
        self.assertEqual(self.client.classify.call_count,1)

    def test_snapshot_builder_whitelist_and_independent_messages(self):
        current = dict(environment='fixture room',instruction='fixture instruction')
        messages = [dict(role='system',content='fixture'),dict(role='user',content=json.dumps(current))]
        builder = SnapshotBuilder([dict(strategy='zero_shot',messages=messages)])
        first = builder(current,'zero_shot')
        first[0]['content']='mutated'
        self.assertEqual(builder(current,'zero_shot'),messages)
        with self.assertRaises(ValueError):
            builder(dict(current,gold_label='FORBIDDEN'),'zero_shot')
        with self.assertRaises(KeyError):
            builder(dict(current,instruction='different'),'zero_shot')

    def test_no_limit_preserves_config_journal_prefix_and_usage(self):
        self.run_job(total_token_stop=200)
        folder = Path(self.tmp.name)/'pilot/pilot-test'
        originals = {name: (folder/name).read_bytes() for name in
                     ('run_config.json', 'attempts.jsonl', 'predictions.jsonl')}
        result = self.run_job(no_token_limit=True)
        self.assertEqual((result['calls_this_run'], result['final_predictions'], result['stopped_reason']), (1,3,None))
        self.assertIsNone(result['total_token_stop'])
        self.assertEqual(result['known_token_usage']['total_tokens'],300)
        self.assertEqual((folder/'run_config.json').read_bytes(), originals['run_config.json'])
        for name in ('attempts.jsonl', 'predictions.jsonl'):
            self.assertTrue((folder/name).read_bytes().startswith(originals[name]))
        event = json.loads((folder/'budget_events.jsonl').read_text().splitlines()[0])
        self.assertEqual((event['event'],event['original_limit'],event['remaining_logical_requests']),
                         ('token_limit_disabled',200,1))
        self.assertEqual(event['opt_in_flag'],'no_token_limit=True')
        self.assertEqual(event['journal_prefixes']['attempts.jsonl']['bytes'],len(originals['attempts.jsonl']))
        self.assertEqual(self.run_job(no_token_limit=True)['calls_this_run'],0)
        # Opt-in is invocation-scoped; removing it restores the original budget.
        self.assertEqual(self.run_job(total_token_stop=200)['stopped_reason'],'token_budget_reached')

    def test_no_limit_new_run_and_conflicting_flags(self):
        with self.assertRaises(ValueError):
            self.run_job(no_token_limit=True,extend_token_budget=True)
        self.client.classify.assert_not_called()
        self.assertEqual(self.run_job(no_token_limit=True)['calls_this_run'],3)
        self.assertEqual(self.run_job(no_token_limit=True)['calls_this_run'],0)
        with self.assertRaises(ValueError):
            self.run_job()

    def test_no_limit_rejects_other_configuration_changes(self):
        self.run_job(total_token_stop=200)
        with self.assertRaises(ValueError):
            self.run_job(no_token_limit=True,prompt_version='changed')
        self.assertEqual(self.client.classify.call_count,2)

    def test_no_limit_never_truncates_incomplete_history(self):
        self.run_job(total_token_stop=200)
        path = Path(self.tmp.name)/'pilot/pilot-test/attempts.jsonl'
        with path.open('ab') as stream:
            stream.write(b'{"event":')
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError,'will not truncate history'):
            self.run_job(no_token_limit=True)
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(self.client.classify.call_count,2)

    def test_no_limit_keeps_all_safety_stops_and_retry(self):
        from src.run_experiment import safety_stop
        success = dict(status='success', token_usage=dict(prompt_tokens=99,completion_tokens=1,total_tokens=100))
        cases = [
            ([dict(status='error',error={'http_status':403})], 'authentication_or_permission_error'),
            ([dict(status='error',error={'code':'authentication_error'})], 'authentication_or_permission_error'),
            ([dict(status='error',error={'code':'rate_limit'})]*3, 'three_rate_limit_errors'),
            ([dict(status='error')]*3, 'three_consecutive_unsuccessful_attempts'),
            ([dict(status='success',token_usage={})], 'missing_or_invalid_token_usage'),
            ([dict(status='success',token_usage=dict(prompt_tokens=16001,completion_tokens=1,total_tokens=16002))], 'abnormal_token_usage')]
        for rows,reason in cases:
            # Earlier budget crossing must not mask later historical safety stops.
            with self.subTest(reason=reason):
                self.assertEqual(safety_stop([success]*3001+rows,None),reason)
        self.client.classify.side_effect = [
            replace(self.client.classify.return_value,status='error',label=None,is_valid=False,token_usage={},
                    error=ModelError('network_error','fixture',True)),
            self.client.classify.return_value, self.client.classify.return_value, self.client.classify.return_value]
        delays = Mock()
        result = self.run_job(no_token_limit=True,sleep=delays)
        self.assertEqual((result['calls_this_run'],result['valid_predictions']),(4,3))
        delays.assert_called_once_with(2)
        self.assertEqual(result['unknown_usage_attempts'],1)

    def test_no_limit_historical_permission_stop_stays_latched(self):
        self.client.classify.return_value = replace(self.client.classify.return_value,status='error',label=None,
            is_valid=False,token_usage={},error=ModelError('permission_denied','fixture',False,403))
        self.run_job(total_token_stop=200)
        result = self.run_job(no_token_limit=True)
        self.assertEqual((result['calls_this_run'],result['stopped_reason']),(0,'authentication_or_permission_error'))

    def test_no_limit_new_safety_stops_stay_latched_on_resume(self):
        success = self.client.classify.return_value
        cases = [
            (replace(success,status='error',label=None,is_valid=False,token_usage={},
                     error=ModelError('network_error','fixture',True)),3,'three_consecutive_unsuccessful_attempts'),
            (replace(success,status='error',label=None,is_valid=False,token_usage={},
                     error=ModelError('rate_limit','fixture',True,429)),3,'three_rate_limit_errors'),
            (replace(success,token_usage={}),1,'missing_or_invalid_token_usage'),
            (replace(success,token_usage=dict(prompt_tokens=99,completion_tokens=65,total_tokens=164)),1,'abnormal_token_usage')]
        for index,(response,calls,reason) in enumerate(cases):
            with self.subTest(reason=reason):
                self.client.classify.reset_mock()
                self.client.classify.return_value = response
                opts = dict(run_id='safety-'+str(index),no_token_limit=True,sleep=lambda _: None)
                result = self.run_job(**opts)
                self.assertEqual((result['calls_this_run'],result['stopped_reason']),(calls,reason))
                self.assertEqual(self.run_job(**opts)['calls_this_run'],0)
                self.assertEqual(self.client.classify.call_count,calls)


if __name__=='__main__':
    unittest.main()
