"""Development batch circuit breaker tests; all responses are local fixtures."""
from dataclasses import replace
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.model_client import ModelClient, ModelError
from src.run_experiment import mock_prompt_builder, run_experiment, safety_stop


class DevelopmentSafetyTests(unittest.TestCase):
    def test_logical_budget_blocks_before_calls(self):
        client = ModelClient()
        client.classify = Mock()
        with self.assertRaisesRegex(ValueError, 'limit exceeded'):
            run_experiment([dict(id=str(i), instruction='fixture') for i in range(21)],
                           client=client, max_logical_requests=60)
        client.classify.assert_not_called()

    def test_permission_error_stops_and_resume_sends_nothing(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(socket.socket, 'connect',
                side_effect=AssertionError('Network forbidden')):
            client = ModelClient()
            response = replace(client.classify('fixture'), status='error', is_valid=False,
                label=None, error=ModelError('permission_denied', 'safe', False, 403))
            client.classify = Mock(return_value=response)
            options = dict(client=client, results_root=folder, builder=mock_prompt_builder,
                           stop_on_anomaly=True, max_logical_requests=60)
            samples = [dict(id='a', instruction='fixture')]
            first = run_experiment(samples, **options)
            self.assertEqual(first['calls_this_run'], 1)
            self.assertEqual(first['stopped_reason'], 'authentication_or_permission_error')
            self.assertEqual(first['remaining_logical_requests'], 2)
            self.assertEqual(run_experiment(samples, **options)['calls_this_run'], 0)
            self.assertEqual(client.classify.call_count, 1)
            self.assertEqual(len((Path(first['output_dir'])/'attempts.jsonl').read_text().splitlines()), 2)

    def test_three_rate_limits_stop_before_fourth_call(self):
        with tempfile.TemporaryDirectory() as folder:
            client = ModelClient()
            response = replace(client.classify('fixture'), status='error', is_valid=False,
                label=None, error=ModelError('rate_limit', 'safe', True, 429))
            client.classify = Mock(return_value=response)
            result = run_experiment([dict(id='a', instruction='fixture')], client=client,
                results_root=folder, builder=mock_prompt_builder, stop_on_anomaly=True, sleep=Mock())
            self.assertEqual(result['calls_this_run'], 3)
            self.assertEqual(result['stopped_reason'], 'three_rate_limit_errors')

    def test_usage_and_failure_thresholds(self):
        good = dict(status='success', token_usage=dict(prompt_tokens=100, completion_tokens=2, total_tokens=102))
        self.assertIsNone(safety_stop([good]*60))
        self.assertEqual(safety_stop([dict(status='success', token_usage={})]), 'missing_or_invalid_token_usage')
        self.assertEqual(safety_stop([dict(status='success', token_usage=dict(prompt_tokens=1,
            completion_tokens=65,total_tokens=66))]), 'abnormal_token_usage')
        self.assertEqual(safety_stop([dict(status='error')]*3), 'three_consecutive_unsuccessful_attempts')
