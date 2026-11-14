"""Real collaborator artifacts exercised offline; no transport permitted."""
import csv
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import Mock, patch
from src import prompts
from src.model_client import ModelClient, ModelConfig
from src.run_experiment import load_samples, run_experiment, read_journal, STRATEGIES
from src.prompt_adapter import official_prompt_builder
from src.evaluation_adapter import convert_predictions
from src.evaluate import score_samples, calculate_metrics, run_parser_tests, SAMPLES

ROOT = Path(__file__).resolve().parents[1]

class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        guard = patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden'))
        guard.start()
        self.addCleanup(guard.stop)
        self.samples = load_samples(ROOT / 'data/processed/test_inputs.csv')[:2]

    def test_official_six_requests_resume_and_evaluate(self):
        client = ModelClient()
        client.classify = Mock(wraps=client.classify)
        rows = [dict(r, gold_label='FORBIDDEN_GOLD', paired_instruction='FORBIDDEN_PAIR', category='FORBIDDEN_CATEGORY') for r in self.samples]
        result = run_experiment(rows, results_root=self.tmp.name, client=client)
        self.assertEqual(result['calls_this_run'], 6)
        for index, call in enumerate(client.classify.call_args_list):
            sample, strategy = self.samples[index // 3], STRATEGIES[index % 3]
            messages = call.args[0]
            self.assertEqual(messages, prompts.build_messages(strategy, sample['environment'], sample['instruction']))
            self.assertEqual(json.loads(messages[1]['content']), {k: sample[k] for k in ('environment','instruction')})
            self.assertNotIn('FORBIDDEN', json.dumps(messages))
        folder = Path(result['output_dir'])
        for row in read_journal(folder / 'predictions.jsonl'):
            self.assertEqual(row['token_usage'], dict(prompt_tokens=None, completion_tokens=None, total_tokens=None))
            self.assertEqual(row['raw_text'], 'CLEAR')
        self.assertEqual(run_experiment(rows, results_root=self.tmp.name, client=client)['calls_this_run'], 0)
        converted = convert_predictions(folder / 'predictions.jsonl', ROOT / 'data/processed/test_labels.csv')
        for group in converted.values():
            self.assertEqual(calculate_metrics(score_samples(group))['total_pairs'], 1)

    def test_csv_bom_multiline_and_rejections(self):
        path = Path(self.tmp.name) / 'input.csv'
        with path.open('w',encoding='utf-8-sig',newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['sample_id','pair_id','environment','instruction'])
            writer.writerow(['001','p','room, bowl\non table','use bowl'])
        self.assertEqual(load_samples(path)[0]['environment'], 'room, bowl\non table')
        valid = path.read_text(encoding='utf-8-sig')
        for bad in [valid + valid.splitlines(keepends=True)[1],
                    'sample_id,pair_id,environment,instruction\na,p,,x\n',
                    'sample_id,pair_id,environment,instruction,gold_label\na,p,e,x,CLEAR\n',
                    'sample_id,pair_id,environment,instruction\na,p,e\n',
                    'sample_id,sample_id,environment,instruction\na,p,e,x\n']:
            path.write_text(bad,encoding='utf-8')
            with self.assertRaises((ValueError,csv.Error)):
                load_samples(path)

    def test_builder_boundary_and_invalid_response_mapping(self):
        def builder(sample, strategy):
            self.assertEqual(set(sample), {'environment','instruction'})
            return official_prompt_builder(sample,strategy)
        result = run_experiment([dict(s,label='CLEAR',paired_instruction='forbidden') for s in self.samples],
            builder=builder, client=ModelClient(ModelConfig(mock_response='CLEAR because...')),
            results_root=self.tmp.name)
        converted = convert_predictions(Path(result['output_dir'])/'predictions.jsonl',ROOT/'data/processed/test_labels.csv')
        self.assertTrue(all(r['prediction']=='INVALID' for g in converted.values() for r in score_samples(g)))

    def test_partial_pairs_rejected(self):
        result = run_experiment(self.samples[:1],results_root=self.tmp.name)
        with self.assertRaisesRegex(ValueError,'expected 2'):
            convert_predictions(Path(result['output_dir'])/'predictions.jsonl', ROOT/'data/processed/test_labels.csv')

    def test_manifest_tampering_fails_before_requests(self):
        client = ModelClient()
        client.classify = Mock(wraps=client.classify)
        with patch('src.prompt_adapter.hashlib.sha256') as sha:
            sha.return_value.hexdigest.return_value = 'wrong'
            with self.assertRaisesRegex(ValueError, 'integrity'):
                run_experiment(self.samples, client=client, results_root=self.tmp.name)
        client.classify.assert_not_called()

    def test_request_failure_mapping(self):
        from dataclasses import replace
        from src.model_client import ModelError
        client = ModelClient()
        failure = replace(client.classify('fixture'), status='error', label=None,
                          is_valid=False, error=ModelError('permission_denied', 'safe', False))
        client.classify = Mock(return_value=failure)
        result = run_experiment(self.samples, client=client, results_root=self.tmp.name)
        converted = convert_predictions(Path(result['output_dir'])/'predictions.jsonl', ROOT/'data/processed/test_labels.csv')
        self.assertTrue(all(r['prediction']=='REQUEST_FAILED' for g in converted.values() for r in score_samples(g)))

    def test_wang_manual_contract(self):
        run_parser_tests()
        metrics = calculate_metrics(score_samples(SAMPLES))
        self.assertEqual(metrics['accuracy'], .5)
        self.assertEqual(metrics['valid_output_rate'], .75)
