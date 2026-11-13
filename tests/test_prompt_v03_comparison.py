"""Windows checkout acceptance; all dispatch fixtures are offline, temporary."""
import builtins
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.model_client import ModelClient, ModelConfig, ModelError
from src.prompt_adapter import official_prompt_builder
from src.prompt_v03_adapter import VerifiedBundle, VERSIONS
from src.run_experiment import load_samples, run_experiment, STRATEGIES
from scripts import prompt_v03_comparison as comparison

ROOT = Path(__file__).resolve().parents[1]


def deny_gold(original):
    def call(path,*args,**kwargs):
        if isinstance(path,(str,bytes,os.PathLike)) and Path(os.fsdecode(path)).name in {'test_labels.csv','ambik_test_900.csv'}:
            raise AssertionError('Test labels forbidden')
        return original(path,*args,**kwargs)
    return call


class GuardedCase(unittest.TestCase):
    def setUp(self):
        for guard in (patch.object(socket.socket,'connect',side_effect=AssertionError('Network forbidden')),
                      patch.object(ModelClient,'_request',side_effect=AssertionError('Real API forbidden')),
                      patch('builtins.open',deny_gold(builtins.open)),
                      patch('io.open',deny_gold(io.open)), patch('os.open',deny_gold(os.open))):
            guard.start()
            self.addCleanup(guard.stop)


class AdapterTests(GuardedCase):
    def test_original_windows_checkout_and_native_default_loader(self):
        sample = load_samples(ROOT/'prompt_v0_3/pilot_10pairs/dev_inputs.csv')[0]
        for variant in VERSIONS:
            with VerifiedBundle(variant) as bundle:
                self.assertEqual(bundle.module.load_examples(),bundle.examples)
                for strategy in STRATEGIES:
                    messages = bundle(comparison.current(sample),strategy)
                    self.assertEqual(messages,bundle.module.build_messages(strategy,**comparison.current(sample)))
                    if variant == 'baseline':
                        self.assertEqual(messages,official_prompt_builder(comparison.current(sample),strategy))
                for record in bundle.audit.values():
                    self.assertEqual(record['lf_sha256'],record['expected_lf_sha256'])
                self.assertTrue(any(r['crlf_count'] for r in bundle.audit.values()))
                # The adapter did not turn off CAI's guard: corrupt its TEMP copy.
                target = bundle.folder/'prompts/few_shot_examples.json'
                target.write_bytes(target.read_bytes()+b' ')
                if variant == 'candidate':
                    with self.assertRaisesRegex(ValueError,'changed since export'):
                        bundle.module.load_examples()

    def test_only_newlines_accepted_all_assets_and_trust_anchors_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)/'bundle'
            shutil.copytree(ROOT/'prompt_v0_3/candidate',base/'candidate')
            for relative in ['prompts/few_shot_examples.json','prompts/zero_shot.txt',
                             'prompts/preview_messages.json','prompts/prompt_manifest.json','prompts.py']:
                target = base/'candidate'/relative
                before = target.read_bytes()
                target.write_bytes(before+b' ')
                with self.assertRaisesRegex(ValueError,'normalized SHA256 mismatch'):
                    VerifiedBundle('candidate',base)
                target.write_bytes(before)
            # The exact LF checkout and CRLF checkout both produce identical messages.
            with VerifiedBundle('candidate',base) as first:
                sample = comparison.current(load_samples(ROOT/'prompt_v0_3/pilot_10pairs/dev_inputs.csv')[0])
                messages = first(sample,'few_shot')
            for target in (base/'candidate').rglob('*'):
                if target.is_file() and target.suffix != '.pyc':
                    target.write_bytes(target.read_bytes().replace(b'\r\n',b'\n'))
            with VerifiedBundle('candidate',base) as second:
                self.assertEqual(messages,second(sample,'few_shot'))

    def test_reject_metadata_and_demonstration_reuse(self):
        with VerifiedBundle('candidate') as bundle:
            e = bundle.examples[0]
            with self.assertRaisesRegex(ValueError,'Only environment'):
                bundle(dict(environment=e['environment'],instruction=e['instruction'],gold_label='CLEAR'),'zero_shot')
            with self.assertRaisesRegex(ValueError,'already a labeled'):
                bundle(dict(environment=e['environment'],instruction=e['instruction']),'few_shot')


class PreparedTests(GuardedCase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = comparison.prepare(Path(self.tmp.name))

    def plan(self,variant):
        return self.folder/variant/'plan.json'

    def fixture(self):
        config = ModelConfig(**comparison.MODEL,api_key='offline-fixture')
        response = replace(ModelClient().classify('fixture'),model_id=config.model,
                           token_usage=dict(prompt_tokens=99,completion_tokens=1,total_tokens=100))
        client = Mock(config=config)
        client.classify = Mock(return_value=response)
        return client,response

    def test_check_only_no_credentials_and_historical_overlap(self):
        with patch.object(ModelConfig,'from_file',side_effect=AssertionError('Credentials forbidden')):
            for variant in VERSIONS:
                report = comparison.check(self.plan(variant))
                self.assertEqual((report['pairs'],report['historical_pairs'],report['previously_unused_pairs']),(85,9,76))
                self.assertFalse((Path(report['output_dir'])/'predictions.jsonl').exists())
        audit = comparison.read_json(self.folder/'development_overlap.json')
        self.assertEqual(len(audit['excluded_demo_pair_ids']),7)
        self.assertFalse(set(audit['excluded_demo_pair_ids']) & {r['pair_id'] for r in audit['pairs']})

    def test_all_1020_mock_messages_and_resume_zero(self):
        for variant in VERSIONS:
            side = self.folder/variant
            rows = comparison.read_json(side/'snapshot/messages.json')
            samples = load_samples(side/'snapshot/dev_inputs.csv')
            client = ModelClient()
            client.classify = Mock(wraps=client.classify)
            options = dict(results_root=Path(self.tmp.name)/'mock-only',run_id=variant,
                split='mock',client=client,builder=comparison.FrozenBuilder(rows),
                prompt_version=VERSIONS[variant])
            # Runner must discard all forbidden current-sample metadata.
            poisoned = [dict(s,gold_label='FORBIDDEN_GOLD',category='FORBIDDEN_CATEGORY',paired_instruction='FORBIDDEN_PAIR') for s in samples]
            result = run_experiment(poisoned,**options)
            self.assertEqual((result['final_predictions'],result['valid_predictions'],result['calls_this_run']),(510,510,510))
            for call,row in zip(client.classify.call_args_list,rows):
                self.assertEqual(call.args[0],row['messages'])
                self.assertNotIn('FORBIDDEN',json.dumps(call.args[0]))
            self.assertEqual(run_experiment(poisoned,**options)['calls_this_run'],0)
            predictions = [json.loads(line) for line in (Path(result['output_dir'])/'predictions.jsonl').read_text().splitlines()]
            self.assertEqual(len({r['logical_id'] for r in predictions}),510)
            self.assertEqual({s:sum(r['strategy']==s for r in predictions) for s in STRATEGIES},{s:170 for s in STRATEGIES})

    def test_real_entrypoint_offline_budget_resume_extension_and_latched_auth_stop(self):
        client,response = self.fixture()
        with patch.object(ModelConfig,'from_file',return_value=client.config), \
             patch.object(comparison,'ModelClient',return_value=client):
            self.assertEqual(comparison.execute(self.plan('candidate'),real_api=True,token_budget=100),3)
            self.assertEqual(client.classify.call_count,1)
            self.assertEqual(comparison.execute(self.plan('candidate'),real_api=True),3)
            self.assertEqual(client.classify.call_count,1)
            self.assertEqual(comparison.execute(self.plan('candidate'),real_api=True,token_budget=60000,extend_token_budget=True),0)
            self.assertEqual(client.classify.call_count,510)
            self.assertEqual(comparison.execute(self.plan('candidate'),real_api=True),0)
            self.assertEqual(client.classify.call_count,510)
            client.classify.return_value = replace(response,status='error',label=None,is_valid=False,
                error=ModelError('permission_denied','offline',False,403))
            self.assertEqual(comparison.execute(self.plan('baseline'),real_api=True),3)
            self.assertEqual(client.classify.call_count,511)
            self.assertEqual(comparison.execute(self.plan('baseline'),real_api=True),3)
            self.assertEqual(client.classify.call_count,511)

    def test_frozen_message_tamper_and_foreign_journal_rejected(self):
        path = self.folder/'candidate/snapshot/messages.json'
        before = path.read_bytes()
        path.write_bytes(before+b' ')
        with self.assertRaisesRegex(ValueError,'Sealed snapshot'):
            comparison.check(self.plan('candidate'))
        path.write_bytes(before)
        report = comparison.check(self.plan('candidate'))
        journal = Path(report['output_dir'])/'predictions.jsonl'
        journal.write_text(json.dumps(dict(run_id='old-experiment'))+'\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Foreign experiment'):
            comparison.check(self.plan('candidate'))

    def test_real_flag_required_before_credentials(self):
        with patch.object(ModelConfig,'from_file',side_effect=AssertionError('Credentials forbidden')):
            with self.assertRaisesRegex(ValueError,'explicit --real-api'):
                comparison.execute(self.plan('baseline'))

    def test_440_successes_resume_only_remaining_70_without_limit(self):
        client,response = self.fixture()
        client.classify.return_value = replace(response,
            token_usage=dict(prompt_tokens=682,completion_tokens=1,total_tokens=683))
        with patch.object(ModelConfig,'from_file',return_value=client.config), \
             patch.object(comparison,'ModelClient',return_value=client), patch('sys.stdout',new_callable=io.StringIO):
            self.assertEqual(comparison.execute(self.plan('candidate'),real_api=True),3)
            self.assertEqual(client.classify.call_count,440)
            before = comparison.check(self.plan('candidate'),no_token_limit=True)
            self.assertEqual((before['final_predictions'],before['valid_predictions'],before['remaining_logical_requests']), (440,440,70))
            self.assertEqual(before['successful_requests_to_skip'],440)
            self.assertEqual(before['historical_stopped_reason'],'token_budget_reached')
            self.assertIsNone(before['effective_stopped_reason'])
            folder = Path(before['output_dir'])
            originals = {p:p.read_bytes() for p in (folder/'attempts.jsonl',folder/'predictions.jsonl',folder/'run_config.json')}
            client.classify.reset_mock()
            self.assertEqual(comparison.execute(self.plan('candidate'),real_api=True,no_token_limit=True),0)
            self.assertEqual(client.classify.call_count,70)
            rows = comparison.read_json(self.folder/'candidate/snapshot/messages.json')
            self.assertEqual([c.args[0] for c in client.classify.call_args_list],[r['messages'] for r in rows[440:]])
            for p,raw in originals.items():
                self.assertTrue(p.read_bytes().startswith(raw))
            self.assertEqual((folder/'run_config.json').read_bytes(),originals[folder/'run_config.json'])
            report = comparison.check(self.plan('candidate'),no_token_limit=True)
            self.assertEqual((report['final_predictions'],report['remaining_logical_requests'],report['known_total_tokens']),(510,0,348330))
            self.assertEqual(comparison.execute(self.plan('candidate'),real_api=True,no_token_limit=True),0)
            self.assertEqual(client.classify.call_count,70)

    def test_no_limit_both_versions_and_conflicts_before_credentials(self):
        with patch.object(ModelConfig,'from_file',side_effect=AssertionError('Credentials forbidden')):
            for kw in (dict(token_budget=500000),dict(extend_token_budget=True)):
                with self.assertRaisesRegex(ValueError,'mutually exclusive'):
                    comparison.execute(self.plan('candidate'),real_api=True,no_token_limit=True,**kw)
        client,_ = self.fixture()
        with patch.object(ModelConfig,'from_file',return_value=client.config), \
             patch.object(comparison,'ModelClient',return_value=client), patch('sys.stdout',new_callable=io.StringIO):
            for variant in VERSIONS:
                self.assertEqual(comparison.execute(self.plan(variant),real_api=True,no_token_limit=True),0)
                report = comparison.check(self.plan(variant),no_token_limit=True)
                self.assertEqual((report['final_predictions'],report['remaining_logical_requests']),(510,0))

    def test_runner_update_requires_scoped_receipt_and_preserves_asset_seals(self):
        plan_path = self.plan('candidate')
        plan = comparison.read_json(plan_path)
        approved = dict(plan['code_raw_sha256'])
        plan['code_raw_sha256']['src/run_experiment.py'] = 'a'*64
        plan_path.write_text(json.dumps(plan),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'authorization required'):
            comparison.check(plan_path,no_token_limit=True)
        receipt_path = plan_path.parent/'no_token_limit_authorization.json'
        receipt = dict(event='authorize_no_token_limit_runner_update',run_id=plan['run_id'],
            plan_sha256=comparison.sha(plan_path),original_code_sha256=plan['code_raw_sha256'],
            approved_code_sha256=approved,required_flag='--no-token-limit',
            authorization='Offline fixture review', evidence_sha256={'development_overlap.json':plan['overlap_sha256']})
        receipt_path.write_text(json.dumps(receipt),encoding='utf-8')
        comparison.check(plan_path,no_token_limit=True)
        with self.assertRaisesRegex(ValueError,'authorization required'):
            comparison.check(plan_path)
        receipt['run_id'] = 'foreign'
        receipt_path.write_text(json.dumps(receipt),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Invalid no-token-limit'):
            comparison.check(plan_path,no_token_limit=True)
        receipt['run_id'] = plan['run_id']
        receipt['evidence_sha256']['development_overlap.json'] = 'b'*64
        receipt_path.write_text(json.dumps(receipt),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'evidence changed'):
            comparison.check(plan_path,no_token_limit=True)


if __name__ == '__main__':
    unittest.main()
