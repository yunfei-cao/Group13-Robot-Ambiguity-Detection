"""Runner contract tests: no network access permitted."""
from dataclasses import replace
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.model_client import ModelClient, ModelConfig, ModelError, ModelClientError
from src.run_experiment import (STRATEGIES, load_samples, read_journal, run_experiment,
                                run_lock)

SAMPLES = [{"id": "mock-1", "instruction": "Move the test cube."},
           {"id": "mock-2", "instruction": "Move that one."}]


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.client = ModelClient()
        self.client.classify = Mock(wraps=self.client.classify)
        self.wait = Mock()
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("No network"))
        guard.start()
        self.addCleanup(guard.stop)

    def run_job(self, samples=None, **kw):
        options = dict(results_root=self.root, client=self.client, sleep=self.wait)
        options.update(kw)
        return run_experiment(SAMPLES if samples is None else samples, **options)

    def rows(self, name):
        return read_journal(self.root / "mock" / "mock-demo" / name)

    def test_six_calls_three_strategies_and_repeat(self):
        result = self.run_job()
        self.assertEqual(result["calls_this_run"], 6)
        self.assertEqual(self.client.classify.call_count, 6)
        predictions = self.rows("predictions.jsonl")
        self.assertEqual(len(predictions), 6)
        self.assertEqual({p["strategy"] for p in predictions}, set(STRATEGIES))
        self.assertEqual(self.run_job()["calls_this_run"], 0)
        self.assertEqual(len(self.rows("predictions.jsonl")), 6)

    def test_log_schema(self):
        self.run_job()
        rows = self.rows("attempts.jsonl")
        self.assertEqual(len(rows), 12)
        finished = [r for r in rows if r["event"] == "completed"]
        for row in finished:
            self.assertTrue({"status", "latency_ms", "token_usage", "error", "retry_count",
                             "attempt_number", "model_id", "prompt_sha256"} <= row.keys())
        config = json.loads((self.root / "mock/mock-demo/run_config.json").read_text())
        self.assertEqual(config["logical_requests"], 6)
        self.assertEqual(config["retry_delays_seconds"], [2, 4])

    def test_independent_messages(self):
        shared = [{"role": "user", "content": "fixture"}]
        def classify(prompt):
            self.assertEqual(len(prompt), 1)
            prompt.append({"role": "assistant", "content": "previous"})
            return ModelClient().classify("fixture")
        self.client.classify.side_effect = classify
        self.run_job(builder=lambda sample, strategy: shared)
        self.assertEqual(len(shared), 1)

    def failure(self, retryable=True):
        return replace(ModelClient().classify("fixture"), status="error", label=None,
                       is_valid=False, error=ModelError("timeout", "safe", retryable))

    def test_retry_then_success(self):
        good = ModelClient().classify("fixture")
        self.client.classify.side_effect = [self.failure(), self.failure(), good, good, good]
        self.run_job(SAMPLES[:1])
        self.assertEqual(self.client.classify.call_count, 5)
        self.assertEqual([c.args[0] for c in self.wait.call_args_list], [2, 4])
        self.assertEqual(self.rows("predictions.jsonl")[0]["attempts"], 3)

    def test_retry_limit_and_failed_final_is_skipped(self):
        self.client.classify.return_value = self.failure()
        self.client.classify.side_effect = None
        self.run_job(SAMPLES[:1])
        self.assertEqual(self.client.classify.call_count, 9)
        self.assertEqual(self.run_job(SAMPLES[:1])["calls_this_run"], 0)

    def test_nonretryable(self):
        self.client.classify.side_effect = None
        self.client.classify.return_value = self.failure(False)
        self.run_job(SAMPLES[:1])
        self.assertEqual(self.client.classify.call_count, 3)
        self.wait.assert_not_called()

    def test_invalid_label_no_retry(self):
        self.client = ModelClient(ModelConfig(mock_response="invalid"))
        self.run_job(SAMPLES[:1])
        self.assertTrue(all(r["status"] == "invalid_output" for r in self.rows("predictions.jsonl")))
        self.wait.assert_not_called()

    def test_raised_client_error(self):
        self.client.classify.side_effect = ModelClientError("safe", code="timeout", retryable=True)
        self.run_job(SAMPLES[:1])
        self.assertEqual(self.client.classify.call_count, 9)

    def test_resume_completed_without_prediction(self):
        self.run_job()
        path = self.root / "mock/mock-demo/predictions.jsonl"
        path.write_bytes(b"")
        self.assertEqual(self.run_job()["calls_this_run"], 0)
        self.assertEqual(len(self.rows("predictions.jsonl")), 6)

    def test_interrupted_request_unknown_not_replayed(self):
        self.client.classify.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.run_job()
        self.client.classify.side_effect = None
        self.assertEqual(self.run_job()["calls_this_run"], 5)
        self.assertEqual(self.rows("predictions.jsonl")[0]["status"], "interrupted")
        result = self.run_job()
        self.assertEqual(result["valid_predictions"], 5)
        self.assertEqual(len(result["unresolved_logical_ids"]), 1)
        self.assertEqual(result["calls_this_run"], 0)

    def test_partial_tail_repair(self):
        self.run_job()
        for name in ("attempts.jsonl", "predictions.jsonl"):
            with (self.root / "mock/mock-demo" / name).open("ab") as stream:
                stream.write(b'{"partial":')
        self.assertEqual(self.run_job()["calls_this_run"], 0)
        self.assertEqual(len(self.rows("predictions.jsonl")), 6)

    def test_interior_corruption_rejected(self):
        path = self.root / "bad.jsonl"
        path.write_bytes(b'{bad}\n{}\n')
        with self.assertRaises(ValueError):
            read_journal(path)

    def test_config_and_dataset_mismatch(self):
        self.run_job()
        with self.assertRaises(ValueError):
            self.run_job(prompt_version="changed")
        with self.assertRaises(ValueError):
            self.run_job(SAMPLES[:1])

    def test_sensitive_data_redacted_everywhere(self):
        self.client = ModelClient(ModelConfig(api_key="private-value", mock_response="private-value Bearer leaked"))
        with patch.dict("os.environ", {"OTHER_SERVICE_TOKEN": "other-value"}):
            self.run_job([{"id": "other-value", "instruction": "private-value"}])
        for path in (self.root / "mock/mock-demo").glob("*.json*"):
            content = path.read_text()
            for secret in ("private-value", "other-value", "leaked"):
                self.assertNotIn(secret, content)

    def test_output_separation(self):
        for split in ("dev", "formal"):
            with self.assertRaises(ValueError):
                self.run_job(split=split)

    def test_input_validation(self):
        for samples in ([], SAMPLES + SAMPLES, [{"id": "a", "instruction": ""}],
                        [{"id": "a", "instruction": "x", "label": "wrong"}]):
            with self.assertRaises(ValueError):
                self.run_job(samples)
        path = self.root / "samples.jsonl"
        path.write_text('\n'.join(json.dumps(s) for s in SAMPLES), encoding="utf-8")
        self.assertEqual(load_samples(path), SAMPLES)

    def test_concurrent_lock(self):
        with run_lock(self.root):
            with self.assertRaises(RuntimeError):
                with run_lock(self.root):
                    pass

    def test_duplicate_predictions_rejected(self):
        self.run_job()
        path = self.root / "mock/mock-demo/predictions.jsonl"
        with path.open("ab") as stream:
            stream.write(path.read_bytes().splitlines(keepends=True)[0])
        with self.assertRaises(ValueError):
            self.run_job()

    def test_real_client_requires_external_builder(self):
        client = ModelClient(ModelConfig(mode="qwen", model="qwen-plus-2025-12-01",
                             base_url="https://example.invalid/v1", api_key="test-key",
                             allow_real_requests=True))
        with self.assertRaises(ValueError):
            self.run_job(client=client, split="dev", prompt_version="approved-v1")

    def test_real_client_rejects_default_prompt_version(self):
        client = ModelClient(ModelConfig(mode="qwen", model="qwen-plus-2025-12-01",
                             base_url="https://example.invalid/v1", api_key="test-key",
                             allow_real_requests=True))
        with self.assertRaisesRegex(ValueError, "non-mock prompt version"):
            self.run_job(client=client, split="dev", builder=lambda s, t: "fixture")

    def test_csv_rejected_with_mapping_guidance(self):
        path = self.root / "test_inputs.csv"
        path.write_text("unknown,columns\na,b\n")
        with self.assertRaisesRegex(ValueError, "column mapping"):
            load_samples(path)

    def test_quoted_headers_and_connection_credentials_redacted(self):
        for index, raw in enumerate(('{"Authorization": "Basic dXNlcjpwYXNz"}',
                    'Authorization: Basic dXNlcjpwYXNz',
                    'https://user:embedded-pass@host.invalid',
                    '{"api_key": "unregistered-value"}')):
            with self.subTest(raw=raw):
                self.client = ModelClient(ModelConfig(mock_response=raw))
                result = self.run_job(run_id="redaction-" + str(index))
                for path in Path(result["output_dir"]).glob("*.json*"):
                    content = path.read_text()
                    for value in ("dXNlcjpwYXNz", "embedded-pass", "unregistered-value"):
                        self.assertNotIn(value, content)

    def test_foreign_journal_identity_rejected(self):
        self.run_job()
        path = self.root / "mock/mock-demo/predictions.jsonl"
        rows = self.rows("predictions.jsonl")
        rows[0]["logical_id"] = "foreign"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        with self.assertRaisesRegex(ValueError, "identity"):
            self.run_job()

    def test_duplicate_completion_rejected(self):
        self.run_job()
        path = self.root / "mock/mock-demo/attempts.jsonl"
        row = self.rows("attempts.jsonl")[1]
        with path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, "matching attempt"):
            self.run_job()

    def test_logical_ids_stable_across_runs(self):
        self.run_job()
        original = {r["logical_id"] for r in self.rows("predictions.jsonl")}
        other = self.run_job(run_id="another-run")
        rows = read_journal(Path(other["output_dir"]) / "predictions.jsonl")
        self.assertEqual(original, {r["logical_id"] for r in rows})

    def test_resume_retryable_completion_respects_remaining_budget(self):
        self.client.classify.side_effect = None
        self.client.classify.return_value = self.failure()
        with self.assertRaises(KeyboardInterrupt):
            self.run_job(SAMPLES[:1], sleep=Mock(side_effect=KeyboardInterrupt))
        self.assertEqual(self.client.classify.call_count, 1)
        self.client.classify.reset_mock()
        self.run_job(SAMPLES[:1])
        self.assertEqual(self.client.classify.call_count, 8)
        first = self.rows("predictions.jsonl")[0]
        self.assertEqual(first["attempts"], 3)
        self.assertEqual([c.args[0] for c in self.wait.call_args_list][:2], [2, 4])

    def test_cli_requires_real_opt_in_before_any_request(self):
        path = self.root / "qwen.json"
        path.write_text(json.dumps(dict(mode="qwen", model="qwen-plus-2025-12-01",
                        base_url="https://example.invalid/v1", allow_real_requests=True)))
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"}):
            result = subprocess.run([sys.executable, "-m", "src.run_experiment",
                        "--input", "tests/fixtures/experiment_mock.jsonl",
                        "--model-config", str(path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Real API requires --real-api", result.stderr)


if __name__ == "__main__":
    unittest.main()
