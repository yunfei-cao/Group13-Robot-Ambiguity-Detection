"""All tests are offline, including Qwen transport contract tests."""
import json
from io import BytesIO
from dataclasses import asdict
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from src.model_client import ModelClient, ModelConfig, parse_label


ROOT = Path(__file__).resolve().parents[1]


class ModelClientTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        guard.start()
        self.addCleanup(guard.stop)

    def qwen(self):
        return ModelClient(ModelConfig(mode="qwen", model="qwen-plus-test-version",
                           base_url="https://example.invalid/compatible-mode/v1/",
                           api_key="test-secret", allow_real_requests=True))

    def test_default_mock_never_constructs_transport(self):
        with patch("src.model_client.request.build_opener") as transport:
            client = ModelClient()
            self.assertEqual(client.classify("Move it").raw_response, client.classify("Move it").raw_response)
            self.assertEqual(client.classify("Move it").label, "CLEAR")
            transport.assert_not_called()

    def test_mock_labels_and_invalid_output(self):
        for raw, expected in [("CLEAR", "CLEAR"), (" NEEDS_CLARIFICATION\n", "NEEDS_CLARIFICATION"),
                              ("", None), ("clear", None), ("CLEAR because...", None),
                              ("CLEAR NEEDS_CLARIFICATION", None), ('{"label":"CLEAR"}', None)]:
            with self.subTest(raw=raw):
                result = ModelClient(ModelConfig(mock_response=raw)).classify("prompt")
                self.assertEqual(result.raw_text, raw)
                self.assertEqual(result.label, expected)
                self.assertEqual(result.is_valid, expected is not None)
                self.assertEqual(parse_label(raw), expected)

    def test_empty_prompts_rejected(self):
        for prompt in ("", "  ", None, 5):
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                ModelClient().classify(prompt)

    def test_invalid_config_rejected(self):
        for settings in ({"mode": "typo"}, {"timeout_seconds": 0}, {"timeout_seconds": float("nan")},
                         {"temperature": 3}, {"temperature": True}, {"max_tokens": 0},
                         {"max_tokens": True}, {"allow_real_requests": "false"}, {"mock_response": None}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                ModelConfig(**settings)

    def test_qwen_requires_explicit_opt_in_and_credentials(self):
        for settings in ({}, {"allow_real_requests": True},
                         {"allow_real_requests": True, "api_key": "key", "model": "version", "base_url": "http://example.invalid"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                ModelConfig(mode="qwen", **settings)

    def test_sample_configs_are_safe(self):
        config = ModelConfig.from_file(ROOT / "configs/model.mock.json", environ={})
        self.assertEqual(ModelClient(config).classify("prompt").mode, "mock")
        with self.assertRaises(ValueError):
            ModelConfig.from_file(ROOT / "configs/model.qwen.example.json", environ={})

    def test_secrets_only_from_environment_and_hidden_in_repr(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{}', encoding="utf-8")
            config = ModelConfig.from_file(path, environ={"DASHSCOPE_API_KEY": "test-secret"})
            self.assertEqual(config.api_key, "test-secret")
            self.assertNotIn("test-secret", repr(config))
            path.write_text('{"api_key":"bad"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                ModelConfig.from_file(path, environ={})

    def response(self, opener, data):
        opener.open.return_value.__enter__.return_value.read.return_value = json.dumps(data).encode()

    @patch("src.model_client.request.build_opener")
    def test_qwen_request_contract(self, factory):
        opener = factory.return_value
        self.response(opener, {"choices": [{"message": {"content": "NEEDS_CLARIFICATION"}, "finish_reason": "stop"}]})
        result = self.qwen().classify("移动红色杯子")
        self.assertEqual(result.label, "NEEDS_CLARIFICATION")
        req = opener.open.call_args.args[0]
        self.assertEqual(req.full_url, "https://example.invalid/compatible-mode/v1/chat/completions")
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.get_header("Authorization"), "Bearer test-secret")
        payload = json.loads(req.data)
        self.assertEqual(payload["messages"], [{"role": "user", "content": "移动红色杯子"}])
        self.assertFalse(payload["enable_thinking"])
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["max_tokens"], 64)
        self.assertEqual(payload["model"], "qwen-plus-test-version")
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 30)
        opener.open.assert_called_once()

    @patch("src.model_client.request.build_opener")
    def test_invalid_label_preserves_raw_response(self, factory):
        self.response(factory.return_value, {"choices": [{"message": {"content": "Maybe"}, "finish_reason": "stop"}]})
        result = self.qwen().classify("prompt")
        self.assertEqual(result.raw_text, "Maybe")
        self.assertFalse(result.is_valid)

    @patch("src.model_client.request.build_opener")
    def test_malformed_and_truncated_responses(self, factory):
        for data in ({}, None, {"choices": []}, {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]},
                     {"choices": [{"message": {"content": "CLEAR"}, "finish_reason": "length"}]}):
            with self.subTest(data=data):
                self.response(factory.return_value, data)
                result = self.qwen().classify("prompt")
                self.assertEqual(result.status, "error")
                self.assertFalse(result.is_valid)
                self.assertFalse(result.error.retryable)

    @patch("src.model_client.request.build_opener")
    def test_non_json_response(self, factory):
        factory.return_value.open.return_value.__enter__.return_value.read.return_value = b"not json"
        result = self.qwen().classify("prompt")
        self.assertEqual(result.error.code, "invalid_response")
        self.assertIn("invalid JSON", result.error.message)

    @patch("src.model_client.request.build_opener")
    def test_transport_errors_are_sanitized_and_not_retried(self, factory):
        for exc in (URLError("test-secret"), TimeoutError("test-secret"),
                    HTTPError("https://example.invalid", 429, "test-secret", {}, None)):
            with self.subTest(error=type(exc).__name__):
                factory.return_value.reset_mock()
                factory.return_value.open.side_effect = exc
                result = self.qwen().classify("prompt")
                self.assertEqual(result.status, "error")
                self.assertNotIn("test-secret", repr(result))
                factory.return_value.open.assert_called_once()

    def test_mock_response_fields(self):
        result = ModelClient().classify("prompt")
        self.assertEqual(result.raw_response, result.raw_text)
        self.assertEqual(result.model_id, "mock")
        self.assertEqual(result.model, result.model_id)
        self.assertEqual(result.status, "success")
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.latency_ms, 0)
        self.assertEqual(result.token_usage, dict.fromkeys(
            ("prompt_tokens", "completion_tokens", "total_tokens")))

    def test_invalid_output_status(self):
        result = ModelClient(ModelConfig(mock_response="maybe")).classify("prompt")
        self.assertEqual(result.status, "invalid_output")
        self.assertIsNone(result.error)
        self.assertIsNone(result.label)

    @patch("src.model_client.time.perf_counter", side_effect=[10.0, 10.125])
    def test_latency_in_milliseconds(self, clock):
        self.assertEqual(ModelClient().classify("prompt").latency_ms, 125.0)
        self.assertEqual(clock.call_count, 2)

    @patch("src.model_client.request.build_opener")
    def test_usage_and_returned_model_id(self, factory):
        self.response(factory.return_value, {"model": "resolved-version", "usage": {
            "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            "choices": [{"message": {"content": "CLEAR"}, "finish_reason": "stop"}]})
        result = self.qwen().classify("prompt")
        self.assertEqual(result.token_usage, {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})
        self.assertEqual(result.model_id, "resolved-version")

    @patch("src.model_client.request.build_opener")
    def test_missing_and_invalid_token_counts(self, factory):
        for usage, expected in [(None, (None, None, None)), ({}, (None, None, None)),
                                ({"prompt_tokens": 0}, (0, None, None)),
                                ({"prompt_tokens": 4, "completion_tokens": 2}, (4, 2, None)),
                                ({"prompt_tokens": True, "completion_tokens": -1, "total_tokens": "5"}, (None, None, None))]:
            with self.subTest(usage=usage):
                self.response(factory.return_value, {"usage": usage, "choices": [
                    {"message": {"content": "CLEAR"}, "finish_reason": "stop"}]})
                result = self.qwen().classify("prompt")
                self.assertEqual(tuple(result.token_usage.values()), expected)

    @patch("src.model_client.request.build_opener")
    def test_messages_preserved_without_mutation(self, factory):
        messages = [{"role": "system", "content": "Return a label"},
                    {"role": "user", "content": "移动杯子", "name": "operator"}]
        original = json.loads(json.dumps(messages))
        self.response(factory.return_value, {"choices": [
            {"message": {"content": "CLEAR"}, "finish_reason": "stop"}]})
        self.qwen().classify(messages)
        self.assertEqual(json.loads(factory.return_value.open.call_args.args[0].data)["messages"], original)
        self.assertEqual(messages, original)
        self.assertEqual(ModelClient().classify(messages).status, "success")

    def test_standard_content_parts_and_tool_messages(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "Move it"}]},
                    {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "call1", "type": "function", "function": {"name": "inspect", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "call1", "content": "done"}]
        self.assertEqual(ModelClient().classify(messages).status, "success")

    def test_invalid_messages_rejected_before_transport(self):
        for messages in ([], [{}], ["text"], [{"role": "unknown", "content": "x"}],
                         [{"role": "user", "content": None}], [{"role": "user", "content": ["text"]}],
                         [{"role": "tool", "content": "x"}], [{"role": "user", "content": "x", "extra": object()}]):
            with self.subTest(messages=messages), patch("src.model_client.request.build_opener") as transport:
                with self.assertRaises(ValueError):
                    self.qwen().classify(messages)
                transport.assert_not_called()

    @patch("src.model_client.request.build_opener")
    def test_http_error_classification(self, factory):
        for status, code, retryable in [(401, "authentication_error", False), (403, "permission_denied", False),
                                        (429, "rate_limit", True), (500, "server_error", True),
                                        (503, "server_error", True), (408, "timeout", True),
                                        (400, "http_error", False), (302, "http_error", False)]:
            with self.subTest(status=status):
                factory.return_value.reset_mock()
                factory.return_value.open.side_effect = HTTPError("https://example.invalid", status, "secret", {}, None)
                result = self.qwen().classify("prompt")
                self.assertEqual(result.error.code, code)
                self.assertEqual(result.error.http_status, status)
                self.assertEqual(result.error.retryable, retryable)
                self.assertEqual(result.status, "error")
                self.assertIsNone(result.label)
                self.assertEqual(result.raw_response, "")
                self.assertTrue(all(value is None for value in result.token_usage.values()))
                factory.return_value.open.assert_called_once()

    @patch("src.model_client.request.build_opener")
    def test_timeout_and_network_classification_with_latency(self, factory):
        for exc, code in [(TimeoutError("secret"), "timeout"),
                          (URLError(TimeoutError("secret")), "timeout"),
                          (URLError("secret"), "network_error")]:
            with self.subTest(code=code), patch("src.model_client.time.perf_counter", side_effect=[1, 1.5]):
                factory.return_value.reset_mock()
                factory.return_value.open.side_effect = exc
                result = self.qwen().classify("prompt")
                self.assertEqual(result.error.code, code)
                self.assertTrue(result.error.retryable)
                self.assertIsNone(result.error.http_status)
                self.assertEqual(result.latency_ms, 500)
                factory.return_value.open.assert_called_once()

    @patch("src.model_client.request.build_opener")
    def test_truncated_response_preserves_text_and_usage(self, factory):
        self.response(factory.return_value, {"usage": {"completion_tokens": 64}, "choices": [
            {"message": {"content": "CLEAR"}, "finish_reason": "length"}]})
        result = self.qwen().classify("prompt")
        self.assertEqual(result.error.code, "incomplete_response")
        self.assertEqual(result.raw_response, "CLEAR")
        self.assertEqual(result.token_usage["completion_tokens"], 64)
        self.assertIsNone(result.label)

    def test_redirects_are_disabled(self):
        from src.model_client import _NoRedirect
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.invalid"))

    @patch("src.model_client.request.build_opener")
    def test_403_json_error_fields(self, factory):
        for nested in (False, True):
            with self.subTest(nested=nested):
                fields = {"code": "AccessDenied.Unpurchased", "message": "Model access is not enabled."}
                body = {"error": fields} if nested else dict(fields)
                body["request_id"] = "req-403-example"
                body["private_debug"] = "must-not-appear"
                stream = BytesIO(json.dumps(body).encode())
                factory.return_value.reset_mock()
                factory.return_value.open.side_effect = HTTPError("https://example.invalid", 403, "Forbidden", {}, stream)
                result = self.qwen().classify("prompt")
                self.assertEqual(result.error.code, "permission_denied")
                self.assertEqual(result.error.provider_code, "AccessDenied.Unpurchased")
                self.assertEqual(result.error.message, fields["message"])
                self.assertEqual(result.error.request_id, "req-403-example")
                self.assertEqual(result.error.http_status, 403)
                self.assertFalse(result.error.retryable)
                self.assertEqual(result.raw_response, "")
                self.assertNotIn("must-not-appear", json.dumps(asdict(result)))
                self.assertTrue(stream.closed)
                factory.return_value.open.assert_called_once()

    @patch("src.model_client.request.build_opener")
    def test_403_sensitive_fields_redacted(self, factory):
        for sensitive in ("test-secret", "Authorization: Basic abc123", "Bearer other-credential",
                          'api_key="other-value"', "sk-another-value", "password=private",
                          "Cookie: session=private", "user@example.com", "https://host/?key=private",
                          "message\nprivate", "api%5fkey=private", "prompt: private text"):
            with self.subTest(sensitive=sensitive):
                body = {"error": {"code": sensitive, "message": sensitive, "request_id": sensitive},
                        "headers": {"Authorization": "Bearer test-secret"}}
                factory.return_value.open.side_effect = HTTPError(
                    "https://example.invalid", 403, "Forbidden", {}, BytesIO(json.dumps(body).encode()))
                result = self.qwen().classify("prompt")
                self.assertEqual(result.error.message, "[REDACTED]")
                self.assertEqual(result.error.provider_code, "[REDACTED]")
                self.assertEqual(result.error.request_id, "[REDACTED]")
                self.assertNotIn(sensitive, json.dumps(asdict(result)))
                self.assertNotIn("test-secret", repr(result))

    @patch("src.model_client.request.build_opener")
    def test_403_bad_error_body_falls_back_safely(self, factory):
        for body in (b"<html>private</html>", b"\xff", b"[]", b'{"error":null}',
                     b'{"code":{},"message":[],"request_id":42}', b"x" * 65537):
            with self.subTest(size=len(body)):
                stream = BytesIO(body)
                factory.return_value.open.side_effect = HTTPError("https://example.invalid", 403, "private", {}, stream)
                result = self.qwen().classify("prompt")
                self.assertEqual(result.error.message, "Qwen HTTP error 403")
                self.assertIsNone(result.error.provider_code)
                self.assertIsNone(result.error.request_id)
                self.assertEqual(result.error.http_status, 403)
                self.assertFalse(result.error.retryable)
                self.assertTrue(stream.closed)

    @patch("src.model_client.request.build_opener")
    def test_403_error_body_read_failure(self, factory):
        stream = BytesIO()
        exc = HTTPError("https://example.invalid", 403, "private", {}, stream)
        factory.return_value.open.side_effect = exc
        with patch.object(exc, "read", side_effect=TimeoutError("test-secret")):
            result = self.qwen().classify("prompt")
        self.assertEqual(result.error.message, "Qwen HTTP error 403")
        self.assertFalse(result.error.retryable)
        self.assertTrue(stream.closed)


if __name__ == "__main__":
    unittest.main()
