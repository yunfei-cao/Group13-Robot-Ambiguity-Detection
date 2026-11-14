"""Offline-first client. Prompts are supplied by the prompt-construction module."""

from dataclasses import dataclass, field
import json
import math
import os
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Mapping
from urllib import error, request
from urllib.parse import urlsplit


LABELS = frozenset({"CLEAR", "NEEDS_CLARIFICATION"})


class ModelClientError(RuntimeError):
    """Transport or response-envelope failure (never includes credentials)."""

    def __init__(self, message: str, *, code: str, retryable: bool = False,
                 http_status: int | None = None, provider_code: str | None = None,
                 request_id: str | None = None):
        super().__init__(message)
        self.details = ModelError(code, message, retryable, http_status, provider_code, request_id)


@dataclass(frozen=True)
class ModelError:
    code: str
    message: str
    retryable: bool
    http_status: int | None = None
    provider_code: str | None = None
    request_id: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    mode: str = "mock"
    model: str = ""
    base_url: str = ""
    timeout_seconds: float = 30.0
    temperature: float = 0.0
    max_tokens: int = 64
    mock_response: str = "CLEAR"
    allow_real_requests: bool = False
    api_key: str = field(default="", repr=False)

    def __post_init__(self):
        if self.mode not in {"mock", "qwen"}:
            raise ValueError("mode must be mock or qwen")
        for name, lower, upper in (("timeout_seconds", 0, math.inf), ("temperature", 0, 2)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if value < lower or value > upper or (name == "timeout_seconds" and value == 0):
                raise ValueError(f"{name} is out of range")
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if type(self.allow_real_requests) is not bool:
            raise ValueError("allow_real_requests must be a boolean")
        for name in ("model", "base_url", "api_key", "mock_response"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")
        if self.mode == "qwen":
            if not self.allow_real_requests:
                raise ValueError("Qwen requires explicit allow_real_requests=true")
            if not self.api_key.strip() or not self.model.strip():
                raise ValueError("Qwen requires an API key and an explicit model version")
            url = urlsplit(self.base_url)
            if (url.scheme != "https" or not url.hostname or url.username or url.password
                    or url.query or url.fragment):
                raise ValueError("base_url must be an HTTPS API base URL without credentials or query")

    @classmethod
    def from_file(cls, path: str | Path, *, environ: Mapping[str, str] | None = None):
        """Load JSON settings; secrets come only from the process environment."""
        settings = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            raise ValueError("Configuration must be a JSON object")
        if "api_key" in settings:
            raise ValueError("Store API keys in DASHSCOPE_API_KEY, not JSON")
        env = os.environ if environ is None else environ
        settings["api_key"] = env.get("DASHSCOPE_API_KEY", "")
        return cls(**settings)


@dataclass(frozen=True)
class ModelResponse:
    raw_text: str
    label: str | None
    is_valid: bool
    model: str
    mode: str
    raw_response: str
    status: str
    latency_ms: float
    token_usage: dict[str, int | None]
    error: ModelError | None
    model_id: str


def _token_usage(data: object) -> dict[str, int | None]:
    usage = data.get("usage") if isinstance(data, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    return {name: value if type(value := usage.get(name)) is int and value >= 0 else None
            for name in ("prompt_tokens", "completion_tokens", "total_tokens")}


def _messages(prompt: str | list[dict]) -> list[dict]:
    if isinstance(prompt, str):
        if not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        return [{"role": "user", "content": prompt}]
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("prompt must be a non-empty string or messages list")
    for message in prompt:
        if not isinstance(message, dict) or message.get("role") not in (
                "system", "developer", "user", "assistant", "tool", "function"):
            raise ValueError("Each message must have a standard OpenAI role")
        content = message.get("content")
        if not isinstance(content, (str, list)) and not (
                message["role"] == "assistant" and
                (message.get("tool_calls") or message.get("function_call"))):
            raise ValueError("Messages require content or assistant tool calls")
        if isinstance(content, list) and (not content or any(
                not isinstance(part, dict) or not isinstance(part.get("type"), str)
                for part in content)):
            raise ValueError("Content parts must be objects with a type")
        if message["role"] == "tool" and not message.get("tool_call_id"):
            raise ValueError("Tool messages require tool_call_id")
    try:
        json.dumps(prompt, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError("Messages must be JSON serializable") from None
    return deepcopy(prompt)


def parse_label(raw_text: str) -> str | None:
    """Accept only an exact label after trimming whitespace; retain invalid outputs."""
    value = raw_text.strip()
    return value if value in LABELS else None


class ModelClient:
    def __init__(self, config: ModelConfig | None = None):
        self.config = config if config is not None else ModelConfig()

    def classify(self, prompt: str | list[dict]) -> ModelResponse:
        """Send a prebuilt prompt, or return the configured offline fixture.

        Mock output is a test fixture, not a linguistic ambiguity prediction.
        Requests are not retried automatically, to avoid hidden experiment calls.
        """
        messages = _messages(prompt)
        config = self.config
        model_id = "mock" if config.mode == "mock" else config.model
        raw, label, failure = "", None, None
        usage = _token_usage(None)
        started = time.perf_counter()
        try:
            if config.mode == "mock":
                raw = config.mock_response
            else:
                data = self._request(messages)
                usage = _token_usage(data)
                if isinstance(data, dict) and isinstance(data.get("model"), str) and data["model"]:
                    model_id = data["model"]
                try:
                    choice = data["choices"][0]
                    content = choice["message"]["content"]
                    if not isinstance(content, str):
                        raise ValueError
                    raw = content
                    if choice.get("finish_reason") != "stop":
                        raise ModelClientError("Qwen returned an incomplete completion", code="incomplete_response")
                except (KeyError, IndexError, TypeError, ValueError):
                    raise ModelClientError("Qwen returned an invalid completion", code="invalid_response") from None
            label = parse_label(raw)
            status = "success" if label is not None else "invalid_output"
        except ModelClientError as exc:
            status, failure = "error", exc.details
        latency_ms = (time.perf_counter() - started) * 1000
        return ModelResponse(raw_text=raw, label=label, is_valid=label is not None,
                             model=model_id, mode=config.mode, raw_response=raw,
                             status=status, latency_ms=latency_ms, token_usage=usage,
                             error=failure, model_id=model_id)

    def _request(self, messages: list[dict]) -> object:
        config = self.config
        payload = {"model": config.model, "messages": messages,
                   "temperature": config.temperature, "max_tokens": config.max_tokens,
                   "enable_thinking": False, "stream": False}
        req = request.Request(config.base_url.rstrip("/") + "/chat/completions",
                              data=json.dumps(payload).encode("utf-8"),
                              headers={"Authorization": "Bearer " + config.api_key,
                                       "Content-Type": "application/json"}, method="POST")
        # Do not forward authenticated requests across HTTP redirects.
        opener = request.build_opener(_NoRedirect())
        try:
            with opener.open(req, timeout=config.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            status = exc.code
            # Allowlist three scalar fields, never retain the raw error body.
            # Bound reads before decoding, and fall back to HTTP classification
            # if the body is malformed, too large, or cannot be read.
            details = {}
            envelope = {}
            try:
                body = exc.read(65537)
                if len(body) <= 65536:
                    parsed = json.loads(body.decode("utf-8"))
                    if isinstance(parsed, dict):
                        envelope = parsed
                        details = parsed.get("error", parsed)
                        if not isinstance(details, dict):
                            details = {}
            except (OSError, ValueError, UnicodeError, RecursionError):
                pass
            finally:
                exc.close()

            def safe_field(value, *, identifier=False):
                if not isinstance(value, str) or not value.strip():
                    return None
                # Discard suspicious fields in full rather than risk partially
                # exposing a credential, echoed header, or private request data.
                if (config.api_key and config.api_key in value) or re.search(
                    r"authorization|bearer|api[\s_-]*key|sk-|password|secret|token|cookie|"
                    r"credential|https?://|[\w.+-]+@[\w.-]+|[\x00-\x1f\x7f]|"
                    r"密钥|密码|令牌|请求头|提示词|prompt|messages|%[0-9a-f]{2}|\\u[0-9a-f]{4}",
                    value, re.IGNORECASE
                ):
                    return "[REDACTED]"
                value = value.strip()
                if identifier and (len(value) > 128 or not re.fullmatch(r"[A-Za-z0-9_.:-]+", value)):
                    return None
                return value[:512]

            provider_code = safe_field(details.get("code"), identifier=True)
            message = safe_field(details.get("message"))
            request_id = safe_field(details.get("request_id") or envelope.get("request_id"), identifier=True)
            code = {401: "authentication_error", 403: "permission_denied",
                    408: "timeout", 429: "rate_limit"}.get(
                        status, "server_error" if 500 <= status < 600 else "http_error")
            raise ModelClientError(message or f"Qwen HTTP error {status}", code=code,
                                   retryable=status in (408, 429) or 500 <= status < 600,
                                   http_status=status, provider_code=provider_code,
                                   request_id=request_id) from None
        except (error.URLError, OSError) as exc:
            reason = exc.reason if isinstance(exc, error.URLError) else exc
            timeout = isinstance(reason, TimeoutError)
            raise ModelClientError("Qwen request timed out" if timeout else "Qwen network request failed",
                                   code="timeout" if timeout else "network_error", retryable=True) from None
        except (ValueError, UnicodeError):
            raise ModelClientError("Qwen returned invalid JSON", code="invalid_response") from None
        return data


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None
