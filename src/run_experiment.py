"""Offline-first experiment runner. See docs/run_experiment.md for the contract."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import time
from typing import Protocol

from .model_client import LABELS, ModelClient, ModelClientError, ModelConfig

STRATEGIES = ("zero_shot", "few_shot", "checklist")


class PromptBuilder(Protocol):
    def __call__(self, sample: dict, strategy: str) -> str | list[dict]: ...


def mock_prompt_builder(sample: dict, strategy: str) -> str:
    """Routing fixture only: deliberately contains no official few-shot examples."""
    return f"MOCK ONLY; strategy={strategy}\nInstruction: {sample['instruction']}"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def load_samples(path: str | Path) -> list[dict]:
    if Path(path).suffix.lower() == ".csv":
        raise ValueError("CSV column mapping is not agreed yet; provide normalized JSONL")
    samples = [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
               if line.strip()]
    validate_samples(samples)
    return samples


def validate_samples(samples):
    seen = set()
    if not samples:
        raise ValueError("At least one sample is required")
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Each sample must be an object")
        sid, instruction = sample.get("id"), sample.get("instruction")
        if not isinstance(sid, str) or not sid.strip() or sid in seen:
            raise ValueError("Sample IDs must be unique nonempty strings")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Each sample needs a nonempty instruction")
        if "label" in sample and sample["label"] not in LABELS:
            raise ValueError("Invalid reference label")
        seen.add(sid)


class Sanitizer:
    def __init__(self, config):
        self.secrets = {v for k, v in os.environ.items() if v and re.search(
            r"key|token|secret|password|credential|authorization", k, re.I)}
        if config.api_key:
            self.secrets.add(config.api_key)

    def __call__(self, value):
        if isinstance(value, str):
            # Drop the whole field: partial regex replacement misses quoted JSON
            # headers, Basic auth and credentials embedded in connection URLs.
            if re.search(r"authorization|\bbearer\b|api[\s_-]*key|password|secret|"
                         r"credential|\btoken\b|\bsk-|://[^\s/]+@", value, re.I):
                return "[REDACTED]"
            for secret in sorted(self.secrets, key=len, reverse=True):
                value = value.replace(secret, "[REDACTED]")
            return value
        if isinstance(value, dict):
            return {self(str(k)): self(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self(v) for v in value]
        return value


@contextmanager
def run_lock(folder):
    """OS lock is released on process death; the persistent file is harmless."""
    with (folder / ".lock").open("a+b") as stream:
        stream.seek(0)
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            lock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            lock = lambda: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(stream, fcntl.LOCK_UN)
        try:
            lock()
        except OSError:
            raise RuntimeError("This experiment is already running") from None
        try:
            yield
        finally:
            stream.seek(0)
            unlock()


def read_journal(path):
    """Only an unterminated final write may be repaired; interior corruption fails."""
    if not path.exists():
        return []
    rows, offset = [], 0
    with path.open("r+b") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                stream.truncate(offset)
                stream.flush()
                os.fsync(stream.fileno())
                break
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Journal rows must be objects")
            rows.append(row)
            offset += len(line)
    return rows


def append(path, row, sanitize):
    data = (json.dumps(sanitize(row), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with path.open("ab") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def run_experiment(samples, *, results_root="results", run_id="mock-demo",
                   split="mock", client=None, builder=None, prompt_version="mock-v1",
                   sleep=time.sleep):
    validate_samples(samples)
    client = client or ModelClient()
    config = client.config
    if split not in {"mock", "dev", "formal"}:
        raise ValueError("split must be mock, dev or formal")
    if (config.mode == "mock") != (split == "mock"):
        raise ValueError("Mock and real experiment outputs must be separated")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run_id must be a safe directory name")
    if not prompt_version.strip():
        raise ValueError("A prompt version is required")
    if config.mode != "mock" and prompt_version == "mock-v1":
        raise ValueError("Real experiments require an explicit non-mock prompt version")
    if builder is None:
        if config.mode != "mock":
            raise ValueError("Real experiments require an explicit prompt builder")
        builder = mock_prompt_builder
    if config.mode != "mock" and builder is mock_prompt_builder:
        raise ValueError("Mock prompts cannot be used for real experiments")
    # Build independently and freeze every prompt before sending any request.
    jobs = []
    for sample in samples:
        for strategy in STRATEGIES:
            prompt = deepcopy(builder(deepcopy(sample), strategy))
            jobs.append((digest([sample["id"], strategy]), sample["id"], strategy, prompt))
    sanitize = Sanitizer(config)
    settings = {"schema_version": 1, "run_id": run_id, "split": split,
                "mode": config.mode, "model": config.model, "temperature": config.temperature,
                "max_tokens": config.max_tokens, "timeout_seconds": config.timeout_seconds,
                "enable_thinking": False, "stream": False,
                "endpoint_sha256": digest(config.base_url), "mock_response_sha256": digest(config.mock_response),
                "prompt_version": prompt_version, "dataset_sha256": digest(samples),
                "strategies": list(STRATEGIES), "logical_requests": len(jobs),
                "prompts_sha256": digest([j[3] for j in jobs]), "retry_delays_seconds": [2, 4]}
    settings = sanitize(settings)
    folder = Path(results_root) / split / run_id
    folder.mkdir(parents=True, exist_ok=True)
    with run_lock(folder):
        config_path = folder / "run_config.json"
        if config_path.exists():
            old = json.loads(config_path.read_text(encoding="utf-8"))
            if {k: v for k, v in old.items() if k != "created_at"} != settings:
                raise ValueError("Run configuration changed; use a new run_id")
        else:
            if any((folder / name).exists() for name in ("attempts.jsonl", "predictions.jsonl")):
                raise ValueError("Journals without run configuration cannot be resumed")
            temporary = folder / "run_config.tmp"
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(dict(settings, created_at=datetime.now(timezone.utc).isoformat()), stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, config_path)
        ap, pp = folder / "attempts.jsonl", folder / "predictions.jsonl"
        attempts, predictions = read_journal(ap), read_journal(pp)
        expected = {lid: sanitize(dict(sample_id=sid, strategy=strategy, run_id=run_id,
                                       prompt_sha256=digest(prompt)))
                    for lid, sid, strategy, prompt in jobs}
        pending, counts = set(), {}
        for row in attempts + predictions:
            identity = expected.get(row.get("logical_id"))
            if identity is None or any(row.get(k) != v for k, v in identity.items()):
                raise ValueError("Journal identity does not match this experiment")
        for row in attempts:
            lid, number = row["logical_id"], row.get("attempt_number")
            if row.get("event") == "started":
                if lid in pending or type(number) is not int or number != counts.get(lid, 0) + 1 or number > 3:
                    raise ValueError("Invalid or duplicate attempt start")
                counts[lid] = number
                pending.add(lid)
            elif row.get("event") == "completed":
                if lid not in pending or number != counts[lid]:
                    raise ValueError("Completion without matching attempt start")
                pending.remove(lid)
            else:
                raise ValueError("Unknown attempt event")
        done = {row["logical_id"] for row in predictions}
        if len(done) != len(predictions):
            raise ValueError("Duplicate predictions detected")
        actual_calls = 0
        for logical_id, sid, strategy, prompt in jobs:
            if logical_id in done:
                continue
            base = dict(logical_id=logical_id, sample_id=sid, strategy=strategy,
                        run_id=run_id, prompt_sha256=digest(prompt))
            history = [row for row in attempts if row["logical_id"] == logical_id]
            starts = [row for row in history if row["event"] == "started"]
            completed = {row["attempt_number"]: row for row in history if row["event"] == "completed"}
            number = max((row["attempt_number"] for row in starts), default=0)
            last = completed.get(number)
            if number and last is None:
                last = dict(status="interrupted", label=None, is_valid=False,
                            error={"code": "outcome_unknown", "retryable": False})
            while True:
                if last and (last["status"] != "error" or not (last.get("error") or {}).get("retryable")):
                    break
                if number >= 3:
                    break
                if number:
                    sleep((2, 4)[number - 1])
                number += 1
                start = dict(base, event="started", attempt_number=number, retry_count=number - 1,
                             timestamp=datetime.now(timezone.utc).isoformat())
                append(ap, start, sanitize)
                actual_calls += 1
                started = time.perf_counter()
                try:
                    response = client.classify(deepcopy(prompt))
                    result = dict(status=response.status, label=response.label,
                                  is_valid=response.is_valid, raw_text=response.raw_text,
                                  model_id=response.model_id, latency_ms=response.latency_ms,
                                  token_usage=response.token_usage,
                                  error=asdict(response.error) if response.error else None)
                except ModelClientError as exc:
                    result = dict(status="error", label=None, is_valid=False, raw_text="",
                                  model_id=config.model, latency_ms=(time.perf_counter()-started)*1000,
                                  token_usage={}, error=asdict(exc.details))
                last = dict(base, **result, event="completed", attempt_number=number,
                            retry_count=number - 1, timestamp=datetime.now(timezone.utc).isoformat())
                append(ap, last, sanitize)
            if last is None:
                last = dict(status="interrupted", label=None, is_valid=False, token_usage={},
                            error={"code": "outcome_unknown", "retryable": False})
            prediction = dict(base, status=last["status"], label=last["label"],
                              is_valid=last["is_valid"], error=last.get("error"),
                              attempts=number, retry_count=max(0, number - 1))
            append(pp, prediction, sanitize)
            done.add(logical_id)
        final_rows = read_journal(pp)
        unresolved = [r["logical_id"] for r in final_rows if not r["is_valid"]]
        return {"output_dir": str(folder), "logical_requests": len(jobs),
                "final_predictions": len(done), "calls_this_run": actual_calls,
                "valid_predictions": len(final_rows) - len(unresolved),
                "unresolved_logical_ids": unresolved}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL: id, instruction, optional label")
    parser.add_argument("--run-id", default="mock-demo")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--split", choices=("mock", "dev", "formal"), default="mock")
    parser.add_argument("--real-api", action="store_true")
    parser.add_argument("--model-config", help="ModelConfig JSON (credentials only from environment)")
    parser.add_argument("--builder", help="Importable module:function implementing PromptBuilder")
    parser.add_argument("--prompt-version", default="mock-v1")
    args = parser.parse_args()
    config = ModelConfig.from_file(args.model_config) if args.model_config else ModelConfig()
    if config.mode != "mock" and not args.real_api:
        parser.error("Real API requires --real-api")
    if args.real_api and config.mode != "qwen":
        parser.error("--real-api requires a Qwen model configuration")
    builder = None
    if args.builder:
        module, name = args.builder.split(":", 1)
        builder = getattr(importlib.import_module(module), name)
    print(json.dumps(run_experiment(load_samples(args.input), results_root=args.results_root,
                     run_id=args.run_id, split=args.split, client=ModelClient(config),
                     builder=builder, prompt_version=args.prompt_version), ensure_ascii=False))


if __name__ == "__main__":
    main()
