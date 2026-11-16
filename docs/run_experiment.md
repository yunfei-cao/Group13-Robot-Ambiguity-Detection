# Experiment runner

Run from the repository root (Python standard library only):

```powershell
python -m src.run_experiment --input tests/fixtures/experiment_mock.jsonl --builder src.run_experiment:mock_prompt_builder --run-id mock-acceptance
python -m unittest discover -s tests -v
```

The default client is offline Mock; the default builder now calls CAI's real
`build_messages(strategy, environment, instruction, examples=None)` through
`src.prompt_adapter.official_prompt_builder(sample, strategy)`. Use
`--builder src.run_experiment:mock_prompt_builder` only for legacy routing fixtures.
Mock labels are fixtures, not model judgments.

## Data and prompt integration

CAI CSV must have exactly `sample_id,pair_id,environment,instruction` (any order).
UTF-8 BOM, quoted commas and multiline values are supported. Empty cells,
malformed records, extra columns, duplicate headers and duplicate IDs fail closed.
`sample_id` maps to internal `id`; no ID is invented. JSONL remains supported.
Only `environment` and `instruction` are passed to any builder. Labels, pair IDs,
categories, source metadata and paired instructions cannot cross this boundary.
The dataset hash still binds all supplied metadata for resume checks.

The official adapter uses the fixed development examples, checks manifest version,
example order and SHA-256 (normalizing checkout CRLF to LF), and calls CAI's
function with structured system/user messages. Version is `v0.1-dev`, currently
DRAFT; no prompt freeze or performance validation is implied. Three strategies:
`zero_shot`, `few_shot`, `checklist`. Legacy instruction-only inputs require the
explicit fixture builder above. No test labels are loaded by the runner.

Six-request offline smoke run and repeat (second invocation makes zero calls):

```powershell
python -m src.run_experiment --input data/processed/test_inputs.csv --limit 2 --run-id group13-integration-3bf202a
```

`src.evaluation_adapter.convert_predictions(predictions_path, labels_path)` joins
labels after inference and returns a dictionary of three separate strategy lists
for WANG's `score_samples` / `calculate_metrics`. It maps `raw_text` to
`raw_response`, success/invalid_output to SUCCESS, error/interrupted to
REQUEST_FAILED. Each list must have complete pairs, unique IDs and equal strategy
coverage. A subset of complete pairs is permitted. Do not combine strategies into
one call (WANG requires unique sample IDs). Legacy predictions without raw_text
are rejected with migration guidance. Gold labels remain outside model inputs.

## Outputs and recovery

Outputs are `results/{mock|dev|formal}/{run_id}/`. Mock clients can write only
`mock`; Qwen clients can write only `dev` or `formal`. A changed dataset, prompt
hash/version, model configuration or retry policy requires a new run ID.

`run_config.json` records safe parameters, hashes, version, run identity and UTC
creation time. Credentials and endpoint text are omitted. `attempts.jsonl`
contains two event rows per normal physical call: `started` is flushed before
calling the client; `completed` includes status, label, sanitized raw text,
model ID, latency, token usage and structured errors. Count `started` events to
count persisted call intents, not confirmed provider requests: a crash can occur
before dispatch. Count `completed` for known client outcomes. Attempt numbers
start at 1; `retry_count` starts at 0. Mock token values are unknown, not zero.
`predictions.jsonl` contains exactly one final row per sample/strategy, including
failures and invalid outputs; `label` is null when classification failed.

Only structured errors with `retryable=True` receive up to two extra calls,
after 2 and 4 seconds. Invalid output is final. A completed attempt missing its
prediction is recovered from the attempt journal without another request.
If only a `started` event survives, the outcome is unknown and the final
prediction is `interrupted`, with null label. It is not automatically resent:
local files cannot guarantee exactly-once delivery across an API/process crash.
`final_predictions` counts terminal records, including failures. It does not
guarantee all classifications are available. `valid_predictions` and
`unresolved_logical_ids` in the returned/CLI summary expose this distinction.
An interrupted record preserves the expected row but has no classification;
formal coverage remains incomplete until explicitly resolved.

For manual recovery, retain the original run unchanged. Review every unresolved
record against its last attempt and available provider request information;
unknown delivery may already have incurred a charge. Record the original run ID,
logical ID, reason, reviewer decision and new run ID in a separate reconciliation
manifest without credentials. Only after explicit authorization for real calls,
select the affected original samples into a new run with identical model and
approved prompt version. The current runner always runs all three strategies for
each selected sample, so a sample-level rerun may also repeat successful strategies.
Do not append its results to the original journals. An evaluation handoff must
explicitly map only the intended replacement `(sample_id, strategy)` records via
the manifest, retain original outcomes, and check full coverage and uniqueness.
No automatic per-strategy recovery/merge tool is implemented yet.

`logical_id` is SHA-256 of `[sample_id, strategy]`, stable and unique within a run.
Across different runs it intentionally repeats; use `(run_id, logical_id)` as
the record identity and include the dataset hash when reconciling datasets.

Every append is flushed and fsynced. An unterminated final JSONL fragment is
truncated on resume; malformed complete lines and duplicate predictions fail
closed. Journal identities and attempt event ordering are also checked before
resuming. An OS lock prevents concurrent writers and releases on process death.
The `.lock` file can remain after completion. Do not edit journals manually.

Secrets from configured API keys and credential-like environment variable names
are redacted from all persisted strings, with additional Authorization/Bearer
filtering; suspicious credential-bearing text fields are dropped in full,
including quoted Authorization/Basic headers and URL userinfo. Environment
mappings, request headers and full model configuration
objects are never serialized.

This protects known secrets and recognizable credential text, not arbitrary
private content echoed by a model or encoded/obfuscated credentials. Inspect
artifacts before sharing. `results/mock/` is ignored by Git; `results/dev/` and
`results/formal/` remain available for reviewed delivery. Git ignore rules do
not remove files that were already tracked.

## Explicit real API opt-in (not exercised during development)

Real CLI execution requires `--real-api`, a Qwen `--model-config` whose
`allow_real_requests` is true, `--split dev` or `formal`, an external
`--builder module:function`, and a meaningful `--prompt-version`. Credentials
are loaded by ModelConfig only from `DASHSCOPE_API_KEY`. Set the pinned model
to `qwen-plus-2025-12-01` in that configuration. Real prompts must be supplied
before real execution is possible; the mock builder is rejected for Qwen.
Programmatic use supplies an explicitly enabled Qwen ModelClient and builder.
