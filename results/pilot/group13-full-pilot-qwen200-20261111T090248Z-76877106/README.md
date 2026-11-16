# Group 13 Full Pilot ? not a final paper experiment

Prepared for the explicitly authorized paid real API pilot. Execute run_full_pilot.ps1 from PowerShell with DASHSCOPE_API_KEY configured. The script changes to the repository root itself. No credential value is written or printed.

Input: data/processed/test_inputs.csv, 200 instructions / 100 complete pairs. Current input contains only environment and instruction; fixed development demonstration labels remain in the few-shot system prompt by design. Test gold labels, answers, categories and paired instructions are never read by the launcher or passed as metadata to the model.

Run ID: group13-full-pilot-qwen200-20261111T090248Z-76877106
Results: results/pilot/group13-full-pilot-qwen200-20261111T090248Z-76877106/
Snapshot: snapshot/messages.json (600 ordered exact messages), original templates/examples/manifest and code copies with SHA-256 in pilot_manifest.json.

Fixed model: qwen-plus-2025-12-01; temperature=0; max_tokens=64; enable_thinking=false; stream=false; timeout=30 seconds; v0.1-dev DRAFT retained. Each strategy has exactly 200 logical jobs. Existing 2/4-second retry delays and at most 3 physical attempts per logical job remain. Calls are serial, each receives an independent frozen message copy.

Pilot execution passes split=pilot, allow_draft_pilot=True, stop_on_anomaly=True, max_logical_requests=600 and the preparation manifest SHA-256 to the runner. Formal execution still requires a reviewed FROZEN/APPROVED/FINAL prompt status; pilot authorization cannot bypass that check. The old formal preparation files were not changed.

Budget starts at 300000 known total tokens. The development known usage multiplied by ten is 275570, but this is only a rough extrapolation: unknown usage, different inputs and retries make it unsuitable as a guaranteed cost bound. The cap is checked after each completed attempt, so one attempt may overshoot it. Unknown timeout/interrupted usage is reported separately and may already be billable. This preparation keeps the 300000 protection rather than automatically raising it.

Resume after ordinary interruption: run the same script. Existing final rows, including failed/interrupted rows, are skipped; completed attempts without a final row are recovered; a started attempt without a completion becomes interrupted/outcome_unknown and is never silently resent. Full coverage of 600 terminal records does not guarantee 600 valid classifications; inspect the summary and unresolved IDs.

After a budget-only stop, inspect execution_summaries.jsonl and token totals. To explicitly approve a higher cumulative limit, run the same script with -TokenBudget 400000 -ExtendTokenBudget. Only a strictly increasing budget can change; all other experiment configuration must match. The original created_at is retained, and budget_events.jsonl records the extension intent before the active config is atomically updated. An interruption between the audit and config update can leave an uncommitted extension intent; run_config.json is the active limit. Subsequent ordinary runs automatically use that active limit. Do not edit the config or journals manually.

Authentication/permission errors, three rate-limit errors, three consecutive unsuccessful attempts, or malformed/abnormal usage stop the run. Budget extension does not clear these stops. Keep all records and diagnose the cause; do not change run IDs to evade a safety stop.

run_full_pilot.ps1 -CheckOnly performs all 600-message/hash/coverage checks without accessing API credentials or constructing a real client. The run command first repeats those checks. Any pinned code/data/prompt/model change blocks execution; restore the pinned files or explicitly prepare a new configuration and run ID rather than mixing variants.

No actual experiment has been executed during preparation. 78 offline tests passed, including pilot opt-in/formal isolation, budget stop/extension, deduplication and retained permission stops. CAI and WANG source files, data, prompts and previous results are unchanged. No Commit/Push.
