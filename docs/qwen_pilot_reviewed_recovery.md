# Group 13 Qwen Plus Pilot reviewed-stop recovery

Run: `group13-full-pilot-qwen200-20261111T090248Z-76877106`.

The recovery entry point is separate from the frozen runner. Existing source,
data, model configuration, messages and manifest must still match their original
hashes. The original entry point retains its historical stop. Recovery never
rewrites existing journal bytes, repairs incomplete tails, raises the budget or
retries an existing final prediction, including the failed Few-shot prediction.

`reviewed_failure_authorization.json` is the existing sealed authorization.
The new `reviewed_stop_audit.json` binds its digest, the three original journal
hashes and byte lengths, the recovery reason and the exact historical stop
record. Its fixed boundary is 45 completed physical attempts / 32 final records
(31 valid). The 568 remaining jobs have not started.

`--resume-after-reviewed-stop` (PowerShell: `-ResumeAfterReviewedStop`) is required
for execution. `--authorize-reviewed-failures` remains a compatible alias for the
existing launcher. CheckOnly reads no credentials, creates no client and writes
no execution audit. A missing execution flag fails before client construction.

Under the run lock, execution repeats the snapshot checks and journal seal
validation, and appends one `recovery_events.jsonl` event **before dispatch**.
This records the explicit flag and both authorization/audit digests. Later
invocations must match that event and reuse the same boundary. Changing the
boundary or deleting the event after new attempts blocks recovery.

Only the consecutive-failure counter at the fixed historical boundary resets.
All post-boundary failures accumulate, and three consecutive unsuccessful
physical attempts latch the stop again, even on a later invocation with the
same recovery flag. Success resets that counter normally. Permission errors,
missing/abnormal usage, three cumulative rate-limit errors and the cumulative
300,000 known Token budget remain active across the entire history. There is
no budget override argument in the recovery entry point. As in the original
runner, the budget is checked after a response; one attempt can cross the cap.
The 14 historical attempts with unknown usage remain recorded; the known-usage
budget does not bound their potentially billable usage.

If a run is interrupted with a started job lacking a final record, this recovery
entry point fails closed for manual review; it does not retry that job. An
incomplete journal tail also blocks recovery without truncation.

From `D:\Group13-Robot-Ambiguity-Detection`, first run:

```powershell
& '.\results\pilot\group13-full-pilot-qwen200-20261111T090248Z-76877106\resume_reviewed_stop.ps1' -CheckOnly
```

After reviewing the output, with `DASHSCOPE_API_KEY` already configured in the
current PowerShell process, the explicit paid execution command is:

```powershell
& '.\results\pilot\group13-full-pilot-qwen200-20261111T090248Z-76877106\resume_reviewed_stop.ps1' -ResumeAfterReviewedStop
```

Do not put the key in a command, JSON file or report. Successful completion of
the remaining jobs would produce 600 final records and 599 valid predictions
if all new jobs succeed; the old failed record stays failed.

Offline verification:

```powershell
python -B -m unittest tests.test_resume_full_pilot -q
python -B -m unittest discover -s tests -q
```

Recovery tests copy the original sealed prefix and frozen inputs to a temporary
directory, block socket connections and the real transport, and inject fake
responses. They simulate all 568 new jobs, verify exact historical prefixes,
deduplication, explicit authorization, new persistent failure stops, retained
permission/usage/budget guards, altered audit/config/snapshot rejection and
incomplete-tail preservation. No real Qwen requests are part of this workflow.
