# Pilot Round 2 independent repetition

Round 1: `group13-full-pilot-qwen200-20261111T090248Z-76877106`.
Round 2 uses a new `group13-full-pilot-round2-*` directory. Its manifest records
the source run, byte hashes of **every** Round 1 file, model configuration,
dataset, all messages, entry point and PowerShell launcher. CheckOnly is read-only,
does not load credentials, and refuses changed snapshots or foreign journals.
Round 2 copies only the immutable snapshot and model configuration. No predictions,
attempts, execution summaries, run configuration or recovery authorization is copied.
The runner's logical IDs identify sample/strategy and intentionally match across
rounds; the run ID, output paths and persisted state distinguish each experiment.

Prompts remain **DRAFT**, `v0.1-dev`; this is a pilot, not a frozen final experiment.
The 200 input rows, ordering, three strategies, fixed development examples and all
600 message hashes are unchanged. Only environment/instruction enter current-input
messages; test labels are not used in preparation or dispatch. Original CAI and
WANG source files are unchanged.

Run `run_round2.ps1 -CheckOnly` in the Round 2 directory for offline validation.
Run `run_round2.ps1` manually in a normal PowerShell with `DASHSCOPE_API_KEY`
already configured to start paid requests. Reuse that exact script to resume only
Round 2. There are 600 logical requests, with up to two network retries per job
(2/4 second backoff), so HTTP attempts and charges can exceed 600. Three consecutive
unsuccessful attempts, three rate limits, authentication/permission errors and
missing/abnormal usage retain the existing stop behavior. The cumulative known-token
soft threshold is 300000; the request crossing it can overshoot, and failed requests
can have unknown usage. This launcher provides no budget-increase option. A safety
stop requires review; it must not be cleared by importing Round 1 recovery state.

## Analysis protocol after Round 2 completes

Do not evaluate or tune prompts during this preparation. After completion, write
all derived artifacts under a new `results/analysis/round1-vs-round2-<timestamp>/`,
keeping both raw run directories read-only. Record both manifest hashes and all
prediction/attempt/summary hashes before and after analysis. Use the existing
evaluation adapter and WANG evaluator by import; do not edit either module.

1. Require 600 unique `(sample_id, strategy)` final records in each round,
   exactly 200 per strategy, matching sample IDs/order and prompt hashes. Verify
   model settings and manifests; report valid/invalid/error totals and stop status.
   A partial run gets a completeness report, not a full-round comparison.
2. Only now join `data/processed/test_labels.csv` by sample ID. Never infer labels
   from IDs or from the paired instruction. Retain the original first-round
   network failure as a failure; do not retry or replace historical predictions.
3. For each strategy and each round, report Accuracy and Macro-F1 on all 200
   logical jobs and additionally on valid outputs with explicit denominators.
   All-job Accuracy counts failures/invalids as incorrect. For all-job Macro-F1,
   use the two gold classes; invalid/failure outputs contribute false negatives
   to their gold class, no false positive to either predicted class. Average the
   two class F1 scores, using zero for a zero denominator. Verify evaluator semantics
   against this definition before reusing its metrics.
4. Produce a 2-by-2 confusion matrix for valid outputs, ordered `CLEAR`,
   `NEEDS_CLARIFICATION`, plus separate gold-by-invalid/failure counts (or an
   extended matrix with an explicit failure column). Include raw counts and
   row-normalized values; do not discard failures invisibly.
5. Join rounds by sample ID and strategy. Report prediction agreement among
   both-valid jobs (`same label / both-valid`), both-valid coverage (`n / 200`),
   and same-valid-label rate on all jobs (`same valid label / 200`). Two failures
   do not count as label agreement. Save a disagreement table with IDs, labels,
   statuses and hashes; keep any later case analysis separate from prompt design.
6. From **completed attempt events**, sum prompt/completion/total known tokens
   for each strategy and round, including retries. Report unknown-usage completed
   attempts and unfinished starts separately, plus total attempts, retries,
   network_error/timeout attempt counts, final failed logical jobs, valid output
   rates, and stop reasons. Do not sum only final-prediction usage or infer unknown
   tokens as zero. Include totals and Round 2 minus Round 1 differences.
7. Report Accuracy/Macro-F1 differences and agreement with denominators. This is
   an independent repeated API execution on the same test set, not an independent
   test sample. Temperature zero does not guarantee identical provider outputs.
   If uncertainty intervals are added, bootstrap the 100 pairs as clusters,
   jointly across rounds and strategies, with a recorded seed and sample count.

Expected deliverables: `integrity.json`, `metrics_by_round_strategy.csv`,
`agreement_by_strategy.csv`, `confusion_matrices.json`, `tokens_and_failures.csv`,
`disagreements.csv`, and `comparison.md`. Round 1 user-reported baseline:
600 finals, 599 valid, 274119 known total tokens, no final stop reason, one
historical final network failure. Reconcile those values from journals before
publishing the comparison; never rewrite them to make the rounds align.
