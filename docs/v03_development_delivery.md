# Qwen Plus v0.3 development delivery

This delivery records the completed `v0.3-candidate` development experiment.
It does not claim an independent blind test or improvement over v0.1.
The earlier integration and authorization notes are historical preparation
records; the final prediction journal and evaluation supersede their progress
counts. The v0.1 comparison side is preparation only in this delivery.

## Completed evidence

- Comparison: `group13-dev-v01-v03-20261113T150521Z-b2a4a074`
- Candidate model: `qwen-plus-2025-12-01`, temperature 0, output limit 64.
- 85 development pairs / 170 instructions, 85 CLEAR and 85 NEEDS_CLARIFICATION.
- 170 unique final predictions per strategy, 510 valid labels overall.
- 9 historically used pairs and 76 previously unused development pairs.
- 521 completed attempts: 510 successes and 11 recovered transport errors.
- Known usage: 347,347 input + 1,730 output = 349,077 tokens.
  Usage from 11 failed attempts is unknown; this is not an exact billing total.

| Strategy | Accuracy | Macro-F1 | Pair Accuracy |
|---|---:|---:|---:|
| Zero-shot | 57.6471% | 57.6236% | 18.8235% |
| Few-shot | 56.4706% | 56.4646% | 17.6471% |
| Checklist | 57.0588% | 56.9381% | 20.0000% |

Zero-shot has the highest instruction Accuracy and Macro-F1. Checklist has
17 fully correct pairs versus 16 for Zero-shot, despite one fewer correct
instruction. These small descriptive differences do not establish significance.

## Locations

- `src/prompt_v03_adapter.py`: independently verified CAI bundle adapter.
- `scripts/prompt_v03_comparison.py` and `scripts/run_prompt_comparison.ps1`:
  isolated preparation, checks and explicitly opted-in execution.
- `src/run_experiment.py` and associated tests: invocation-scoped no-token-limit
  support with the other safety stops retained.
- `results/prompt_comparison/<comparison>/candidate/`: complete candidate run,
  sealed snapshots, plans, configuration and append-only journal.
- `results/prompt_comparison/<comparison>/runner_updates/`: scoped authorization
  evidence required to verify the reviewed runner changes against original plans.
- `results/analysis/v03-development-evaluation-20261113/`: formal offline metrics,
  confusion counts, pair scores, strict ID coverage and source audit.
- `results/reports/v03_development_evaluation/`: 16-page editable DOCX and native
  Word PDF, 11 main figures plus three standalone confusion matrices (PNG/SVG),
  generation script, README and source-derived CSV/JSON data.

## Offline verification

No API key is needed for these commands. They do not execute a model experiment.

```powershell
python -B -m unittest discover -s tests -v
python -B -m scripts.prompt_v03_comparison check --plan results/prompt_comparison/group13-dev-v01-v03-20261113T150521Z-b2a4a074/candidate/plan.json --no-token-limit
```

See the report README for document rebuild dependencies and commands. The formal
evaluation script refuses to overwrite outputs; reproduce it in a fresh analysis
directory with the same relative repository structure. Raw checkout hashes in
the runner receipts describe the original Windows checkout; LF prompt snapshots
and normalized prompt asset digests are also retained.

Internal rendering screenshots, local dependencies, caches and `.lock` files
are excluded from this delivery. CAI/WANG prompt and evaluator source files,
the two historical 600-prediction rounds and the previous report are unchanged.
