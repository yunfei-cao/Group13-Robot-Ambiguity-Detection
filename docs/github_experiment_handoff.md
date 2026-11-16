# Group 13 model experiment handoff

This branch contains the completed model-client integration, safe experiment
runner, Pilot preparation and recovery, offline tests, and two completed Qwen
Plus Pilot repetitions. Both repetitions use the same 200 test instructions,
100 pairs, three strategies and identical **v0.1-dev DRAFT** prompts. They are
not a frozen final experiment or an independent test-set replication.

## Code and validation

`src/model_client.py`, `src/prompt_adapter.py` and
`src/evaluation_adapter.py` were already committed in the preceding model and
integration work. This handoff adds the Pilot safety and recovery extensions
to `src/run_experiment.py`, supporting scripts, offline tests and documentation.
CAI's source prompts and data and WANG's `evaluate.py` / `analysis.py` remain
unchanged. A Round 2 test fixture rebases absolute execution-summary paths only
inside its temporary copy so completed-run evidence can be tested without
modifying the original journals or relaxing production validation.

Run offline tests from the repository root:

```powershell
python -B -m unittest discover -s tests -q
```

The client HTTP contract tests use local mocked responses. Integration, Pilot
and recovery tests use local fixtures and temporary directories. Never invoke
the real-run launchers merely to validate this handoff.

## Immutable execution evidence

- Round 1: `results/pilot/group13-full-pilot-qwen200-20261111T090248Z-76877106/`
- Round 2: `results/pilot/group13-full-pilot-round2-20261111T153626Z-bc85f07d/`

Both directories include final predictions, per-attempt journals, execution
summaries, configuration, preparation records, launchers and frozen snapshots.
Round 1 also retains the reviewed stop, recovery authorization and recovery
audit. Configuration files refer to the credential environment variable; no
credential value is stored. Keep original journals intact, including the one
historical final failure. Round 1 has 599 valid predictions and Round 2 has 600.

The two **one-byte `.lock` files** are deliberately retained as sealed historical
evidence: the Round 2 manifest records every Round 1 file, and the offline
integrity reports hash both files. Their content is the digit `0`; they contain
no process/session information or credentials. Dropping them would break
read-only reproducibility checks. The development run's one-byte `.lock` is
also retained because its verified source-integrity report seals that file.
These three historical evidence files are the only lock-file exceptions;
runtime caches and temporary files are excluded.

`.gitattributes` preserves exact bytes for historical result files and pinned
Pilot entry points. Specific source/input checkout line endings reproduce the
original SHA-256 seals without editing collaborator content. The remaining
pre-existing source files are retained in their already committed form.

Historical execution summaries include the original absolute Windows output
directory. Exact launcher checks are intended for the recorded workspace;
other locations should use offline evaluation/report generation rather than
rewrite original logs to bypass path checks. No new real execution is needed.

## Analysis and final report

`results/analysis/round1-vs-round2-20261111T155346Z/` contains the independent
offline comparison, scoring details, agreement, confusion matrices, token and
failure summaries, and integrity seals. First-round valid-only and all-job
failure-inclusive results are kept separate. The two classes are averaged for
Macro-F1, Macro-Precision and Macro-Recall. The historical failed request is not
assigned a fabricated label. The reports identify 13 valid label flips and one
failure-to-valid change.

`results/reports/Group13_Qwen_Plus_Experimental_Results.docx` is the editable
15-page English course report. The adjacent PDF was exported using native
Microsoft Word, then all final pages were inspected. `generate_report.py`
recomputes metrics from raw JSONL/CSV and produces the figures and DOCX; its
README describes dependencies and optional PDF export.

To keep Git focused, this handoff includes the 16 main vector SVG figures and
key report CSVs, but excludes duplicate PNG exports, six repeated standalone
confusion heatmaps, report QA rasterizations, local dependencies and the ZIP
copy of the deliverables. The Word/PDF already embed all 16 main figures.
The generator can recreate all 22 PNG/SVG plots when needed. Report
`artifact_sha256.json` describes the original full generated output set, so it
also lists intentionally omitted regenerable exports; it is not an assertion
that every generated export is version controlled.

The completed development run and its verified acceptance tables are included
as supporting evidence. Unverified/incomplete development analysis and obsolete,
unexecuted formal-preparation directories remain local. No formal run is claimed.

## Upload audit

Run `python -B scripts/prepare_github_handoff.py` to build an explicit candidate
file list and scan source text, DOCX XML and PDF text without printing secrets.
It writes the inventory, hashes, size totals and scan status only under ignored
`results/upload_audit/`. Git staging uses that exact NUL-separated path list;
there is no blanket `git add .` or forced ignored-directory add.

This handoff publishes already completed work. It does not rerun model
experiments, optimize prompts on test results, edit `main`, or merge branches.
