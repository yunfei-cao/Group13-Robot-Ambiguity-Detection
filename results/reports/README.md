# Group 13 Qwen Plus experimental report

This is an offline, same-test-set repetition of two completed Pilot executions.
Prompts remain v0.1-dev DRAFT. This report does not modify the experiment,
test labels, prompts, evaluate.py or analysis.py, and makes no model calls.

## Rebuild

Run from the repository root (Python 3.13 was used):

```powershell
python -m pip install --target results/reports/_dependencies python-docx pymupdf pypdf pdf2image
python -m pip install matplotlib numpy Pillow pywin32
python -B results/reports/generate_report.py
python -B results/reports/generate_report.py --render
```

The first command installs document dependencies only in this results directory.
Use the dependency packages recorded in package_versions.json for reproducibility.
The --render option requires Microsoft Word and pywin32 on Windows. It runs Word
invisibly, opens only the generated DOCX read-only, exports the same content to
PDF, closes its own Word instance, and renders every PDF page using PyMuPDF.
The packaged DOCX renderer is attempted first; its log documents the absence of
LibreOffice in this environment. Native Word is used as the pagination authority.

## Outputs

- Group13_Qwen_Plus_Experimental_Results.docx: editable paragraphs and tables.
- Group13_Qwen_Plus_Experimental_Results.pdf: native Word PDF export.
- figures/: 16 main figures plus six standalone confusion matrices, PNG/SVG.
- data/: recomputed metrics, per-class diagnostics, prediction distributions,
  agreements, aligned predictions, disagreements, attempts, error cases and
  figure_source_data.csv containing the numeric values used for every main plot.
- integrity.json: source hashes before/after, checks and document structure.
- figure_manifest.json: figure titles and file paths.
- artifact_sha256.json: output file hashes, excluding temporary dependencies/QA.
- _qa/: renderer diagnostics, per-page PNGs and text inventory for layout review.

## Statistical scope

All-job metrics retain all 200 jobs per strategy and count failures as errors.
Valid-only R1 has 200/199/200 outputs, or 599 overall; R2 has 600 valid outputs.
Macro-Precision, Macro-Recall and Macro-F1 average the two gold classes.
Failure contributes FN to its true class, with no invented predicted label.
Confusion heatmaps contain accepted labels only and disclose the R1 failure.
Primary agreement is matching valid labels divided by both-valid jobs.
All-job agreement composition separates 13 label flips from one validity change.
Token sums use every completed attempt and preserve unknown usage separately.
Latency includes completed failed attempts, excludes waits, and uses linear P95.
No significance or generalization advantage is asserted; pair clustering matters.

## Source inputs

- results/pilot/group13-full-pilot-qwen200-20261111T090248Z-76877106/
- results/pilot/group13-full-pilot-round2-20261111T153626Z-bc85f07d/
- data/processed/test_inputs.csv and test_labels.csv
- src/evaluate.py and src/evaluation_adapter.py, imported without edits
- results/analysis/round1-vs-round2-20261111T155346Z/, cross-checked read-only

## Final verification

The final native Word export has 15 A4 pages, 16 embedded main figures and 14
editable tables. All 15 final page PNGs were visually inspected after the last
formatting update. No clipping, overlap, missing glyphs or missing figures was
observed. All 22 figure PNGs use 350 DPI; all have SVG counterparts. See
quality_assurance.json for checks and figure_manifest.json for figure names.
The canonical LibreOffice renderer was unavailable; native Word export succeeded.
