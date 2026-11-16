# Group 13 Qwen Plus v0.3 Development Evaluation

This directory contains an offline English research report on 510 completed
Qwen Plus v0.3-candidate predictions: 85 development pairs, 170 instructions,
three prompt strategies. This is development evaluation, not a blind test.

## Sources

- results/analysis/v03-development-evaluation-20261113/ (all metrics and audit)
- results/prompt_comparison/group13-dev-v01-v03-20261113T150521Z-b2a4a074/candidate/
  (sealed inputs/messages, plans, model configuration, predictions and journals)
- prompt_v0_3/candidate/ and prompt_v0_3/development_comparison_85pairs/
- src/evaluate.py and src/evaluation_adapter.py (reused without modification)
- Previous DOCX/PDF and generation script in results/reports/ (design reference)

## Rebuild

From the repository root, with Python 3.13 and python-docx, matplotlib, numpy,
Pillow and PyMuPDF available:

```powershell
python -B results/reports/v03_development_evaluation/generate_report.py
python -B results/reports/v03_development_evaluation/generate_report.py --render
```

PDF export additionally requires Microsoft Word and pywin32 on Windows.
The current host has no exposed workspace dependency loader or bundled Python;
the existing report-local document dependencies are reused as a fallback, with
the existing Python environment supplying plotting and Word automation packages.
No dependency is installed or changed. The packaged render_docx.py is attempted
and its diagnostic retained under _qa/canonical_renderer.log; native Word is the
pagination authority when the packaged LibreOffice route is unavailable. Word
runs invisibly, opens only the new report, refreshes page fields and exports PDF.
Each final PDF page is rasterized at 108 DPI for visual review; charts use 350 DPI.
Reruns replace only this directory's generated outputs. No API or experiment runs.

## Outputs

- Group13_Qwen_Plus_v0.3_Development_Evaluation.docx: editable body and native tables.
- Group13_Qwen_Plus_v0.3_Development_Evaluation.pdf: Word-exported equivalent.
- charts/: 11 main figures and three standalone confusion matrices, PNG and SVG.
- data/: recomputed metrics, full sample scoring, pair composition, token usage,
  selected real cases, model configuration and numeric figure source records.
- generate_report.py: reproducible offline report builder with network blocking.
- figure_manifest.json, integrity.json, quality_assurance.json: traceability and checks.
- _qa/: renderer logs, source design evidence and final page images for internal QA.

Known usage is 349,077 tokens. Eleven failed attempts lack usage; no missing
tokens are imputed as zero and no monetary billing total is asserted. Historical
and previously unused partitions remain development data. No v0.1 improvement
or statistical significance claim is made. Inputs, labels, CAI/WANG source code
and historical reports are never rewritten. No commit or push is performed.
