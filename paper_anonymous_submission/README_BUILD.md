# Communication-Aware Federated Brain Tumor MRI Classification

COMPAS 2026 Anonymous IEEE Submission

This folder contains the anonymous review manuscript using the official IEEE conference class. No model training or experiment reruns are required to build the PDF.

## Contents

- `main.tex`: anonymous IEEE conference manuscript.
- `references.bib`: cited bibliography assembled from the provided NotebookLM source notes, with source comments where metadata still needs verification.
- `IEEEtran.cls`: official IEEE conference class from the COMPAS-linked template archive.
- `IEEEtran.bst`: official IEEE BibTeX style.
- `figures/`: manuscript figures, including regenerated three-seed convergence and full-state payload charts.
- `tables/`: supporting screening and federated-result table assets.
- Submission title: `Communication-Aware Federated Brain Tumor MRI Classification`.
- CMT-ready title metadata: use exactly `Communication-Aware Federated Brain Tumor MRI Classification`.

## Build

From this directory:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The class and bibliography style are vendored in this folder.

## Submission State

- Dataset provenance is recorded as the 7,200-image balanced Kaggle Version 2 snapshot, verified on June 28, 2026; the citation records an access date of July 14, 2026 and the CC BY 4.0 listing.
- The confirmed CFL-SparseMed and Federated Cross Learning author/venue errors were corrected. The Saraswat and Radhika venue metadata should still receive a final source-level check before submission.
- Review placeholder: `Anonymous Authors`.
- PDF identity fields and build dates are suppressed in `main.tex`.
- Displayed communication units are MiB; numerical values are unchanged.
- Communication values denote full-state communication payload per 20-round federated run.
- The active pipeline is the author-supplied vector file `figures/federated_mri_pipeline.pdf`.
- Do not regenerate the active pipeline with `../scripts/generate_final_pipeline.py` unless deliberately restoring the earlier code-generated design.
- `figures/pipeline_overview_submission.tex` is retained only as a historical fallback and is not included by `main.tex`.
