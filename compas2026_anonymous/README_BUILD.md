# COMPAS 2026 Anonymous IEEE Submission

This folder contains the anonymous initial-submission manuscript. It uses the official IEEE conference class linked by the COMPAS 2026 template page. No model training or experiment reruns are required to build the PDF.

## Contents

- `main.tex`: anonymous IEEE conference manuscript.
- `references.bib`: cited bibliography assembled from the provided NotebookLM source notes, with source comments where metadata still needs verification.
- `IEEEtran.cls`: official IEEE conference class from the COMPAS-linked template archive.
- `IEEEtran.bst`: official IEEE BibTeX style distributed through CTAN.
- `figures/`: all figures retained from the authored draft.
- `tables/`: supporting Markdown/CSV table assets retained from the authored draft.

## Build

From this directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The class and bibliography style are vendored so the build does not depend on a system IEEEtran installation.

## Submission State

- Verify the exact dataset citation, version, and license.
- Verify incomplete arXiv/venue metadata in `references.bib` against the final submission bibliography.
- The author block and other identity-bearing sections are absent for double-blind review.
- PDF author, title, subject, keywords, creator, producer, date, and trailer-ID metadata are suppressed in `main.tex`.
- Communication quantities are binary byte counts and are labeled MiB.
