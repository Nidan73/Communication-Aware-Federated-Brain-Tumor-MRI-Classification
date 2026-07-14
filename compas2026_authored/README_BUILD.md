# Paper Draft Build Notes

This folder contains a double-blind first draft generated from completed saved assets only. No model training or experiment reruns are required to build the PDF.

## Contents

- `main.tex`: expanded 5-6 page target paper draft.
- `references.bib`: cited bibliography assembled from the provided NotebookLM source notes, with source comments where metadata still needs verification.
- `figures/`: selected final PNG figures copied from `paper_assets_final/`, plus `pipeline_overview.tex` for the TikZ/PGF workflow figure.
- `tables/`: selected final Markdown/CSV tables copied from `paper_assets_final/`.

## Build

From this directory:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

If `pdflatex` is unavailable, use any LaTeX environment that supports standard packages: `graphicx`, `booktabs`, `tabularx`, `caption`, `microtype`, `tikz`/PGF, and `hyperref`.

## Manual Verification Before Submission

- Confirm the target COMPAS template, page limit, font size, margins, and bibliography style.
- Replace the generic `article` layout with the official conference template if required.
- Verify the exact dataset citation, version, URL, license, and required acknowledgement wording.
- Verify incomplete arXiv/venue metadata in `references.bib` against the final submission bibliography.
- Check whether the double-blind policy permits dataset names and Kaggle citations in the main text.
- Confirm that all table values match the final 20-round outputs in `federated_results_20rounds/`.
- Confirm that the manuscript wording consistently describes the clients as simulated or virtual clients.
- Check that no author names, affiliations, repositories, local paths, or acknowledgments appear in the final PDF metadata or text.
- Review all claims for cautious wording; avoid clinical deployment, full privacy compliance, and exaggerated performance claims.
