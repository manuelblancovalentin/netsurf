# NetSuRF software-only internal-review manuscript

Upload the source archive as a new Overleaf project. Set `main.tex` as the main
document and use pdfLaTeX. The standard ACM `acmart` class and bibliography style
are supplied by TeX Live / Overleaf; no local patched class is required.

All six saved scientific plots, their vector alternatives, generated table rows
and thirteen bibliography entries are included. The workflow is native TikZ. No
model, dataset, Python environment or NetSuRF installation is needed to compile.

Locally: `latexmk -pdf main.tex`. The included `.latexmkrc` writes to `build/`.
The internal-review setting omits fabricated publication/copyright metadata; an
ACM reference-format warning is expected until production metadata is supplied.

The review package accompanying this source archive provides verified source
receipts, plot data, a claim-to-evidence matrix and an adversarial review. The
paper is software only: exact INT8 kernel faults and ideal logical event
suppression at matched protected-bit counts. It makes no physical implementation
or radiation-rate claim. Partial-tier FKeras comparisons now include a separately
labeled 128-clip exploratory cohort; the larger common-cohort grid is unexecuted. The
small conditional comparisons do not establish universal optimizer superiority.

Historical collaborators are restored to the author block, pending their
approval of current authorship/order/affiliation and acknowledgments. This is a
complete internal draft, not an assertion of journal submission readiness.
