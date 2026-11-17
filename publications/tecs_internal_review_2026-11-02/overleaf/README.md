# NetSuRF TECS internal review source

Complete internal-review manuscript consolidated through finalized M11 evidence on 2 November 2026. The authoritative starting point is the user-supplied Overleaf ZIP, preserved unchanged as original_overleaf.zip in the enclosing review package.

Open main.tex as the Overleaf root document. The ACM acmart manuscript/review format, TECS journal selection, nine numbered sections and mathematical appendix are preserved. The exact requested title is in include/metadata.tex.

## Build

Use a LaTeX distribution providing acmart, Libertine, Inconsolata, newtx, TikZ, PGFPlots and placeins. The verified local build uses pdfLaTeX and BibTeX through latexmk:

    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

No Python, TensorFlow, datasets or experiment execution is needed to compile the manuscript. PDF figures and table fragments are included.

## Read the evidence correctly

The positive exploratory frozen-pool replication is reported alongside all six later cross-pool/BER arms, whose selected-versus-exact-isolated marginal intervals include zero. Exact response tables support selection-data interaction attribution, not attribution of unmeasured final response subcubes. M11 includes mathematical bounds and bounded numerical checks, not a large-scale efficacy campaign.

The enclosing review package contains the claim matrix, source identities and checksums, table data, figure provenance, bibliography verification and adversarial review. The historical runs remain at their original locations; checkpoints, datasets and proprietary assets are not redistributed.

## Internal review and submission

There are no placeholder results or unfinished result sections. Submission readiness still requires team review of novelty, uncertainty, practical effect size, authorship and journal requirements. Only the original named author has been retained; coauthors must be confirmed before submission.

The standard acmart footer saying “Manuscript submitted to ACM” is template boilerplate; no submission was performed. ACM reference-block suppression and omitted CCS concepts are deliberately retained from the internal review setup. Set submission metadata and verify current journal requirements before uploading to a journal.
