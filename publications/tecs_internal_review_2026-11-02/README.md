# NetSuRF TECS internal review package

This package contains the complete manuscript consolidated from the user-supplied Overleaf project and finalized NetSuRF evidence through M11. Its central conclusion is deliberately bounded: conditional vulnerability changes, but independent cross-pool evidence does not yet resolve general superiority over exact isolated-bit allocation.

## Start here

- NetSuRF_TECS_internal_review.pdf: compiled manuscript.
- NetSuRF_TECS_overleaf.zip: uploadable source; main.tex is the root.
- ADVERSARIAL_REVIEW.md: concrete objections, corrections already made and optional additional-week priorities.
- BIBLIOGRAPHY_VERIFICATION.md: primary sources and publication-status qualifications.
- evidence/CLAIM_EVIDENCE.csv: 24 scientific claims, source selectors and payload hashes.
- evidence/DISPLAY_PROVENANCE.json: every numbered table and figure, with hashes and derivation.
- evidence/sources.json: 125 finalized bundle/ledger verification receipts, paths and recorded source identities.
- evidence/FINAL_VALIDATION.json: build, arithmetic, pooled-statistic and source-integrity checks.
- original_overleaf.zip: untouched supplied authority, SHA-256 3d8fba9df4891bd05792b9a8157712c30fc6dd837526dc7c13e9e2a4ed84a5f6.

The evidence directory also contains full-precision selected numerical inputs, every plotted pool contrast, all static paired-comparison CSVs, table extracts and source revisions. It is not a replacement for the original scientific archives or a redistribution of models/datasets.

## Rebuild only the manuscript

From the overleaf directory, with a normal LaTeX installation:

    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../build main.tex

The successful Kona build uses pdfTeX/TeX Live 2023 and acmart 2.03, with the fonts installed by the researcher. It requires no Python or neural-network environment. Upload the contents of NetSuRF_TECS_overleaf.zip to Overleaf and select main.tex.

## Recheck the evidence on Kona

These manuscript-only programs read the existing finalized archives. They do not load models, generate masks, or run fault experiments. Their output is confined to this publication directory. They expect the historical paths in sources.json to remain available.

    cd /projects/netsurf/manuelbv
    /mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python publications/tecs_internal_review_2026-11-02/prepare_evidence.py
    cd publications/tecs_internal_review_2026-11-02/overleaf
    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../build main.tex
    cd ..
    /mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python complete_review.py

No installation commands are executed or required for the already configured server. Away from Kona, compiling the source ZIP works independently; rechecking original evidence additionally requires access to the recorded historical archives.

## Verification scope

All cited canonical scientific payload digests are checked. All inventoried M9.10/M11 derived files and conditional bundles are checked; all 112 campaign selection/final payloads, manifests and archived checksum identities are checked. Consumed static CSVs are checked. Unused multi-gigabyte prediction files are not claimed to have been rehashed during manuscript authoring.

Eighteen aggregate pool contrasts and their marginal t intervals are independently recomputed from saved individual-pool means. The analytical counterexample and sharp-bound construction are checked with exact rational arithmetic. No new empirical data are generated.

Historical dirty-state fingerprints remain in provenance rather than being relabeled as clean committed runs. The internal package records the source actually used, not the current HEAD of a worktree.

## Remaining review decisions

Read ADVERSARIAL_REVIEW.md before submission. The major risks concern novelty boundaries, practical effect size, few independent non-KWS pools, imperfect companions, candidate/data exposure and the missing equal-compute search baseline. A complete draft does not resolve those scientific limitations.

Coauthors, acknowledgments, current TECS requirements and final publication metadata require researcher approval. The acmart submission footer is standard template text; no manuscript has been submitted. There are no claimed hardware measurements, radiation rates or unexecuted large-scale M11 results.

## Preservation

No experiment source code, researcher TOML, asset lock, checkpoint, finalized bundle or Git history was changed. All authored files are under publications/tecs_internal_review_2026-11-02. No package installation, training, model inference, merge or push was performed.
