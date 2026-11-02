# NetSuRF software-only TECS internal review

Title: **NetSuRF: A Bit-Accurate Evaluation of Selective Protection in Quantized
Edge Neural Networks**.

This is a rebuilt manuscript, not a revision of the rejected conditional-theory
draft. The uploaded current Overleaf project supplies the `acmart` manuscript /
screen / review format and sectioned project structure. The new text follows the
approved original-to-current reconstruction audit and the software-only scope.
The historical draft and scientific bundles remain unchanged.

## Read and build

- `overleaf/main.tex`: complete manuscript; ten main sections, proofs and
  auxiliary loss curves in appendices, verified bibliography.
- `ADVERSARIAL_REVIEW.md`: novelty/validity objections, concrete fixes and
  unresolved submission risks.
- `OPTIONAL_WEEK.md`: prioritized missing comparisons with counts and qualified
  cost projections. No research runs were launched.
- `BIBLIOGRAPHY_VERIFICATION.md`: source and publication status for each citation.
- `CLAIM_EVIDENCE.csv`: manuscript claims, source artifacts, transformations and
  limitations.
- `evidence/`: newly derived verification receipts, numerical plot inputs,
  paired comparisons and figure/table provenance (included in review archive,
  not large historical prediction files).

Compile on Kona with the already installed TeX Live / ACM class:

```bash
cd /projects/netsurf/manuelbv/workspace/implementation-worktrees/paper1-engineering/publications/tecs_software_review_2026-11-02/overleaf
latexmk -pdf main.tex
```

Output: `build/main.pdf`. For Overleaf, upload the source ZIP and select
`main.tex` with pdfLaTeX. All figures and generated table rows are included;
model execution, Python and access to Kona are **not** needed to compile.

Offline evidence rebuild on the source server, never inference:

```bash
cd /projects/netsurf/manuelbv/workspace/implementation-worktrees/paper1-engineering
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=workspace/cache/matplotlib \
 /mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python \
 publications/tecs_software_review_2026-11-02/build_evidence.py
```

This refuses an existing derived evidence directory. `--refresh-draft` explicitly
replaces **only this unfinalized presentation's outputs**, rechecking original
source hashes first. Do not use it against a distributed archive and describe
the result as unchanged. Source paths resolve the historical receipts only as
locators; actual finalized artifacts and their hashes are rechecked. Source
datasets, masks, checkpoints and locks are never written.

## Scientific and editorial decisions

Protection cost means eligible-bit cardinality only. No physical mechanism or
physical cost/radiation claim is made. The main static grids retain all methods
and observed BERs. Missing partial tiers are identified, not fabricated. FKeras
is a validated score/mapping baseline with a small execution check, not a
completed publication-scale efficacy study. Historical qPolarGrad is described
faithfully and is not assigned unmeasured modern-model results.

One audit correction is substantive: ToyCar's 16 calibration units are complete
files, **3,136 windows**, not 16 windows. Its reported per-example construction
time is interpreted accordingly. No underlying result changed.

The title emphasizes what is actually measured, avoiding either an arbitrary new
acronym or an implication of a superior optimizer. The four historical authors
are restored rather than retaining a single-author placeholder. **Authorship
order, current affiliations, acknowledgments and contributions require the
collaborators' approval before submission.** No funding statement or acceptance
metadata has been invented. The ACM reference-format block is intentionally
disabled for this internal review; production metadata must be supplied by the
authors/publisher, not fabricated to suppress that class warning.

## Preservation and limits

Work is isolated on `paper1-engineering-reconstruction`; no merge/push or
scientific-source edits. The audit checks 125 finalized records / 592 consumed
files, not every unused file in every directory. Static summaries and paired
effects are recomputed from 7,616 raw rows. Exact large mask artifacts are
hash-verified; no masks are regenerated. Counts and checksums are audit material,
not the paper's central contribution.

This is an internal-review manuscript, not a claim of submission readiness.
Important risks remain: contribution beyond prior work, partial-tier FKeras
efficacy, low ToyCar field count, exposed inputs, one checkpoint per model,
approximate companions and unresolved conditional-versus-isolated benefit.
