# Compact software-reliability internal draft

**Is There a Universal Bit Ranking? Evaluating Fault Protection in Quantized Neural Networks**

Start with `NetSuRF_internal_review.pdf` in the delivery archive. The source entry point is `source/main.tex`; the separate Overleaf ZIP is self-contained. NetSuRF remains the software framework's name, not a claimed universally superior allocation method.

## Current revision

The compact third delivery is 13 pages in the unchanged ACM manuscript/review format, down from 24. The conclusion is on page 10; references occupy the rest of page 10 and part of 11, followed by appendices through page 13. This is not an IEEE two-column page count. No fonts or margins were reduced to achieve the compression.

Four main figures show the workflow, six-model BER curves, protection-budget curves, and the complete 32-pool conditional-selection distribution. Three numbered equations remain in the main text; derivations and numerical controls are in the appendix. The delta matrix and all qPolarGrad discussion are absent from the manuscript. Previous artwork and audits remain in the review archive for provenance only.

The draft follows the engineering sequence: frozen network → exact stored-bit perturbations → ranking → fixed-budget masks → paired fault injection → task-level reliability. It does not report implemented hardware correction or radiation qualification. The four-author list remains provisional pending researcher confirmation of authorship, affiliations and corresponding author.

## Evidence and boundaries

- Six current INT8 configurations: ResNet-8, DS-CNN, ToyCar, DistilBERT, Pythia-160M and Qwen3-0.6B.
- 1,254 measured conditions and 35,136 labeled policy/field observations, not that many independent fields. Complete-tier duplicate masks and reused controls are documented in `EVIDENCE_HANDOFF.md`.
- Published-form FKeras on three TFLite benchmarks, treated as one method among the structural and gradient comparators. Missing transformer FKeras and matched PrioriFI/RL allocation measurements remain limitations.
- Paired reversals show that preferences between measured methods depend on operating conditions. They do not establish that every static ranking fails or identify the cause of each reversal.
- The 32-pool KWS test changes allocations but does not resolve an advantage of conditional selection over exact isolated selection. Both improve slightly over the original gradient allocation in separate secondary comparisons.
- ToyCar uses complete-file AUC risk. Classification CE and capped token NLL retain separate units and cohorts; the earlier restricted FP32/BF16 pilots are not pooled into these curves.

The six-page complete budget-grid supplement remains available. No numerical curve has been invented to fill a missing cell. Current compact figures use the same saved means and intervals as the preceding delivery, with reduced labels and lighter styling.

## Review materials

`review/compact_revision_notes.md` maps the requested edits to changes. `review/claim_to_evidence.md` and `review/figure_table_manifest.md` describe the current draft. `review/compact_figure_provenance.json` binds the new rendering script, exact inputs, filters and outputs. Earlier source/figure audits remain available but do not dictate current manuscript contents.

Related work includes FORTUNE, FKeras, PrioriFI, EISFINN, progressive bit search, DeepHammer, FlipLLM, genuine RL protection allocation, selective hardening, and the published FCCM predecessor. Attack discovery and expected-risk allocation are distinguished without dismissing learned protection. The public selective-ECC release is identified as a companion dataset, not independently verified peer-reviewed publication.

## Reproduce offline

~~~bash
cd /projects/netsurf/manuelbv/workspace/implementation-worktrees/paper1-complete-review-v1/publications/paper1_review/source
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
~~~

`scripts/build_compact_figures.py` reads only the included saved derived data; it performs no model inference. Earlier source-audit scripts require the original artifacts at the recorded Kona paths. Large masks, checkpoints and trial payloads are linked by their manifests, not redistributed or claimed to be independently rehashed by every manuscript build.

`scripts/package_review.py` validates citations, outputs, printed mathematics and provenance, and creates a new write-once delivery with source/full ZIPs and SHA-256 manifests. It does not import experiment code. Previous deliveries remain unchanged.

## Venue and remaining risk

TECS remains plausible but ambitious; the central risk is whether the controlled methodology and decision evidence contribute enough beyond prior ranking comparisons and the published FCCM flow. The compact format does not resolve that scientific question. See `review/compact_venue_literature.md` and the adversarial review. TCAD's verified current hard limit is 14 IEEE two-column pages, not nine; the current TECS page cap could not be independently verified. This package retains the supplied ACM format and does not choose a venue by silently changing typography.

No inference, training, new campaign, asset change, automatic merge or push was performed. Original evidence and supplied source files are unchanged.
