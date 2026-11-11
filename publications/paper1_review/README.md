# NetSuRF — complete software-methodology internal draft

Start with the compiled NetSuRF_internal_review.pdf in the delivery archive. The source entry point is source/main.tex; the separate Overleaf ZIP is self-contained.

**Title:** NetSuRF: Bit-Accurate Reliability Analysis and Budgeted Protection of Quantized Neural Networks.

The draft follows the original engineering sequence: frozen network → exact encoded perturbations → susceptibility ranking → protection budget → paired fault injection → task-level reliability. It uses the original manuscript's direct technical organization, but does not inherit its unsupported estimator, hardware or radiation claims. It restores the original four-author list provisionally; the researchers must confirm authorship, affiliation and corresponding author before submission.

## What is delivered

- Complete ACM TECS internal-review source and compiled PDF, five main figures, five main/appendix tables and mathematical appendix.
- Six-page full-budget figure supplement, plus vector PDF/SVG and PNG versions of the statistical figures.
- Current evidence for ResNet-8, DS-CNN, ToyCar, DistilBERT, Pythia-160M and Qwen3-0.6B.
- 1,254 measured conditions and 35,136 labeled policy-field observations. These are not 35,136 independent fault fields; the accounting and identical-mask controls are documented in EVIDENCE_HANDOFF.md.
- Published-form FKeras on all three TFLite benchmarks. Missing transformer FKeras and matched PrioriFI-derived results remain explicit.
- A visible negative primary result for conditional versus exact isolated selection on 32 independently sampled KWS pools.
- Original-to-current contribution audit, claim-to-evidence matrix, figure/table provenance, hostile scientific review, venue/novelty assessment and prioritized optional-week ledger under review/.

## Main conclusion and its boundary

The preferred measured allocation changes with checkpoint/task, BER and budget. Exact isolated damage has no general guarantee of equaling conditional protection value. These results support checkpoint- and operating-condition-specific reliability assessment. They do not prove that all static methods fail, that interactions cause every observed reversal, or that the conditional selector is universally better.

The full-eligible INT8 studies are distinct from the earlier restricted FP32/BF16 pilots. ToyCar uses complete-file ROC-AUC; token and classification losses are never pooled with it. The transformer NLL cap and Qwen's finite-loss saturation diagnostic remain explicit. The obsolete Pythia high-BER run with clean-state mismatch is excluded in favor of the new clean-gated finalized run.

## Bibliography

FORTUNE is included as quantization plus redundant MSB protection and majority voting, not as a measured ideal-suppression baseline. FKeras and PrioriFI are described using their published objectives. EISFINN, Ranger, FT-CNN, selective TMR, the published FCCM predecessor and the public selective-ECC companion release are compared explicitly. The latter is labeled a dataset/source release, not a verified peer-reviewed manuscript. Model and dataset references are included. Unused inherited bibliography records are retained in source but do not appear in the rendered reference list; the review specifies which cited entries were verified.

The Zotero skill's read-only check found no active local service, so reference verification used original author/publisher records. Hugging Face paper metadata was used read-only for the model-reference cross-check. No reference manager configuration, model asset or environment was changed.

## Reproduce without inference

From this worktree:

~~~bash
cd /projects/netsurf/manuelbv/workspace/implementation-worktrees/paper1-complete-review-v1/publications/paper1_review/source
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
~~~

Figure scripts under scripts/ read saved artifacts; their canonical paths are recorded in the script and provenance JSONs. They cannot be replayed on a different host without those source artifacts. The delivery includes the compact derived data and publication figures; it does not redistribute all large model, fault-event and mask payloads. Their original manifests remain authoritative. Fresh verification of the inputs actually read is distinguished from provenance links to omitted bytes.

scripts/package_review.py checks citations, required outputs, manuscript placeholders, printed mathematics and source diff, then creates a new write-once delivery directory with Overleaf/full ZIPs and SHA-256 manifests. It never imports model or experiment code. It requires an unused output directory.

## Submission assessment

TECS is a defensible but ambitious choice. The research contribution is an exact, reproducible evaluation methodology and quantified limits of transfer across operating conditions. Its novelty beyond the published FCCM flow and existing fault-analysis work remains the main reviewer risk. Journal of Systems Architecture is the closest alternative, not a guaranteed easier acceptance. The reviewer reports give concrete fixes rather than promising acceptance.

No scientific inference, new campaign, training, asset change, automatic merge or push was performed for this draft. Existing finalized evidence and the supplied manuscript ZIP/PDF remain unchanged.
