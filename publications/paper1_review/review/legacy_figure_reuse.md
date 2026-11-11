# Original-figure reuse audit

The researcher supplied 24 PNGs from the original paper. Their exact paths and SHA-256 values are recorded in legacy_figure_reuse.json. The attachments are unchanged. Reuse here means preserving the original explanation and graphic structure where scientifically appropriate, not adopting historical results as new measurements.

## What was retained

| Original figure family | Treatment in this revision | Scientific boundary |
|---|---|---|
| draft_framework.png and earlier variants | Figure 1 restores the three large stages, brown headers, light grouped panels, documents, bit/mask motif and connecting arrows as editable vector artwork. | Frozen checkpoint and inputs replace training; exact software evaluation replaces HLS, synthesis and TMR. No hardware implementation is implied. |
| delta_computation.png / delta_computation2.png | Figure 2 retains blue numerical weights → pink encoded words → green delta matrix. Exact illustrative INT8 values replace the six-bit worked example. | Delta is an encoded numerical jump, not a task-damage prediction. The illustrative scale is not claimed to be shared by actual model tensors. |
| susceptibility_computation.png | Its additive score × jump × fault-indicator concept is explained by the first-order estimator equations. | It is not inserted as an exact multi-fault risk formula. Nonlinear inference and conditional effects cannot generally be replaced by that sum. |
| 2dAUC variants and three-benchmark comparison layouts | The manuscript retains the engineering questions, shared-axis small multiples, readable controls and direct method labels. Current numerical plots are unchanged. | No historical point, uncertainty band, selected best gain or runtime is imported. |

The original delta examples are internally consistent: signed six-bit two's-complement storage with scale 1/32 and zero point zero. Their sign coefficient is −1, followed by 1/2, 1/4, 1/8, 1/16 and 1/32. That specific illustration is not the current INT8 fault population and should not be relabeled as such. The replacement records its INT8 scale and bit ordering explicitly.

## What was not reused as evidence

- area_power.png, areapower_interpolation.png, netsurf_iccad.png and the hardware panels in mnist.png / tinyml_anomaly_det.png are excluded from the software-only contribution.
- The FORTUNE comparison PNGs use an all-MSBs-protected marker. This cannot stand in for a faithful matched implementation of FORTUNE's quantization, redundancy and voting behavior. FORTUNE remains properly distinguished in related work.
- The old Hessian plots cannot be relabeled as the current published-form FKeras comparison. Their historical score, ordering and runtime are different evidence.
- The old qPolarGrad advantages do not establish an advantage in the present experiments. The historical definition is not silently changed.
- The old ToyAD/MSE curves are not complete-file ROC-AUC. MSE is not substituted for the current anomaly-detection task risk.
- The boxplots and pruning correlations belong to historical protocols and models. They do not fill missing cells in the current reliability database.
- Historical “best gain” callouts are not copied into current figures. The current plots retain full measured grids and separately identified paired contrasts.

## Verification and preservation

This is a presentation-only revision. No scientific inference, ranking construction, fault schedule, asset lock, finalized bundle, numerical plot or data table is changed. The first delivery remains write-once; the new PDF and archives are delivered separately. The two diagrams use native LaTeX/TikZ so the Overleaf source remains editable, searchable and sharp at publication scale.

Validation consists of a clean LaTeX build, rendered inspection of both diagrams, exact encoded-jump arithmetic checks, archive checksums and a diff confirming that the measured data and numerical figure assets are unchanged.

