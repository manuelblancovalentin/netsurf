# Figure and table provenance: reader questions and boundaries

The machine-readable counterpart is `evidence/figure_table_provenance.json`.
All plotted inputs, paired effects and exact source receipt hashes are in
`evidence/`. Internal identifiers below are supplementary locators, not labels
on the manuscript's scientific figures.

| Item | Scientific question / supported claim | Source and derivation | Does not establish |
|---|---|---|---|
| Figure 1: flow and encoding | How does one stored-bit fault become a measured protected-inference outcome? A small-magnitude code can have a large sign-bit jump. | Original diagram of executable adapter/runner contracts; exact analytical -3 INT8 example at scale 1/16. No borrowed historical figure or measured parameter. | Hardware implementation, physical cost, universal gradient validity. |
| Figure 2: score validation | Do scores order damaging isolated bits on the same candidates? Associations differ by task and score. | `empirical_metrics.json`, source `emp_resnet`, `emp_kws`, `emp_toycar`; saved positive-damage Spearman and within-stratum bootstrap intervals for 320 candidates. | Population-wide ranking or protection superiority. ToyCar's target here is MSE, not AUC. |
| Figure 3: budget | What quality do two available logical budgets retain at fixed BER? Gradient, MSB and random masks behave differently across models. | All methods from `static_curves.csv`, BER .001/.01; recomputed raw field means and t intervals. Clean/full control shown separately. | Missing 6.25%/18.75% results, uncertainty over new models, interpolation as measurement. |
| Figure 4: BER | How does quality vary over every observed fault-probability condition? Relative policy usefulness changes across architectures/regimes. | Same raw-derived table, coverages .125/.25, complete measured BER grids. Classifiers 32 fields; ToyCar eight. | Physical SEU rate or a monotone ToyCar response. |
| Figure 5: matched KWS comparison | Does within-tier sensitivity change protection quality? Five baselines including one published-form FKeras; all four budgets and three BERs retained. | `static_fkeras128`, 128 clips and 32 fields/BER; verified raw curves and paired effects. Historical repository variant remains archived but is not a manuscript baseline. | Pooling with 1000 clips; results for absent random/jump policies; universal winner. |
| Figure 6: local allocation | Do bounded context-aware choices improve held-out risk? All six selected-minus-isolated pool intervals include zero. | `conditional_effects.json`, verified `M910/aggregate_summary.json` with original pool-run receipt chain; equal-pool t intervals. CE and AUC risk use separate axes. | Whole-network optimum, causal superiority, independent KWS pools across common-pool BERs. |
| Figure 7 (appendix): loss | Does auxiliary loss agree with task quality? ToyCar MSE and file AUC can disagree. | Raw trials → `loss_curves.csv`; all methods/BERs. Numerical interval columns retained. | AUC from window loss; cross-task pooled effects. |
| Table 1: definitions | What exactly is ranked? | Mathematical definitions, pinned FKeras source and historical qPolarGrad audit. | Every method executable on every architecture. |
| Table 2: models | Which actual populations/cohorts were tested? | `static_inventory.json`; finalized manifests and payloads. | Official benchmark submission. |
| Table 3: paired effects | How large are absolute-gradient minus MSB effects? | `illustrative_paired_effects.json`; complete pairing identity and original Holm family. | Significance from marginal bars; independent-mask population inference. |
| Table 4: FKeras masks | When does a Hessian score change the selected set? | `fkeras_compact.json`, 128-clip study, all four budgets. | Whole-network optimum or formula superiority. |
| Table 5: targeted FP32 effects | Does isolated calibration damage guide the 1024-bit, negative-logit-only classifier allocation? | `distilbert_paired_loss.csv`, recomputed from 96 hash-verified field records, all 12 original contrasts and original Holm family. All effects and intervals are in nats of 30-capped cross-entropy; 144 capped dependent example outcomes and extreme-field contributions are retained. | Full-transformer reliability; causal interactions; input or random-mask population uncertainty. |
| Table 6: ranking cost | What computation was recorded? | `empirical_metrics.json`; shared passes counted once; ToyCar 3136 windows. | Platform-independent speedups or equal calibration workload. |
| Supplementary heatmap prototype (not a main figure) | Where does FKeras improve over fixed MSB? | `fkeras_heatmap_cells.csv`: 3 BERs × 4 coverages × 2 metrics; all measured cells from the same 128-clip study. Positive is improvement; accuracy in pp and CE reduction in nats. Every cell retains paired 95% CI and original Holm p. | A new experiment, interpolation, or finalized common-grid study. Coincident whole-tier masks have exact zero effects. |
| Section 6 risk table | Why can isolated ordering lack a universal guarantee? | Exact synthetic eight-state table, checked by enumeration; not a neural-network measurement. | That every static ranking fails or realistic decision regret is large. |

The compiled manuscript uses 400-dpi PNGs from the same plotting calls, with
vector PDFs retained for later typesetting. A local PDF-embedding/rendering issue
clipped BER labels in an intermediate compile; the raster-backed final figures
were inspected at page scale. This is a presentation correction only: numerical
plot inputs and all source payloads are unchanged.

Protected cardinalities and actual mean applied/proposed flips are also retained
in `static_curves.csv`; expected flip counts in the text are analytical Np values,
not substituted realized-event measurements. Descriptive bars intersect t
intervals with the metric's [0,1] range; untrimmed values remain in the tables.

DistilBERT source/figure-table bindings are independently recorded in
`evidence/distilbert_table_provenance.json` and `distilbert_audit.json`; its
1024-bit FP32 subset is never combined with INT8 whole-kernel axes. All 768
archived accuracy and capped-loss outcomes were additionally recomputed from
full logits and hash-checked labels (`distilbert_logit_verification.json`).
This is offline verification, not another inference experiment.

The independent DistilBERT replication paragraph uses the finalized
validation-cohort plan and 32 new matched fields per BER. Its paired capped-loss
effects are not pooled with Table 5's original training-calibrated study.

The new targeted language-model table is sourced from three separate finalized
96-bit pilot bundles. Its all-token NLL and nonfinite-failure fractions are
the archived analysis values; finite-token NLL is recomputed from the saved
target-logit/logsumexp pairs across all 32 high-BER field records. The
read-only verification script and source/ZIP hashes are recorded in
evidence/targeted_cross_precision_receipts.json. This table does not estimate
whole-model reliability or full-corpus perplexity, and it cannot attribute
the observed mask contrast to multi-bit interactions.
