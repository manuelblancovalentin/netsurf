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
| Figure 5: local allocation | Do bounded context-aware choices improve held-out risk? All six selected-minus-isolated pool intervals include zero. | `conditional_effects.json`, verified `M910/aggregate_summary.json` with original pool-run receipt chain; equal-pool t intervals. CE and AUC risk use separate axes. | Whole-network optimum, causal superiority, independent KWS pool samples across common-pool BERs. |
| Figure 6 (appendix): loss | Does auxiliary loss agree with task quality? ToyCar reconstruction MSE and file AUC can disagree. | Raw trials → `loss_curves.csv`; all methods/BERs, mean losses in separate units. Numerical interval columns are retained even though bars are omitted here. | AUC from window loss, cross-task pooled effects. |
| Table 1: models | Which actual kernel populations and cohorts were tested? | `static_inventory.json`; finalized manifests and scientific payloads; exact checkpoint hashes and sample indices retained. | Official benchmark submission or a reproduced prior fabricated design. |
| Table 2: paired effects | How large are per-example absolute minus MSB effects at both budgets? | `illustrative_paired_effects.json`; full BER/replication/trial/schedule matching; original Holm family retained. | Significance from marginal bars, new confirmatory testing, independent-mask population inference. |
| Table 3: FKeras masks | When does a Hessian score actually change the selected set? | `fkeras_compact.json`; saved symmetric differences and protected cardinalities at all four budgets. | Meaningful four-clip efficacy comparison. |
| Table 4: ranking cost | What computation was recorded for constructing the scores? | `empirical_metrics.json` runtime fields; shared per-example pass counted once; ToyCar 3,136 windows. | Platform-independent speedup or equal calibration workload across architectures. |
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
