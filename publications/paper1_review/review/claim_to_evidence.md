# Scientific claim-to-evidence matrix (TECS revision, 11 November 2026)

Paths are relative to `publications/paper1_review/` unless absolute. This matrix describes the revised manuscript *Which Stored Bits Are Worth Protecting?*. The matrix for the previous draft is in git history (commit `fdba0d0`). No model inference was run for this revision: every number is recomputed from saved policy-field records by `scripts/tecs_grid_audit.py`, `scripts/tecs_kws_conditional_audit.py` and `scripts/tecs_residual_drift_check.py`, whose outputs are stored under `data/tecs_audit/` and hash-bound in `data/tecs_revision_provenance.json`.

All measured claims are conditional on one frozen checkpoint, one evaluation cohort and one constructed mask per deterministic ranking and budget.

## Evidence levels

- **Analytical:** exact under stated assumptions; not a network measurement.
- **Measured, conditional:** saved outcomes on a frozen checkpoint, executable, cohort, mask and fault law.
- **Offline engineering check:** computed from saved scores, masks and stored codes without inference.
- **Not claimed:** listed so that no reader infers it.

| Claim (manuscript location) | Evidence and locator | Scope and permitted conclusion |
|---|---|---|
| Protecting a bit is worth `p * V_b(p, P)`; this is the improvement potential of reliability theory (Sec. 5, Eq. 2; App. B.1) | Proof in `source/sections/10_mathematical_appendix.tex`; numerical identity test on a random 3-bit risk table in `tests/test_tecs_grid_audit.py`. Rausand and Hoyland ch. 5 (author's lecture version read): `I_IP(i) = I_B(i) * (1 - p_i)`. | Analytical; standard. No novelty claimed. Holds for independent faults only. |
| A p-independent order cannot be best at every p; optimal masks need not be nested (Sec. 5, Eq. 3 and the nested example) | Exhaustive expectations in `tests/test_tecs_grid_audit.py` and `scripts/package_review.py::math_checks`: crossover at `p = 1/15`; best single bit `{1}` (0.210 vs 0.215) but best pair `{2,3}` (0.100 vs 0.105) at `p = 0.1`. | Analytical counterexamples. They prove absence of a guarantee, not practical regret, and not that static rankings fail. |
| TFLite cross-entropy is computed from the 8-bit softmax output with a 1e-12 floor (Sec. 6.2; App. A.2) | `/projects/netsurf/manuelbv/workspace/implementation-worktrees/m12-fkeras-multibench/netsurf/adapters/mlperf_tiny.py` lines 460-463 and the contract string at 674-679. | Code-verified. Per-example loss is at most 5.55 nats or exactly 27.63. Accuracy is the classifier task metric in all conclusions. |
| No nonfinite output in 9,984 saved language-model outcomes; cap frequency not recorded in the main grids (Sec. 6.2) | `data/tecs_audit/conditions_with_quality.csv`, column `mean_failed_fraction` (all zero for `pythia`, `qwen`); source `derived/trials.csv` columns `failed_units`, `total_units`. | Measured. Finite-token cap frequency exists only in the 144-row Qwen follow-up (`data/qwen_crossover.json`). |
| Eligible population is INT8 kernels (TFLite) or Linear `qweight` codes (W8A8); embeddings, biases, scales, normalization are not faulted; Qwen `lm_head` is a separately stored copy (Sec. 3; App. A.1, Table 6) | `/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json` records for `distilbert_sst2_int8`, `pythia160m_int8`, `qwen3_0_6b_int8` (fields `eligible_stored_int8_population`, `quantization`, `model.artifact.source`, `reference_results`); each model's `quantization_manifest.json`; `w8a8_assets.py` Qwen head check; TFLite `mask_composition.csv` (10 kernel tensors per model). | Identity facts. Full-set clean references (90.8% vs 91.1%; PPL 26.3 vs 25.4; 12.5 vs 12.1) come from the readiness manifest, not from this study's cohorts. |
| Event accounting is consistent: proposed = applied + suppressed, protected runs never apply more faults than their unprotected partner (Sec. 7.4) | `data/tecs_audit/event_accounting.csv`: zero violations in 35,136 rows. | Measured over every saved policy-field record. |
| Task quality at 12.5% (Fig. 2) and the quoted accuracies and NLL increases (Sec. 7.1) | `data/tecs_audit/conditions_with_quality.csv` (`mean_quality`, `mean_loss`, `clean_*`); paired ResNet-8 contrast in `data/tecs_audit/pairwise_contrasts.csv` (`model=resnet8, risk=err, ber=0.003, coverage=0.125, abs_grad vs msb`: -0.1268 [-0.1692, -0.0844]). | Measured, conditional. |
| Highest useful BER per ranking (Table 3); usefulness is monotone in BER | `data/tecs_audit/conditions_with_quality.csv`, column `useful_primary`; monotonicity checked for every (model, method, budget). | Half-decade grid. Usefulness criterion fixed in `scripts/tecs_grid_audit.py` before regret was computed, but after the data existed and two slices had been plotted. |
| Decision-useful cell counts 19, 22, 30, 17, 21 of 32 and 4 of 20 (Sec. 7.1) | `data/tecs_audit/summary.json`, key `cell_counts`. | Measured, conditional. |
| Regret and resolved win/loss against absolute gradient (Table 4; Fig. 3; Sec. 7.2) | `data/tecs_audit/summary.json`, keys `regret_by_method` and `absgrad_vs_others` (`<model>:err` for classifiers, `<model>:loss` for language models and ToyCar); cell-level rows in `per_cell_regret.csv` and `pairwise_contrasts.csv` (`p_holm_absgrad_family_within_model`). | Measured, conditional. Regret of non-best rankings is biased upward because the best is chosen in-sample. "Strong default" refers only to the seven measured rankings. |
| Reversals inside the decision-useful region occur only on the language models and are dominated by positive gradient (Sec. 7.3) | `pairwise_contrasts.csv`, `p_holm_all_pairs_within_model`, restricted to `decision_useful_primary=1`: zero bidirectional pairs for ResNet-8, DS-CNN, ToyCar; one (random baseline) for DistilBERT; seven for Pythia; four for Qwen. | Measured, conditional. No causal attribution. |
| Partial-tier MSB-first is determined by tensor storage order (Sec. 4; Sec. 7.3) | Source `measurements/mask_composition.csv` of the three TFLite studies: at 6.25% ResNet-8 MSB-first covers sign bits of tensors 0007-0014 fully and tensors 0015-0016 not at all (0015 holds 294,912 of 618,880 bits); config `priority = [bit_significance_desc, estimator_score_desc, stable_identity]`. | Measured mask anatomy. FKeras-versus-MSB partial-tier contrasts are not interpreted. A randomized within-tier control does not exist yet. |
| Positive-gradient protection is worse than none in 6 (accuracy) / 17 (loss) DistilBERT cells and 8 Pythia cells (Sec. 7.4; Fig. 4) | `data/tecs_audit/protection_vs_unprotected.csv` (`ci_low > 0`); largest DistilBERT accuracy harm 0.0598 [0.0467, 0.0729] at `ber=0.03, coverage=0.0625`. | Measured, conditional; unadjusted paired 95% intervals. |
| The positive-part score is computed with the correct sign (Sec. 7.4; App. A.3) | `w8a8_rank.py` lines 214-224 (SHA-256 `18f3a15b...`, the hash recorded in each frozen plan): `pos = max((a+g)/2, 0)`, `neg = max((a-g)/2, 0)`, `where(delta >= 0, pos*delta, neg*-delta)`; exact XOR jump from stored codes. | Code-verified identity. |
| One-sided residual under the positive-gradient mask (Table 5) | `data/tecs_audit/distilbert_residual_drift.json`, rows with `coverage=0.125`; accuracies from `conditions_with_quality.csv` at `ber=0.01`. | Offline engineering check on DistilBERT only. Uses surrogate FP32 gradients on 64 fine-tuning-split sentences. A mechanism consistent with the measurements, not a proof of causation, and not checked on Pythia or Qwen. |
| Local exact re-selection moves risk by too little to matter (Sec. 7.5; App. B.2; Fig. 5) | `data/tecs_audit/kws_conditional_audit.json` (`summary`): background 7.08 nats/clip, attainable range median 0.013, runner-up margin below two paired SE in 32/32 pools, in-sample 0.0053 vs held-out 0.00008, transitivity residual mean -1.2e-4; source payload SHA-256 `e9e777f1...`; held-out contrasts in `data/kws_pool_statistics.json`. | Measured, conditional, at a non-useful BER. Supports no claim that conditional selection is better or worse than isolated selection. |
| Score-fidelity correlations and their intervals (Sec. 7.6; Table 7) | `data/empirical_validation.csv`, `data/toycar_auc_correlations.csv`, `data/fkeras_candidate_audit_v1.json`. | Sample-design statistics on 320 stratified bits; not whole-network correlations. |
| Score-construction times (Sec. 7.6) | `data/score_construction_costs.csv`. | TFLite models only; transformer construction time was not recorded. |
| Source-study Holm p-values were resolution floors (App. A.4) | Source `paired_comparisons.csv`: raw p = 9.999e-05 = 1/10001, family 112; W8A8 plan: 10,000 draws, family 160. | Explains 0.0112 and 0.016 in the previous draft. |
| Comparisons outside the useful region (App. C) | `data/wide_paired_comparisons.csv`, `data/qwen_crossover.json`, `data/fkeras_paired_comparisons.csv` and the focused studies listed in `EVIDENCE_HANDOFF.md`. | Recorded for completeness; explicitly excluded from the conclusions. |

## Deliberately not claimed

- That a universal ranking has been disproved empirically, or that one exists.
- That absolute gradient is optimal, or better than search-based or learned allocators.
- That conditional (Monte Carlo-guided) selection is better or worse than isolated selection.
- Any statement across trained checkpoints, calibration draws or fresh cohorts.
- Any hardware cost, physical mechanism (ECC, TMR), area, power or radiation-rate result.
- Anything about `legacy_qpolargrad`; it is absent from the comparison and from the text.

## Unverified items a submitter must check

- TECS page limit and formatting (the author-guidelines page returned HTTP 403).
- Birnbaum (1969) bibliographic details were taken from secondary sources; the original chapter was not inspected.
- FORTUNE and EISFINN descriptions rest on the earlier bibliography audit, not on a re-reading in this revision.
- Authorship, corresponding author and exposure disclosure.
