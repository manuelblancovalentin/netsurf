# Adversarial scientific review of the rebuilt internal draft

## Review scope and bottom line

This review covers the rewritten Introduction, Related Work, Design Flow and Fault Model, Susceptibility, Conditional Diagnostic, Experimental Methodology, Results, Discussion, Conclusion, mathematical appendix, and metadata. It also checks the original rejected manuscript, historical conditional artifacts, and the finalized bounded-scaling analysis. It is an independent scientific review, not an external peer-review decision. The 23-page assembled PDF was visually inspected on pages4,5,14,15,17 and22: the workflow, encoding, main reliability and paired-result figures and corrected attribution table are legible and not clipped. A caption correction was requested to call the budget figure's whiskers “error bars,” not “bands.” LaTeX reports no overfull boxes in this build; ACM submission metadata still requires author approval.

The rebuilt paper has a defensible engineering question, actual large-model measurements, explicit executable scope, a correct conditional-risk formulation, and useful negative results. It is no longer dependent on a claimed universal dynamic optimizer. It is also not automatically a strong TECS submission. Its main exposure is **incremental novelty**: selective protection, gradients, Hessian sensitivity, feedback-guided injection, and context-dependent interactions all have prior art. The paper must be judged on whether its matched, exact-storage comparison yields a practically useful and reproducible protection decision, not on the novelty of conditional expectation or a new name for a design flow.

The draft's restrained claim is supportable: the relative quality of particular frozen masks changes with the tested operating condition, and clean isolated effects do not possess a general protection-optimality guarantee. It must not expand that claim to “no static ranker can work,” “dynamic protection is required,” or “interactions caused every observed reversal.”

## Highest-priority reviewer objections and concrete fixes

| Likely objection | Evidence-based response or manuscript fix | Residual risk |
|---|---|---|
| “This extends an existing framework but contributes no new algorithm.” | State the contribution as exact encoded execution plus a matched decision-oriented benchmark and a controlled test of isolated versus conditional allocation. Show practical effects and costs, including null conditional results. Explicitly separate inherited FCCM design-flow concepts. | High: a tools/evaluation contribution still needs a clear reusable insight and sufficient venue fit. |
| “The paper attacks an implausibly broad claim that no prior method actually made.” | Restrict the counterexample to absence of a general guarantee for a context-independent **isolated** ordering. Do not claim FKeras ignores off-diagonal curvature or PrioriFI is a static optimizer. | Avoidable through the present careful wording. |
| “The largest-model curves are a clipping artifact.” | Report finite cap frequency and uncapped diagnostics where actually recorded; preserve 30-nat capped estimands. Qwen has a separate diagnostic; Pythia does not. Do not causally attribute the Pythia reversal or advertise enormous penalized perplexities. | Important remaining limitation; uncapped Pythia diagnostics require a new separately authorized study, not retrospective invention. |
| “The input cohorts are small and already exposed.” | Report fixed cohort sizes and earlier exposure. Keep field-level confidence intervals conditional on those cohorts. A new fault field is not a new trained network or independent language corpus. | High for generalization beyond the evaluated checkpoint and text segment. |
| “Your random baseline has pseudoreplication.” | TFLite studies contain four random mask seeds, while deterministic methods have one unique mask per budget. Label pooled intervals as conditional on the recorded mask mixture and preserve replication-level data. Do not infer random-mask population uncertainty from 32 pooled fields alone. | Can be fixed in wording without rerunning data. |
| “Two leading Hessian eigenpairs are not FKeras in general.” | Give the published equation and strict tier ordering, name the architecture-specific continuous surrogate, report two eigenpairs and solver gates, and characterize it as a bounded implementation of the published projection—not exhaustive curvature. Report different surrogate objectives explicitly. | Eigenpair truncation and companion mismatch limit interpretability; no claim that this is the best FKeras tuning. |
| “Conditional optimization changes masks but does not improve protection.” | Report the independent 32-pool null result in the abstract/results, not only in limitations. Treat the method as a diagnostic and bounded comparator. Do not use selection-side objective decreases as held-out success. | This limits a separate optimizer-superiority paper; it does not invalidate the evaluation paper. |
| “Prior art already contains context-dependent sensitivity.” | Cite FKeras, PrioriFI, selective hardening, EISFINN, and the public selective-ECC release accurately. Credit established Boolean-function/reliability foundations. Compare the executable intervention, unit of allocation, fault law, and independent outcome rather than claiming first discovery. | High novelty risk, especially against the companion release's allocation and interaction diagnostics. |
| “Why no activations or device-level radiation validation?” | State persistent stored-weight corruption as the selected software fault domain. Do not justify it by unsupported weight-versus-activation dominance. Treat logical BER and ideal suppression as abstractions, not radiation qualification. | Scope limitation, not a mathematical defect; broader claims would make it fatal. |

## Mathematical review

The exact affine two's-complement jump in the draft is correct. For an INT8 sign bit, the coefficient is negative, and the affine zero point cancels. The draft correctly distinguishes a scalar's exact numerical delta from the potentially nonlinear network/task response.

For independent Bernoulli faults on unprotected bits, the protection-benefit identity

\[
J_p(P)-J_p(P\cup\{b\})=p\,\mathbb E_G[R(G\cup\{b\})-R(G)]
\]

is exact. The add-protection comparison is valid for arbitrary finite scalar task risk, including complete-cohort AUC risk. The swap identity is also correct when its common background excludes the two exchanged bits and all coordinates protected under both masks. The same master-field unit must contain the sum along any multi-swap path before uncertainty is computed.

The three-bit counterexample is valid for positive BER: with
`R=1.2*z1+z2+0.1*z3+3*z2*z3`, protecting bit one gives `1.1*p+3*p*p`, protecting bit two gives `1.3*p`, and protecting bit three gives `2.2*p`. The first two cross at `p=1/15`. At `p=0` all masks tie; this does not affect the lack-of-guarantee statement. The example is a constructed task loss, not a measured network or bounded [0,1] risk.

The response-table Bernoulli integration and decomposition are algebraically correct. The manuscript should explicitly define the outside field as supported on `bits \ (C union P_out)`. The finite table retains all candidate interaction orders. Its outside expectation is sampled; its selected optimum is restricted to a frozen pool and selection sample. Neither a whole-network optimum nor population-risk-optimal mask follows.

An independent-event law does not make task consequences additive. Conversely, nonadditivity does not prove that any particular sophisticated score will improve held-out risk, nor that a deployment mask must change over time. Those qualifications are correctly present in the rebuilt draft.

## Method and execution precision

### FKeras

The manuscript uses only the published projection `sum(lambda_i * (v_i.T theta) * v_i)` with parameter-component magnitude within significance tiers. It does not substitute the repository's different effective accumulation, a diagonal Hessian, or gradients of the integer program. Complete-tier mask equivalence at 12.5% and 25% is correct for the homogeneous INT8 population and provides a strong executable control.

The production configuration uses two signed largest-magnitude eigenpairs, tolerance `1e-5`, maximum 100 eigensolver iterations and 150 HVP products. These choices should be stated in the experimental methods or appendix. In the inspected implementation, the FKeras score timer starts **before** continuous companion construction and the eigensolve; the small measured construction times are not cached-eigenpair times. They still exclude unrelated model initialization, experiment inference, and export, and must not be presented as full campaign cost.

The continuous float64 Hessian companions omit activation rounding/clipping. This is a different surrogate from the float32 straight-through gradient companion. The bounded ResNet audit reports class agreement on 61/64 examples but nontrivial probability/loss discrepancy (reference CE 0.91815 versus continuous CE 0.62556 nats; maximum probability discrepancy 0.33188). That is a limitation of the susceptibility surrogate, not failure of the authoritative TFLite inference. High class agreement alone must not be described as near-exact forward equivalence. ToyCar's Hessian objective is window MSE, while its protection objective remains complete-file AUC risk.

### qPolarGrad and the original manuscript

The new draft correctly treats the recovered `legacy_qpolargrad` expression separately from the printed paper expression, which contains an additional finite-perturbation factor. Neither is validated by the original paper's claimed superiority. The original strict tier-first allocation description and its claimed advantage over all-MSB protection at exactly one six-bit tier are mutually inconsistent under a shared population/budget. The historical result must not reappear as current evidence.

### Metrics and uncertainty

The draft correctly separates classification CE, token NLL and file-level `1-AUC`. AUC tie half-credit and complete-file aggregation are indispensable. MSE is not an additive substitute for AUC. Probabilities/CE clipping and the 30-nat transformer failure/cap convention must remain explicit. “No failures” does not imply “no capped finite losses.”

The primary independent units differ across analyses: fault fields within a fixed mask/cohort; independently sampled candidate pools; and independent input cohorts or checkpoints, which are not replicated by these trials. These levels cannot be exchanged to increase nominal sample size. Paired comparisons must use shared proposed-field identities, not independent-policy error bars. Repeated full-protection values do not create additional clean-model uncertainty.

## Historical conditional evidence independently reverified

`data/conditional_historical_summary.csv` is regenerated from the finalized signed-attribution study. The audit verified its canonical scientific payload, finalization marker and listed checksums, then rehashed all 112 linked selection/final bundles: 1,908 listed scientific files, 2,994,477,591 bytes. Six campaign summaries cover 56 pool/BER conditions. KWS pools recur across BERs; the row count is not 56 independent candidate populations.

Pool means and marginal Student-t intervals were independently reconstructed from saved per-pool held-out contrasts. Strict-order reversal counts were independently resummed. All six selected-minus-exact-isolated intervals include zero. The main mechanism result is selection-side: accounting for faults outside the candidate pool removes most clean-additive ordering reversals, while candidate-internal corrections are relatively small in these recorded bounded subproblems. This is not an attribution of final-cohort regret or a causal redundancy analysis.

The isolated/conditional distinction is scientifically meaningful even where no superiority is resolved. What is not justified is selecting favorable pools, treating repeated pools as independent, treating mask-pair reversals as independent Bernoulli observations, or claiming that a selection winner must transfer.

## Bounded-scaling evidence and its limit

The finalized KWS and ResNet bounded-scaling analyses now exist. Their analysis payloads, finalization markers and listed analysis checksums were freshly verified for this review; their linked source payload identities are retained, but all underlying source bundle bytes were **not** freshly rehashed by this optional export. This narrower scope is explicit in `data/m11_bounded_scaling_provenance.json`.

Each model has 48 pool/BER rows: four candidate-pool sizes (6, 12, 32, 64), four pool seeds, and expected applied background counts 10, 100 and 1,000. At candidate sizes 6 and 12, every saved outside-additive selection regret against exact restricted selection is zero. At sizes 32 and 64, there is no exact optimum or measured exact approximation error. All 54 saved simultaneous equal-pool intervals per model include zero. Identically zero natural-field observations do not prove equality of expected protection risk.

The computational contrast is useful but not an optimizer result: 12-bit exact selection requires 32,782 cohort requests per pool and approximately 1.65–2.04 hours in the recorded runs; the 64-bit outside-additive procedure uses 586 requests and approximately 105–127 seconds. The latter estimates a different, truncated objective; cost alone cannot establish equal decision quality.

The mathematical remainder bound is a worst-case statement for bounded scalar risks. It becomes uninformative in regimes where it approaches the risk range, and it is not a certificate of the measured surrogate or an automatic scaling theorem for whole-network optimal protection. This material is appropriate for a compact appendix, not a claim that the large-model conditional optimizer has been demonstrated.

## Authorship and submission readiness

The supplied old paper names Manuel Blanco Valentín, Houxuan Guo, Xiuyuan He, and Seda Ogrenci in that order; the rebuilt metadata restores those names and Northwestern ECE. Current authorship and contributions must still be approved by the researchers. The original cover identifies Houxuan as corresponding author while the manuscript footnote identifies Seda. The internal draft should not silently resolve that disagreement. No new funding or acknowledgment should be invented.

Before submission, the authors should approve the final claim boundary, the manuscript's distinction from the FCCM publication, author/contribution metadata, and the exposure/capping limitations. The draft is appropriate for internal review now. It is not evidence that all reviewer objections have been eliminated.

## Final Results and appendix cross-check

The main-text BER0.01 means at12.5% protection match `data/reliability_conditions.csv` to the printed precision across all six models. The Pythia paired budget reversal, two partial-tier FKeras comparisons, Qwen crossover and new KWS primary/secondary results match their saved comparison files. The new mathematical appendix's statewise `sM` bound and binomial expectation are valid. The historical attribution table must map the four KWS studies correctly: one earlier BER0.01 study with396/8 reversals, followed by the three-arm study with328/0 at0.001,305/0 at0.003,376/4 at0.01; ResNet has201/1 at0.01 and ToyCar196/0 at0.001. There is no ToyCar0.01 group in that six-group artifact.

The KWS secondary means do not subtract exactly to its primary mean because separately estimated paired-swap paths have finite Monte Carlo noise. The manuscript should not imply transitivity of these finite estimators or treat that small discrepancy as a new interaction effect. The stronger320-field ResNet FKeras follow-up, if discussed, must remain separate from the32-field current grid; it has the same adverse direction at6.25% and BER0.01. These are presentation boundaries, not reasons to alter any saved scientific output.

## Highest-value optional additional-week work

1. Independently verify the strongest reported crossover with a prespecified, practically meaningful effect and input-population uncertainty where feasible; do not expand an entire grid merely to produce more rows.
2. Diagnose finite-loss caps on the informative Pythia conditions and retain both capped primary and finite uncapped diagnostic outputs under a new frozen protocol. Preserve the existing result unchanged.
3. Test one additional independently trained checkpoint or independent input cohort at selected decisive cells. More fault fields on the same checkpoint cannot answer checkpoint/generalization objections.
4. If PrioriFI-derived allocation is advanced, first freeze the conversion from feedback discovery into an equal-budget mask and count all probe/calibration cost. An unfinished scheduler is not a missing curve that may be filled by analogy.
5. Improve reproducibility packaging and eliminate ambiguity in current versus historical cohort identity. This is lower risk and more immediately valuable than speculative new architectures or a new optimizer before submission.

None of these actions is authorized or launched by this review. They are prioritized residual scientific risks, not prerequisites for delivering the complete internal manuscript.
