# NetSuRF internal adversarial review

Review date: 2 November 2026. This is a complete internal-review draft, not a recommendation to submit without author review. The strongest defensible paper is a software-methodology study of the gap between sensitivity, conditional vulnerability and protection decisions. The evidence does not support a universally superior conditional allocator.

## Submission assessment

The mathematical formulation, exact execution contracts, cross-task negative results and signed attribution form a coherent paper. The central empirical constraint is consequential: selected-versus-exact-isolated marginal pool-level intervals include zero in all six completed cross-pool/BER arms. The favorable eight-bit replication cannot be generalized to those arms. The draft now states both findings in the abstract and results.

The main remaining risks are novelty relative to reliability importance and recent protection-allocation work; small decision effects relative to construction cost; weak independent-pool precision outside KWS; and the absence of an equal-compute unguided alternative-search baseline. These are not repaired by adding more estimator names.

## 1. Novelty could be challenged as established Boolean reliability theory

**Likely objection.** A multilinear expectation, its forced-state derivative, Möbius inversion and biased Fourier expansion are established mathematics. A three-bit counterexample is not a new theorem about neural networks.

**Evidence and correction applied.** Sections 2, 4 and Appendix A attribute the foundations to Boolean-function analysis and reliability importance. The manuscript claims an operational separation of exact encoded-bit objects, outside-context correction and candidate interaction, coupled to reproducible cross-task measurements. It does not claim invention of influence, interaction or adaptive allocation.

**Remaining risk.** The sharp bounded-risk singleton remainder and finite-field regret result may be useful as a scoped proposition, but priority over all approximation literature is not established by this audit. Do not label it “the first” or imply a new general reliability calculus. Have a mathematically independent reader check the proof and terminology.

## 2. FKeras and PrioriFI are stronger prior art than a straw-man static baseline

FKeras contains cross-parameter Hessian information. Its score is not a diagonal Hessian, and saying it “ignores interactions” is wrong. PrioriFI uses prior fault outcomes to guide injection; feedback-based analysis is not novel to NetSuRF.

**Correction applied.** Section 4 distinguishes the printed FKeras vector from the pinned repository's effective sum of absolute eigenvalue-squared contributions. It records the continuous KWS companion and largest-magnitude signed eigenvalue convention. Section 7 reports validated eigenpairs and actual mask differences, without calling the two-field/four-input check a performance comparison.

At 12.5% and 25%, homogeneous INT8 strict-tier FKeras and corrected MSB protect identical sets. A favorable or unfavorable performance difference there would indicate a protocol or implementation mismatch. Partial-tier budgets distinguish masks but have only engineering evidence.

**Remaining risk.** There is no powered FKeras or PrioriFI cost-matched protection study here. The manuscript does not claim one. A reviewer may still require a larger partial-tier comparison; this would need a separate approved protocol, not reinterpretation of the smoke.

## 3. The selective-ECC release already studies learned allocations and layer interactions

The [Zenodo companion release](https://zenodo.org/records/23054820) is verified as a dataset, not as independent proof of peer-reviewed acceptance of its manuscript. It includes learned/rule-based allocations, cost-matched held-out comparisons and layer-interaction diagnostics. The source snapshot inspected in the scientific audit includes subset-additivity and contextual marginal analysis.

**Correction applied.** Section 2 explicitly acknowledges this overlap. NetSuRF's exact encoded-bit fixed-cardinality problem, independent Bernoulli risk and bounded candidate-state attribution differ from layer-wise policies, word-level decoding and burst/interleaving laws. No quantitative superiority over that release is claimed.

**Remaining risk.** “We consider context and others do not” is indefensible. The narrower contribution must remain explicit. Public dataset status does not diminish its relevance as prior art.

## 4. Candidate-pool selection can manufacture a favorable narrative

M9.5.7 gives a strongly resolved fixed-mask contrast, but its eight-bit pool descends from exploratory conditional-damage work. Independent final fields validate that frozen comparison; they do not make the pool an unbiased network sample.

**Correction applied.** The abstract, protocol, replication table and discussion retain the selection history. All later six-bit pools are included, including identical masks, negative effects and zeros. The two-protected/four-unprotected sampling design is named explicitly.

**Remaining risk.** Twelve KWS pools recur across cohorts/BERs. They are not 48 independently sampled pools. Four ResNet and four ToyCar pools leave wide uncertainty. The pool design is conditional on a specific S5 mask and cannot establish whole-network optimality or population-uniform candidate effects.

## 5. The conditional-versus-exact-isolated question remains unresolved

The selected-minus-isolated mean is favorable in the four KWS arms, adverse in ResNet and ToyCar, and its marginal interval includes zero in every arm. The manuscript presents all arms, not just a favorable mean or the earlier exploratory result.

Improvement over S5 alone mixes two questions: how well a surrogate predicts exact isolated damage, and whether background conditioning improves allocation beyond exact isolated knowledge. This is why the exact-isolated comparator is essential.

**Correction applied.** The conclusion says “unresolved,” not “equivalent” or “no effect.” Results are in CE units or AUC-risk units, never pooled. No significance claim is inferred from marginal-policy intervals.

**Remaining risk.** The study may demonstrate a real conditional effect without a practically useful allocation gain. That is an admissible negative result, but it changes the contribution from an optimizer paper to a measurement-and-limits paper.

## 6. Selection attribution is not held-out causal mechanism identification

The signed clean/outside/within decomposition closes numerically. Outside-conditioned singletons reverse far fewer mask orderings than clean singletons on the recorded selection tables. That supports the algebraic explanation on those tables.

**Correction applied.** Sections 5 and 7 say that final response subcubes and matching same-final-cohort clean controls are absent. Final risk differences cannot be decomposed using selection-cohort terms. Architecture, checkpoint, metric, input distribution and BER are confounded.

**Remaining risk.** “Redundancy explains KWS versus ToyCar” and “ReLU gating causes the gain” would require interventions or additional controlled measurements. The draft makes neither claim. A useful next analysis must distinguish field noise from cohort shift before naming an architectural cause.

## 7. Small effects and expensive exact selection may limit system value

The table of saved engine times is paired with effect sizes. Exhaustive candidate integration costs 64 cohort evaluations per K=6 outside field; it gives every legal mask, but does not scale to unrestricted whole-network search. A ToyCar selection state entails 4,704 window evaluations.

**Correction applied.** The paper reports operation counts separately from measured wall time and avoids converting their ratios into measured speedups. It calls the exact calculation a bounded reference, not a practical universal allocator.

**Remaining risk.** No equal-compute unguided swap baseline isolates the benefit of conditional guidance from merely trying alternatives. There is also no demonstrated physical area/power tradeoff. Those omissions preclude deployment-efficiency claims but need not preclude a carefully scoped software methodology contribution.

## 8. M11's bound is sharp but not a practical certificate here

The remainder bound allows nonmonotone beneficial faults and correctly accounts for additive extrapolation leaving the risk range. A probability-of-two-faults bound multiplied only by L would be incorrect. The regret proof keeps approximation bias separate from field-estimation uncertainty.

At K=6, r=2 and p=.01, the classification bound is about 0.03283 nats, much larger than observed errors and many measured decision effects. The finite-field term is looser still.

**Correction applied.** Sections 5 and 7 explicitly call this a conservative assumption/limit certificate. Full conditional vulnerability averages other candidate faults; the seven-state approximation forces them clean. The two quantities are never equated. Pure higher-order falsifiers and the nonzero K=8 engineering regret remain included.

**Remaining risk.** A reviewer may ask why a loose worst-case bound belongs in the main paper. Keep the statement brief and the proof in the appendix; its value is preventing invalid guarantees, not claiming statistical resolution. No unexecuted K=32/64 campaign is presented as evidence.

## 9. Uncertainty, multiplicity and exposure cannot be collapsed into one confidence interval

Field uncertainty conditions on a fixed cohort and mask. Pool uncertainty conditions on the candidate-sampling design and shared inputs. Complete-file input resampling is a third object. Policies sharing fields and normal/anomalous pairs sharing files are dependent.

**Correction applied.** Captions identify units; source comparison families are preserved; displayed marginal intervals are not relabeled simultaneous. The full static comparison CSVs accompany the selected table. ToyCar's task objective is complete-file AUC with tie half-credit, not reconstruction MSE or window AUC.

The verified ToyCar ledger records fault-outcome exposure for 200/248 official files and clean inference for all 248. Remaining files are class-imbalanced (40 normal, eight anomalous), so they are not an automatic adequate confirmation reserve. Source-ordered disjoint windows do not establish exchangeability.

**Remaining risk.** Pool-t approximations with four pools are fragile; the completed studies cannot be retrospectively made prespecified by rewriting prose. Expanded claims would require independently frozen cohorts/fields and honest exposure records.

## 10. TECS scope and submission readiness remain author decisions

The embedded connection is exact low-precision storage and allocation reasoning before choosing a deployment mitigation. The manuscript has no radiation calibration, memory residency model, measured ECC/TMR implementation or hardware costs.

**Correction applied.** Scope and absence of deployment guarantees are explicit. The contribution is not sold as a hardware protection design.

Before submission, confirm coauthors, affiliations and acknowledgments with the team; only the author supplied in the authoritative ZIP is retained. Confirm current TECS submission format/length, CCS concepts and permissions on submission day. The acmart “Manuscript submitted to ACM” footer is template boilerplate, not evidence of an actual submission. The current package is internal review.

## Numerical and editorial corrections made

- ResNet at BER .01 and 12.5%: the verified paired CSV gives S5-minus-MSB accuracy +0.3300625, not the supplied draft's +0.3239.
- ToyCar at BER .01 and 12.5%: the effect is -0.0052490234375 AUC, with an interval crossing zero; it is not a positive result.
- ToyCar at 25%: the positive point effect is +0.241455078125 AUC, but the saved within-BER/metric Holm value is .84375. An unadjusted interval cannot override the declared family.
- Removed stale claims that the now-completed exact conditional and restricted studies had not been executed.
- Removed unsupported blanket multiplicity and optimizer-superiority language; retained source-specific test families.
- Correct KWS population is 176,128 bits; ResNet's 618,880 is not a KWS count.
- M11 historical selection regret is not independent final regret.
- Runtime numbers lacking a directly finalized hashed measurement were omitted from empirical claims rather than promoted from prose notes.

## Optional additional week, in priority order

1. **No new experiments required:** independent mathematical/statistical review of the bound, comparison families, signed decompositions and candidate-exposure narrative; resolve authorship and journal fit.
2. **No new inference required:** author-approved paper compression and nomenclature cleanup; move engineering details to a supplement while preserving all adverse/null results and machine-readable tables.
3. **New protocol required:** specify a practically meaningful effect threshold and an equal-compute unguided swap baseline. Decide sample size using an explicit prospective design; do not keep sampling until a favorable p-value appears.
4. **If computation is approved:** increase independent selection information or independently sampled pools where decision gaps are unstable, followed by untouched frozen-mask evaluation. Do not adapt candidate pools using final outcomes.
5. **For ToyCar confirmation:** obtain legitimate additional labeled recordings or justify an external validation set; do not relabel already exposed official files.
6. **Only if reviewers need it:** a bounded but adequately specified FKeras partial-tier comparison with construction cost and matched fields. Keep it separate from claims about integer-program Hessians.
7. **Not a prerequisite for this scope:** hardware, physical SEU exposure, new estimators and large M11 optimization campaigns. Do not add them merely to make the paper sound broader.

The defensible submission argument is that exact analysis prevents incorrect conclusions about protection: interactions can change vulnerability, yet accurate conditional prediction and statistically resolved allocation improvement are different achievements.
