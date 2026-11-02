# Software-only manuscript: adversarial internal review

## Verdict

This is a complete engineering investigation, but **not yet a low-risk TECS
submission**. Its strongest contribution is a controlled comparison of how
familiar susceptibility scores translate into actual protection decisions under
exact stored-code faults. It is not a new universal ranking method, physical
protection design, or demonstrated superior conditional optimizer. The recovered
128-clip FKeras comparison closes the earlier absence of a meaningful partial-tier
experiment, but not the need for a common larger cohort with random and numerical-jump
controls. The broader benchmark-expansion gate remains incomplete.

The PDF deliberately contains unfavorable and unresolved results. Completing a
manuscript is not evidence that its contribution clears the journal's novelty bar.

## Hostile questions and concrete manuscript responses

| Reviewer objection | Assessment | Fix already incorporated | Remaining risk / action |
|---|---|---|---|
| “Your own FCCM paper already presented NetSuRF and gradient/delta-guided protection.” | Correct; substantive overlap, not merely shared motivation. | Introduction and §2 explicitly credit the published flow. New claims concern frozen INT8 execution, corrected/matched baselines, multiple tasks and decision validation. No old implementation numbers or figures are reused. | Authors should approve an extension statement and explain the additional evidence in the cover letter. Do not market the framework name as novelty. |
| “FKeras already ranks important bits and uses a weight Hessian.” | Correct. A full Hessian includes mixed curvature. | §5 gives both printed and effective repository formulas and preserves tier-first ordering. §8.4 includes all three BERs and four budgets of the recovered 128-clip study: gains, ties and unresolved comparisons. | Full-tier masks are identical. Partial-tier outcomes depend on cohort, budget and formula. The frozen larger-cohort campaign adds missing common controls; it has not run. |
| “Recent reinforcement learning already learns bit allocations.” | Correct; Su et al. (Sensors 2025) adds directly relevant peer-reviewed evidence. | §2 credits its layer-bit actions and distinguishes its memory-entry law and objective. | No local matched RL baseline. The paper cannot claim superiority over adaptive allocation or assert that RL ignores interactions. |
| “PrioriFI and EISFINN already prioritize important faults.” | Correct, but fault discovery and expected mask-risk are different estimands. | §2 contrasts their sampling objectives with a fixed mask under coordinate-independent Bernoulli faults, without claiming to invent guided exploration. | No head-to-head result on their native objectives is provided. That is a scope boundary, not evidence of superiority. |
| “Progressive attacks already use encoding-aware gradients and changing model state.” | Correct. | §2 explicitly acknowledges this. Deliberately selected damaging faults are not treated as samples of the Bernoulli law. | An encoding-aware gradient alone is not a novel contribution. |
| “The selective-ECC companion release already studies contextual effects and allocation.” | Substantial overlap. Publication status does not erase prior art. | §2 cites it as a public dataset/software release, credits context analysis, and distinguishes layer/codeword choices, modeled ECC semantics and fault law. | Granularity and explicit contracts may be incremental. Do not claim the first interaction-aware allocation or dismiss learning methods as incapable of interactions. |
| “The conditional identity is elementary.” | Correct. | §6 calls it ordinary conditional expectation; Appendix A credits Boolean/reliability foundations. No spectral program is presented as a new practical method. | The counterexample rules out a guarantee; it cannot by itself sustain a methods paper. |
| “You selected a favorable pool to show conditional superiority.” | The earlier eight-bit pool was exploratory. | §8.5 distinguishes that historical result from all independent-pool arms, and plots every arm. All six selected-minus-isolated pool intervals include zero. | No general practical advantage over exact isolated allocation is resolved. The paper must not imply otherwise. |
| “Your gradient is not a derivative of the deployed integer program.” | Correct. | §§3,5,8.1 separate companions from authoritative inference, report forward disagreement qualitatively, and validate finite isolated effects and actual masks. | Quantitative correspondence should be preserved in a finalized, hash-bound dedicated forward-audit artifact before expanding that claim. No unverified forward-error number is quoted in the manuscript. |
| “You use MSE to argue about anomaly-detection robustness.” | That inference would be invalid. | ToyCar's authoritative protection metric is complete-file ROC-AUC; MSE is labeled a separate surrogate. The nonmonotonic budget examples explicitly show their divergence. | Only eight fault fields on a fixed cohort. No causal conclusion about anomaly-score ordering is established. |
| “The random baseline has 32 independent masks.” | False; it has four classifier masks and two ToyCar masks. | §7 separates mask seeds from fault fields. Pooled bars are descriptive and conditional on the realized masks, not a random-mask population interval. | A dedicated hierarchical/random-mask analysis would be needed for a population claim. Do not increase nominal replication by counting policies or inference samples. |
| “A positive marginal interval conflicts with a large adjusted p-value.” | Not a contradiction: the original multiplicity family and discrete tests differ from a single marginal interval. | Table 2 retains both the source familywise values and paired intervals. ToyCar's positive 25% point effect is not called significant. | Approximate field-t intervals with eight, potentially heavy-tailed outcomes are fragile. Further independent fields are needed for precision. |
| “The three tasks have incomparable exposure and uncertainty.” | Correct. | §8 states expected flip counts; AUC and CE have separate axes. Field, candidate and pool uncertainty are separately labeled. | Cross-architecture differences are observational, not causal evidence for redundancy or topology. |
| “The cost is not a real protection cost.” | It is a logical cardinality budget, by design. | Abstract, §§4,7,9 and conclusion explicitly use protected eligible bits only. Runtime means analysis runtime. | The paper supports software allocation decisions, not physical implementation savings. No area, power, energy, latency or radiation estimates are offered. |

## Analytical versus empirical claims

- **Analytical:** exact affine XOR jump; Bernoulli mask expectation; marginal and
  paired-swap identities; explicit three-bit counterexample; full-tier mask
  equivalence; uniform approximation implies at most twice that error in regret.
  These are proven or directly enumerable, not experimentally inferred.
- **Empirical:** sampled score/damage associations; actual quality versus BER and
  budget; task-dependent method ordering; source construction times; bounded
  mask differences and local paired outcomes. All are conditional on their saved
  checkpoints, cohorts and schedules.
- **Not established:** superiority on other checkpoints; causal topology effects;
  a universally best static score; conditional selection beating exact isolated
  allocation across independently sampled pools; scalability of an interaction
  optimizer; physical mitigation efficacy.

## Are the small decision effects worth their cost?

The independent-pool selected-minus-isolated differences are on the order of
10^-4 cross-entropy or 10^-6 AUC-risk units and are unresolved by their pool
intervals. A six-bit response table needs 64 full-cohort evaluations per outside
field, before independent final comparisons. No measured benefit justifies a
general optimizer claim at that cost. The draft uses this as a limitation of
turning formal context dependence into a practical allocation improvement.

The large ResNet-8 static stress differences are more practically visible, but
one frozen checkpoint and logical stress conditions do not establish broad
generalization. The paper must report both scales of effect, not hide the small
conditional differences behind the larger static result.

## Is the worst-case approximation bound useful?

The 2-epsilon regret result is valid if a **uniform** error bound holds over all
admissible masks. A sum of absolute omitted interactions can be so large that it
does not resolve a decision margin. Bounded numerical validation does not make
this a useful scalable certificate. Appendix A therefore treats it as a limit
on interpretation, not an empirical contribution or speedup claim.

## Submission risks ranked

1. **Novelty/extension:** a comparison of known scores may not meet TECS's
   contribution threshold even when exceptionally reproducible. The new
   experimentally established decision lessons must carry the paper.
2. **FKeras comparison scope:** the restored 128-clip partial-tier study is real
   evidence, not a four-clip execution check. It remains one ranking realization
   on a different cohort from the main curves; do not splice those means together.
3. **Precision and generalization:** ToyCar has eight fields, all tasks have one
   checkpoint, and prior data exposure prevents calling these fresh confirmation.
4. **Statistical presentation:** fixed-cohort/mask field intervals are not input
   population or ranking-seed uncertainty. Most effects remain exploratory.
5. **Companion validity:** exact kernel mapping does not imply gradient validity
   through integer discontinuities. Keep empirical validation prominent.

6. **Expansion acceptance:** one short streaming fixture has a reproducible
   comparison, but no independent full streaming corpus or transformer result.
   Neither a metadata record nor the VWW blocked probe counts as an executed
   additional benchmark study. Do not advertise broad-architecture generality.

The integrated milestone supersedes the old optional-week plan. Frozen campaigns,
costs, asset blockers and foreground commands are in
`docs/PAPER1_EXPANDED_CAMPAIGN.md`. No production campaign was launched.
