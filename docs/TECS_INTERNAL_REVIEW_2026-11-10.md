# Internal TECS-style review, 2026-11-10

Manuscript reviewed: *Is There a Universal Bit Ranking? Evaluating Fault Protection in Quantized Neural Networks* (14-page ACM review layout).
Comparison manuscript: *NetSuRF: A Resource-Aware Fault-Tolerant Design Flow for Machine Learning on Edge Devices* (TCAD-2026-0117.R1 proof).

This is a reviewer opinion, not a result manifest. It contains no publication numbers of its own; every number quoted is copied from one of the two PDFs or is simple arithmetic on those numbers.

## Scope of the review

**What the reviewer had.** Only the two PDFs. The code, database and logs were not opened, so every checksum, artifact and reproducibility statement in the manuscript is author-reported.

**One correction to the brief.** The TCAD file is the R1 proof and embeds the author response, which quotes the round-one comments of two reviewers. The final decision letter is not in it. The review uses those quoted comments and infers nothing about the final rejection.

**External sources read.**
- FKeras journal paper: full text.
- PrioriFI: FastML'25 slides and the lab announcement only, not the ASPLOS paper.
- *Learned or Ranked?*: the Zenodo record page.
- Stutz et al. (MLSys 2021): abstract.
- FORTUNE and EISFINN: bibliographic records and abstract only, so the manuscript's descriptions of them are unverified.
- Su et al., FlipLLM and the FCCM paper: not checked.
- TECS author guidelines: the page returned 403, so the page limit is unverified. The only figure found is ESWEEK's journal-track rule of 20 pages including references and appendix, which may not apply to regular submissions.

## A. Executive verdict

**Contribution as understood.** The paper is a paired, equal-budget software benchmark of six or seven cheap static bit rankings on six frozen INT8 models under independent Bernoulli stored-bit faults with ideal suppression. It adds a standard conditioning identity and a three-bit example showing that clean isolated damage is not the marginal protection value under simultaneous faults. It adds a 32-pool KWS test in which exact conditional re-selection changes masks but yields no resolved held-out gain over exact isolated selection.

**Recommendation: major revision.** It is not acceptable as written. If a revision kept the current title, the abstract's reversal claim and Table 4 as the central evidence, the recommendation would be reject.

**Confidence.** High on the inferential and statistical points, which rest on the PDF alone. Moderate on novelty, since the literature was only partly verified. None on anything that depends on the artifacts.

**Salvageable for TECS in present scope?** Yes, conditionally. Software-only scope with ideal suppression is legitimate. But the paper must deliver an actionable finding in a regime where the models still work, and must say what an irregular per-bit mask costs to implement.

**Principal barrier: validity of the central inference.** The evidence shown does not support the headline, and partly points the other way. Novelty is second; presentation is a distant third.

## B. Ranked major concerns

| # | Location | Problem | Why it matters | Severity | Concrete remedy | New experiments? |
|---|---|---|---|---|---|---|
| 1 | Table 4; §7.1 l.345–348; §6.2 l.292–294 | Every featured reversal sits where the model is non-functional. ResNet-8 unprotected is 20.1 nats/image (uniform guessing is 2.30). Pythia at BER 1% is 7.9 vs 12.7 nats/token (uniform over the vocabulary is about 10.8). Qwen at 7.5% is 21–23 nats/token (uniform about 11.9). No accuracy under faults is reported, although §6.2 says it was measured. | A preference between two destroyed models is a counterexample to universality, not evidence of practically important regret. | Critical | Report accuracy/AUC beside loss. Define a usable region (e.g. accuracy drop ≤ x pp) and restate every claim inside it. | No, if accuracy is in the records |
| 2 | Figs. 2–3 vs. abstract l.13–15, §8 | In the two slices plotted, absolute gradient is best or tied-best almost everywhere. The exceptions are ToyCar, Pythia at 25% (≤0.8 nats) and saturated BER ≥ 7.5%. The text reports six hand-picked "explanatory" cells from thousands. | The figures suggest a robust default exists among tested methods, the opposite of the title's implication. This is a selective-reporting exposure. | Critical | Per-cell regret of each fixed method against the best tested method, with paired CIs. Win/loss/unresolved matrices over the full grid. A best-method map over (BER, budget). | No |
| 3 | §5, §7.3, Fig. 4, l.310–327 | The theory contrasts isolated with conditional damage, but no grid method is exact isolated damage. The only test swaps 2 of 6 bits among ~154k unprotected bits at BER 5% (about 7,700 applied faults, ΔCE around 8 nats), using 8 selection fields. | The grid results and the theory are two separate papers. The null is near-guaranteed by design: the effect ceiling is about 10⁻³ nats. It cannot show whether conditional information is useless or merely unreachable here. | Critical | Report the attainable effect per pool (best minus worst of the 15 masks, held-out) and a minimum detectable effect. Then either demote the study or run G3/G4 in section G. | Yes, to make it informative |
| 4 | l.286–288; abstract l.19 | One checkpoint per model and one mask per deterministic method. Intervals condition on model, inputs and masks. | "Method A beats B" is really "this mask beat that mask". "Checkpoint-specific" has zero supporting data because checkpoint is confounded with architecture and task. Calibration-draw variance is unmeasured. | High | Delete checkpoint claims now. Add ranking reconstructions over calibration draws and several trained MLPerf Tiny checkpoints. | Yes |
| 5 | l.190–191; Table 4 rows 1–2 | MSB-first at 6.25% and 18.75% is decided by "canonical tensor, word and bit" tie-breaking, so it protects sign bits of whichever tensors come first. | The KWS vs ResNet-8 "FKeras − MSB" reversal may be a layer-order artifact, not Hessian information. FKeras's own MSB baseline uses index order too, so this is inherited, but it is still a confound. | High | MSB-first with random within-tier selection (≥8 seeds); test FKeras against that distribution. | Yes, cheap on TFLite |
| 6 | Figs. 2d–e, 3d–f | Positive-gradient protection is far worse than no protection on DistilBERT and Pythia (e.g. Pythia Δ ≈ 23 vs 18.5). The text never mentions it. Table 3 validates score fidelity only on TFLite models. | Suppressing faults should not, on average, make things much worse. It is either a real biased-residual effect (the most interesting result in the paper) or a sign-convention error in the transformer path. Readers will suspect the latter. | High | Sign unit test on single flips. Exact isolated damage on a stratified transformer bit sample. Residual first-order drift per mask. Then discuss openly. | Small |
| 7 | §5, App. B, §2 | Eq. (2) is the classical Birnbaum-importance / improvement-potential identity from system reliability theory, where dependence of rankings on component failure probability is textbook. FKeras and PrioriFI already document exceptions to bit monotonicity. FKeras also reports a multi-bit selective-protection result in the *opposite* direction (see E). Only 24 references. | The analytical contribution is not new, and directly relevant counter-evidence is uncited. | High | Cite the reliability-importance literature and discuss the FKeras multi-bit result. Present Eqs. 2–7 as background. | No |
| 8 | l.153–155, 276–277; Fig. 3 caption | Cost is the bit count. Biases are excluded. Reviewer arithmetic shows the transformer populations equal exactly the `nn.Linear` weights, so input embeddings are excluded (about 36% of DistilBERT's parameters). Irregular per-bit masks need per-bit metadata or placement; tier masks do not. | Excluded state is implicitly protected for free. The cost metric favours irregular gradient masks. For a TECS reader this is the embedded-systems question. | High | State the exclusions per model. Add a mask-anatomy analysis (tier × layer). Add at least one structured-granularity policy (G3). | Partly |
| 9 | l.297, Table 4, l.470–472, Table 3, Table 5 | Holm families are undefined. Contrasts with very different standardized effects share identical adjusted p (.0112, .0160, .0012), which looks like a resampling floor times family size. Table 3 has no intervals. Table 5 has no denominators and is in-sample. Three paired means "need not subtract exactly". | The p-values convey no evidence. Several statistical statements cannot be interpreted. | Medium–High | See D. | No |
| 10 | Table 2, App. A | Model identity is by name only. Missing: checkpoint and file hashes, transformer quantization recipe (per-channel or per-tensor, calibration), float reference quality, list of faultable tensors, Qwen tied `lm_head` handling, ToyCar normal/anomalous split, LM segment structure, hardware platform. | The executed program cannot be identified from the paper. | Medium | One identity table in the appendix. | No |

Eligible-bit arithmetic behind concern 8 (public architecture dimensions recalled by the reviewer, not re-fetched):
- DistilBERT: 6 × (4·768² + 2·768·3072) + 768² + 768·2 = 43,058,688 weights = 344,469,504 bits. Matches Table 2 exactly.
- Pythia-160M: 12 × 7,077,888 + 50,304·768 = 123,568,128 weights = 988,545,024 bits. Matches exactly.
- Qwen3-0.6B: 28 × 15,728,640 + 151,936·1,024 = 595,984,384 weights = 4,767,875,072 bits. Matches exactly.

## C. Claim-to-evidence audit

| Claim | Status | Evidence | Defensible wording |
|---|---|---|---|
| Isolated damage is not the marginal protection value (Eq. 2) | Supported | Derivation is correct; standard | Keep; attribute to reliability-importance theory |
| An isolated ordering has no general optimality guarantee (Eq. 3) | Supported, exactly as stated | Arithmetic checks: 1.1p+3p² vs 1.3p, crossover p = 1/15 | Keep. Add that no p-independent ordering is optimal for this function at B = 1 |
| "Paired comparisons reverse method preferences across models, budgets and fault probabilities" | Narrower wording only | Budget: one Pythia pair at severe degradation. BER: Qwen, post hoc, non-functional. Model: FKeras vs MSB, confounded by tie-break and by model | "In a few cells, mostly at severe degradation, the paired preference between two tested masks changes sign" |
| Preferences change with the checkpoint | Unsupported | One checkpoint per model | Delete |
| "Ranking choice therefore matters well beyond small networks" | Narrower wording only | Large paired gaps, at loss levels beyond usability | "Allocations differ by several nats at BER 1%; whether they differ where the model remains usable is reported in …" |
| Gradient scores track isolated damage better than significance | Unresolved | n = 320, so the standard error of a Spearman coefficient is about 0.056. KWS gaps are 0.04–0.08. No paired intervals | Give bootstrap CIs on the differences, or drop the comparative |
| Surrogate quality is not a protection guarantee (ToyCar 0.637 vs 0.504) | Supported descriptively | No intervals; the AUC column is not in Table 3 | Add the column and CIs |
| Tier-first and MSB-first give identical masks at 12.5% and 25% | Supported | Holds by construction for homogeneous 8-bit words in one eligible pool | Keep |
| Conditional selection changes 27 of 32 masks without resolved held-out gain | Supported, low information | −8.2×10⁻⁵ [−0.0017, 0.0015]; the numbers are internally consistent | Add "the largest attainable effect in these pools was …" |
| "Accounting for outside faults removes most reversals" (Table 5) | Unresolved | In-sample; the predictor and the target share the same sampled outside fields | Use a split-half version with denominators |
| Qwen reversal is "not solely a capping artifact" | Unresolved | The finite-only comparison conditions on survivors; non-finite rates are not given; which share belongs to which method is unstated | Report non-finite and capped fractions per method, or drop |
| Results support "condition-specific protection assessment" | Narrower wording only | The plotted slices equally support a robust default | See H |
| Implied answer to the title: no universal ranking | Unresolved; partly contradicted by Figs. 2–3 | Finite comparisons cannot settle it | Retitle |

## D. Mathematical and statistical findings

**Demonstrable errors.** None found in the equations.
- The INT8 example checks: 5/16 = 0.3125, code −123 = −7.6875, jump −8.
- Eq. (4) is correct for both bit states.
- Eqs. (2), (5) and (6) follow under product-Bernoulli faults with *b* unprotected and *G* excluding the swapped coordinates.
- Eq. (7) is correct. The "at most sM" step is tight because the statewise error is h(A) + (s−1)h(∅) − Σh({b}), a point the proof sketch omits.

**Numerical inconsistency needing explanation.** Line 472 says the paired estimates "need not subtract exactly". Improvements of 0.00268 and 0.00248 imply a difference of 2.0×10⁻⁴, but the reported contrast is 8.2×10⁻⁵. If all three masks were evaluated on the same fields and clips with a mean estimator, they must subtract exactly. Either the contrasts come from different data, or the estimator needs stating. The gap exceeds the headline effect.

**Missing conditions and ambiguities.**
- V_b(p, 𝒫) depends on the current mask. Ranking by V_b at one mask is therefore also not a mask-optimality guarantee, and the paper should say so.
- A ranking forces nested masks across budgets, but optimal masks need not be nested. Example: R = z₁ + 1.05z₂ + 1.05z₃ + z₁z₂ + z₁z₃ at p = 0.1. The best one-bit mask is {1} (0.210 vs 0.215); the best two-bit mask is {2,3} (0.100 vs 0.105). This is a cleaner answer to the title than Eq. (3) and speaks directly to the Pythia budget reversal.
- "Positive isolated damage" (Table 3) is undefined. So are the stratification variable and the handling of ties; MSB-first has eight distinct score values.
- Is TFLite cross-entropy computed from dequantized 8-bit softmax outputs? If so, per-example loss is at most about 5.5 nats or else jumps to 27.6 at the 10⁻¹² floor. ΔCE then mostly counts floor events, a saturation effect just like the 30-nat cap, but undiscussed. This cannot be verified from the PDF. Appendix A's 0.626 vs 0.918 CE gap is consistent with it.
- Fig. 2 "fault-field intervals": of the mean, or of outcomes?
- The symbol *p* is both BER and p-value; *B* and *b* are budget and bit.

**Statistical design.**
- The paper correctly separates outcome spread from paired-difference uncertainty, and it does not treat non-significance as equality. Keep both.
- Intervals cover fault-field randomness only. The claims need more: ranking-reconstruction variance (for "method" claims), checkpoint variance (for "checkpoint" claims) and input-cohort variance (for task-quality statements).
- The cohorts are small for that purpose: ToyCar has 64 files, and the language models have 4,096 dependent tokens.
- Table 4 reports a 32-field ResNet-8 estimate (+3.035) while Appendix C has a 320-field estimate (+2.241 [1.890, 2.593]). Use the better one.
- The Qwen follow-up and "explanatory conditions" are post hoc. Label the whole table so, or pre-specify a confirmation set.

**Fault scope.**
- Equal BER is acknowledged not to mean equal fault count, but the consequence is not drawn out. Even at 10⁻⁵, Qwen receives about 47,700 simultaneous faults, Pythia 9,900 and DistilBERT 3,400. ResNet-8 receives 6 and KWS 1.8. The transformers are never observed near the isolated-fault regime. Add an expected-fault-count axis.
- The BER range up to 10% is unmotivated. There is no radiation over-claim, which is good, but name the regime that produces such rates (voltage-scaled or emerging memories).
- Weights-only is a legitimate bound; no claim made requires activation faults.

## E. Novelty and competing methods

**Strongest defensible contribution.** A paired, exact-encoding, equal-budget protocol applied from MLPerf Tiny to W8A8 transformers. With it come two empirical observations that are genuinely useful if they survive checking:
1. Selective protection can be *worse than none* (positive gradient on transformers; all methods on ToyCar AUC at 1%).
2. Exact local re-selection, isolated or conditional, buys almost nothing over a gradient mask.

**Closest precedents.**
- **FKeras** (read in full). Its score matches the manuscript's formula. The passage read ranks by "higher score" and does not state an absolute value, so confirm |H′| against their code. It already compares random, MSB→LSB, gradient and Hessian on MLPerf-Tiny-class models, and it states that tier-first ordering "is not ideal". Its multi-bit experiment reports that protecting the bits found sensitive under the single-bit model "can prevent all of the errors" from simultaneous flips. That is evidence *for* isolated-damage sufficiency in one setting, and the manuscript does not mention it.
- **PrioriFI** (slides only). Its premise is "many exceptions to bit-level monotonicity". It is a fault-discovery method, not an allocator. The statement that no matched allocation exists is fair.
- **Learned or Ranked?** (Zenodo record, verified; publication status not stated). It is a cost-matched, held-out comparison of 13 allocators, including greedy, annealing, genetic and QMIX, with BER sweeps and layer-interaction diagnostics. It covers the "static mask built from multi-fault measurements" class that the manuscript lacks at whole-network scale. The manuscript's description of it is accurate.
- **Reliability-importance theory** (Birnbaum importance, joint reliability importance). This is the origin of §5 and is uncited.
- **Bit-error-rate studies on quantized weights** such as Stutz et al. (abstract verified). Uncited, and relevant to motivating the BER range.
- **Statistical fault-injection sizing and single-vs-multiple bit-flip studies.** Recalled by the reviewer but not verified here; check before citing.

**On RL.** Omitting it is adequately justified as a scope limit. The larger hole is the absence of *any* whole-network, multi-fault-informed static baseline, which need not be RL (see G3).

**Most damaging novelty objection.** "The theory is textbook. That MSB-first and single-fault rankings are imperfect is already published. The new evidence is a set of sign changes among cheap proxies in non-operational regimes, plus an underpowered null."

**Candid answer.** As written, the paper establishes little new knowledge firmly, and not enough for TECS. Three things would change that: a usable-regime regret result with replication, a validated account of harmful protection, and one measured link between the conditional theory and network data.

## F. Figures and tables

| Item | Verdict | Change |
|---|---|---|
| Fig. 1 workflow | Keep, simplify | Replace internal jargon ("authoritative", "seal", "immutable"). Show ranking → mask → fault field as three labelled objects. |
| Table 1 rankings | Revise | Add columns for data/labels needed, construction cost and tie-break rule. State that all gradient scores share the \|Δ_b\| factor. |
| Table 2 models | Revise | Add weights, expected faults at lowest and highest BER, float reference quality, cohort composition, hash. Report cohort accuracy with a binomial interval. |
| Table 3 correlations | Revise or move to appendix | Add CIs and paired differences, a ToyCar AUC column and a transformer column. Recompute the missing FKeras entries: at seconds per run, "could not be established for the older … audits" is not acceptable in a journal paper. Prefer precision-at-budget to Spearman, given the ties. |
| Fig. 2 loss vs BER | Revise | 12.5% is exactly the budget where FKeras is invisible; use a partial tier or two budgets. Drop Random (it coincides with Unprotected). Add accuracy panels and an expected-fault axis. Shade the unusable region. Explain the grey box in (c). State the asinh linear threshold. |
| Table 4 contrasts | Replace | Use the regret and win/loss summary over the full grid. Keep at most two or three illustrative reversals, flagged post hoc, with test, raw p and family size. |
| Fig. 3 loss vs budget | Keep, revise | Mark partial-tier MSB points as tie-break dependent. Add a paired-difference panel against a reference method. Call out positive gradient above Unprotected. |
| Fig. 4 pool effects | Move to appendix or shrink | Add the attainable-effect range so readers can see the ceiling. |
| Table 5 reversals | Revise | Add denominators, a split-half version and weighting by \|ΔJ\|. If it survives, it is the best evidence about background shift and belongs in the main text. |

The fonts are readable at normal size. The overlapping grey and purple lines are not distinguishable.

## G. Revision plan

### 1. Within 48 hours, no new inference

- Retitle. Rewrite the abstract around what is measured.
- Delete "checkpoint-specific".
- State excluded tensors (biases, embeddings) per model.
- Define the CE computation, the Holm families and the tests.
- Add the missing citations and discuss the FKeras multi-bit result.
- Present §5 as background.
- Discuss the positive-gradient and ToyCar harm openly.
- Remove experiment-diary language: "saved Holm-adjusted", "older audits", "preflight", "request", "master fields", "numerical contracts".
- If any tested score descends from the earlier `legacy_qpolargrad` expression, say so in one sentence.

### 2. High-value analysis of existing records

- Regret and dominance tables over the full grid.
- Accuracy and floor/cap/non-finite fractions per cell.
- Mask anatomy (tier × layer) and pairwise mask overlap.
- Table 3 intervals.
- One coherent three-mask table for the conditional study, with attainable effect and minimum detectable effect.
- Table 5 split-half.
- Substitute the 320- and 384-field estimates into the main table.

### 3. Minimum new experiments that could change the verdict

None of these should be launched without a bounded command, prerequisites, output path and resource estimate, per `AGENTS.md`. G1 includes retraining and needs an explicit decision.

**G1. Usable-regime regret with replication (TFLite models).**
- Hypothesis: one fixed method has regret ≤ δ in every usable cell, with δ fixed in advance (e.g. 1 pp accuracy).
- Comparator: the best tested mask per cell.
- Independent units: fault field (about 320) within ranking reconstruction (≥5 calibration draws) within trained checkpoint (≥3–5).
- Controls: common fields across masks; fresh evaluation cohort.
- Decision: if the upper CI of regret stays below δ everywhere, the paper's answer becomes "a robust default exists among tested methods". Otherwise, map where it fails.

**G2. Tie-break control.**
- Hypothesis: FKeras within-tier selection differs from random within-tier MSB.
- Comparator: ≥8 random within-tier seeds, and the canonical order.
- Unit: field, paired.
- Decision: a Holm-corrected paired difference on both KWS and ResNet-8.

**G3. Group-level measured allocation.**
- Hypothesis: a tier × layer allocation ranked by measured multi-fault benefit per bit at the target BER matches or beats per-bit gradient masks in the usable region.
- Comparators: absolute gradient, MSB-first.
- Cost: roughly 80 groups × 32 fields on ResNet-8, plus a held-out evaluation.
- Why it matters: it is the missing multi-fault-informed static baseline, and it yields hardware-plausible masks. Either outcome is publishable.

**G4. Isolated vs conditional as a function of expected fault count.**
- Hypothesis: agreement between I_b and V_b decays as Np grows, producing held-out regret.
- Method: estimate both on the 320 stratified bits for Np from about 0.1 to 10³, with split fields.
- Decision: the held-out regret of top-k by I_b against top-k by V_b excludes zero somewhere in the usable region.
- Why it matters: this ties §5 to data.

**G5. Positive-gradient diagnosis on DistilBERT.**
- Sign unit test, a sign-flipped mask and residual first-order drift per mask.

### 4. Defer to another paper

Whole-network conditional optimisation, RL comparisons, PrioriFI-derived allocation, activation and datapath faults, physical ECC/TMR cost, radiation calibration.

## H. Final recommendation and replacement framing

**Contribution statement.** "We present a paired, equal-budget evaluation of static bit-protection masks under independent stored-bit faults, executed on exact INT8 programs from MLPerf Tiny models to W8A8 transformers. We report, with fault-field, calibration and checkpoint uncertainty, how much task quality each low-cost ranking preserves where the model remains usable, how much is lost by committing to one ranking across BER and budget, and where protection is counter-productive. A standard reliability-importance identity explains why single-fault damage need not predict protection value; we measure when that gap matters and find that local exact re-selection, with or without faulty-background conditioning, yields no resolved benefit in the tested setting."

**Title.** The question cannot be answered by a finite comparison, and the present data lean toward "nearly, among those tested". Suggested: *How Far Do Static Bit Rankings Transfer? A Paired Evaluation of Selective Protection in INT8 Neural Networks.*

**Three claims to keep.**
1. Isolated damage carries no general optimality guarantee under simultaneous faults, and optimal masks need not be nested across budgets.
2. Identical masks from differently named methods are one decision, and a mask comparison is not a ranking-correlation comparison.
3. Exact local re-selection changed most masks without a resolved held-out gain. State it with its effect ceiling.

**Three claims to stop making.**
1. That preferences are "checkpoint-specific".
2. That the featured reversals show practically relevant instability, or by implication that no robust default exists.
3. That Table 5 or the Qwen uncapped comparison shows what outside faults or capping do.

**Judgment after revision.** If the revision includes the existing-records analysis together with G1 and G2, and at least one of G3 or G4, a publishable TECS paper is likely, probably after one more minor round. With rewording alone it stays below the bar, because the central inference would still rest on non-operational cells and single masks.

## Comparison with the rejected TCAD manuscript

**Round-one concerns quoted in the R1 author response, and where the new paper stands.**
- "No rigorous evaluation of qPolarGrad's accuracy … does it truly identify the most vulnerable bits" (Reviewer 2). The new paper removes the metric and separates score fidelity (Table 3) from protection outcome. Addressed in structure; Table 3 still lacks intervals.
- "Such precision may be unnecessary for selective hardening" (Reviewer 2). Partly addressed: large gaps between rankings are shown, but only where the models are already broken. Unresolved in the usable regime.
- Simple datasets and no modern architectures (both reviewers). Addressed in breadth by adding MLPerf Tiny reference models and three transformers.
- TMR as a weak baseline against ABFT or Ranger (Reviewer 2). Now scoped out rather than answered; a TECS reviewer may raise it again.
- Weights versus activations (Reviewer 2). Still weights only, but the new paper claims nothing about activations. A legitimate bound.
- Component-level comparison against partial-redundancy methods such as FORTUNE (Reviewer 1). Dropped along with all hardware content.

**Engineering strengths retained.** Budgeted selective protection under a BER sweep; MSB and Hessian baselines; Monte Carlo evaluation, now paired and with stated uncertainty.

**Engineering strengths lost.** The synthesis-backed area/power cost model (Table IV of the old paper) and the register-level instrumentation. These were the parts most distinctive for a design-automation or embedded venue.

**Unsupported claims appropriately removed.**
- "Consistently yields better accuracy" for one metric, and the "up to 21.7 pp" gains quoted at the budget where the lead was widest.
- The statement that FKeras approximates the Hessian diagonal. The FKeras paper uses top-k eigenpairs, and the new manuscript says so.
- The Hessian runtime of 9.1 hours on KWS. The new manuscript reports 1.43 seconds for rank-two scoring on DS-CNN; the old cost advantage does not survive.
- The claim that the metric "implicitly accounts for" multi-bit interference, and the #P-completeness framing.

**Weaknesses that persist.**
- Single checkpoint per model.
- Reporting at selected cells: the old "best gain" markers have become "explanatory conditions".
- No operating point tied to a deployment scenario; BER range unmotivated.
- Ideal protection with no implementation cost.

**Identity.** The new paper is more careful and more honest than the old one, but it currently reads as two papers joined by caveats: a proxy-ranking benchmark, and a conditional-damage argument with one small null test. The regret analysis and one of G3/G4 would give it a single identity.

## Sources

- FKeras: A Sensitivity Analysis Tool for Edge Neural Networks (author PDF): https://kastner.ucsd.edu/wp-content/uploads/2024/05/admin/jats24-fkeras.pdf
- PrioriFI FastML'25 slides: https://indico.cern.ch/event/1496673/contributions/6637923/attachments/3127187/5546996/PrioriFI%20FastML'25.pdf
- PrioriFI @ ASPLOS'26 lab post: https://kastner.ucsd.edu/2026/03/
- Selective ECC protection companion dataset, Zenodo: https://zenodo.org/doi/10.5281/zenodo.23054820
- Bit Error Robustness for Energy-Efficient DNN Accelerators, MLSys 2021: https://proceedings.mlsys.org/paper_files/paper/2021/hash/2f9b1b6b29361118f630783c19891ea0-Abstract.html
- EISFINN record, TU Wien: https://repositum.tuwien.at/handle/20.500.12708/208038
- FORTUNE record, TalTech: https://ws.lib.ttu.ee/digibase/en/Publ/Item/4eb61fef-187e-46a1-aa50-6fcba8a70f58
- ESWEEK 2025 author information (TECS journal track): https://2025.esweek.org/author-information/
