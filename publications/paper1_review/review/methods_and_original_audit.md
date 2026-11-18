# Original-paper and scientific-method audit

This audit supports the complete internal draft. It is not a new experiment. The original submitted PDF was read through the main manuscript (PDF pages 4–17), the response letter (18–22), and the duplicated tracked manuscript/COI material. All eight original figures were inspected as rendered pages. The supplied Overleaf main file and section sources were inspected. Current contracts take precedence over historical prose.

## Authorship, narrative, and voice

The historical author order is **Manuel Blanco Valentín, Houxuan Guo, Xiuyuan He, Seda Ogrenci**, Department of Electrical and Computer Engineering, Northwestern University, Evanston, Illinois, USA. The submission cover identifies Houxuan as corresponding author; the manuscript footnote identifies Seda. This discrepancy requires author confirmation before submission. The attached Overleaf source has a single-author placeholder; preserving it would silently erase the historical coauthors. Actual author contributions and new work must be confirmed by the authors, not inferred by software.

The older paper's strongest writing pattern is a concrete engineering question followed by a staged construction: identify the stored bit, calculate its numerical change, construct a priority order, spend a protection budget, inject faults, and inspect task performance. Its explanation repeatedly connects an equation to the next operation in the flow. Preserve this progression and the direct first-person plural voice. Preserve the reader-facing emphasis on the performance-versus-budget and performance-versus-BER questions. The sentence rhythm can remain conventional engineering prose without copying the old unsupported assertions.

Do not preserve the old claims of guaranteed optimal protection, an intrinsically superior qPolarGrad score, a direct mapping to fabricated hardware reliability, or weight faults dominating activation faults. The new manuscript can say that persistent parameter storage is a declared fault domain. The study supplies no measurement comparing its exposure or consequences with activation/datapath faults.

The supplied Overleaf source is scientifically stale: its abstract calls the three-model results preliminary, its results and allocation sections contain unexecuted-experiment TODOs, it labels FKeras outstanding, and it omits completed larger-model studies. It is an authoritative source package to update, not authoritative evidence that those experiments are still missing. Its numbered section structure and mathematical notation are reusable.

## Original contribution and figure mapping

| Original element | Intended engineering message | Current action and scientific reason |
|---|---|---|
| End-to-end resource-aware design flow, Algorithm 1 and Fig. 3 | Translate a trained network and resource limit into a reliability tradeoff | Retain the software workflow. Replace HLS and hardware endpoint interpolation with exact eligible-bit cardinality and measured ranking/evaluation cost. Frozen masks and matched field evaluation are real implemented operations. |
| Fig. 1: weights, encoded words, delta matrix | A stored bit has a precise numerical perturbation | Retain with affine INT8 examples, sign-aware XOR, per-channel scales, and explicit tensor/word/bit identity. The old sign convention and fractional scale are inconsistent with its compact Eq. (3). |
| Fig. 2: susceptibility/protection mask times delta times fault field | Protection suppresses selected bit events; several flips compose | Retain this semantic separation. Current notation should distinguish a protection set P from a susceptibility score and a random proposed field Z. Exact XOR is applied to Z minus P. |
| Fig. 3: Keras/hls4ml, synthesis, ranking/TMR pipeline | A designer can traverse software-to-hardware stages | Replace with the software-only flow diagram delivered as `source/figures/workflow.tex`. Cite earlier FCCM hardware flow as previous work, not a new completed contribution. |
| Fig. 4 and Table IV: area/power proportionality | Use synthesis endpoints to estimate intermediate cost | Omit from this paper. No current physical cost claim is needed or justified by bit coverage. Historical synthesis cannot establish a universal cost law. |
| Figs. 5–6: budget/BER performance surfaces plus slices | Make protection decisions from an operating-condition map | Retain the engineering question. Use only measured grid cells and shared controls. Label uncertainty as a CI of a mean or a trial-outcome SD, never interchange them. ToyCar must use file AUC rather than calling reconstruction MSE task reliability. |
| Fig. 7: qPolarGrad vs MSB/FORTUNE | Compare allocations at matched budgets | Replace with current verified static comparisons, including adverse conditions and mask identity. Do not retain best-case markers selected to maximize the reported gain. |
| Fig. 8: qPolarGrad vs Hessian | Compare quality and construction cost against a stronger baseline | Use the published FKeras score, its tier-first ordering, validated continuous companion, and actual measured costs. Repository-form FKeras may appear in a provenance note, not as another main-paper method. |
| Table I: qPolarGrad impact equations | Generalize a heuristic across layer types | Historical context only. Current tested estimator support is adapter-specific. No general CNN/transformer qPolarGrad result follows from this table. |
| Table II: models and parameter counts | Establish benchmark scope | Replace with exact modern checkpoints, task cohorts, eligible stored bits, and backends. The historical 7,845 MNIST count is a nonzero-value count, not the allocated parameter population. KWS MSWC and modern Speech Commands are different experiments. |
| Table III: optimal coverage under a hardware limit | Give a usable design decision | Replace with paired reliability-versus-budget measurements. A sampled best mask/coverage is not a proven optimum and bit fraction is not area fraction. |

The visual audit found a concrete scientific contradiction in the old Fig. 7: Section III-A specifies strict bit-position-first ordering with qPolarGrad only within tiers, while the figure claims gains over protecting all MSBs at exactly one complete six-bit tier (1/6 coverage). With a common eligible population and exact cardinality, those protected sets must be identical. The historical numerical claim cannot be carried forward without resolving the actual policy and population. The current full-tier FKeras/MSB mask identity tests directly prevent this ambiguity.

The original paper's broad claims that ordinary gradient methods require an optimally trained stationary point, that FKeras is diagonal-only, or that activation-only layers are inherently immune are incorrect. Their removal is a scientific correction, not merely an editorial choice. The original assertion that an intractable global optimum prevents empirical isolated-bit ground truth is also incorrect: exhaustive isolated damage is finite and measurable; it is a different object from optimal multi-fault protection.

## Exact mathematical statements for the manuscript

Let U be the declared eligible stored-bit universe, P its protected subset, Z a proposed independent Bernoulli field, and R(A) the authoritative task risk for exactly the applied set A. Then

`J_p(P) = E[R(Z \ P)]`, with `E|Z \ P| = (|U|-|P|)p`.

The fault set persists over a complete trial cohort. For ToyCar this includes every window of every file used to compute the cohort AUC. Nothing in these definitions makes risk additive across bits or files.

For b not in P, draw G over `U \ (P union {b})`. Conditioning on whether b flips gives

`J_p(P)-J_p(P union {b}) = p E[R(G union {b})-R(G)]`.

The sign is retained: a beneficial fault has negative marginal damage and suppressing it can increase recorded task loss. Full protection returns the clean model; it is not automatically the minimum of an arbitrary empirical task loss over every corrupted model.

For an equal-budget exchange that removes a from protection and adds b, with a in P and b not in P, draw G over `U \ (P union {b})`. Then

`J_p((P \ {a}) union {b})-J_p(P) = p E[R(G union {a})-R(G union {b})]`.

This identity is exact under the independent product law. The two terms must share the same background. Path steps are summed within each independent master field before calculating uncertainty. A sequence of path steps is not a set of independent trials. Ordinary Monte Carlo estimation of this expectation is not Markov-chain Monte Carlo over masks.

The supplied analytical counterexample is valid without adjustment:

`R(z)=1.2 z1 + z2 + 0.1 z3 + 3 z2 z3`.

The exact isolated ranking protects bit 1. Under a one-bit budget, `J({1})=1.1p+3p²`, `J({2})=1.3p`, and `J({3})=2.2p`. The winner changes from protecting bit 1 to bit 2 at `p=1/15`. This is a toy nonnegative loss, not a measured classification error or AUC. It proves that clean isolated scores lack a universal guarantee across BERs. It does not prove that every static mask or every static ranking fails: the optimal mask at any fixed condition can itself be static.

The exact finite-bit delta for an affine INT8 code is `s(decode(q XOR 2^j)-q)`; the zero point cancels. For j<7 it is `s(1-2a_j)2^j`; for j=7 it is `-s(1-2a_7)128`. Multiple distinct same-word flips compose by XOR; each proposed bit occurs once under the Bernoulli law. Zero-valued allocated words remain faultable. `source/figures/encoded_xor.tex` uses q=5, s=1/16, z=0, with exact sign-bit displacement -8 and bit-2 displacement -0.25.

## Static definitions and interpretation

For exact finite jump delta_b and per-example companion gradient g_b(x), keep these definitions:

- Global first-order: `|E[g_b(x) delta_b]|`.
- Per-example absolute: `E[|g_b(x) delta_b|]`.
- Per-example positive: `E[max(0,g_b(x) delta_b)]`.

The signed average, absolute average, and positive average are different quantities. `S_positive=(S_absolute+E[g delta])/2`, and `S_absolute >= |E[g delta]|`, but neither relation implies superior protection. Derivatives of a straight-through companion are ranking proxies, not derivatives of the discontinuous exact integer executable. Actual damage and protection results use the authoritative backend.

Published FKeras uses `H' = sum_i lambda_i (v_i^T theta) v_i`; the implemented ordering uses `|H'|` within strict significance tiers. The validated solver retains signed largest-magnitude eigenvalues and reports residuals, symmetry and orthogonality. It uses a separately defined continuous float64 loss model, not the first-order STE companion or an integer-program Hessian. Do not multiply its score by an extra XOR delta or quantization scale. The pinned repository sum form is `sum_i |lambda_i^2 (v_i^T theta) v_i|`; its extra eigenvalue and pre-sum magnitude can change rankings when k>1. The main paper baseline is the published form only.

At homogeneous INT8 coverages 12.5% and 25%, strict-tier FKeras and corrected MSB must select the identical whole tiers. Equivalent masks and their paired outcomes are duplicate allocations, not independent evidence. The 6.25% and 18.75% budgets are the informative within-tier comparisons. Hessian scores can contain cross-parameter curvature; the justified limitation is that the published score is not conditioned on the deployment mask, BER, or realized outside field.

Historical qPolarGrad is `abs(P_b * dL/dw)`, where in the recovered dense path `P_b=mean(layer_input_i under a global all-bit attack)*delta_b`. The manuscript's printed Eq. (7) additionally multiplies by `Delta theta`; it is a different expression. The historical attacked activation source and absolute-value location matter. Neither definition has a current validated main-grid result on the six modern benchmarks. Describe it as the original motivation/historical heuristic if needed, with no transferred superiority claim.

## Task losses and uncertainty

Classification cross-entropy is in nats/example; LM token NLL is in nats/token; 1 minus ROC-AUC is dimensionless. Natural logarithms define a nat. These quantities cannot be averaged across models as a common numerical endpoint. The TFLite probability-floor classification contract caps CE at `-log(1e-12)`; W8A8 LM records cap per-token NLL at 30 nats, including finite values. A zero nonfinite-failure count does not imply absence of finite-loss clipping. Pythia's new wide study does not separately retain cap frequency. The Qwen crossover does, and its finite uncapped summaries can help diagnose cap effects, without reconstructing unsaved data.

ToyCar's authoritative risk is one minus complete-file ROC-AUC. MSE is the differentiable ranking objective and a separate diagnostic. Files have many windows but contribute one anomaly score each; normal/anomalous score ties receive half credit. Neither windows nor normal/anomalous score pairs are independent AUC replications.

The main grid uncertainty is over independent whole proposed fault fields, conditional on the checkpoint, cohort and constructed masks. Random-mask seed replication and independently trained checkpoints are different missing/generalization dimensions. A narrow field CI cannot demonstrate model-population generality. Candidate-pool experiments principally generalize over their prespecified protected/unprotected-stratified pool design; pool means, not pool-by-field cells, are its units. Input resampling, where supplied, is another level. Keep each original multiplicity family intact. Separate plots of marginal means cannot replace paired hypothesis tests.

## Interaction attribution and scaling boundaries

The historical all-pool interaction audit verifies 56 pool/BER conditions and 112 selection/final bundles. Its twelve KWS pools recur across four arms; those are not 48 independent pool samples. For an outside field G and unprotected candidate set C minus Q, the exact integrated risk decomposes into the clean isolated prediction, outside-field intercept, outside-conditioned singleton correction, and candidate interaction residual. The common intercept cancels in mask contrasts. This algebra retains all candidate orders, but is not a causal attribution of architectural redundancy.

The recorded selection-side strict reversal counts change from 396/328/305/376/201/196 using clean additive scores to 8/0/0/4/1/0 after outside conditioning (six campaign arms in the published audit order). These are correlated mask-pair comparisons, not independent binomial observations. All six historical held-out selected-minus-isolated pool intervals include zero. One must not relabel the same-selection-field optimum as held-out regret or infer causal mechanisms from selection/final discrepancies.

The observed prominence of outside conditioning says that a small pool is embedded in a corrupted network: its individual bits can have different effects when other coordinates are already faulty. It does not prove that within-pool interactions vanish. Their Bernoulli weights include p² and higher powers; small candidate pools may rarely experience coflips even while many outside bits flip.

The M11 bound is mathematically valid for a fixed cohort and bounded risk `0<=R<=L`: with m unprotected candidates, the remainder of outside-conditioned singleton prediction is at most `L m p [1-(1-p)^(m-1)]`. Equal-budget restricted selection regret is at most twice a uniform approximation error, capped by L. The bound is sharp over arbitrary bounded set functions, but can be loose or trivial in a real network. It is not a novel foundation of Boolean-function or reliability theory. State its assumptions and do not use it as empirical proof that a low-order method works at deployment scale.

The inherited M11 delivery documentation predates subsequently executed research campaigns; its status prose must not override finalized artifacts. The primary manuscript need not contain the full spectral program. A short appendix bound and compact interaction diagnostic are enough to support the engineering question. Production numbers require the current evidence audit rather than copying old implementation-stage tables.

## Draft claim that survives the audit

NetSuRF evaluates resource-constrained protection of declared stored INT8 parameter-bit populations using exact software execution and common random fault fields. Its measurements show that protection rankings can exchange relative performance as the checkpoint, task, BER, or budget changes. Exact conditional experiments distinguish changed protection decisions from measurable held-out improvements; the new independent KWS pools do not establish an advantage over exact isolated selection. The contribution is a reusable decision-evaluation methodology and a measured operating-condition map, with an analytical explanation of why isolated damage lacks a universal allocation guarantee. A claim of universally superior dynamic optimization, or physical radiation reliability, is unsupported.

## Source locations used

- Attached original PDF, SHA and source extraction recorded by the parent manuscript build.
- Attached Overleaf `main.tex`, `include/metadata.tex`, and all numbered section sources.
- `AGENTS.md`, `docs/SCIENTIFIC_CONTRACTS.md`, `docs/M97_FKERAS_REPRODUCTION.md`, `docs/M957_FROZEN_MASK_REPLICATION.md`.
- `netsurf/estimators_m3.py`, `netsurf/adapters/numpy_reference.py`, and the legacy/MNIST adapter qPolarGrad branches.
- Historical `m11-conditional-scaling/docs/M910_INTERACTION_ATTRIBUTION.md`, `M11_THEORY.md`, `M11_RESULTS_AND_LIMITATIONS.md`.

The numerical result text in this audit is a methodology cross-check of existing reports; the manuscript's result provenance must link independently verified finalized artifacts and the current evidence tables.
