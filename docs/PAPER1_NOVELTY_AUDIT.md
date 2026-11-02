# Paper 1 adversarial novelty and baseline audit

Companion to [the original-to-current audit](PAPER1_ORIGINAL_TO_CURRENT_AUDIT.md).
This is a method-level comparison, not evidence that NetSuRF outperforms the cited
systems under their own fault laws. Sources checked on 2 November 2026.

## Primary sources and actual overlap

### FKeras — already sensitivity analysis and selective protection

Read the [JATS 2024 paper, §§3.1–3.2, Eq. (3)](https://kastner.ucsd.edu/wp-content/uploads/2024/05/admin/jats24-fkeras.pdf)
and [pinned source](https://github.com/KastnerRG/fkeras/blob/68e775c77670c9a1aab4e1b5295b523347d97013/fkeras/metrics/hessian.py).
It already combines weight-Hessian information, bit-tier prioritization, fault
injection and selective redundancy. Its Hessian contains cross-parameter terms;
calling it a diagonal estimator that cannot contain interactions is wrong.

The printed projection is `sum_i lambda_i (v_i^T theta) v_i`. NetSuRF preserves
this signed expression and uses its magnitude for ordering. The pinned `sum`
implementation effectively accumulates `sum_i abs(lambda_i^2 (v_i^T theta) v_i)`:
both the extra eigenvalue and absolute-before-sum matter. Neither variant is
selected for favorable outcomes. The validated solver is a numerical correction,
not a faithful reproduction of the historical power-iteration defects.

**NetSuRF distinction:** exact deployed-code risk versus a continuous companion
Hessian score, and held-out matched-budget decision validation. It is not the first
Hessian/bit-ranking/co-design flow. A clean-point Hessian includes local mixed
curvature but is not automatically the deployed Bernoulli conditional marginal.
Strict INT8 tier-first masks equal corrected MSB masks at 12.5% and 25%; use partial
tiers to test score-dependent decisions. See `M97_FKERAS_REPRODUCTION.md`.

### PrioriFI — adaptive fault prioritization is not new

Read [ASPLOS 2026 paper, §3 and Algorithm 1](https://kastner.ucsd.edu/wp-content/uploads/2026/05/admin/asplos2026-priorifi.pdf),
[DOI](https://doi.org/10.1145/3779212.3790204). PrioriFI initializes Hessian-guided
rankings within significance tiers, samples initial tier representatives, and
uses recent observed degradation to prioritize subsequent injections. It already
examines limitations of simple bit-significance assumptions.

**Distinction:** prioritizing which faults to test is not the same estimand as
expected risk of a fixed-budget protection mask under a specified simultaneous
Bernoulli field. NetSuRF should compare these objectives explicitly, not claim
novelty for adaptive exploration or observation that MSB heuristics can miss damage.

### EISFINN — full paper supplied and read

[IOLTS 2024 DOI](https://doi.org/10.1109/IOLTS60994.2024.10616075).
Alessio Colucci, Andreas Steininger and Muhammad Shafique. The complete supplied
three-page PDF, including figures, Table I and references, has SHA-256
`996060e82bc8e2d60cd9b20909b4bdd8dc923eb809a792a080c3edb6dbdd406f`.
The full-text access blocker is **resolved**.

Section II assigns attribution-derived sampling weights to **neuron-output tensors
and their bits**. It permits a mixture with uniform sampling. Attribution choices
include layer conductance; Figure 3 compares gradient/exponential/linear/random
bit weighting and attributed/random neuron selection. A fault injector compares
baseline and faulted test-set accuracy. Its objective is finding critical faults
efficiently; preliminary results vary across VGG11/ResNet18 and CIFAR-10/GTSRB.
Table I takes BinFI/AVFI results from their papers, not a common rerun.

The paper **does not specify** a target/proposal likelihood ratio, an unbiased
Bernoulli-risk estimator, a fixed-budget protection-allocation experiment, or an
exact persistent INT8 kernel-storage contract. Its “importance sampling” language
must not be silently equated with statistically corrected expected-risk estimation.
The asserted eventual coverage/recall requires support and sufficient sampling;
it is not a finite-campaign guarantee. Exact attribution-to-bit probability
normalization, mixtures and full experimental replication details are not supplied
in this short paper. This limits a faithful implementation comparison, not its
validity as a precedent for sensitivity-guided fault discovery.

**Consequence:** remove any NetSuRF claim to invent attribution-guided injection,
bit-weighted fault selection, or combined precision/coverage exploration. The
defensible distinction is the protection decision and exact target-risk contract,
not a generic “we consider important faults” formulation. Do not compare EISFINN's
critical-fault precision directly with NetSuRF's accuracy preserved per protected bit.

### Progressive bit-flip attack — exact bits and gradient search predate NetSuRF

Read [Rakin, He and Fan, ICCV 2019, §3](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rakin_Bit-Flip_Attack_Crushing_Neural_Network_With_Progressive_Bit_Search_ICCV_2019_paper.pdf).
The method works with quantized two's-complement weights, uses bit gradients and
feasible flip directions, and alternates intra-/inter-layer search to increase
loss with few chosen flips. It updates the attack state progressively.

**Distinction:** adversarially selected faults optimize a different objective from
independent random faults and cost-matched protection. They cannot be presented as
unweighted estimates of natural-fault risk. NetSuRF cannot claim to invent
encoding-aware gradients or state-dependent multi-bit search.

### “Learned or Ranked?” — substantial interaction/allocation overlap

The [Zenodo companion release](https://zenodo.org/records/23054820),
[DOI](https://doi.org/10.5281/zenodo.23054820), is a public dataset/software release;
peer review of the accompanying manuscript was **not independently established**.
The source archive `code__eccmarl-2ea13f9.tar.gz` was read, not executed (SHA-256
`82eed19a02e3242afc908ac249e549778cc0174b9b87ddfc1bfe5f4e72f7e000`).

`interaction.py` already studies paired common-random-number layer effects,
pairwise/nonadditive combinations, random-context marginals and ranking changes.
`reward.py` combines/clips accuracy degradation and costs. `faultmodels.py`
supports layer-level protection modes, packed ECC words, modeled miscorrection/
detected-uncorrectable behavior, bursts/interleaving and an ideal mode. Its
count-then-with-replacement flip locations can cancel through XOR; this is not
NetSuRF's independent-coordinate Bernoulli law. Decoder behavior is a modeled
approximation, not a verified physical ECC implementation.

**High novelty risk:** neither context-dependent importance nor learned/search
allocation versus rankings is new. NetSuRF's possible distinction is exact encoded
bit granularity, explicit Bernoulli identities, signed bounded-candidate attribution
and controlled scalar task-risk comparisons. Granularity, fault law, costs and
task must be compared—not dismissed because one release uses RL or lacks a verified
journal publication. No claim that RL cannot represent higher-order interactions.

### FORTUNE — not an ideal MSB mask on the same checkpoint

Read the [authors' ATS 2024 accepted manuscript, §II](https://d197for5662m48.cloudfront.net/documents/publicationstatus/231624/preprint_pdf/19c1f8a46e5ee0c13975ff55f41587bc.pdf).
It searches quantization precision, maps weights to **unsigned affine codes**, and
adds two redundant copies of each code's MSB with majority voting. Its memory
trade-off includes the change in precision; 3 data bits plus two redundant bits
are not the original 8-bit model. Thus an isolated MSB-only ideal-protection point
does not reproduce FORTUNE's pipeline, encoding, faulted redundancy or overhead.
The original NetSuRF Fig. 7 must not be retained as a fair implementation comparison.

### Bit-level redundancy protection — real hardware precedents

Read [Catalán, Flich and Hernández, §§3–5](https://link.springer.com/article/10.1007/s11227-024-06693-7)
(online 2024, journal volume 2025). Their fixed/variable mechanisms store invariant
bit patterns for weight groups and restore matching positions, including treatment
of mask protection. They examine FP32/INT8 redundancy and an FPGA implementation.
This is a physical recovery mechanism with specific representation and grouping
assumptions, not an arbitrary ranked subset receiving free perfect suppression.
It defeats broad novelty claims for selective encoded-bit hardening or low-cost
bit masks; a fair quantitative comparison needs its actual storage/logic contract.

### Ranger, ABFT, ECC and parity — different mechanisms and fault domains

The user's `arxiv:2003.13874` link is **Ranger, not progressive BFA**. Read
[Ranger §§II-C, III, IV](https://arxiv.org/html/2003.13874v4): it targets transient
datapath faults, assumes memory/cache/register-file protection, profiles activation
ranges and inserts clamps in activations and selected dependent operations.
Containment can preserve application output without restoring a corrupted weight.
Its software overhead is not an ASIC-cost measurement for NetSuRF.

For parity, detection requires a specified recovery/failure policy. For SECDED,
packed codeword layout, parity-bit exposure, correction/detection/miscorrection,
latency and decoder costs matter. For ABFT, checksums, trusted state, coverage and
recovery must be specified; a detector is not a correction guarantee. No current
matched NetSuRF hardware implementation supports quantitative comparisons among
these mechanisms. Keep them in related work and limitations, not invented curves.

### Own prior publication — full-text overlap confirmed

The researcher supplied Guo, Blanco Valentín, He and Ogrenci's **FCCM 2025 invited
paper**, “Toward Reconfigurable In-Pixel Computing: A Fault-Tolerant Design Flow
for Machine Learning Accelerators,” pp. 261–267,
[DOI](https://doi.org/10.1109/FCCM62733.2025.00014). All seven pages, equations,
references and three rendered figures were read. PDF SHA-256:
`80b08d098e1f3adc9fbd347c67b745f597bf0e4bee13d560dafd7500f11b5cec`.
The full-text blocker is **resolved**; the overlap is substantive.

| Already published in FCCM | Consequence for the new paper |
|---|---|
| §III: bit-level weight susceptibility, fault injection, user-defined protection budget and a named `GradDelta` technique | Do not claim to first introduce bit-level ranking or gradient/delta-aware selective allocation. `GradDelta` is described qualitatively, without a score equation; it cannot be equated to current S4 or historical qPolarGrad solely by its name |
| §§III–IV / Fig. 2: QKeras → hls4ml → Catapult → register-based reconfigurable weights → partial TMR → synthesis | The engineering flow is prior work to be credited and extended, not a new TECS contribution |
| §II / Fig. 1: Full-TMR, Reg-TMR and selected register triplication, voter and radiation assumptions | v2 ideal suppression does not reproduce the physical mechanism or validate its flux/cross-section assumptions |
| §IV: one six-bit, three-dense-layer network; 45 nm RVT synthesis; 20%/40% partial TMR and MAE comparisons | New frozen INT8 models, exact encoded representation, matched fields, independent selection/evaluation and cross-task metrics are genuinely additional evidence. No claim that these reproduce that old unspecified model/dataset exactly |
| Fig. 3: area/cell-count/power ratios and relative savings | Cite as historical reported results only, not new v2 resource measurements. Raw provenance must be re-established before reuse |
| §V: possible future tape-out | Not a fabricated-silicon validation of this framework |

The main **extension statement** should be explicit: prior work introduced the
reconfigurable selective-TMR flow; this study evaluates the reliability of the
ranking-to-allocation decision using exact frozen-code inference, corrected and
independently checked baselines, common fault fields, multiple tasks, and measured
limits of isolated rankings. It does not claim a new implementation of that
hardware flow. No old figure should be reused without attribution/permissions.

There are also normalization issues not to inherit: Fig. 3 shows Full-TMR area
294% and power 308% of baseline, whereas nearby text says area nearly doubles and
power increases about 175%. The quoted 19.7%, 15.3% and 47.3% area reductions match
ratios of **total areas** (147/183, 155/183, 155/294), not reductions of added
overheads. The paper does not give a checkpoint hash, dataset identity, BER grid,
sample/trial protocol or exact GradDelta formula sufficient to reproduce those
MAE claims. This is not grounds to alter the publication; it is a reason not to
transfer its numbers into the new evidence tables. Authorship order also differs
from the rejected TCAD draft and requires collaborator approval, not automatic
copying of either historical order.

## Baseline readiness for the rebuilt paper

| Method / mechanism | Implemented and validated scope | What the main paper can currently compare | Missing / blocked |
|---|---|---|---|
| Random | Seeded ordering, independent mask seeds, all three models | Complete static curves; distinguish 4/2 masks from 32/8 fields | More mask replications for random-policy population inference |
| Corrected MSB-first | Word-relative tier priority, deterministic ties | Complete static curves, full-tier equality control | Partial-tier sensitivity to tie ordering |
| Numerical jump | Exact affine XOR jump magnitude | Complete static/empirical comparisons | No task sensitivity implied by jump magnitude |
| Global first-order | `abs(E[g]*delta)` ranking, signed predictions separate | Complete static and sampled empirical results | Surrogate forward/derivative approximation |
| Per-example absolute | `E[abs(g_x*delta)]` | Complete static and sampled empirical results | Shared-gradient cost; not expected damage itself |
| Per-example positive | `E[max(0,g_x*delta)]` | Same three models | Positive per-example contribution is not positive aggregate damage |
| Historical qPolarGrad | Recovered dense first-call semantics, separate alternatives | Historical/synthetic ablation only | No validated DS-CNN/ResNet/ToyCar qPolarGrad pathway |
| Exact isolated | Restricted sampled bits / small pools | Isolated prediction reference and bounded-mask comparator | Not a computed whole-network isolated ranking |
| FKeras paper/repository | Validated continuous KWS Hessian and tier-first mapping | Source/formula comparison and numerical smoke | Only 2 ranking examples, 4 test examples, 2 fields: no useful efficacy evidence; no principal-model-wide baseline |
| Exact bounded conditional allocation | Frozen context, equal global cardinality, selection/final separation, three models | All-pool diagnostic comparisons | Small-pool result is not whole-network optimality; gains versus exact isolated unresolved |
| TMR | Historical register/voter flow | Related work / historical audit | No matched v2 physical overhead or residual-failure curve |
| Parity / SECDED / ABFT / Ranger | Not implemented as matched v2 mechanisms | Scope-aware related work | Actual fault domain, handling, implementation and cost prerequisites |

## Hostile-review verdict and concrete fixes

1. **“This is a toolkit comparison, not a new method.”** A real risk. State the
   methodological contribution as an auditable connection from exact stored-code
   susceptibility to resource-constrained task outcomes; identify the new controlled
   evidence and decision-validity diagnostic, not generic Monte Carlo or new acronyms.
2. **“The strongest gain is one favorable ResNet condition.”** Show every measured
   BER, both budgets and all three models; report ToyCar reversals and KWS nulls.
   Do not select headline peak gaps after viewing the grid.
3. **“Hessian was not fairly tested.”** Publish tier degeneracy and both pinned
   variants; prioritize partial-tier KWS evaluation with adequate calibration/fields.
4. **“Conditional methods beat a weak surrogate, not isolated truth.”** Correct:
   independent-pool held-out selected-minus-exact-isolated intervals all cross zero.
   Say so; use conditional work to delimit static guarantees, not claim superiority.
5. **“No hardware resources are measured.”** Define budget as stored eligible bits;
   remove area/power/latency improvement statements. Discuss physical mapping as a
   separate necessary validation, not a generic plug-in solved by interpolation.
6. **“Data reuse invalidates confirmation.”** Label existing studies exploratory,
   keep exposure ledgers, and distinguish independent fault fields from untouched
   inputs. Increasing fields does not create new models or new evaluation data.
7. **“M11 adds formalism without practical gain.”** Put exact identity/proof in the
   appendix; report that worst-case bounds can be vacuous and bounded validation is
   not scalable efficacy evidence. Omit unexecuted spectral/scaling claims.

The available evidence supports an honest internal engineering draft. Novelty and
submission strength remain contingent on fair partial-tier baselines, explicit
prior-work/own-publication differentiation, and useful cross-model decision guidance.
