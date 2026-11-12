# Run P (DRAFT, not frozen): mean-variance protection masks

Status: draft, revised 11 November 2026 after the researcher's answers to the first draft. Nothing has been run.
Nothing here is a result. The plan becomes frozen only after the researcher approves the name and the go-ahead
for Phase 0; any later change is recorded as a dated amendment before outcomes exist.

Naming rule (AGENTS.md): this is a new proposal, not qPolarGrad. The recovered expression stays
`legacy_qpolargrad` and is not used here. The final name is open (Section 9); the code name until then is
`mean_variance_mask`.

Researcher's decisions recorded on 11 November 2026:
- Use statistical terms (mean, variance) in descriptions.
- Test both a mask built for a single fault rate and one built for a range, in the same cells and plots as every
  other method, so that all can be compared.
- Calibration size: to be determined (rule in Phase 0).
- If the pilot works, run the method on all models and show it alongside every case already tested.

## 1. Why this study

- The positive-gradient ranking makes DistilBERT and Pythia worse than no protection. The first-order effects of
  the bits it leaves unprotected sum to +6.7 against -20.6 on DistilBERT at 12.5%; for every other mask they
  cancel to within 0.5% (Table 7 of the draft).
- An exploratory prototype (`workflows/meanfield_drift_prototype.py`, results in
  `workspace/prototypes/meanfield_drift_v1*`) evaluated DistilBERT once with every weight set to its *mean*
  under the faults, so with the mean displacement of the weights and none of their variance. For the
  positive-gradient mask at 12.5% that single pass reproduces the measured accuracy: 0.844 vs 0.849 at 3e-3,
  0.586 vs 0.585 at 1e-2, 0.465 vs 0.470 at 3e-2. For the unprotected and random masks it predicts accuracy of
  about 0.91 at every rate, while the measured accuracy falls to 0.52-0.60.
- So the damage has two separable sources: the **mean displacement** of the weights, which dominates when the
  unprotected faults are lopsided, and the **variance** of the displacement, which dominates for unprotected and
  random masks. The absolute-gradient ranking reduces the variance term well. Nothing guarantees that its mean
  displacement is harmless.
- The absolute gradient ignores the sign, so it protects the two directions in whatever proportion they occur
  among the largest effects (50% measured on DistilBERT; not guaranteed). Two checkpoints are likely lopsided:
  pruned ResNet-8 (50% and 70% of codes are zero; for a zero code bits 0-6 all move the weight up and only the
  sign bit moves it down), and under-trained ResNet-8 (the mean gradient is not zero, and this is the checkpoint
  on which the preferred ranking reversed).

Question: does a mask that controls both the mean and the variance of the leftover displacement beat the
absolute gradient where fixed rankings fail, without losing anywhere else?

## 2. The objective and where it comes from

Notation: weight `w_i = s_i q_i`; bit `b` of weight `i` has exact jump `Delta_b = s_i c_b` with
`c_b = +-2^j` (or `-+128` for the sign bit) set by the stored bit value; faults are independent Bernoulli(p); the
mask P protects B bits. The displacement of weight i is `delta_i = sum over unprotected bits b of i of
Z_b Delta_b`, with `Z_b ~ Bernoulli(p)`. Jumps of the bits of one word add exactly in two's complement.

- Mean: `mu_i = p * sum_{b unprotected} Delta_b` (exact).
- Variance: `Var(delta_i) = p (1 - p) * sum_{b unprotected} Delta_b^2` (exact; independent bits, Bernoulli
  variance times squared jump). Covariances between weights are zero.

Second-order expansion of the calibration loss around the *mean* weights (the linear term vanishes because the
expansion point is the mean):

    E[L(theta + delta)]  ~  L(theta + mu)  +  1/2 * sum_i H_ii(theta + mu) * Var(delta_i)

The first term is evaluated exactly with one forward pass of the differentiable companion at the mean weights
(no re-rounding of weights). The second needs the diagonal curvature `H_ii`. For cross-entropy near a minimum the
Hessian is approximated by the Fisher information, `H ~ E_x[g(x) g(x)^T]`, so `H_ii ~ E_x[g_i(x)^2]`, and the
variance term becomes

    1/2 * p (1 - p) * sum_{b unprotected} E_x[(g_i(x) * Delta_b)^2]  =  1/2 p (1 - p) sum E_x[e_b(x)^2].

This is why the variance term uses **squared** per-example effects: variances of independent jumps add as
squares, and the Fisher curvature is a squared gradient. The paper's absolute gradient `E_x|e_b(x)|` is the first
absolute moment of the same per-example effects; it differs from the second moment when a bit's effect varies a
lot between inputs (`E[e^2] = (E|e|)^2 + Var(|e|)`). Limits of the approximation: the Fisher matrix is positive
semidefinite and matches the Hessian only near a minimum with a well-calibrated model; it is weakest on the
under-trained checkpoint, where the mean term (which is exact) carries the reversal. As a check, Phase 0 also
estimates `H_ii` directly with Hutchinson's estimator (Hessian-vector products at the mean weights) on the pinned
and under-trained checkpoints and reports the rank agreement with the Fisher diagonal.

Without the mean term the objective ranks bits by `E_x[e_b(x)^2]` ("squared-effect ranking", an ablation arm).
With it, the method protects bits whose contribution to the mean displacement is harmful and lets the remaining
contributions offset each other, weighted by how sensitive the loss is in each direction. Near a minimum any mean
displacement raises the loss, so the method makes it cancel. Away from a minimum (under-trained), a small mean
displacement in the loss-lowering direction helps; the objective keeps it at small p and removes it where it
would overshoot at large p.

Construction (deterministic greedy swaps):
1. Start from the absolute-gradient mask at budget B.
2. Compute the mean weights and the gradient of the calibration loss there (one forward and one backward pass).
3. Change in the objective from protecting unprotected bit b: mean term `-p Delta_b dL/dw_i` (first order at the
   mean weights), variance term `-1/2 p(1-p) E_x[e_b(x)^2]`; releasing a protected bit: the opposite.
4. Swap the m best additions with the m cheapest releases (budget stays exactly B); accept only if the objective,
   recomputed on the calibration images, decreases; halve m after a rejection. Stop after 40 rounds or m < 16.

Two versions, both evaluated in every cell of the grid:
- **Single-rate version:** one mask per (BER, budget) cell, built for that BER.
- **Range version:** one mask per budget, built by averaging the objective over 8 rates on a log-uniform grid.
  Primary range W = [3e-4, 3e-2], the band in which masks differ in usefulness on the measured checkpoints
  (it includes the under-trained reversal at 1e-3). Secondary range N = [3e-3, 3e-2], identical to the
  range-optimized relaxed mask of Run N (c2r), for a like-for-like comparison with it.

## 3. Phase 0: offline checks on saved data (no fault injection; under an hour)

Inputs, read-only: finalized Run A, A2 and K bundles (`artifacts/protection_masks.json`,
`artifacts/fault_schedules.json`, `artifacts/sample_partitions.json`), the Run J checkpoints, the five Run D
seeds, the DistilBERT drift data.

- **P0a, lopsidedness.** For the absolute-gradient, global-gradient, positive-gradient and MSB-first masks of
  every ResNet-8 checkpoint (pinned, five retrained seeds, pruned 50% and 70%, under-trained) and DS-CNN: share of
  protected bits whose predicted effect raises the loss, summed loss-raising and loss-lowering first-order effects
  of the unprotected bits, and the mean displacement per tensor in code units.
- **P0b, does the mean term explain ResNet-8?** Accuracy and loss at the mean weights for every saved Run K mask at
  12.5% and 25% and BER 3e-4 to 1e-2, against the measured values. In particular: does it reproduce the
  under-trained reversal (positive gradient better than absolute gradient at 1e-3, worse at 1e-2)?
- **P0c, does the objective rank masks?** Spearman correlation between the objective and the measured accuracy
  loss across the seven saved masks within each decision-useful cell.
- **P0d, calibration size.** On pinned and under-trained ResNet-8 (replication 1), build range-W masks from
  n = 64, 128, 256 and 512 calibration images and evaluate their objective on 256 held-out images (training
  images 27000-27255, as in Run N). Rule fixed now: use the smallest n whose held-out reduction of the objective
  relative to the absolute-gradient mask is at least 80% of the reduction obtained with n = 512. If n = 64, the
  new masks use each replication's own 64-image calibration draw, the same data as the seven existing rankings.
  If n > 64, an extra arm (absolute gradient from the same n images) is added, so that more data and the new
  objective are not confounded.
- **P0e, Fisher vs Hessian diagonal** (Section 2): rank agreement on two checkpoints; reported, not a gate.

**Gate, fixed now.** Phases 1-2 run only if (i) P0b reproduces the sign of the under-trained reversal at both 1e-3
and 1e-2, and (ii) the median within-cell Spearman correlation in P0c is at least 0.7 on the pinned and
under-trained checkpoints. Otherwise the study stops, and the report states that the model behind the method does
not describe ResNet-8 well enough to build masks from it.

## 4. Phase 1: build masks (minutes per checkpoint)

Per checkpoint and replication: single-rate masks for the 8 grid BERs x 4 budgets; range-W and range-N masks for
the 4 budgets; squared-effect ranking for the 4 budgets. Every mask is saved with its hash and its objective on the
calibration and held-out images.

## 5. Phase 2: evaluation on the saved fault fields of earlier runs

Design principle: the new masks are evaluated on **exactly the fault fields, input windows and calibration draws
already used for the seven existing rankings**, replayed from `fault_schedules.json`. Every comparison is then
paired with the saved outcomes of all seven rankings, and every existing table and figure gains the new methods
as extra rows or curves without re-running anything old.

Replay check, fixed now: before any new mask is evaluated, replaying the saved absolute-gradient mask on the saved
fields must reproduce the saved per-field accuracy exactly, for every field of at least one replication per
checkpoint. Any mismatch stops the run.

- **Block A, repeated evaluations (full grid: 8 BERs x 4 budgets).**
  Pilot (primary): under-trained and pruned-70% ResNet-8 (Run K: 8 replications x 2 fields per BER) and pinned
  ResNet-8 (Run A: 9 replications x 4 fields per BER).
  Pilot (secondary): pruned-50% ResNet-8 (Run K), DS-CNN (Run A2: 11 x 8), and the five retrained seeds (Run K on
  Run D) once their bundles are finalized.
  New arms: single-rate, range W, squared-effect ranking (plus absolute gradient from n images if P0d gives n > 64).
- **Block B, method campaigns (Run N protocol).** Pinned ResNet-8, test images 2000-2999, the 16 Run M fault fields
  per rate, rates 3e-3, 1e-2, 3e-2, budgets 12.5% and 25%: single-rate and range-N masks next to MC search,
  relaxed, relaxed-range, group CEM, Thales and HarDNN.

## 6. Decision rules (fixed before any Phase 1 or 2 outcome)

- **Primary contrast.** Range W against absolute gradient, accuracy, decision-useful cells, on the three primary
  checkpoints; one Holm family over all their range-W vs absolute-gradient contrasts on the full grid. Paired
  fields pooled for resolution; Student-t bounds over replications reported alongside.
- **The method helps** if, on the under-trained or pruned-70% checkpoint, range W is resolved better than the
  absolute gradient by at least 1 accuracy point in a decision-useful cell, and is not resolved worse by 0.5 point
  or more in any decision-useful cell of any primary checkpoint.
- **No gain** if no difference of 0.5 point or more is resolved in either direction. Reading: the absolute gradient
  already leaves a harmless mean displacement on these checkpoints (to be checked against P0a).
- **Harmful** if range W is resolved worse by 0.5 point or more in any decision-useful cell.
- **Secondary, exploratory:** single-rate vs range W (cost of committing to one mask without knowing p); squared-
  effect vs absolute gradient (does the second moment matter); range W vs squared-effect (what the mean term adds);
  all new arms against the other six rankings; Block B against the Run N methods.

## 7. Cost, prerequisites, outputs

- **Phase 0:** under an hour, CPU, no fault injection.
- **Phase 2, pilot primary:** Run K checkpoints 3 arms x 32 cells x 16 fields = 1,536 evaluations each; pinned
  ResNet-8 3 x 32 x 36 = 3,456; total about 6,500 evaluations, about 22 s each per worker (Run K timing), about
  5 hours with 8 workers. Pilot secondary: about 9 further hours (DS-CNN timing to be measured in the replay check).
  Block B: 4 masks x 3 rates x 2 budgets x 16 fields = 384 evaluations, under 30 minutes.
- **Memory** under 4 GB per worker; **storage** under 2 GB (masks are about 0.1 MB each, packed).
- **Prerequisites:** Run K on the retrained seeds and Run N finished; new code on a separate branch with the tests
  below passing; launch only with the researcher's go-ahead.
- **Outputs:** `workspace/runs/paper1_runP_<block>_<checkpoint>_v1/`, write-once after finalization; plan hash in
  the launcher; registered in the run monitor.
- **Tests, written before any run:**
  1. Mean and variance of a word's displacement: formula against enumeration of all 256 fault states, for every
     mask pattern of random words, including the sign bit.
  2. Zero displacement reproduces the clean companion loss; companion agreement with the INT8 executable reported.
  3. On a synthetic quadratic loss, mean term + variance term equals the exact expected loss by enumeration.
  4. Greedy swaps keep exactly B bits; with the mean term removed the procedure returns the top-B of the
     squared-effect ranking; identical inputs give the identical mask hash.
  5. Replay of saved fields reproduces saved outcomes (the replay check above).

## 8. If the pilot succeeds (Phase 3, separate frozen plan)

Run the method on all models and show it alongside every case already tested, with the same tables and figures:
the main grid of the six models (replaying the saved main-grid fields), the repeated evaluations, the ResNet-8
variants and retrained seeds, the method campaigns, and the correction-mechanism comparison (triplication of the
bits the new mask selects against SEC-DED and ABFT). The transformers need the W8A8 mean-weight pass of the
prototype plus a backward pass; cost will be estimated from the pilot.

## 9. Open choice: the name

Candidates (none may be "qPolarGrad"); descriptive term for the paper text: "mean-variance protection".
See the conversation of 11 November 2026 for the shortlist.

## 10. Amendments

**A1 (11 November 2026, 17:50 CST, before any Phase 0 computation).** Gate (ii) restricts the within-cell Spearman
correlation to decision-useful cells with BER of at least 1e-3. Below that rate the seven masks differ by less than
the field noise (all within about one accuracy point on every Run K checkpoint), so a rank correlation there
measures noise and would make the gate depend on how many low-rate cells happen to be decision-useful. The measured
accuracy loss in each cell is the mean over all replications and fields; the objective is evaluated per replication
on that replication's calibration images and averaged over replications. P0b and P0c use 12.5% and 25% for the
reversal test and all four budgets for the correlation. Gate (i) requires the predicted sign of (positive-gradient
minus absolute-gradient accuracy) at the mean weights to match the measured sign in all four cells (two budgets x
1e-3 and 1e-2). Predicted accuracies come from the differentiable companion at the mean weights and are compared
as changes from the companion's own clean accuracy. P0d takes the larger of the two checkpoints' selected n; if the
held-out reduction at n = 512 is not positive on a checkpoint, the report says the objective does not transfer to
held-out images there, and n = 512 is used if Phase 2 is run at all. The researcher approved Phase 0 at 17:45 CST ("Yes, start
Phase 0"); the name remains open and the code name is `mean_variance_mask`.

**A2 (11 November 2026, about 18:25 CST, after the P0e Hutchinson output and before any P0a-P0d outcome was
inspected).** The 64-probe Hutchinson estimate of P0e is dominated by off-diagonal noise (49.9% and 49.8% of the
estimated diagonal entries are negative on the pinned and under-trained checkpoints; Spearman with the Fisher
diagonal 0.018 and 0.016), so it cannot test the Fisher approximation. P0e is repeated with the exact diagonal
`e_i^T H e_i` (one Hessian-vector product per weight) on 512 weights per checkpoint: 256 drawn uniformly and 256
drawn uniformly from the top decile of the Fisher diagonal (seed 20261112), 64 calibration images of replication 1.
Reported: Spearman correlation, share of negative entries, median ratio Hessian/Fisher. P0e remains reported, not
a gate. The Hutchinson output is kept as recorded.

## 11. What would not be claimed

- Nothing about `legacy_qpolargrad`.
- A positive ResNet-8 result says nothing about the transformers until Phase 3.
- The variance term is an approximation (second-order expansion, Fisher curvature); Phase 0 tests its usefulness.
- Protection remains ideal logical suppression; no hardware cost.
