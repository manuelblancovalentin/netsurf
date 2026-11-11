# Run P (DRAFT, not frozen): masks that keep the push of the leftover faults harmless

Status: draft for discussion with the researcher, written 11 November 2026 at their request ("Yes, write the
plan. You don't have to follow the old equations"). Nothing here has been run. Nothing here is a result. The
plan becomes frozen only after the researcher approves the open choices in Section 8; any change after that is
recorded as a dated amendment before outcomes exist.

Naming rule (AGENTS.md): this is a new proposal, not qPolarGrad. The recovered expression stays
`legacy_qpolargrad` and is not used here. Working name of the new method: **push-balanced mask**
(code name `push_balanced`). The researcher may rename it, but not to qPolarGrad.

## 1. Why this study

What we already know, all on saved data:

- The positive-gradient ranking (protect only flips predicted to raise the loss) makes DistilBERT and Pythia
  worse than no protection. In those cells the leftover faults move the weights coherently one way: summed
  first-order effects +6.7 against -20.6 on DistilBERT at 12.5%, against a near-exact cancellation for every
  other mask (Table 7 of the draft).
- An exploratory prototype (`workflows/meanfield_drift_prototype.py`, `workspace/prototypes/meanfield_drift_v1*`)
  ran DistilBERT once with every weight replaced by its *expected* value under the faults, that is, with only
  the average push of the leftover faults and none of their randomness. For the positive-gradient mask at 12.5%
  that single pass reproduces the measured accuracy almost exactly: 0.844 vs 0.849 measured at 3e-3, 0.586 vs
  0.585 at 1e-2, 0.465 vs 0.470 at 3e-2. For the unprotected and random masks it predicts almost no damage
  (0.91 at every rate) while the measured accuracy falls to 0.52-0.60.
- So there are two different kinds of damage. **Push:** the average displacement of the weights caused by the
  leftover faults; it dominates when the leftover faults are lopsided. **Shake:** the random spread around that
  average; it dominates for unprotected and random masks. The absolute-gradient ranking is good at reducing
  shake. Whether it leaves a harmless push is not guaranteed.

The researcher's objection to the draft's wording ("it removes both signs about equally") is correct. The
absolute gradient ignores the sign, so it protects the two directions in whatever proportion they occur among
the largest effects. On DistilBERT that proportion was measured at 50%. Nothing guarantees it elsewhere. Two
checkpoints are likely to be lopsided:

- Pruned ResNet-8: 50% and 70% of the stored codes are zero. For a zero code, flips of bits 0-6 all move the
  weight up (+1 to +64 code units) and the sign-bit flip moves it down by 128. The push of a zero weight is
  lopsided by construction.
- Under-trained ResNet-8: its loss is not at a minimum, so the average gradient is not zero and loss-raising and
  loss-lowering flips are not symmetric. This is the checkpoint on which the preferred ranking reversed.

The question: **can a mask that keeps the absolute gradient's shake reduction, and also chooses bits so that the
average push of the leftover faults is harmless, beat the absolute gradient where fixed rankings fail, without
losing anywhere else?**

## 2. The method, in plain words

For a mask P and fault rate p, every stored code has an exactly known expected value:
`E[q'] = q + p * (sum over unprotected bits j of that word of the jump of bit j)`. The jumps of different bits of
one word add exactly in two's complement, so this is exact, not an approximation. The **push** of P is the
vector of these expected displacements over all weights.

Goal: choose B protected bits to minimize the expected calibration loss of the faulty model, approximated as

    J_p(P) = [loss at the pushed weights] - [clean loss]      (push; one forward pass, nonlinear, exact
                                                               for the average displacement)
           + (1/2) p (1 - p) * sum over unprotected bits b of E_x[(g_i(x) * Delta_b)^2]
                                                              (shake; per-bit spread times curvature,
                                                               curvature taken from per-example gradients)

Why this form: for a loss that is exactly quadratic in the weights, the expected loss under independent faults
is exactly "loss at the average weights + half the curvature-weighted spread". The push part is kept exact
(the real network at the pushed weights) because the prototype shows that is where the positive-gradient harm
comes from. The shake part uses the per-example gradients we already compute for the absolute gradient (the
same data, squared instead of absolute). With the push part removed, J ranks bits by `E_x[(g Delta)^2]`, a
close relative of the absolute gradient `E_x|g Delta|`.

What this does on each kind of checkpoint:

- Near a loss minimum, any push raises the loss, so the method protects bits whose leftover push is harmful
  and lets the remaining pushes cancel. This is the polar idea made precise: the cancellation that matters is
  of the *signed sizes* of the leftover jumps, weighted by how sensitive the loss is in that direction, not a
  count of signs.
- Away from a minimum (under-trained), a small push in the loss-lowering direction helps. The method keeps
  it at small p and removes it where it would overshoot at large p. This is the measured reversal of the
  positive-gradient ranking, turned into a design rule instead of an accident.

Construction (greedy swaps, deterministic):

1. Start from the absolute-gradient mask at budget B.
2. Compute the push, the pushed weights, and the gradient of the calibration loss at the pushed weights (one
   forward and one backward pass of the differentiable companion, with weights not re-rounded).
3. For every bit, estimate the change in J from flipping its protection status: protecting unprotected bit b
   removes `p * Delta_b` from its weight's push (first-order change: `-p * Delta_b * dL/dw_i` at the pushed
   point) and removes its shake term; releasing a protected bit does the opposite.
4. Swap the m best additions with the m least costly releases (budget stays exactly B); accept the swap only if
   the recomputed J on the calibration set decreases; halve m when a swap is rejected. Stop after 40 rounds or
   when m < 16.

Two versions:

- **Point version:** one mask per target fault rate p (an upper reference: what one gains by knowing p).
- **Range version:** one mask for an unknown rate. J is averaged over p drawn on a fixed grid of 8 rates
  log-uniform in [3e-4, 1e-2], the band in which the masks differ in usefulness on the ResNet-8 checkpoints
  (Table 5). This is the version a designer can commit to before knowing the fault rate, and it is the primary
  comparison.

Overfitting guard. With 64 calibration images the method could make the push cancel for those images only.
All gradient-based masks in this study therefore use 512 calibration images, and every mask's pushed loss is
also reported on 256 held-out validation images (training images 27000-27255, as in Run N).

## 3. Phase 0: offline checks on saved data (no fault injection; minutes)

Inputs, all read-only: the finalized Run K bundles (`artifacts/protection_masks.json` per replication), the
pinned ResNet-8 and DS-CNN Run A bundles, the Run J checkpoints, the five Run D seeds, and the DistilBERT drift
data.

- **P0a, lopsidedness (the researcher's point).** For the absolute-gradient, global-gradient, positive-gradient
  and MSB-first masks of every checkpoint (pinned ResNet-8, five retrained seeds, pruned 50% and 70%,
  under-trained, pinned DS-CNN), report: share of protected bits whose predicted effect raises the loss; summed
  loss-raising and loss-lowering first-order effects of the leftover bits; the size of the push in code units per
  tensor. Same quantities as Table 7, for every ResNet-8 checkpoint.
- **P0b, does the push explain the ResNet-8 results?** Push-only accuracy and loss (expected weights, one pass)
  for every saved Run K mask at 12.5% and 25% and BER 3e-4, 1e-3, 3e-3, 1e-2, compared with the measured
  values. Specifically: does the push alone reproduce the under-trained reversal (positive gradient better than
  absolute gradient at 1e-3, worse at 1e-2)?
- **P0c, does J rank masks correctly?** Spearman correlation between J (push + shake) and the measured accuracy
  loss across the seven saved masks within each decision-useful cell of each checkpoint.

**Gate, fixed now.** Phases 1 and 2 run only if (i) P0b reproduces the sign of the under-trained reversal at both
rates, and (ii) the median within-cell Spearman correlation in P0c is at least 0.7 on the pinned and
under-trained checkpoints. Otherwise the study stops, and the report says the model behind the method does not
describe ResNet-8 well enough to build masks from it.

## 4. Phase 1: build the masks (minutes per checkpoint)

Per checkpoint and replication: 512 calibration images from that replication's training range (Run K windows,
replication i uses training images [1000 i, 1000 i + 512)); absolute-gradient, squared-effect, positive-gradient
masks from those images; push-balanced point masks at 3e-4, 1e-3, 3e-3 and 1e-2; one push-balanced range mask.
Budgets 12.5% and 25%. Every mask's J on the calibration and held-out images is saved with its hash.

## 5. Phase 2: paired fault-injection evaluation

- **Checkpoints.** Primary: under-trained, pruned 70%, pinned ResNet-8. Secondary: pruned 50%, pinned DS-CNN.
- **Arms.** A1 absolute gradient (512 images; the reference). A2 squared-effect ranking (shake term only; an
  ablation separating "squared instead of absolute" from "push"). A3 push-balanced, point version. A4
  push-balanced, range version (primary). A5 positive gradient (512 images; the lopsided reference). A6
  MSB-first. Unprotected control.
- **Grid.** BER 3e-4, 1e-3, 3e-3, 1e-2; budgets 12.5% and 25%.
- **Replications.** Eight per checkpoint (Run K test windows 0-7999), four new fault fields per BER per
  replication (32 per BER), common to all arms. New fault seeds, fixed before launch.
- **Engine.** INT8 executable through the Run K engine (branch `paper1-runK-retrained-engine`), masks supplied
  from Phase 1. No change to the asset locks.

## 6. Decision rules (fixed before any Phase 1 or 2 outcome)

- **Primary contrast.** A4 against A1, accuracy, decision-useful cells, on the three primary checkpoints
  (24 contrasts, one Holm family). Paired fault fields pooled for resolution; Student-t bounds over replications
  reported alongside.
- **Push balancing helps** if, on the under-trained or pruned-70% checkpoint, A4 is resolved better than A1 by at
  least 1 accuracy point in a decision-useful cell, and A4 is not resolved worse than A1 by 0.5 point or more in
  any decision-useful cell of any primary checkpoint.
- **No gain** if no A4-A1 difference of 0.5 point or more is resolved in either direction. Reading: the absolute
  gradient already leaves a harmless push on these checkpoints (supported by P0a).
- **Harmful** if A4 is resolved worse than A1 by 0.5 point or more in any decision-useful cell.
- **Secondary, exploratory.** A2 vs A1 (does squaring matter); A4 vs A2 (what the push term adds); A3 vs A4 (how
  much is lost by committing to one mask without knowing p); A5 vs A1 (replicates the positive-gradient result
  with 512 calibration images). Secondary checkpoints are reported, not counted toward the primary reading.

## 7. Cost, prerequisites, outputs

- **Phase 0.** Under 30 minutes on CPU; no new fault field.
- **Phase 2.** About 1,664 evaluations per checkpoint (6 arms x 2 budgets x 4 BERs x 32 fields + 128
  unprotected); about 22 s per evaluation per worker (Run K timing); about 1.3 h per checkpoint with 8 workers;
  about 4 h for the primary checkpoints and 6 h in all. Memory under 4 GB per worker. Storage under 1 GB.
- **Prerequisites.** Run K on the retrained seeds and Run N finished (both are running); new code for the push
  and the mask builder on a separate branch, with the tests below passing; launch only with the researcher's
  go-ahead.
- **Outputs.** `workspace/runs/paper1_runP_push_balanced_<checkpoint>_v1/`, write-once after finalization; plan
  hash recorded in the launcher; registered in the run monitor.
- **Tests, written before any run.**
  1. Expected code: formula against enumeration of all 256 fault states of random words, for every mask
     pattern of a word.
  2. Additivity of per-bit jumps within a word, including the sign bit.
  3. Zero push reproduces the clean companion loss; report the companion's agreement with the INT8 executable.
  4. On a synthetic quadratic loss, push + shake equals the exact expected loss computed by enumeration.
  5. Greedy swaps keep exactly B bits; with the push term removed the procedure returns the top-B of the
     squared-effect ranking; same inputs give the same mask hash.

## 8. Open choices for the researcher (to settle before freezing)

1. Name of the method (default `push_balanced`; not qPolarGrad).
2. The rate range of the range version (default log-uniform [3e-4, 1e-2]).
3. Calibration size (default 512 for every gradient-based arm, which also changes the reference absolute-gradient
   mask from the paper's 64 images).
4. Whether the shake term should use squared per-example effects (default, from the quadratic expansion) or the
   absolute-gradient score (keeps the paper's default exactly and changes only the push).
5. Whether DistilBERT, where the harm was found, should be a Phase 3 after a positive ResNet-8 result. It needs
   the W8A8 mean-field pass of the prototype plus a backward pass at the pushed weights.

## 9. What would not be claimed

- Nothing about `legacy_qpolargrad`.
- A positive result on ResNet-8 checkpoints says nothing about the transformers until Phase 3 is run.
- J is an approximation (exact push, quadratic shake); its quality is what Phase 0 tests.
- Protection remains ideal logical suppression; no hardware cost.
