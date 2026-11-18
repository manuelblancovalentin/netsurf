# A fault-rate-aware ranking: derivation and an exploratory check (12 November 2026)

Status: exploratory and post hoc.
- This note answers the researcher's question of why the preferred gradient ranking changes with the BER, and
  whether a rule derived from first principles reproduces the change automatically.
- The comparison in Section 2 was not planned before the data existed, so it is not a confirmed result and is not in
  the paper.
- The method is the Run P mean-variance mask (code name `mean_variance_mask`, single-rate version `mv_single`). It is
  not `legacy_qpolargrad`.

## 1. Why the preferred ranking changes with the BER

**Set-up.** Consider one input. Each unprotected bit b flips independently with probability p and changes the loss
by about e_b = g · Δ_b, its signed first-order effect. A fault field is a random set of such flips, and the expected
loss change splits into two parts (Run P plan, Section 2).

**Mean displacement ("drift").** The weights move on average by μ = p Σ_unprotected Δ_b. The loss change this causes
is g·μ + ½ μᵀHμ:
- The first term is linear in p and signed. Flips that lower the loss on average make it negative.
- The second term grows as p², and as the square of how one-sided the unprotected set is. It is the cost of many
  faults pushing the weights coherently in the same direction, overshooting.

**Spread around it (variance).** This term is ½ p(1−p) Σ_unprotected (curvature × Δ_b²). It is linear in p and
blind to the sign of the effect.

How the regime shapes the best mask:

| Regime | Dominant terms | Best mask | Closest static ranking |
|---|---|---|---|
| Sparse faults (small p) | linear drift and spread | Leaves loss-lowering flips unprotected, which act like a small training step, and spends the budget on loss-raising flips | Positive gradient, since E max(0, e) = ½ (E\|e\| + E e): half magnitude, half signed mean |
| Dense faults (large p) | the p² overshoot term | An unprotected set that is balanced (drift ≈ 0) and has small spread | Absolute gradient, which protects by magnitude regardless of sign |
| Crossover | — | — | Roughly where p · \|g·v\| ≈ ½ p² vᵀHv, with v the one-sided displacement direction |

The crossover is at a higher rate the farther the model is from a minimum (larger g). This matches the low-BER lead
of the positive gradient: +1.2 to +2.5 points on the under-trained checkpoint, about +0.5 on the well-trained ones.

**Why 64 images hide the effect.** The signed mean E e is a sum of terms that largely cancel across inputs, so it
needs many inputs to estimate. That is why the positive gradient needs hundreds of calibration images, as Run Q
shows.

**What the derivation gives.** It is not a threshold rule. The objective J_p = [L(θ + μ) − L(θ)] + ½ p(1−p) Σ curvature Δ²
contains p, and a mask that minimizes it for a given p weighs drift against spread automatically.
- Run P built exactly this mask (`mv_single`, greedy swaps, 512 images, curvature from mean squared gradients).
- Run P also built a version for a range of rates (`mv_range`).

## 2. Exploratory check on existing replays

Script `workflows/runPQ_envelope_explore.py` (branch `paper1-runP-mean-variance`, commit 755a405). Output
`workspace/runs/paper1_runPQ_envelope_explore_v1/summary.json`, read-only, sha256
`76372b50ca8284ef3bcfeaf8f5f20cae04da41cd918373a242ac5b19d0ca5d7c`.

The comparison uses:
- the same saved fault schedules, replications (Q-A1 applied) and usable cells as Run Q;
- `mv_single` built for the test rate from 512 images, from Run P;
- `pos512` from Run Q.

Values are medians over usable cells of the nine ResNet-8 checkpoints (accuracy points), with the number of cells
where the first mask is ahead:

| BER | Cells | pos512 − abs512 | mv_single − abs512 | mv_single − pos512 | mv_single − better of (abs512, pos512) | mv_range − pos512 |
|---|---|---|---|---|---|---|
| 1e-3 | 36 | +0.65 (36 ahead) | +0.74 (36) | +0.09 (22) | +0.09 (22); lowest −0.38 | −0.59 (0) |
| 3e-3 | 29 | +0.17 (15) | +0.45 (27) | +0.33 (23) | +0.11 (21); lowest −0.42 | −0.04 (13) |
| 1e-2 | 15 | −10.2 (0) | +0.56 (12) | +10.9 (15) | +0.56 (12); lowest −0.36 | +10.5 (15) |
| 3e-2 | 2 | −52.4 (0) | +0.81 (2) | +53.2 (2) | +0.81 (2) | +53.0 (2) |

**Reading.**
- The mask derived for the known rate behaves like the positive gradient at 1e-3 and like, or slightly better than,
  the absolute gradient at 1e-2 and 3e-2.
- In none of the 82 usable cells is it more than 0.42 points below the better of the two static rankings, even though
  that better ranking is picked per cell after seeing the outcomes. It is ahead of that per-cell pick in 57 of 82
  cells.
- The range version (`mv_range`) gives up the low-rate lead (−0.59 against the positive gradient at 1e-3). Run P
  found that it loses 7 to 32 points at 1e-1, outside its design range.

**Limits.**
- Post hoc, on the same fields used before. There are no intervals and no multiplicity correction.
- ResNet-8 only; BER 1e-3 to 3e-2; 512 calibration images.
- The single-rate mask needs the fault rate to be known.
- Gains over the better static ranking are small on well-trained checkpoints (median 0.1 to 0.8 points).
- Run P's frozen verdict compared the mask with abs512 alone, using its own margins. It is unchanged, and this note
  does not overturn it.

## 3. What would confirm it (proposal, not launched)

A frozen confirmatory run on fresh fault fields:
- **Checkpoints:** the nine ResNet-8 checkpoints, using their existing masks: abs64, abs512, pos512 and mv_single.
  No new training and no new mask search.
- **Rates:** BER 3e-4 to 3e-2. Run Q has no data below 1e-3, so 3e-4 is included.
- **Rule fixed in advance:** for example, mv_single is never resolved below the better static ranking by 0.5 point in
  a usable cell, and it is resolved ahead of the absolute gradient at the low rates.
- **Cost:** about 20 fresh fields per replication; 72 jobs at 26 in parallel; about 2 hours; well under 1 GB.

It needs the researcher's go-ahead and a frozen plan before launch.

## 4. Predicted crossover against the Run Q measurements (added 12 November 2026, exploratory)

Scripts `workflows/runPQ_crossover_theory.py` (commit f6a5a66) and `workflows/runPQ_crossover_plot.py` (commit
80f081e), branch `paper1-runP-mean-variance`. Outputs, read-only:
`workspace/runs/paper1_runPQ_crossover_theory_v1/` (one file per checkpoint) and
`workspace/runs/paper1_runPQ_crossover_theory_v1_analysis/` (`summary.json` sha256 `1b139636d3248fa5…`,
`summary_maxp0.03.json` sha256 `851cae16dd1b2520…`, two figures).

**What was computed.** For each checkpoint, replication, budget and calibration size (512, 1,024), the objective
J_p of the positive and of the absolute mask on 21 rates from 1e-4 to 1e-1, on the 512 held-out training images
28000-28511. No fault outcome is read. The predicted crossover p* is the rate above which the positive mask is
predicted worse at every larger grid rate.

**Predicted crossovers (1,024 images, four budgets).**
- Pinned and five retrained seeds: 2.4e-3 to 5.2e-3.
- Under-trained: 6.9e-3 to 8.8e-3, about twice as high, as expected for a model farther from a minimum.
- Pruned: 3.1e-3 to 6.0e-3 on the measured rate range (see the caveat below).

**Agreement with the measured sign of positive minus absolute (1,024 images).**

| BER | Usable cells | Sign agrees | All cells | Sign agrees |
|---|---|---|---|---|
| 1e-3 | 36 | 36 | 36 | 36 |
| 3e-3 | 29 | 21 | 36 | 27 |
| 1e-2 | 15 | 15 | 36 | 36 |
| 3e-2 | 2 | 2 | 36 | 36 |

- 74 of 82 usable cells agree, and all 56 usable cells whose 95% interval over replications excludes zero agree.
- The 8 disagreements are all at 3e-3, all unresolved, and all within 15% below the predicted crossover (BER / p*
  from 0.85 to 0.96). The measured switch is therefore at or slightly below the predicted one.
- With 512 images: 73 of 82 usable cells, and all 53 resolved ones.

**Caveats.**
- Post hoc: the derivation was written after Run Q was seen. The prediction itself uses no outcome.
- The definition of p* was written before any output. On the two pruned checkpoints the predicted difference turns
  negative again above 5e-2, where nothing was measured and every mask has failed, so the original definition puts
  p* at the end of the grid there. The numbers above for all cells use the definition restricted to the measured
  range (rates up to 3e-2), added after seeing this. With the original definition the usable-cell counts are
  identical (74 of 82, 56 of 56) and the all-cell count is 123 of 144, the misses being unusable pruned cells.
- The measured grid has four rates, so the measured switch is located only between 1e-3 and 1e-2.
- The prediction is in loss and the measurement in accuracy; only the sign and the crossover are compared.
- Below about 1e-3 the predicted difference is noisy, because the companion model rounds activations.
- ResNet-8 only.
