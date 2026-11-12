# Run R, Stage 0 record (12 November 2026)

Plan: `RUN_R_RATE_AWARE_PLAN_2026-11-12.md` (amendment R-A2). Stage 0 was started on the researcher's go-ahead.
Code: branch and worktree `paper1-runR-rate-aware`, created from `paper1-runP-mean-variance` at a3ce54d.

## 1. Implementation decisions (made before any Stage 0 output)

- **New module.** The method lives in a new module `netsurf/rate_aware.py` instead of an edit of
  `netsurf/mean_variance.py`, so Run P's code and hashes stay unchanged. The plan said "in `mean_variance.py`"; only
  the file differs.
- **Ladders.** Protection is a ladder of levels with equal storage steps (`none -> ideal`, step 1 bit;
  `none -> 3 copies -> 5 copies`, step 2 bits). Every move changes one bit by one level, so a swap of $m$ step-ups
  against $m$ step-downs keeps the storage budget exact, as the plan requires. Five copies are reached in two steps.
- **Shrinkage.** $\mathrm{se}_w$ is computed once, from the per-input gradients at the clean weights, and applied
  to $\hat{g}_w$ at every round. The per-input gradients at $\theta+\mu$ are not recomputed each round.
- **Wrapped repairs.** An ABFT "repair" by a multiple of 256 wraps back to the same INT8 code and changes nothing;
  it is not counted as a wrong repair.
- **Checksum flips in the ABFT residual curve.** With flipping checksum words, a checksum flip together with one
  faulty weight also blocks the correct repair, so the residual curve becomes
  $p\,[1-(1-p)^{8(G-1)+64}]$ for two 32-bit checksum words. The paper's model is the case with no checksum flips.
- **Search trace (researcher's request).** Every search saves, per round: the objective and its parts (drift,
  variance, MC correction and its standard error, norm of $\mu$), the predicted change, the swap size $m$, the
  temperature, the storage used, the number of bits at each level, the rates $\rho_s$, and the sum, mean and
  standard deviation of $v_b$ over all possible step-ups, over all possible step-downs, and over the chosen ones.
- **MC fields for Stage 2.** Seeds `[20261118, ckpt, rep]`, one draw of $K$ fields per replication.

## 2. Rule for the Stage 2 constants (fixed before running it)

On pinned replication 1, budget 12.5%, held-out training images 28000-28511 only, ideal suppression:
1. Build the analytical mask from the absolute-gradient start at $p = 3\times10^{-3}$ and $p = 10^{-2}$, and record
   the improvement of the objective, $\hat{J}(\text{start}) - \hat{J}(\text{final})$.
2. At the absolute-gradient start, evaluate the loss on 32 MC fields (seed `[20261118, 0]`) and compute the
   standard error of the MC objective with the first $K$ of them, for $K = 8, 16, 32$.
3. $K$ is the smallest of 8, 16 and 32 whose standard error is below one tenth of the analytical improvement at both
   rates; 32 if none is.
4. Build the MC-corrected mask at $p = 3\times10^{-3}$ with that $K$ and refresh every evaluation. The refresh
   interval is 1 if that build takes under 10 minutes, and 4 otherwise.
5. Round limits stay at Run P's values (at most 40 rounds, stop when $m < 16$); the annealing factor stays 0.9.

## 3. Results

(Filled in after the checks run.)
