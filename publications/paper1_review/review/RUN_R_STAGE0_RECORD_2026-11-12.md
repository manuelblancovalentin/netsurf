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

Outputs, read-only: `workspace/runs/paper1_runR_stage0_v1/` (`checks/stage0_checks.json`, sha256
`0efc4a0332676451eb2be43a66294c551dbd8bae3ab032089fdd736bbaf97c36`; search traces; Stage 1 timing smoke). Code:
branch `paper1-runR-rate-aware` at 6e7a0e7 (module and checks), 4baea4f to 0deffd9 (amendment R-A3 scripts).

**Unit tests.** `tests/test_rate_aware.py`, 25 tests, all pass, with the 28 tests of `test_mean_variance.py`. They
cover every residual curve against exhaustive enumeration, the ABFT pair counts against a brute-force decoder (this
caught one error: a repair by a multiple of 256 wraps to the same code and is not a wrong repair), the MC
estimate against the exact expectation, exact storage, determinism, and parity with Run P's greedy search on a toy
model. The full repository suite has 4 failures, all from a missing MNIST benchmark checkpoint in this worktree;
they fail the same way on `paper1-runP-mean-variance`.

**Parity on the real model: passed.** With ideal suppression and no correction, the new search rebuilt Run P's
`mv_single@0.125000@0.001` mask of pinned replication 1 bit for bit (sha256 `3557e740…`, 16 rounds, 34 s).

**ABFT pair counts (pinned ResNet-8, 32-bit checksum words, all channels).** Pairs of data-bit flips in different
weights of one channel: 18.9% end in a wrong repair, 6.2% cancel and go undetected. Pairs of one data-bit flip and
one checksum-bit flip: 13.6% end in a wrong repair. Per layer the shares grow with the channel size, from 15.1%
($G=16$) to 19.0% ($G=576$), and from 6.5% to 14.8%.

**Stage 1 timing smoke: passed.** The engine check reproduced Run Q's saved accuracy (86.4%) exactly, the clean
accuracy of the window is 87.4%, and one evaluation takes about 6.2 s (32 evaluations in 222 s including loading).

**Stage 2 constants, by the rule of Section 2.**

| Rate | Analytic improvement | MC standard error, $K=8$ / 16 / 32 | Rule met at |
|---|---|---|---|
| $3\times10^{-3}$ | 0.0877 | 0.0074 / 0.0070 / 0.0077 | $K=8$ |
| $10^{-2}$ | 0.0792 | 0.0389 / 0.0334 / 0.0217 | none |

- $K = 32$ (no $K$ met the rule at $10^{-2}$, so the rule gives 32).
- The MC-corrected build with $K=32$ took 439 s, under 10 minutes, so the refresh interval is 1.
- Round limits 40 and $m \ge 16$; annealing factor 0.9 (unchanged).
- Analytic builds took 13 to 14 s (19 to 20 rounds). The start from the values at $\mu=0$ was better than the
  absolute-gradient start at $3\times10^{-3}$ ($-0.0092$ against $-0.0068$) and far worse at $10^{-2}$ (0.627
  against 0.153), as the theory predicts; the search keeps the better of the two.
- At $3\times10^{-3}$ the MC-corrected search lowered the MC objective from 0.073 to 0.003 in 21 rounds.

**Annealing does not work as specified.** The annealing search ended at its starting objective (0.073) after 24
rounds and 1,393 s. The search trace shows why: once $T$ is set by the first rejected block ($T \approx 0.024$), the
sampling weight $\exp(v_b/T)$ is nearly uniform, because single-bit values are about $10^{-7}$ (mean of $v_b$ over all
541,520 possible step-ups $1.2\times10^{-7}$, over the 1,208 chosen ones $1.1\times10^{-7}$). The moves are then close
to random, and a block of random swaps raises the objective. One temperature cannot serve both the acceptance of a
whole block and the sampling of single bits. A corrected schedule (for example a per-move temperature $T/m$) would
be a change to the frozen method and needs an amendment before Stage 2; as specified, the arm would be dropped by
rule R2c.

**Cost estimate for Stage 2** (for its addendum): per mask about 14 s analytic, 440 s with the MC correction;
$9 \times 8$ replications $\times$ 8 rates $\times$ 4 budgets = 2,304 masks per variant, about 4 hours for the MC
variant at 76 processes, plus about 4 hours of evaluation.

**Verdict.** Every Stage 0 test required by the plan passed. Stage 1 can be launched on the researcher's go-ahead.
