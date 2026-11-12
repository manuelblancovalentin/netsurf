# Run P, Phase 0 results (11 November 2026)

Plan: `RUN_P_MEAN_VARIANCE_DRAFT_PLAN_2026-11-11.md` with amendments A1 (gate cells, before any computation) and A2
(exact curvature check, after the Hutchinson output only). Code: branch `paper1-runP-mean-variance`, commits
`ba17257` (module, tests, driver), `c80be42` (analysis), A2 commit after it; tests `tests/test_mean_variance.py`
(10 passed). Outputs: `workspace/runs/paper1_runP_phase0_v1/` in the paper worktree, `phase0_analysis.json`.
No fault injection was run. All numbers are offline computations on saved masks and finalized measured tables.

## Data identity

The saved absolute-gradient masks of every replication of every checkpoint are reproduced from the recomputed
per-example gradients (overlap 1.0 at all four budgets, one replication of the pinned checkpoint 0.99997 at
6.25%). The engine's decoded bit jumps equal one scale times the two's-complement code jumps for every bit.

## Gate

- **(i) Under-trained reversal at the mean weights: passed.** Positive-gradient minus absolute-gradient accuracy,
  predicted from one companion forward pass at each mask's mean weights, against measured:
  12.5%, 1e-3: +1.56 vs +1.30 points; 12.5%, 1e-2: -20.5 vs -22.6; 25%, 1e-3: +1.16 vs +1.09; 25%, 1e-2: -10.8 vs
  -12.3. All four signs match.
- **(ii) Objective ranks the seven saved masks: failed.** Median within-cell Spearman correlation between the
  objective and the measured accuracy loss, decision-useful cells with BER >= 1e-3: pinned 0.44 (10 cells),
  under-trained 1.00 (13 cells); threshold 0.7 on both. Reported only: pruned 70% 0.96 (6 cells), pruned 50% 0.96
  (8 cells).

**By the rule fixed in the plan, Phases 1 and 2 are not run under this plan.**

## Why (ii) failed on the pinned checkpoint (diagnosis after the outcome; exploratory)

- The objective was evaluated on each replication's own 64 calibration images, the same images from which the
  saved gradient-based masks were built. For the positive-gradient mask the mean term on those images is
  -0.10 to -0.15 nats (the mean displacement lowers the calibration loss), while on the evaluation images the same
  mean weights cost 0.5 to 6.1 accuracy points. The objective therefore ranks the positive-gradient mask best in
  cells where it is measured among the worst. This is an in-sample bias of the check, of the same kind P0d found
  for the method itself.
- A second, smaller discrepancy: the variance term ranks MSB-first far below the absolute gradient (0.57 vs 0.07 at
  3e-3 and 12.5%), while the measured accuracy losses are close (1.97 vs 1.37 points).
- On the under-trained and pruned checkpoints the measured spread between masks is larger (10 to 65 points against
  2 to 9 on the pinned checkpoint at 1e-3 and 3e-3), and the objective ranks them almost perfectly.

## Other Phase 0 results

- **P0a, lopsidedness.** First-order loss effects of the bits left unprotected by the absolute-gradient mask cancel
  to within 2% on every checkpoint (50% of protected bits are loss-raising); the positive-gradient mask leaves
  -31% to -46%. In weight space the leftovers are lopsided on the pruned checkpoints: |net code shift| / total code
  shift is 0.19 and 0.24 for the absolute gradient (0.03 pinned, 0.04 under-trained) and 0.49 and 0.70 for
  MSB-first (0.01 pinned), because flips of the low bits of a zero code all move the weight up.
- **P0d, calibration size.** Held-out reduction of the range-W objective relative to the absolute-gradient mask
  (12.5%, replication 1): under-trained -0.028, -0.005, +0.013, +0.029 for n = 64, 128, 256, 512; pinned -0.065,
  +0.017, +0.044, +0.044. With 64 images the method overfits. Rule: n = 512 (under-trained), 256 (pinned);
  larger of the two: **512**.
- **P0e, curvature.** 64-probe Hutchinson: noise-dominated (half negative), not informative. Exact diagonal on
  512 sampled weights (A2): Spearman with the Fisher diagonal 0.91 (pinned) and 0.93 (under-trained), 0.60 and 0.62
  within the top Fisher decile, no negative entries, median Hessian/Fisher ratio 1.29 and 0.98.

## Options for the researcher (not decided)

1. Stop here; use Phase 0 as mechanism evidence (the mean displacement explains the under-trained reversal; the
   Fisher curvature is adequate).
2. A version-2 plan, explicitly designed after seeing this failure: the objective evaluated on held-out images
   (never those used to build the masks), 512 calibration images, and a new gate tested on data that played no
   part in the diagnosis (the saved masks of the five retrained seeds).
