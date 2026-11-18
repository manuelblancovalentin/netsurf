# Run R, Stage 1 results (12 November 2026)

Plan: `RUN_R_RATE_AWARE_PLAN_2026-11-12.md`, Section 4, amendments R-A3, analysis note A1, fix F2.
Question: does the expected-loss formula predict the bit-error probability at which the positive-gradient mask stops
being better than the absolute-gradient mask?

## Verdict: confirmed (R1a, R1b and R1c all hold)

| Rule | Requirement | Result |
|---|---|---|
| R1a (sign) | no Holm-resolved usable cell outside $[p^*/1.25,\,1.25p^*]$ with the wrong sign | 0 violations; 404 resolved cells away from the switch, all with the predicted sign |
| R1b (coverage) | at least 80% of usable cells away from the switch with the predicted sign | 515 of 519 (99.2%) |
| R1c (location) | 95% interval of the mean $p^*_{\text{meas}}/p^*$ over nine checkpoints inside $[1/1.5,\,1.5]$ | 0.877, interval 0.804 to 0.957 |

**Bias.** The formula predicts the switch about 12% late on average (geometric mean of measured over predicted
0.876). Per checkpoint, mean ratio over budgets with a measured switch:

| Pinned | 7101 | 7102 | 7103 | 7104 | 7105 | Under-trained | Pruned 50% | Pruned 70% |
|---|---|---|---|---|---|---|---|---|
| 0.71 | 0.86 | 0.83 | 0.89 | 0.92 | 1.01 | 1.03 | 0.82 | 0.89 |

**Secondary (descriptive).** The under-trained checkpoint switches later than every well-trained one: geometric
mean $9.5\times10^{-3}$ against $4.8$ to $5.8\times10^{-3}$.

**Cells.** 864 cells (9 checkpoints, 8 budgets, 12 rates), 586 usable (best mask within 5 points of clean), 419
resolved after Holm correction. 24 checkpoint-budget pairs have no measured switch: all 9 at 1%, all 9 at 3.125%, 5
at 6.25% and 1 at 50%. In those, the network falls more than 5 points below clean before the masks swap order, so
the switch lies outside the usable cells. One prediction lies at the grid end (under-trained, 50%).

**Switch rule (descriptive, noted before the final data; not a decision rule).** Accuracy lost against the better of
the two masks in each usable cell, choosing positive below $p^*$ and absolute above:

| Rule | All 586 usable cells: mean / worst | 67 cells near the switch | 519 cells away from it |
|---|---|---|---|
| Switch rule | 0.01 / 1.91 points | 0.12 / 1.91 | 0.00 / 0.19 |
| Always absolute | 0.50 / 3.14 | 0.14 / 1.91 | 0.55 / 3.14 |
| Always positive | 1.43 / 56.6 | 0.44 / 2.28 | 1.56 / 56.6 |

## Conduct

- 576 jobs, no failures; every job's engine check reproduced the saved Run Q accuracy exactly.
- Two descriptive interim looks were taken at the researcher's request (304 and 475 jobs), recorded in the plan;
  nothing was changed after them. The first look found that the analysis script could not hold the pinned cells
  (72 paired differences per cell, 64 columns); fix F2 resized the sign-flip matrix (same seed and test) before the
  final analysis. The frozen analysis was run once, on all 576 jobs.
- Compute: 345 CPU hours for the evaluation at the time of the ledger (12.2 s per evaluation with 76 jobs on 40
  physical cores; `COMPUTE_ENVIRONMENT.md`), about 5 h wall.

## Outputs (read-only)

- Jobs: `workspace/runs/paper1_runR_stage1_crossover_v1/`.
- Analysis: `workspace/runs/paper1_runR_stage1_analysis_v1/`: `summary.json` sha256
  `1916fa046ecc6ef6fe61c9310e81c07c5bc4061e462f2581a0ddafbe0751b67d`, `cells.csv` sha256
  `ca299036707659709aa13de6689cfece1e6c05fda2331cb7f12f5f8b929be571`, and the two figures of the analysis script.
  Script `workflows/runR_stage1_analyze.py` at 407069b (Run R branch); predictions sha256 `bf38ffd6…` checked.
- Descriptive figures (predicted switch as lines with measured switches per replication; measured against predicted
  with the collapsed difference curves) and the switch-rule summary, with the scripts that made them:
  `workspace/runs/paper1_runR_stage1_figures_v1/` (`SHA256SUMS`).

## What it allows the paper to say

On nine INT8 ResNet-8 checkpoints (five retrained seeds, under-trained, two pruned), a prediction computed from
calibration gradients before any fault was injected located the bit-error probability at which the better static
ranking changes from positive to absolute gradient within a factor 1.5 on every checkpoint, about 12% late on
average, and the sign of the difference matched the prediction in every resolved cell away from the switch. Not
allowed: any statement about other models (Stage 4), or about budgets below 6.25%, where the switch falls where the
network has already lost more than 5 points.
