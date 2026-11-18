# Run P version 2: gate result (11 November 2026)

Plan: `RUN_P_MEAN_VARIANCE_PLAN_V2_2026-11-11.md` (commit 23ba6d0, frozen before computation). Code: branch
`paper1-runP-mean-variance`, commits 8977430 (gate stage) and bf40881 (analysis). Outputs:
`workspace/runs/paper1_runP_v2_gate/` with `SHA256SUMS`. Offline; no fault injection.

**Gate passed.** Median within-cell Spearman correlation between the held-out objective (512 images, 28000-28511)
and the measured accuracy loss across the seven saved rankings, decision-useful cells with BER >= 1e-3:

| Checkpoint | Cells | Objective | Mean term only | Variance term only |
|---|---|---|---|---|
| Retrained seed 7101 | 9 | 0.96 | 0.31 | 0.86 |
| Retrained seed 7102 | 10 | 1.00 | 0.37 | 0.96 |
| Retrained seed 7103 | 9 | 1.00 | 0.09 | 0.96 |
| Retrained seed 7104 | 10 | 1.00 | -0.18 | 0.96 |
| Retrained seed 7105 | 9 | 1.00 | 0.27 | 0.96 |
| Pinned (used in the diagnosis; reported only) | 10 | 0.69 | -0.03 | 0.51 |
| Under-trained (used in the diagnosis; reported only) | 13 | 1.00 | -0.04 | 0.96 |
| Pruned 70% (reported only) | 6 | 0.96 | 0.10 | 0.96 |
| Pruned 50% (reported only) | 8 | 0.98 | -0.04 | 0.96 |

Five of five seeds pass (rule: at least four of five at 0.7 or more). On the pinned checkpoint the held-out
objective reaches 0.69, against 0.44 when evaluated on the masks' own calibration images, so the in-sample bias
explained most of the version-1 failure; the pinned checkpoint remains the hardest case.

Reading: on held-out images the objective ranks existing masks almost perfectly, and most of that ranking comes
from the variance term; the mean term matters for the few masks whose leftovers are lopsided. This says nothing
yet about whether masks *built* from the objective are better. Phases 1 and 2 need the researcher's go-ahead.
