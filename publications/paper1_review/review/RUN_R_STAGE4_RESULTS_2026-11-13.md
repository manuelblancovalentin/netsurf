# Run R, Stage 4 results on the keyword-spotting DS-CNN (13 November 2026)

Plan: `RUN_R_STAGE4_DSCNN_PLAN_2026-11-12.md` (frozen) and amendment R-A5 of the Run R plan. 48 jobs, no failures;
every job reproduced the saved v14 engine-check trial exactly. Clean accuracy of the windows 91.6%.

## Verdict

| Rule | Result |
|---|---|
| R4a (sign) | **fails**: 13 resolved usable cells away from the predicted switch have the wrong sign, all of them the positive mask behind where it was predicted ahead (0.17 to 0.58 points) |
| R4b (coverage) | **fails**: 13 of 39 usable cells away from the switch (33%) have the predicted sign |
| R4c (location) | **fails**: measured over predicted switch 0.42 (95% interval 0.24 to 0.75); 9 of 24 replication-budget pairs without a measured switch |
| Switch prediction confirmed | **no** |
| R4d (analytic rate-aware mask, v1) | **fails**: 109 violations; the absolute mask is ahead by up to 10.2 points (median gap 0.2 points at $3\times10^{-4}$, 0.6 at $3\times10^{-3}$, 1.4 at $10^{-2}$, 6.4 at $3\times10^{-2}$) |
| R4d′ (selected rate-aware mask, v2, R-A5) | **fails narrowly**: 3 violations, all against the absolute mask, 0.50, 0.58 and 0.68 points (18.75% at $5\times10^{-3}$ and $7\times10^{-3}$, 25% at $10^{-2}$); against every other static mask v2 is at most 0.20 points behind |

## What was measured

- **The positive mask is never ahead on DS-CNN.** With all fields pooled, positive minus absolute is $+0.04$ points at
  $3\times10^{-4}$ and 12.5% (not resolved) and negative everywhere else, growing with the rate (12.5%: $-0.07$ at
  $10^{-3}$, $-0.58$ at $3\times10^{-3}$, $-3.8$ at $10^{-2}$; 25%: $-0.09$, $-0.29$, $-1.5$, $-10.9$ at $3\times10^{-2}$).
  The predicted switch ($3.8$ to $7.1\times10^{-3}$) is far too late: the measured switch is at or below the lowest
  tested rate. Geometric mean of measured over predicted where measured: 0.12.
- **v1 overfits**, as the training-side check predicted (`RUN_R_V2_DEVELOPMENT_2026-11-13.md`): it is behind every
  static mask except at the lowest rate, by up to 10 points at $3\times10^{-2}$.
- **v2 removes almost all of that loss** (v2 is 0.59 points better than v1 at the median and up to 10.2 points better),
  but its held-out choice sometimes keeps the positive mask where the absolute mask is better on test data, because
  the predicted objective itself favours the positive mask on DS-CNN, the same bias that misplaced the switch.
- Switch rule (positive below $p^*$, absolute above): mean loss 0.15 points, worst 0.88 against the better of the two.

## Reading (hypotheses, not tested here)

The same bias appears on ResNet-8, much smaller: the switch there came 12% earlier than predicted. On DS-CNN the
formula expects the sign-keeping positive mask to help below about $4\times10^{-3}$, and it never does. Two candidate
causes, both open:
1. **Loss versus accuracy.** The formula predicts loss. Leaving loss-lowering flips unprotected can lower the loss of
   a well-trained classifier (more confident correct outputs) without changing which class wins, so a loss gain need
   not become an accuracy gain. The prediction was computed out of sample (validation clips), so this is not the
   calibration overfitting of v1.
2. **The model is close to a minimum.** The theory says the sign is never worth exploiting at a minimum ($p^*\to0$);
   a small residual mean gradient, partly calibration noise, may be enough to predict a positive $p^*$.

## Outputs (read-only)

- Jobs: `workspace/runs/paper1_runR_stage4_kws_eval_v1/`.
- Frozen analysis: `paper1_runR_stage4_kws_analysis_v1/summary.json` sha256
  `1aea33849ba5df6bbb4a166ec5146862749e535a7f5cfa0b81d29c526698cac1` (script `workflows/runR_stage4_analyze.py`,
  Run R branch, 5719146).
- R-A5 analysis: `paper1_runR_v2_stage4_analysis_v1/summary.json` sha256
  `837de601179fc77e3f0fd58dfcbcb49fc000faec2768744818ed6188ba8a5b62` (script `workflows/runR_v2_stage4_analyze.py`,
  branch `paper1-runR-v2-validated`, 0b57e51; choices `paper1_runR_v2_picks_v1/picks.json` sha256 `8c1a864f…`).

## What it allows the paper to say

The switch prediction confirmed on nine ResNet-8 checkpoints does not hold on the DS-CNN: there the absolute mask is
never worse than the positive one, and the formula predicts a switch that does not exist. The analytic rate-aware
mask overfits its calibration data on the DS-CNN and falls far behind; choosing on held-out data removes most of that
loss, but not all of it. Not allowed: any claim that the switch prediction or the rate-aware mask generalizes beyond
ResNet-8.
