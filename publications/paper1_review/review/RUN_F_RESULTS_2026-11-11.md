# Runs F and F2: fault-rate-aware masks on DistilBERT (exploratory, 11 November 2026)

Not used in the manuscript. Designs: `RUN_A2_D_F_FROZEN_PLANS_2026-11-11.md` (Run F) and the header of `workflows/fault_aware_masks_f2.py` (Run F2, frozen after Run F was read). Analyses: `data/exploratory_fault_aware/`.

## What was built

"Rebuilt mask": start from the sealed absolute-gradient mask; twice, draw four new fault fields at the target fault rate, apply the faults the current mask leaves unprotected, recompute the study's gradients on the same 64 calibration sentences on each faulted model, and re-rank all bits by mean absolute gradient times the bit's jump. Both build gates passed with zero difference in every cell (fault-free gradients equal the saved moments; re-ranking reproduces the sealed mask).

## Run F: saved fault fields, frozen 512-sentence cohort (24 fields per fault rate)

Primary contrasts, rebuilt mask minus fixed absolute-gradient mask at the target cell:

| Target fault rate | Budget | Fixed | Rebuilt | Gain, points (95% interval) |
|---|---|---|---|---|
| 0.01 | 12.5% | 90.7 | 90.4 | -0.24 (-0.44, -0.03) |
| 0.03 | 12.5% | 89.6 | 89.5 | -0.14 (-0.47, +0.19) |
| 0.1 | 12.5% | 72.9 | 82.1 | +9.26 (+7.52, +11.00) |
| 0.01 | 25% | 90.8 | 90.9 | +0.11 (-0.09, +0.30) |
| 0.03 | 25% | 90.7 | 90.8 | +0.14 (-0.03, +0.31) |
| 0.1 | 25% | 89.5 | 89.8 | +0.24 (-0.20, +0.69) |

- One of six gains matters by the pre-set rule (at least one point, interval excluding zero). Clean accuracy is 91.4%.
- At fault rate 0.1 and 12.5%, masks built for 0.01 and 0.03 gain 2.7 and 1.6 points; the mask built for 0.1 gains 9.3. Tested at lower fault rates, the mask built for 0.1 is within 0.3 points of the fixed mask.
- The second candidate (two-moment benefit, signed) lost 27 to 43 points in all six cells and is dropped.

## Run F2, part A: confirmation on unused sentences and new fault fields

Same two masks, unchanged; 328 validation sentences outside the frozen cohort (source indices 544 to 871; earlier pilots not audited); 24 new fields per fault rate. Clean accuracy on these sentences: 89.3%.

| Fault rate | No protection | Fixed | Rebuilt for 0.1 | Gain, points (95% interval) |
|---|---|---|---|---|
| 0.01 | 57.0 | 89.7 | 89.5 | -0.17 (-0.54, +0.21) |
| 0.03 | 48.9 | 89.5 | 89.6 | +0.03 (-0.32, +0.37) |
| 0.1 | 49.7 | 69.0 | 80.5 | +11.46 (+8.52, +14.40) |

The primary contrast (fault rate 0.1) meets the pre-set confirmation rule.

## Run F2, part B: the remaining cells where the fixed mask loses more than two points

| Built for | Budget | Fixed | Rebuilt | Gain at target, points (95% interval) |
|---|---|---|---|---|
| 0.03 | 6.25% | 87.3 | 87.5 | +0.15 (-0.29, +0.58) |
| 0.1 | 6.25% | 54.9 | 55.2 | +0.35 (-0.74, +1.44) |
| 0.1 | 18.75% | 86.9 | 88.1 | +1.26 (+0.73, +1.79) |

Off target, the 6.25% mask built for 0.1 is worse than the fixed mask by 1.24 points at 0.03 and 0.48 at 0.01.

## Scope

One model, one trained checkpoint. Run F and part B use a cohort that earlier runs used. The gains at 12.5% are outside the manuscript's five-point usefulness band (82.1% against 91.4% clean); the 18.75% gain is inside it. No claim enters the manuscript without a second model and the researcher's decision.
