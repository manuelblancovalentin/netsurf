# Top-bit analysis, Phase 2 results (14 November 2026, EXPLORATORY)

Plan: `TOPBIT_ANALYSIS_PLAN_2026-11-14.md`.

The targets are the Stage 4 TFLite accuracies of replication 0, which had already been read:
- 288 cells: 6 masks × 4 budgets × 12 rates;
- 8 fault fields per cell;
- 800 test clips.

Inputs, all locked:
- `paper1_topbit_logits_kws_v1`: every bit-6 and bit-7 single flip, with all 12 class codes per clip;
- `paper1_topbit_background_kws_v1`;
- `paper1_topbit_cells_kws_v1`.

Run: Quest job 8671897, 79 s, code b68bdee. Output: `workspace/runs/paper1_topbit_analysis_kws_v1/`, locked,
hashes in `_logs/`.

## Checks that passed

- **The companion replaying each field's actual flips reproduces TFLite.** Mean absolute error 0.22 points
  per field and 0.08 points per cell, with a mean error of −0.01. The measurement path is right.
- **With no top-bit flips, the additive model with the real low-bit background matches TFLite.** Error
  +0.01 points over 400 fields. The implementation is right where additivity is not tested.
- **The exact lattice readout dropped at most 5e-12 of probability.**

## Q1: single-flip effects do not add up beyond about ten simultaneous top-bit flips

Field-level error of the additive prediction (E: top-bit events only; E+B: with the field's real low-bit
background), against TFLite, in points:

| Top-bit flips in the field | Fields | E | E+B real | Companion |
|---|---|---|---|---|
| 0 | 400 | +0.60 | +0.01 | +0.01 |
| 1–2 | 39 | +0.15 | +0.03 | +0.13 |
| 3–5 | 117 | −0.14 | −0.14 | +0.07 |
| 6–10 | 138 | −0.27 | −0.21 | +0.03 |
| 11–20 | 186 | −0.63 | −0.62 | −0.01 |
| 21–50 | 491 | −2.38 | −2.82 | −0.03 |
| 51–100 | 396 | −9.74 | −10.90 | −0.04 |
| 101–300 | 374 | −33.7 | −35.0 | −0.02 |
| over 300 | 163 | −48.5 | −49.4 | +0.03 |

By rate, the additive model loses:
- 0.6 points at 1e-3, with about 12 flips per field;
- 4.7 points at 3e-3, with about 39 flips;
- 23 points at 1e-2, with about 123 flips.

The sum of measured single-flip effects **overstates** the combined damage, increasingly with the number of
flips. The network absorbs many simultaneous top-bit faults far better than their individual effects add
up to: the interaction is strongly sub-additive.

The likely cause is saturation and switching inside the network. Activations clip at the INT8 limits, and
ReLUs stop negative changes, so later faults move units that are already saturated or switched off. This is
not tested here.

The background makes it slightly worse rather than explaining it.

The rate-level readouts, Fourier (exact lattice) and Gaussian (bell curve), use the margin against the clean
runner-up only. They are 4.9 points too low on average.
- **The two readouts agree to within 0.01 points.** On this model, the bell-curve readout loses nothing
  against the exact distribution.
- **The additive premise both rest on is what fails.**

## Q2: the predictors do not choose masks well

Choosing, in each budget and rate, the mask with the highest predicted accuracy:

| Chooser | Picks the measured best | Mean regret | Worst regret |
|---|---|---|---|
| Fourier | 8% of the 48 budget-rate pairs | 0.96 points | 12.1 points |
| Gaussian | (same as Fourier) | (same) | (same) |
| Always abs1024 | | 0.05 points | |

## What this changes

- **Combining measured single-flip effects at the output does not work on the DS-CNN at the rates of
  interest.** This applies to any formula that adds per-bit effects on the output or the loss: the Fourier
  readout, the bell curve, and gradient-based linear sums. The interactions dominate once more than about
  10–20 top bits are faulty.
- **What has worked is combining in weight space and passing through the network.**
  - Weight displacements are combined first, then passed through the network: the blur pilot's Gaussian and
    hybrid variants, with mean absolute error 0.5–0.9 points.
  - The companion replay of real fields is also computed this way, with 0.08 points.

  The network's own nonlinearity does the combining. A mask-choosing method should therefore evaluate
  candidates with network passes (the group-search plan's design), not with sums of per-bit scores.
- **The plan's Q3 and Q4 are unaffected and still open.**
  - Q3: the measured-damage masks, evaluated by TFLite.
  - Q4: the gradient scores against measured damage. A first look in the smoke run gave rank agreement
    0.25 (abs) and 0.36 (pos), network-wide, on loss.
- **Limits.**
  - One model and one replication, post hoc.
  - Bits 0–5 enter only as background.
  - The flips were measured on the GPU path, which differs from the CPU path by rounding noise. The replay
    check suggests this does not matter at the accuracy level.
