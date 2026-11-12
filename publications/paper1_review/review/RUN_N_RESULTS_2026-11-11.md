# Run N results: search-based, learned and literature-derived masks on ResNet-8 (11 November 2026)

Plan: `RUN_N_METHOD_CAMPAIGNS_FROZEN_PLAN_2026-11-11.md` (with the c2r amendment). Code:
`workflows/method_campaigns_resnet8.py`. Outputs: `workspace/runs/paper1_runN_method_campaigns_resnet8_v1/`,
analysis `workspace/runs/paper1_runN_method_campaigns_resnet8_v1_analysis.json`. Not in the manuscript.

Setting: pinned ResNet-8, test images 2000-2999 (clean accuracy 85.6%), the 16 Run M fault fields per rate, rates
3e-3, 1e-2, 3e-2, budgets 12.5% and 25%. Entries: mean accuracy and paired difference against the engine's
absolute-gradient mask, in points, with the 95% field-level interval (16 fields; no multiplicity correction).
Single-rate masks are shown at the rate they were built for. Usefulness margin: within 5 points of clean
(80.6%); usable cells are 3e-3 at both budgets and 1e-2 at 25%.

| Mask | 3e-3, 12.5% | 3e-3, 25% | 1e-2, 25% | 1e-2, 12.5% | 3e-2, 12.5% | 3e-2, 25% |
|---|---|---|---|---|---|---|
| Absolute gradient (reference) | 83.0 | 85.0 | 83.7 | 76.8 | 54.1 | 79.7 |
| MC-guided search (c1) | +0.05 [-0.34, 0.44] | -0.08 | +0.27 [-0.01, 0.54] | +0.69 | **+4.00 [1.15, 6.85]** | +0.49 |
| Rebuild of Run M | +0.18 | +0.01 | -0.28 | +0.51 | +3.66 [-0.90, 8.21] | +0.41 |
| Group CEM, layer x bit position (c3) | **-0.66** | -0.38 | **-0.99** | -1.34 | -2.99 | **-1.89** |
| Relaxed per-bit optimization (c2) | **-5.76** | **-4.56** | **-8.17** | **-12.68** | **-12.43** | **-7.26** |
| Relaxed, rate range [3e-3, 3e-2] (c2r) | **-5.22** | **-3.72** | **-12.32** | **-22.58** | **-33.22** | **-35.17** |
| Thales single-flip ranking (c4) | **-21.72** | **-24.04** | **-66.44** | **-58.44** | **-43.51** | **-68.84** |
| HarDNN, loss (whole filters) | **-5.27** | **-3.78** | **-15.09** | **-18.32** | **-29.76** | **-40.99** |
| HarDNN, gradient (whole filters) | **-14.76** | **-11.76** | **-40.96** | **-47.94** | **-41.25** | **-64.16** |
| MSB-first | **-12.88** | **-2.46** | **-12.74** | **-38.57** | **-37.76** | **-33.04** |

Bold: field-level interval excludes zero.

Off-target (mask built for one rate, tested at another): MC-search masks built for 1e-2 and 3e-2 gain +0.69
[0.39, 1.00] and +0.73 [0.36, 1.10] at 3e-3 and 12.5%; rebuilt masks +0.45 and +0.94. All other off-target
differences are within about one point and unresolved.

Reading:
- No mask beats the absolute-gradient mask by a point or more in a usable cell. The largest usable-cell gain is
  below one point (MC-search masks built for higher rates, at 3e-3 and 12.5%). The only resolved gain above a
  point, +4.0 for MC-guided search, is at 3e-2 and 12.5%, where every mask has already failed (54-58% against
  85.6% clean).
- The relaxed per-bit optimizers, which optimize a protection probability per bit through the differentiable
  surrogate and are the closest of the five to policy-gradient reinforcement learning, are 4-35 points worse,
  and the range-optimized one is worst at the high rates it was built for. This is one implementation with one
  set of hyperparameters; it does not show that learned allocators in general are worse.
- Rankings ported from other fault models fail at the stored-bit level. Thales ranks bits by single-flip accuracy
  loss, which is zero for most bits on 64 calibration images, and HarDNN protects whole filters; both lose 4-69
  points.
- A group-structured mask (whole layer x bit-position groups, last group filled by absolute gradient) costs 0.4 to
  1.9 points against the irregular absolute-gradient mask. That bears on the bit-count-cost limit: regular masks
  that hardware can implement cheaply lose little here.

Scope: one checkpoint, one test window, 16 fields, search budgets as frozen (8 rounds, 4000 steps, 8 CEM
iterations). Exploratory.
