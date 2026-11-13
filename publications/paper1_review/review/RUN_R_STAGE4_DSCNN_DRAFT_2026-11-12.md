# Run R, Stage 4 on the keyword-spotting DS-CNN: DRAFT plan (12 November 2026)

Status: draft for the researcher's review. Nothing in it is frozen and no DS-CNN fault outcome has been produced.
Main plan: `RUN_R_RATE_AWARE_PLAN_2026-11-12.md` (Section 7 outlines Stage 4).

## 1. Purpose

A lighter check of generality on a second model, not a second confirmatory study:
1. Does the formula predict the switch between positive and absolute masks on DS-CNN? (as Stage 1)
2. Is the analytic rate-aware mask never far behind the better static mask? (as Stage 2, fewer arms)
3. The MC-corrected mask, only if rule R2c holds on ResNet-8.

## 2. Preparation done (no fault outcome)

Outputs: `workspace/runs/paper1_runR_kws_prep_v1/`; code `workflows/runR_kws.py` (Run R branch, f778236).

- **Model.** Pinned MLPerf Tiny DS-CNN, INT8 TFLite (sha256 `aeea4368…`), 10 kernel tensors, 22,016 kernel
  weights (176,128 stored bits).
- **Gradient companion** (TFLite constants, requantization with a straight-through gradient), against the TFLite
  reference on 1,000 test clips of the clean model: same predicted class on 99.3%, identical INT8 outputs on 63.6%,
  largest output-code difference 22; accuracy 90.2% against 90.1%. ResNet-8's companion, used in Stage 1, gives 97%,
  45% and 58 on 200 images, so DS-CNN's is at least as faithful. Faulty models are always evaluated on the TFLite
  engine.
- **Masks.** `abs1024` and `pos1024` at the eight Stage 1 budgets from three draws of 1,024 training clips (offsets
  0, 1024, 2048), Run Q's scoring code. About 1.5 min per draw.
- **Predicted switch**, on 512 validation clips (0-511), mean over the three draws, same definition as Stage 1
  (last sign change of the predicted difference up to $3\times10^{-2}$):

| Budget | 1% | 3.125% | 6.25% | 12.5% | 18.75% | 25% | 37.5% | 50% |
|---|---|---|---|---|---|---|---|---|
| $p^*$ (mean of draws) | 1.1e-3 | 5.3e-4 | 2.4e-3 | 3.3e-3 | 4.3e-3 | 7.0e-3 | 7.7e-3 | 1.9e-2 |
| $p^*$ per draw | 2.2e-3, 1.2e-3, 7.4e-4 | 3.2e-3, 8.6e-4, 3.4e-3 | 3.2e-3, 4.7e-4, 4.0e-3 | 3.5e-3, 2.4e-3, 4.4e-3 | 3.0e-3, 4.0e-3, 7.3e-3 | 5.3e-3, 7.3e-3, 9.6e-3 | 7.5e-3, 6.4e-3, 9.1e-3 | 1.7e-2, 1.0e-2, 2.5e-2 |

**Finding: the DS-CNN predictions are noisier than ResNet-8's.** The predicted difference crosses zero 2 to 5 times
below $3\times10^{-2}$, and the draws disagree by up to a factor 7 at 3.125% and 6.25%, and by about a factor 2 from
12.5% up. With 22,016 weights the two masks share most bits, so the predicted difference is close to zero at low
rates and its sign there is noise.

**Usable range** (from the v14 DS-CNN data, absolute-gradient accuracy only; no positive-versus-absolute
difference was read): the absolute mask stays within 5 points of clean up to about $3\times10^{-3}$ at 6.25% and
12.5%, and up to about $10^{-2}$ at 18.75% and 25%. The predicted switch therefore lies inside the usable range from
12.5% up, and at or beyond its edge below.

## 3. Proposed design (to be decided)

- **Budgets: 12.5%, 18.75%, 25%, 37.5%**: the predicted switch is inside the usable range and the draws agree within
  about a factor 2. Smaller budgets are left out on the grounds above, before any outcome.
- **Replications: six**, each with its own calibration draw (training clips from $1024 \cdot r$) and its own window
  of 800 test clips (offset $800 \cdot r$; the test clips were used before in v14). The prediction becomes the mean
  over the six draws (three more draws, about 12 min), recomputed and frozen by hash before any evaluation.
- **Rates:** the twelve Stage 1 rates. **Fields:** eight per replication, fresh seeds.
- **Arms, part 1 (switch):** `abs1024`, `pos1024`. **Part 2 (method):** `abs1024`, `pos1024`, the paper's
  `abs64` from the v14 bundle, and the analytic rate-aware mask built for each rate (same search and starts as
  Stage 2). **Part 3:** the MC-corrected mask, only if R2c holds on ResNet-8.
- **Rules:** R1a and R1b as in Stage 1. With one checkpoint the location rule R1c is replaced by: the 95% Student-t
  interval of the mean of $\log(p^*_{\text{meas}}/p^*)$ over the four budgets lies inside $[1/1.5,\,1.5]$, reported
  with the caveat that budgets are not independent. R2a as in Stage 2 for the analytic mask against the three static
  arms.

## 4. Still open

- A timing smoke of DS-CNN evaluation and of the analytic search, for the cost and wall time.
- The researcher's choice of budgets, number of replications, and whether FKeras and significance-first are added
  as static arms (they lose by large margins on ResNet-8 and in v14 on DS-CNN).
- Whether the noise in the prediction calls for more calibration clips (2,048) rather than more draws.
