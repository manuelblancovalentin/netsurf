# Run R, Stage 4 on the keyword-spotting DS-CNN: frozen plan (12 November 2026)

Supersedes the draft `RUN_R_STAGE4_DSCNN_DRAFT_2026-11-12.md`. Design choices by the researcher: budgets 12.5%,
18.75%, 25% and 37.5%; six replications; FKeras and significance-first included. Frozen before any DS-CNN fault
field of this stage is drawn. Launch needs the researcher's go-ahead.

## 1. Questions

1. Does the expected-loss formula predict the switch between the positive- and absolute-gradient masks on a second
   model? (Stage 1 on DS-CNN)
2. Is the analytic rate-aware mask never far behind the best of five static masks? (Stage 2 on DS-CNN, analytic only)
3. The MC-corrected mask is added later only if rule R2c holds on ResNet-8, by its own amendment.

## 2. Model, data and replications

- Pinned MLPerf Tiny DS-CNN, INT8 TFLite (sha256 `aeea4368…`), 22,016 kernel weights, 176,128 stored bits, TFLite
  2.14.1 reference kernels. Gradient companion checked against TFLite (99.3% same class on 1,000 clips).
- Replication $r = 0,\dots,5$: calibration draw of 1,024 training clips from offset $1024\,r$; test window of 800 clips
  from offset $800\,r$ (`full_test_4890`; these clips were used before in v14). Prediction images: 512 validation
  clips (0-511).

## 3. Masks (all built, hash-recorded, no fault outcome read)

Output `workspace/runs/paper1_runR_kws_prep_v1/` (read-only):
- `abs1024`, `pos1024` at eight budgets per draw, Run Q's scoring code (`masks/rep<r>/`).
- `abs64` (first 64 clips of the draw), significance-first and FKeras (two Hessian eigenpairs, seed 970201, the
  engine's ordering: bit significance, then score, then stable identity), at the four Stage 4 budgets
  (`static/rep<r>/`). The builder reproduces the v14 DS-CNN masks of Run A2 replication 1 bit for bit for
  significance-first and FKeras at all four v14 budgets; the 64-clip absolute mask differs by one or two swapped bits
  per budget (2 to 4 differing bits out of 11,008 to 44,032), a floating-point near-tie at the cut-off between two
  score computations (`parity_v14.json` in the scratch check; recorded here).
- Analytic rate-aware masks are built in phase `build` (same search, starts and constants as Stage 2).

## 4. Predictions, frozen

`predictions_6draws.json`, sha256 `c59fcfcd234c120db79307599eec31263b11fe1ed5447c68e8a870bf42df0093`: $p^*$ from the
mean over six draws of the predicted objective of both masks, same definition as Stage 1 (last sign change up to
$3\times10^{-2}$).

| Budget | 12.5% | 18.75% | 25% | 37.5% |
|---|---|---|---|---|
| $p^*$ | 3.8e-3 | 4.5e-3 | 7.1e-3 | 6.0e-3 |
| per draw | 3.5, 2.4, 4.4, 5.5, 1.3, 2.2 e-3 | 3.0, 4.0, 7.3, 7.8, 3.4, 1.7 e-3 | 5.3, 7.3, 9.6, 10, 6.8, 5.3 e-3 | 7.5, 6.4, 9.1, 15, 2.5, 11 e-3 |

The predictions are noisier than on ResNet-8 (draws differ by up to a factor 6), and $p^*$ is not monotone in the
budget (37.5% below 25%). Both are reported as found.

## 5. Fault regime and evaluation

- Exact INT8 stored-bit XOR faults on kernels, independent with probability $p$, ideal suppression.
- Rates: the twelve Stage 1 rates, $3\times10^{-4}$ to $3\times10^{-2}$.
- Fields: eight per replication, `default_rng([20261119, r, field])`, one uniform number per bit, nested across rates
  and shared by every mask and budget.
- Arms per cell: `abs1024`, `pos1024`, `abs64`, significance-first, FKeras, and the analytic rate-aware mask built for
  the tested rate.
- Engine check: every evaluation job first replays one saved v14 trial (Run A2 replication 1, significance-first at
  12.5%, BER $3\times10^{-3}$, 509 events, saved accuracy 0.85) and must reproduce it exactly, or it stops.

## 6. Analysis and decision rules

Usable cell: the better of `abs1024` and `pos1024` (switch rules) or the best arm (method rule) within 0.05 of the
mean clean accuracy of the windows. Paired differences on the field; pooled-field sign-flip test, 100,000
assignments, seed 20261121; Holm correction; Student-t intervals over replications reported alongside. Zero
differences count against the prediction; measured switch as in Stage 1 (last sign change on the usable rates,
log-linear interpolation); a budget without a sign change has no measured switch.

- **R4a (sign):** no Holm-resolved usable cell outside $[p^*/1.25,\,1.25p^*]$ has the sign opposite to the prediction.
- **R4b (coverage):** at least 80% of usable cells outside that zone have the predicted sign.
- **R4c (location):** the 95% Student-t interval over replications of the mean, over budgets, of
  $\log(p^*_{\text{meas},r}/p^*)$ lies inside $[1/1.5,\,1.5]$, where $p^*_{\text{meas},r}$ is replication $r$'s
  measured switch; replications without a measured switch at a budget leave it out, and the count is reported.
- **R4d (method):** no static arm is ahead of the analytic rate-aware mask by 0.5 point or more in a usable cell after
  Holm correction over the usable cells and arms.
- The prediction is confirmed on DS-CNN only if R4a, R4b and R4c hold. Bias (geometric mean measured over predicted)
  and the switch-rule loss are reported whatever the verdict.

Allowed claims: the switch prediction and the rate-aware mask on a second model under this design. Not allowed:
anything about budgets below 12.5%, other models, or the MC correction.

## 7. Cost (measured in the timing smoke, under full machine load)

| Phase | Jobs | Per job | Total CPU |
|---|---|---|---|
| Build analytic masks (replication x budget, 12 rates) | 24 | ~40 min, 2 GB | ~16 h |
| Evaluate (replication x field, 288 evaluations) | 48 | ~15 min, 0.5 GB | ~12 h |

About 28 CPU hours. Wall time: about 4 h with 8 processes alongside Stage 2, or about 1.2 h with 24 after it.
Storage under 200 MB in `workspace/runs/paper1_runR_stage4_kws_masks_v1/` and `..._eval_v1/`, write-once.
Interpreter: the TensorFlow 2.14.1 reference environment. Code: `workflows/runR_stage4_kws.py` (Run R branch).
