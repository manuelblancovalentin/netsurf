# DS-CNN calibration-size test: results (15 November 2026)

**Pre-registration:** `KWS_CALIBRATION_SIZE_PREREG_2026-11-15.md` (1be5878).

**Code:** `workflows/runG_kws_calsize.py` (5366e6b), run once on kona's CPUs with `.venv-mlperf`.

**Runs:**

| Stage | Detail |
|---|---|
| Mask builds | 6 replications, 21 minutes |
| Fault runs | 48 evaluation tasks, 12 minutes; every engine check matched (48 of 48) |
| Analysis | run once |
| Scoring check | the streamed scoring formula equals the Stage 4 formula exactly (relative difference 0) |

**Outputs**, locked, with SHA-256 sums in `paper1_kws_calsize_logs/`:
- `paper1_kws_calsize_masks_v1`
- `paper1_kws_calsize_eval_v1`
- `paper1_kws_calsize_analysis_v1`

## Result

**Agreement rose as the noise explanation predicts:**

| Calibration clips | D1 |
|---|---|
| 1,024 | 0.21 |
| 4,096 | 0.57 |
| 8,192 | 0.73 |

**The signed ranking still does not help (primary test: no).** pos-8192 minus abs-8192, in points; usable cells, 48
paired fields, Holm over the 12 cells:

| Budget | 3e-4 | 1e-3 | 1.5e-3 | 2e-3 | 2.5e-3 | 3e-3 |
|---|---|---|---|---|---|---|
| 12.5% | −0.01 | −0.13 | −0.13 | −0.17 | **−0.29** | **−0.47** |
| 25% | +0.02 | −0.11 | −0.09 | −0.08 | **−0.23** | **−0.23** |

Bold cells have abs-8192 ahead by at least 0.2 point with Holm p < 0.05. No cell has pos-8192 ahead.

**Secondary comparisons:**
- **pos-4096 minus abs-4096:** the same pattern. abs is ahead by 0.26–0.61 point at 2e-3 to 3e-3; pos is never
  ahead.
- **pos-8192 minus the stored pos1024:** −0.05 to +0.13 point. More data improves the signed mask a little at 3e-3,
  but not enough to matter.
- **abs-8192 minus the stored abs1024:** −0.08 to +0.08 point. The size-based mask hardly changes, although about 900
  of its bits at 12.5% differ.

## Reading, by the rule fixed in advance

**"It does not help although D1 at 8,192 is above 0.40."** The agreement check alone is not enough; the DS-CNN differs
for another reason. Here the average gradient is now reliable (D1 0.73, higher than on most ResNet-8 checkpoints),
and the signed ranking is still behind abs1024.

Together with the gradient-trust result (`GRADIENT_TRUST_RESULTS_2026-11-15.md`), the conclusion is the same on
five architectures:
- **A reliable average gradient does not make the signed ranking better.** What decides it is whether the
  first-order (signed) effect of faults is large compared with their spread effect. That was the case only on the
  fragile ResNet-8. Elsewhere, even at low rates, the spread term dominates, and the ranking that ignores the sign is
  as good or better.
- **Practical advice for the paper:** use abs1024. A signed or rate-specific ranking is worth considering only for a
  model whose low-rate fault damage stays large after protection, as on ResNet-8. That damage can be measured
  directly with a few fault runs, which is the check to recommend.
