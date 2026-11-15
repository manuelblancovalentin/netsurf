# DS-CNN: does a larger calibration set make the signed ranking useful? (pre-registration, 15 November 2026)

**Status:** written and committed before any run below exists. The researcher requested it on 15 November 2026:
"Yes, pre-register it and run after the fault runs".

**When it runs:** after the gradient-trust Phase 2 fault runs.

**What has been seen of this model:**
- the Stage 4 results;
- the Stage 4 diagnostics (D1 = 0.21 at 1,024 calibration clips);
- the gradient-trust regression check, which reproduced D1 = 0.21 and also measured D1 = 0.42 at 2,048 clips.

## Question

On the DS-CNN, pos1024 (bits ranked by the signed gradient) never beat abs1024 (bits ranked by gradient size). The
explanation offered is that with 1,024 calibration clips the average gradient is mostly noise, so its sign is
unreliable. If that is right, building the masks from many more clips should make the sign reliable. pos1024 should
then win at low fault rates, as it does on ResNet-8.

## Calibration sets (nested, disjoint across replications, training split as in Stage 4)

For replication r = 0–5, the existing Stage 4 calibration clips are training clips 1024 r … 1024 r + 1023. The new sets
add clips from a separate block for each replication, starting at training clip 6144 + 7168 r:

| Size | Clips for replication r |
|---|---|
| 1,024 | the Stage 4 set, unchanged |
| 4,096 | the Stage 4 set plus the first 3,072 clips of the replication's block |
| 8,192 | the Stage 4 set plus all 7,168 clips of the replication's block |

The largest set uses training clips up to 49,151; the training split has about 85,000.

**Caveat, stated in advance:** the clips come from the training split, as in Stage 4. At convergence the average
gradient over training data is pushed toward zero by training itself. So a negative result could also mean that
training-split calibration has little signal at any size. This is reported as a limitation, not tested here.

## Masks

At each size, pos1024 and abs1024 are built with the Stage 4 scoring formula (`runQ_calibration.scores`). The
budgets are 12.5% and 25%. The per-example gradients come from the graph companion, which matches the pinned DS-CNN
companion to float rounding (`runG_companion_check.py`). Everything runs on CPU.

## Fault runs

- **Protocol:** Stage 4, unchanged:
  - TFLite with the reference kernels and the engine check;
  - the replication's 800-clip test window;
  - fields seeded [20261119, r, f], f = 0–7.

  These are the same windows and fields as the stored Stage 4 masks, so every comparison is paired.
- **Rates:** the Stage 4 grid up to 3e-3, where the signed ranking is expected to help: 3e-4, 1e-3, 1.5e-3, 2e-3,
  2.5e-3, 3e-3.
- **Arms:** pos and abs at 4,096 and at 8,192. The stored pos1024 and abs1024 come from Stage 4.
- **Cost:**
  - Gradients: about 6 × 8,192 clips on CPU, minutes.
  - Fault runs: 48 Quest CPU tasks, each 6 rates × 2 budgets × 4 arms = 48 TFLite evaluations of 800 clips, a few
    minutes per task.
- **Output:**
  - `workspace/runs/paper1_kws_calsize_masks_v1/`
  - `workspace/runs/paper1_kws_calsize_eval_v1/`
  - `workspace/runs/paper1_kws_calsize_analysis_v1/`

## Analysis (fixed now; same statistics as the gradient-trust Phase 3)

- **Primary:** pos-8192 minus abs-8192.
  - The cells are 6 rates × 2 budgets = 12.
  - The test is the paired sign-flip test on the 48 per-field differences in clips, with 100,000 assignments and seed
    20261123. Holm correction is over the usable cells.
  - A cell is usable if the better of the two masks is within 5 points of clean accuracy.
  - **The signed ranking helps** if at least one usable cell has pos-8192 ahead by at least 0.2 point with Holm
    p < 0.05.
- **Secondary**, each with its own Holm correction:
  - pos-4096 minus abs-4096;
  - pos-8192 minus the stored pos1024 (does more data improve the signed mask itself?);
  - abs-8192 minus the stored abs1024 (does more data change the size-based mask?).
- **Descriptive:**
  - D1 at 1,024, 4,096 and 8,192 (median correlation of the average gradient across the 6 disjoint sets at each size);
  - bits differing between the masks at each size.

## How the outcomes will be read (fixed now)

| Outcome | Reading |
|---|---|
| The signed ranking helps at 8,192 and D1 has risen | Supports the noise explanation. The paper can recommend checking D1 and, if it is low, either using abs1024 or enlarging the calibration set until D1 is high. |
| It does not help although D1 at 8,192 is above 0.40 | The agreement check alone is not enough. The DS-CNN differs for another reason, for example the loss-cost misestimate seen in the Stage 4 diagnostics. |
| It does not help and D1 stays at or below 0.40 | Inconclusive about the cause, since more data did not produce a reliable sign. The training-split caveat applies. |

**Scope:** one model, already inspected, with the prediction fixed here before any new run.
