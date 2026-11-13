# Run R: overfitting of the analytic search and the selected rate-aware mask (development record, 13 November 2026)

Exploratory and developmental. Training-side data only: no test input and no fault outcome was used. Stage 2
(ResNet-8) and Stage 4 (DS-CNN) evaluations were running and unread throughout. Code: branch
`paper1-runR-v2-validated` (worktree `paper1-runR-v2-validated`), created so that the frozen Run R code is untouched.
Scratch scripts of the first check: `kws_heldout_objective.py`, `resnet_heldout_objective.py` (session scratchpad).

## 1. The check that found it

Every mask was scored with the predicted objective
$\hat J_p = [L(\theta+\mu)-L(\theta)] + \tfrac12\sum_w F_wV_w$ (ideal suppression) on two sets: the 1,024 calibration
inputs it was built from, and held-out inputs (DS-CNN: validation clips 0-511, all 6 replications, 4 Stage 4
budgets, 12 rates; ResNet-8: training images 28000-28511, first replication of each of the 9 checkpoints, 4 Stage 2
budgets, 8 rates). Statistic: rate-aware (v1) minus the best static mask, median over replications and budgets.

| Rate | DS-CNN in-sample | DS-CNN held-out | ResNet-8 in-sample | ResNet-8 held-out |
|---|---|---|---|---|
| $10^{-3}$ | $-0.006$ (best in 100%) | $+0.001$ (12%) | $-0.017$ (100%) | $-0.006$ (72%) |
| $3\times10^{-3}$ | $-0.007$ (100%) | $+0.003$ (4%) | $-0.036$ (100%) | $-0.004$ (64%) |
| $10^{-2}$ | $-0.020$ (100%) | $+0.016$ (0%) | $-0.065$ (100%) | $-0.018$ (78%) |
| $3\times10^{-2}$ | $-0.030$ (100%) | $+0.062$ (0%) | $-0.049$ (97%) | $-0.024$ (92%) |

The search overfits its calibration inputs. On ResNet-8 about half of the in-sample gain survives; on DS-CNN none
does. DS-CNN has a third of ResNet-8's weights and its positive and absolute masks share most bits, so differences
between masks are small next to the calibration noise; this agrees with the noisy DS-CNN switch predictions.

## 2. Variants tried (`workflows/runR_v2_dev.py`, `runR_v2_dev_select.py`)

Development grid: DS-CNN all 6 replications, ResNet-8 first replication of the 9 checkpoints; budgets 12.5% and 25%;
rates $10^{-3}$, $3\times10^{-3}$, $10^{-2}$, $3\times10^{-2}$. Selection inputs (for the held-out choices):
ResNet-8 training images $30000+512\,r$, DS-CNN validation clips $1024+512\,r$; scoring on the held-out sets above.

- **Gate:** accept a swap only if mean(d) + 2 sd(d)/$\sqrt N$ < 0 for the per-input change d of the objective
  (`descend(..., accept_z=2)`, `per_input_evaluator`). The search then accepts 1-2 swaps per rate.
- **Gate + choice:** the gated mask, `abs1024` or `pos1024`, whichever has the lowest objective on the selection inputs.
- **v1 + choice (selected):** the ungated v1 mask, `abs1024` or `pos1024`, chosen the same way.

Minus the best static mask on the held-out inputs, median / worst case over all development cells:

| | v1 | Gate | Gate + choice | v1 + choice |
|---|---|---|---|---|
| DS-CNN (48 cells) | $+0.008$ / $+0.110$ | worse than v1 at $10^{-3}$-$3\times10^{-3}$ | $0.000$ / $+0.007$ | $0.000$ / $+0.007$ |
| ResNet-8 (72 cells) | $-0.0075$ / $+0.046$ | gains mostly lost at $10^{-3}$-$3\times10^{-3}$ | $-0.0044$ to $0.000$ at low rates / $+0.019$ | $-0.0075$ / $+0.018$ |

The gate fixes DS-CNN but discards most of ResNet-8's real gains, because one fixed strictness cannot suit both
models. Choosing on held-out inputs adapts per case: on ResNet-8 it kept v1 in 58 of 72 cells and lost none of its
median gain while cutting the worst case from $+0.046$ to $+0.018$; on DS-CNN it fell back to `abs1024` or `pos1024`
in almost every cell and removed the losses. A further finding: the choice of starting point inside the search is
itself made on calibration inputs and overfits on DS-CNN (at $10^{-3}$ the gated search still changed about 39% of the
protected bits, through its starting point).

**Chosen for confirmation:** v1 + choice, named `rate_aware_mask_selected` (v2). It needs one extra forward pass per
candidate on 512 held-out inputs. Amendment R-A5 of the Run R plan fixes it and its test.

## 3. Caveats

The variants were compared on held-out training-side data from the same models, and the choice of v2 was made after
seeing those comparisons; the confirmatory test uses the untouched test windows and fault fields of Stages 2 and 4.
Objectives are predicted losses, not accuracy. Development numbers use one replication per ResNet-8 checkpoint and two
budgets.
