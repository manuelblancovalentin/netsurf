# Run R: why the switch prediction failed on the DS-CNN (post-hoc diagnostics, 13 November 2026)

Exploratory. Written after the Stage 4 outcomes were read (`RUN_R_STAGE4_RESULTS_2026-11-13.md`). No new fault
field was drawn; no Stage 2 outcome was read. Inputs: the finished Stage 1 (ResNet-8, nine checkpoints) and Stage 4
(DS-CNN) fault runs, which recorded mean cross-entropy next to accuracy; the frozen predictions; the calibration
statistics saved by the mask builds (mean gradient `gbar`, its standard error `se`, mean squared gradient `g2`); the
R-A5 selection scores. Script `workflows/runR_stage4_diagnostics.py` (branch `paper1-runR-v2-validated`, f2ad1e8),
about one minute on one process. Output `workspace/runs/paper1_runR_stage4_diagnostics_v1/` (read-only):
`summary.json` sha256 `a9a8f6d44a1ffff1b19b740df40898ee25f7ee796c670793db5fd8797fea5682`, `cells.csv` sha256
`6ff54b834f5bbe0779098430ad536e2df7b1b4fec4ec3cb14d55c7402ced2330`.

## H1, loss versus accuracy: rejected as the cause

If the formula predicted loss correctly and only the step from loss to accuracy failed, the measured loss would favour
the positive mask where it was predicted to win. It does not. Usable cells, pooled fields, 95% t-intervals:

| | cells predicted pos ahead | measured loss: pos ahead / abs ahead (resolved) | accuracy: pos ahead / abs ahead (resolved) |
|---|---|---|---|
| DS-CNN | 28 | 0 / 22 | 0 / 19 |
| ResNet-8, nine checkpoints | 40-71 each | 37-71 / 0-1 each | 34-69 / 0-4 each |

On the DS-CNN the measured loss says the same as accuracy: the loss prediction itself is wrong. On ResNet-8 the
measured loss switch is close to the predicted one (12.5% budget, e.g. pinned: predicted $3.5\times10^{-3}$, loss
$2.9\times10^{-3}$, accuracy $2.4\times10^{-3}$; retrained 7103: $3.2$, $3.1$, $3.0\times10^{-3}$), so part of the
small ResNet-8 bias is the step from loss to accuracy.

## H2, close to a minimum: supported

| | DS-CNN | ResNet-8 (range over nine checkpoints; pinned) |
|---|---|---|
| share of the measured mean-gradient energy that is real, $(\sum\bar g^2-\sum se^2)/\sum\bar g^2$ | 0.23 | 0.37-0.85; 0.37 |
| correlation of the mean gradient between independent calibration draws | 0.21 | 0.47-0.92; 0.47 |
| first-order advantage of pos over abs kept when scored with the other draws' gradient | 39% | 73-92%; 76% |
| predicted pos - abs on the six R-A5 held-out sets, rates $\le 5\times10^{-3}$: share with pos ahead | 58-67% | 75-100% (rates $10^{-4}$ to $3\times10^{-3}$) |
| measured over predicted loss increase, rates $\le 3\times10^{-3}$: abs1024 / pos1024 | 1.7 / 8.8 | 1.2-3.0 / 0.7-3.3, about equal per checkpoint |
| clean loss, test windows over calibration inputs | 3.1 | 1.7-2.7 |

Reading:
1. The DS-CNN mean gradient is mostly calibration noise. About three quarters of its energy is noise, and two
   independent draws of 1,024 clips barely agree. The positive mask is chosen by the sign of that gradient, so most of
   what made it look good on its own calibration clips does not carry over.
2. The predicted advantage of the positive mask is therefore tiny and unstable: on other held-out sets it points
   the predicted way only slightly more often than a coin.
3. The formula underestimates the damage of the positive mask on the DS-CNN about five times more than that of the
   absolute mask (8.8 against 1.7). On ResNet-8 the two errors are about equal, so they cancel in the difference
   that sets the switch. A tiny predicted gain plus a large, mask-specific underestimate of cost gives a predicted
   switch that does not exist.
4. Across the nine ResNet-8 checkpoints the direction agrees: pinned has the least real gradient and the largest
   overprediction of the switch. Rank correlation with the Stage 1 bias: 0.55 (real share of the gradient), 0.58
   (agreement between draws), 0.38 (kept advantage), -0.10 (test/calibration loss). With nine checkpoints none of
   these is significant.

## What is not explained yet

Why the positive mask's cost is underestimated on the DS-CNN. A candidate: the positive mask leaves unprotected some
large-jump bits whose sign looked favourable on the calibration clips, and the quadratic cost
$\tfrac12 F\Delta^2$ misjudges large jumps. A deterministic check (each bit protected by abs1024 but not by pos1024,
flipped alone on held-out clips, measured against $\bar g\Delta + \tfrac12 F\Delta^2$) would settle it without any
fault field.

## What it allows the paper to say

- The DS-CNN failure is not a loss-versus-accuracy artefact: the formula's loss prediction is wrong there.
- It coincides with a model whose mean gradient is mostly noise on 1,024 clips. The switch prediction needs a real
  mean gradient, and a well-trained model may not have one.
- A check that needs no fault runs, the agreement of the mean gradient between two independent calibration draws, cleanly separated the
  failing model (0.21) from the nine that worked (0.47 and above). This is a candidate trust check, found after
  the fact on one failing model; any threshold needs confirmation on further models before it is claimed.
- Not allowed: any claim that this check is validated, or that the cause of the cost underestimate is known.
