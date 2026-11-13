# Blur pilot: are two moments per weight, passed through the network, enough? (exploratory, 13 November 2026)

Post hoc and exploratory: designed after the Stage 1, Stage 4 and Stage 2 interim outcomes were read. Not a
confirmatory stage; no number here is a manuscript result.

Scripts (branch `paper1-runR-v2-validated`): `workflows/runR_blur_pilot.py` (7057778), analysis
`workflows/runR_blur_pilot_analyze.py`. Outputs (read-only): `workspace/runs/paper1_blur_pilot_v1/` and
`paper1_blur_pilot_analysis_v1/` (`summary.json` sha256 `cf06073c53e8b77caac25f9c5340ce71739bb5ed3ee15c75648d2322fc23770f`).
Eight jobs, 56 minutes, 32 threads; companion passes only, no TFLite fault run.

Models: DS-CNN (replications 0 and 1) and pinned ResNet-8 (replications 1 and 2). Masks abs1024 and pos1024 at 12.5% and
25%, rates $3\times10^{-4}$ to $3\times10^{-2}$. Variants, all through the differentiable companion (activation rounding
and clipping reproduced):

- **mean**: weights shifted by the exact mean displacement $\mu_w$ (the drift term of the current formula, without its spread term);
- **gauss**: $\mu_w+\sqrt{V_w}\,\epsilon_w$, one Gaussian draw per weight (exact first two moments, no gradient);
- **hybrid**: bits 6 and 7 as real flips, bits 0-5 by their two moments;
- **real**: the measured runs' own fault fields replayed through the companion (a check of the companion, not a predictor).

## 1. The companion reproduces TFLite

Same fields, same inputs: accuracy within 0.07 (DS-CNN) and 0.11 (ResNet-8) points on average, worst 0.19 and 0.46;
positive minus absolute within 0.10 and 0.13 points. Clean accuracy agrees within 0.5 points.

## 2. Two moments reproduce accuracy and the switch, without any gradient

Accuracy error against the measured TFLite runs (points; 48 cells per model, 8 draws per cell):

| Variant | DS-CNN mean / worst | ResNet-8 mean / worst | pos-abs error, DS-CNN / ResNet-8 (mean) |
|---|---|---|---|
| mean shift only | 3.1 / 22.6 | 3.3 / 27.9 | 0.68 / 2.61 |
| two-moment Gaussian | 0.9 / 12.8 | 0.65 / 5.0 | 0.85 / 1.03 |
| top two bits real, rest Gaussian | 0.55 / 5.6 | 0.48 / 3.9 | 0.66 / 0.70 |

The largest errors are at $10^{-2}$ and $3\times10^{-2}$, where one draw varies by several points, so part of them is
the noise of 8 draws.

The switch, positive minus absolute (points), at the low and middle rates where it is decided:

- **DS-CNN**: measured $-0.09$ to $+0.02$ at $3\times10^{-4}$ and $10^{-3}$, then negative. Gaussian: $-0.25$ to
  $+0.06$, then negative. **No switch, as measured.** The mean shift alone puts the positive mask ahead in 4 of those 8
  low-rate cells ($+0.12$ to $+0.25$), the same bias as the current formula.
- **ResNet-8**: measured positive ahead at low rates ($+0.2$ to $+0.6$), switching between $10^{-3}$ and $5\times10^{-3}$
  depending on budget and replication. The Gaussian and the hybrid place it on the same side in most cells and the
  switch within one rate step (Gaussian late by one step in 2 of 4 budget-replication pairs, hybrid in 1 of 4). The
  mean shift alone has the wrong sign at $3\times10^{-3}$ in 2 of 4 pairs.

Reading: what the current formula gets wrong is its shortcut for the spread ($\tfrac12\sum F_wV_w$), not the
description of faults by two moments per weight. Passing the spread through the network removes the DS-CNN bias.

## 3. Bell shape

64 draws of real flips against 64 of the Gaussian, 400 inputs, 12.5% masks and no mask. Per-input change of the
margin (true logit minus best other):

- **With a mask** (absolute or positive): the two distributions nearly coincide (largest difference between the two
  cumulative distributions 0.008-0.032), accuracy over draws agrees within about 1 point in mean and spread, and the
  per-input average change correlates 0.74-0.99. Real flips have slightly heavier tails (excess kurtosis 0.2-1.1
  against 0.1-0.8).
- **No mask**: real flips have heavy tails (excess kurtosis 8.1 against 0.3 on the DS-CNN at $3\times10^{-3}$; 6.0
  against 1.1 on ResNet-8 at $10^{-3}$) and a wider spread of accuracy between draws (ResNet-8 at $10^{-3}$: 0.71 ± 0.13
  against 0.65 ± 0.09). These are the rare sign-bit events; the Gaussian misses them.

## 4. Single flips against the gradient

80 weights per bit position (10 per layer), the real loss change on 512 calibration inputs against $\bar g\Delta$ and
$\bar g\Delta+\tfrac12F\Delta^2$, on the same inputs (so the difference is approximation error, not sampling noise).

| Bits | Rank correlation real vs gradient-plus-curvature, DS-CNN / ResNet-8 | Sign agreement |
|---|---|---|
| 0-2 (smallest jumps) | $-0.01$ to $0.17$ / $-0.13$ to $0.29$ | about 0.5 (a coin) |
| 3-5 | 0.24-0.61 / 0.41-0.60 | 0.61-0.77 |
| 6-7 (largest jumps) | 0.69-0.77 / 0.72-0.81 | 0.76-0.81 |

Share of the real change explained, bit 7: 0.33 on the DS-CNN, 0.86 on ResNet-8.

For bits 0-2 the real loss change is 10-25 times larger than the gradient predicts and of random sign: the rounding of
activations turns a small weight change into a step. Those flips act as noise, which is why they are well described
as part of the background. For the top bits the gradient with curvature is a fair guide on ResNet-8 and a poor one on
the DS-CNN.

## What it supports, and what not

- Supports: a method that describes faults by two moments per weight and evaluates them through the network, with
  the top bits as discrete events, reproduces both the ResNet-8 switch and the DS-CNN's lack of one, with no gradient.
- Not shown: that a mask *chosen* this way beats the static masks; the pilot only scores existing masks. Two models,
  two replications each, one checkpoint of ResNet-8.
- Cost: one network pass per draw. Choosing a mask this way needs many passes, which is where a GPU would help.
