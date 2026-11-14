# What the measured top-bit flips can tell us: analysis plan (exploratory, not run)

Status: PLAN ONLY. Each phase needs its own go-ahead.
Classification: EXPLORATORY. The Stage 4 DS-CNN outcomes used as targets below have already been read.

Data: `workspace/runs/paper1_topbit_flips_kws_v1/` (locked). It holds every bit-6 and bit-7 single flip of the
DS-CNN (44,032 flips), with per-clip margin and correctness on:
- the 800 test clips of the Stage 4 window for replication 0;
- the 512 blur-pilot calibration clips.

It was measured on the GPU path. Against the CPU path it reproduces:
- 95–99% of the variation in loss change;
- accuracy changes within one clip of 512.

Loss effects below about 3e-4 are within rounding noise. See the `_logs/README.md`.

## 1. Questions

- **Q1, combining.** If each fault's measured single-flip effect is added up per clip, does that predict
  accuracy when many bits flip at once?
  - The top bits are handled as discrete events.
  - The low bits (0–5) are handled as a Gaussian background.

  This is the bit-level formulation of the Fourier/characteristic-function discussion. Where it fails,
  whether at high rates or for particular masks, the bits interact and single-flip measurements are not
  enough.
- **Q2, choosing.** If the predicted accuracy is used to pick a mask for each rate and budget, does it pick
  the mask that measures best?
- **Q3, the reviewers' baseline.** Does ranking the top bits by their measured single-flip damage beat the
  gradient rankings, at any rate? This is exhaustive single-flip injection, the obvious "why not just
  simulate" alternative. It must be in the paper either way.
- **Q4, where the cheap scores fail.** How well do the absolute and positive gradient scores rank the
  measured top-bit damage:
  - across the whole network;
  - within each layer?

  If the scores do well within a layer and badly across layers, that supports the group-search plan:
  layer-level decisions measured, within-layer choice by the cheap score.

## 2. Targets and data separation

- **Accuracy to predict (Q1, Q2):** the Stage 4 TFLite results for replication 0, which are already on
  disk:
  - 6 masks: abs1024, pos1024, abs64, significance-first, FKeras, and the analytic mask built for each rate;
  - 4 budgets: 12.5%, 18.75%, 25%, 37.5%;
  - 12 rates, from 3e-4 to 3e-2;
  - 8 fault fields, on the same 800 test clips the top-bit job measured.

  That makes 288 cells, each with 8 fields.
- **Building anything** (the Q3 mask, any choice in Q2) uses only the 512 calibration clips. The test
  clips are used only for prediction and comparison.
- **Two reference paths.** Each prediction is compared with:
  - the TFLite accuracy;
  - for the 48 cells the blur pilot replayed, the CPU companion's own replay of the same fault fields.

  The companion replay separates prediction error from the companion-versus-TFLite difference, which is
  about 0.07 points.

## 3. Phases

### Phase 0: code and deterministic tests (no compute run)

The script is `workflows/runR_topbit_analysis.py` (v2 worktree). Tests:
- With one fault event and no background, the additive prediction equals the measured single-flip result
  exactly.
- On synthetic per-clip effects, the analytic readout (Gil-Pelaez integral of the per-clip characteristic
  function) agrees with direct simulation of the same additive model to Monte Carlo precision.
- The Q3 masks have exactly floor(f·176,128) bits, and are nested across budgets.

### Phase 1: GPU job on Quest (about 15 minutes, one A100)

**1a. Re-measure, keeping every class score.**

The current files keep only the margin: the true class's score minus the best other class. Margins do not
add up correctly when a fault changes which class is runner-up. Keeping the change of all 12 class scores
makes the additive model exact in the quantity that decides correctness.

| Item | Value |
|---|---|
| Flips | the same 44,032 |
| Precision | float16 changes of the 12 class scores, per clip |
| Size | about 1.4 GB |
| Time | about 5 minutes (measured: 297 s for the first run) |
| Output | `paper1_topbit_logits_kws_v1/` |
| Check | the margins must equal the locked v1 margins exactly, the same GPU path |

**1b. Gaussian background for the low bits.**

For each of the 288 cells, 16 network passes with Gaussian weight noise, using the exact mean and variance
of the unprotected bits 0–5, as in the blur pilot. From these passes, keep each clip's mean and spread of
each class score.

| Item | Value |
|---|---|
| Passes | 288 × 16 = 4,608 full passes on 1,312 clips |
| Time | about 1–2 minutes at about 10–20 ms each |
| Output | `paper1_topbit_background_kws_v1/` |
| Seeds | `[20261212, cell, draw]`, new |

### Phase 2: analysis on kona's CPU (minutes, no network passes)

For every cell and clip, combine three things:
- the unprotected top bits' flip probability p;
- their measured per-clip score changes;
- the clip's low-bit background.

Then predict accuracy four ways, from simplest to fullest:

| Predictor | Top bits | Low bits | Readout |
|---|---|---|---|
| E | additive events | ignored | exact for the additive model (simulation over flip patterns, no network) |
| E+B | additive events | Gaussian background | same |
| G | first two moments of events plus background | (included) | Gaussian per clip: Φ(mean/spread) of the margin |
| CF | additive events plus background, on the margin | (included) | Gil-Pelaez integral (the analytic formula) |

For reference, the plan also includes the blur pilot's "gauss" and "hybrid" network-pass predictions where
they exist (48 cells), and the old analytic formula's predicted loss.

**Reports:**
- **Q1:**
  - predicted against measured accuracy, per rate, mask and budget;
  - mean absolute error, and the worst cell;
  - the error as a function of the expected number of simultaneous top-bit flips.

  If predictor E+B fails only where many top bits flip at once, that locates the interactions.
- **Q2:** in each cell family (rate and budget), whether the predicted best mask is the measured best. Also
  the accuracy lost by following the prediction, the same regret measure Stage 4 used.
- **Q4:** rank agreement between gradient scores and measured top-bit damage on the calibration clips,
  across the network and within each layer, per bit position.

### Phase 3: evaluate the measured-damage mask (Q3), CPU

The mask is built from the calibration clips only:
- top bits are protected in order of measured single-flip loss increase;
- 25% of all bits is exactly all 44,032 top bits;
- at 37.5%, the rest is filled with bit 5 in abs1024 order.

| Budget | Masks |
|---|---|
| 12.5%, 18.75%, 25%, 37.5% | 4 masks |

Two variants:
- "loss" order: mean loss increase.
- "accuracy" order: clips turned wrong.

That's 8 masks. Evaluation uses the Stage 4 protocol: TFLite replay, engine check, the same 8 fields, all
12 rates.

| Item | Value |
|---|---|
| Evaluations | 8 masks × 12 rates × 8 fields = 768 |
| Time | at about 5.7 s each, about 75 CPU-minutes; split across 8 jobs on a Quest CPU node (account p33112), about 10 minutes |
| Output | `paper1_topbit_mask_eval_kws_v1/` |

All outputs are write-once, then hashed, locked and registered. Every job writes an unbuffered progress log
under `<output>_logs/`.

## 4. How to read the outcomes

| Outcome | Meaning | Next |
|---|---|---|
| E+B or CF predicts measured accuracy within about 0.5 points across rates, and picks the best mask in most cells | measured single flips plus a background explain multi-bit faults on this model; the bit-level formulation works where the cheap formula failed | pilot a rate-aware mask search with this predictor, then ResNet-8 (about 20 GPU-minutes of flips) |
| E+B good at low rates, breaking above some rate | interactions between top-bit faults become important at high rates; the error against the number of simultaneous flips shows where | report the range of validity; add pairwise interactions only if they are cheap to measure |
| G as good as CF | the bell-curve readout is enough; the full characteristic function adds nothing on this model | use G (simpler) |
| measured-damage mask beats the gradient masks | exhaustive single-flip injection is better than gradient rankings, at 5 GPU-minutes of cost | the paper must include it as a baseline, and the method must beat it |
| measured-damage mask no better than abs1024 | ranking by single-flip damage is not what matters; combinations and fault rate are | supports the rate-dependence framing |
| gradient scores rank well within layers but not across layers | cheap scores fail mostly on layer-to-layer amplification | supports the group-search plan's two-level design |

## 5. Limits

- **One model and one replication.** This is a diagnosis, not a result for the paper.
- **Both paths are float approximations.** The GPU-path flips are compared against TFLite, so a remaining
  error of about 0.1 points cannot be attributed to the predictor.
- **Bits 0–5 are not measured flip by flip.** They are described only by the background. Q1 tests whether
  that is enough, but cannot separate a background error from an event error. A per-flip measurement of
  bit 5 (another 22,016 flips, about 3 GPU-minutes) can be added if the errors point there.
- **What is not claimed.**
  - A mask chosen by a predictor is not shown to be optimal.
  - A single flip's effect is not the effect of swapping two bits.
  - Fault rates are logical bit-error rates, not physical upset rates.
