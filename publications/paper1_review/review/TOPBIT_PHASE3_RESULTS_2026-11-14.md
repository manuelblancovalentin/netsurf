# Top-bit analysis, Phase 3 results (14 November 2026, EXPLORATORY)

Plan: `TOPBIT_ANALYSIS_PLAN_2026-11-14.md`, questions Q3 and Q4. Stage 4 outcomes of replication 0 had already
been read, so everything here is descriptive.

**Runs** (all locked under `workspace/runs/`, with hashes in the `_logs/` folders):
- Quest masks job 8672840 → `paper1_topbit_masks_kws_v1`.
- Quest TFLite array 8672841 → `paper1_topbit_mask_eval_kws_v1`. It ran 8 fields; every engine check reproduced
  the saved trial exactly.
- The comparison → `paper1_topbit_phase3_compare_v1`, made with `workflows/runR_topbit_phase3_compare.py`.

**Checks:**
- On Quest, the recomputed absolute-gradient scores rebuild the stored abs1024 masks exactly, with 0 differing
  bits.
- An earlier kona smoke run differed by 85–98 bits. I first attributed that to CPU round-off. **Correction:** that
  smoke run used the conda `netsurf` environment, which has TensorFlow 2.21, instead of the TensorFlow 2.14.1
  reference environment (`.venv-mlperf`). The difference comes from the TensorFlow version. The group-search smoke
  check found the same cause for a one-clip TFLite difference: in the reference environment, it reproduces the
  Stage 4 cell exactly.
- No locked output is affected. Every locked output ran in TensorFlow 2.14.1, either on Quest or with
  `.venv-mlperf`.

## Q3: ranking by measured single-flip damage is much worse than the gradient rankings

The masks were built from the 512 calibration clips only:
- the 44,032 top bits were ordered by the measured loss increase of their single flip ("loss"), or by clips
  turned wrong ("acc");
- bit 5 then follows in abs1024 order.

They were evaluated with TFLite, paired by field with the Stage 4 masks: 8 fields and 800 test clips per cell.
The values below are the mean difference, the measured-damage mask minus abs1024, in points. A negative value
means the measured mask is worse.

| Rate | 12.5% (loss / acc) | 18.75% | 25% | 37.5% |
|---|---|---|---|---|
| 3e-4 | −0.2 / −0.3 | −0.2 / +0.1 | −0.2 | 0.0 |
| 1e-3 | −0.7 / −0.3 | −1.0 / +0.1 | −0.1 | 0.0 |
| 3e-3 | −4.6 / −1.3 | −4.2 / −1.0 | −0.1 | −0.3 |
| 1e-2 | −30.9 / −18.7 | −26.2 / −9.8 | −1.5 | −0.3 |
| 3e-2 | −51.6 / −45.1 | −61.1 / −42.1 | −6.2 | −0.9 |

- **Across the 48 budget-rate cells**, the "loss" mask is never ahead of abs1024 by two standard errors, and is
  behind in 27 cells. It is behind the best fixed ranking in 29.
- **Significance-first** (all sign bits first) also beats it by up to 38–49 points.
- **At 25%**, the measured mask protects all 44,032 top bits. That is exactly the significance-first and FKeras
  mask (difference 0.00).

**Why it fails.** A single top-bit flip barely moves this network's loss on the calibration clips:
- the median change is −1.2e-4 for bit 6 and +1.7e-4 for bit 7;
- 56% of bit-6 flips and 46% of sign-bit flips lower the loss or leave it unchanged.

So the ranking leaves about 10,000 sign bits unprotected at 12.5%, because each one looks harmless alone. Their
jumps are the largest in the word, though, and many of them together are devastating at higher rates. The
gradient masks keep more sign bits because the score includes the jump size: abs1024 has 13,615 sign bits at
12.5%, against the measured mask's 12,083.

This is the same lesson as Phase 2 from the other side. An effect measured with everything else intact does not
describe the bit's contribution once many other bits are faulty.

**For the paper.** The reviewers' "why not just inject every single flip" alternative was tested here:
exhaustive single-flip injection of every top bit. On this model it is not a gold standard. It is clearly worse
than the gradient rankings and significance-first at moderate and high rates, despite costing about 5
GPU-minutes. One model, one replication, post hoc.

## Q4: how the gradient scores rank measured single-flip damage

Spearman rank agreement with the measured loss increase, on the calibration clips:

| Score | All top bits | Bit 7, network-wide | Bit 7, median within layer | Bit 6, network-wide | Bit 6, median within layer |
|---|---|---|---|---|---|
| abs | 0.25 | 0.35 | 0.37 | 0.10 | 0.11 |
| pos | 0.36 | 0.47 | 0.50 | 0.21 | 0.25 |

- Agreement within a layer is no better than across the network. The group-search idea of measuring layers
  and choosing bits within a layer by the cheap score gets no support from this.
- Q3 shows that single-flip damage is itself a poor guide to which bits to protect, so a weak agreement with it
  is not a defect of the gradient scores.

## What this changes

Phases 2 and 3 agree: on the DS-CNN, a bit's isolated effect, whether measured or gradient-predicted, is a poor
guide to its value under many simultaneous faults. Two kinds of guide have worked:
- **Combining faults in weight space and passing them through the network.** This is the blur pilot's
  Gaussian/hybrid variants and the companion replays.
- **Rules driven by jump size**, such as significance-first or the gradient score times the jump, at higher
  rates.

A method should evaluate candidate masks with network passes under the target fault rate. The group-search plan
does that. Its two-level design (cheap within-layer choice) is not supported by Q4, so within-layer choices
should also be evaluated by network passes, or the cheap level dropped.
