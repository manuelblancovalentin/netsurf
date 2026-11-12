# Run P, Phase 2: secondary checkpoints and the full picture

Status, 11 November 2026, 22:40 CST: all 74 jobs exited with code 0, including the primary jobs and Block B (see
`RUN_P_PHASE2_PRIMARY_RESULTS_2026-11-11.md`). Every job passed the exact replay check.

- Analysis: same code as the primary record (commit d409b1f).
- Full snapshot: `workspace/runs/paper1_runP_phase12_v1/phase2_analysis_all_2026-11-11.json`, read-only, sha256
  `90f1c464700e9a6ae6e546a9b4876dc7a613e780afae619eb87258441d287538`.

The frozen verdict uses the primary checkpoints only and is unchanged: the method helps (Section 2 of the primary
record). Everything below is secondary or exploratory, as the plan fixed in advance. It does not change the
verdict, but it limits what can be claimed.

Arms are as in the primary record: abs (the paper's 64-image mask), abs512, squared, range W and single.

## 1. Five retrained copies, pooled per cell

Mean over the five seeds, in accuracy points, with a Student-t 95% interval over seeds (4 degrees of freedom).
Every cell listed is usable on all five seeds, except 3e-3 at 6.25%, which is usable on 2 of 5.

| Cell | Range W − abs | abs512 − abs | Range W − abs512 | Range W − squared | Single − range W |
|---|---|---|---|---|---|
| 1e-3, 6.25% | +0.43 [+0.2, +0.7] | +0.23 [+0.1, +0.4] | +0.21 [+0.1, +0.4] | +0.19 [−0.1, +0.5] | +0.61 [+0.4, +0.8] |
| 1e-3, 12.5% | +0.20 [−0.1, +0.5] | +0.20 [−0.0, +0.4] | +0.00 [−0.1, +0.1] | −0.02 [−0.2, +0.1] | +0.59 [+0.2, +1.0] |
| 3e-3, 12.5% | +0.54 [+0.1, +1.0] | +0.47 [−0.0, +1.0] | +0.07 [−0.1, +0.2] | +0.09 [−0.0, +0.2] | +0.41 [−0.2, +1.0] |
| 3e-3, 18.75% | +0.30 [+0.0, +0.6] | +0.11 [−0.1, +0.3] | +0.18 [+0.1, +0.3] | +0.11 [−0.1, +0.3] | +0.29 [−0.0, +0.6] |
| 1e-2, 18.75% | +0.94 [+0.8, +1.1] | +0.64 [+0.3, +0.9] | +0.29 [−0.1, +0.7] | +0.10 [−0.1, +0.3] | −0.02 [−0.4, +0.3] |
| 1e-2, 25% | +0.90 [+0.3, +1.5] | +0.50 [+0.1, +0.9] | +0.40 [−0.0, +0.8] | +0.20 [−0.2, +0.6] | +0.09 [−0.3, +0.5] |

Outside the usable cells (none of the seven rankings keeps the model within 5 points):

| Cell | Range W − abs512 | Single − range W |
|---|---|---|
| 3e-2, all budgets | −0.2 to −2.2 | |
| 1e-1, 18.75% | −10.4 [−14.9, −5.8] | +10.5 |
| 1e-1, 25% | −16.6 [−21.6, −11.6] | +16.4 |

Readings:

- On well-trained copies of the pinned recipe, range W beats the paper's mask by 0.2–0.9 points in usable cells.
  Most of that comes from using 512 calibration images instead of 64 (abs512 − abs: +0.1 to +0.6). The new
  objective adds 0.0–0.4 points over the absolute gradient built from the same 512 images.
- The mean term adds little on these copies (range W − squared ≤ 0.2 points in usable cells). This is consistent
  with Phase 0: the absolute gradient already leaves the first-order effects balanced on well-trained checkpoints.
- Exploratory Holm correction over all 192 secondary cells: range W beats abs512 in two usable cells (seed 7104 at
  1e-2 and 25%, +0.96; pruned 50% at 1e-2 and 6.25%, +1.69) and loses in seven cells, all at 1e-1.

## 2. Pruned 50%

No usable cell shows a difference from either reference that survives correction. The point estimates against
abs512 are −0.0 to −0.3 in usable cells, and −1.2 at 1e-2 and 18.75%, where it is not usable. The single-rate mask
beats range W by +0.4 to +2.0, often resolved without correction. As on pruned 70%, the range-W objective helps
least on pruned checkpoints.

## 3. Outside the design range

Range W is built from rates in [3e-4, 3e-2]. At 1e-1 it loses heavily to the absolute gradient on several
checkpoints:

| Checkpoint | Loss vs abs512 at 1e-1 |
|---|---|
| Under-trained | 7–32 points |
| Pinned | 8–17 points (18.75% and 25%) |
| Retrained seeds | 10–17 points (18.75% and 25%) |

No checkpoint is usable at 1e-1, so the decision rule never sees these cells. A designer who builds the mask for
one range of rates and meets a higher rate pays for it. The plain absolute gradient does not have this failure.
The single-rate mask built at 1e-1 does not have it either.

## 4. Calibration size: the effect that matters most here

Phase 0 showed that the companion rebuilds the paper's 64-image masks exactly, from images [1000 r, 1000 r + 64).
abs512 uses images [1000 r, 1000 r + 512), a superset of those 64. So abs512 − abs measures calibration size
alone. On the nine ResNet-8 checkpoints:

- In usable cells it is +0.1 to +1.0 points, often resolved without correction.
- In unusable cells at 1e-2 and 3e-2 it reaches +1.4 to +4.5 points (seeds pooled), up to +9 on the pinned
  checkpoint at 1e-1.

The paper builds every data-driven ranking from 64 images. Whether the other gradient rankings gain in the same
way, and whether the ranking order holds at larger calibration sizes, was not measured. See
`RUN_Q_CALIBRATION_SIZE_PLAN_2026-11-11.md`.

## 5. What can be claimed from Run P

- The mean-variance objective improves on the same-data absolute gradient where the mean displacement matters: by
  +1.0 to +1.5 points at 1e-2 on the under-trained checkpoint, where the preferred ranking reverses.
- Elsewhere on ResNet-8 it adds 0.0 to 0.4 points in usable cells, and nothing on the pruned checkpoints.
- It can lose 7 to 32 points at a rate above the range it was built for.
- It is not a better default. It is evidence that the reversal works through the mean displacement, because a mask
  that accounts for the mean recovers part of it.
- Knowing the fault rate is worth 0.3 to 2 points on the under-trained, pruned and retrained checkpoints, and
  nothing on the pinned one.
- Not run: DS-CNN and the other models (amendment A3).
