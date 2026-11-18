# Run Q results: calibration size of the gradient rankings (ResNet-8)

Status: written 12 November 2026, after every Run Q job had finished.
- The analysis follows the frozen plan `RUN_Q_CALIBRATION_SIZE_PLAN_2026-11-11.md` (Section 3, with amendment Q-A1).
- It was produced by `workflows/runQ_analyze.py` (branch `paper1-runP-mean-variance`, commit feb8db1, file sha256
  `03e903a6c9196831e3f72643de1281be477244f61e6130334907db01ae432d1e`), unchanged since before any outcome existed.
- Section 2 gives the frozen verdicts. Sections 3 and 4 are descriptive readings of the same data and test no new
  hypothesis.

## 1. Integrity

| Check | Result |
|---|---|
| Jobs | 72 of 72 finished with exit code 0, each with its `DONE` file (22:46 to 01:18 CST) |
| Runner | every `DONE` file records script sha256 `7de191f7f9aae9283c41db2bc73b379e127365176138f01f2a10ebe22817cea2` (commit c442d09) |
| Identity gate | passed: minimum overlap 0.99977 between rebuilt and saved 64-image masks, excluding the Q-A1 replication |
| Equality with Run P | the 512-image absolute-gradient mask equals Run P's, bit for bit, in every job (assertion) |
| Exact replay check | 648 of 648 schedules reproduce the saved accuracy and cross-entropy of the saved 12.5% absolute-gradient mask; no `replay_check_FAILED.json` |
| Evaluations | 28,512 new mask evaluations + 648 replay checks = 29,160 (the planned 29,520 minus the excluded replication's 360) |
| Storage | 262 MB |

Output: `workspace/runs/paper1_runQ_calibration_size_v1/`. Read-only snapshots:

| File | sha256 |
|---|---|
| `analysis_final_2026-11-12.json` (identical to `analysis.json`) | `ee7818f3b3e5963c2485e0c680c96825628fce1319694665b1a39cf43cd9fcee` |
| `analysis_table_2026-11-12.txt` (printed per-cell table) | `cf03c90b281272228ffbf966053936c0c860b5214578c03e76cfa4e20c1aac8b` |

The 1024-image set of replication r overlaps the first 24 calibration images of replication r + 1, as the plan
accepted. Evaluation images and fault schedules are the saved ones of each replication.

## 2. Frozen verdicts

### Q1, order of the gradient rankings: the default does **not** hold

The rule: the default holds if neither competitor is resolved ahead of the absolute gradient by 0.5 point or more,
in a usable cell where it was not already resolved ahead at 64 images.
- Family: 328 contrasts (positive and global minus absolute, at 512 and 1,024 images, in the usable cells of all nine
  checkpoints), Holm-corrected.
- The global gradient is never resolved ahead.
- The positive gradient is resolved ahead in 16 contrasts, which fall in 10 cells on 5 checkpoints:

| Checkpoint | BER | Protection | Positive − absolute, 512 images | Positive − absolute, 1,024 images | At 64 images |
|---|---|---|---|---|---|
| Under-trained | 1e-3 | 6.25% | +1.32 (replications +0.81 to +1.84) | +1.24 | +0.74, not resolved |
| Under-trained | 3e-3 | 6.25% | +2.03 | +2.01 | −0.36 |
| Under-trained | 3e-3 | 12.5% | +2.41 | +2.46 | +1.17, not resolved |
| Under-trained | 3e-3 | 18.75% | +2.03 | +1.67 | +1.45, not resolved |
| Under-trained | 3e-3 | 25% | +1.98 | +1.99 | +1.46, not resolved |
| Pruned 50% | 1e-3 | 18.75% | (+0.70, not resolved) | +0.92 | +0.31 |
| Retrained R1 (seed 7101) | 1e-3 | 18.75% | +0.89 | (+0.69, not resolved) | −0.35 |
| Retrained R1 (seed 7101) | 3e-3 | 25% | (+0.71, not resolved) | +1.24 | −3.31, resolved **behind** |
| Retrained R3 (seed 7103) | 1e-3 | 25% | +0.53 | (+0.68, not resolved) | +0.30 |
| Retrained R4 (seed 7104) | 1e-3 | 18.75% | +0.69 | +0.96 | +0.52 |

All values are accuracy points. Holm-adjusted p-values range from 0.012 to 0.038.

Per the plan, the paper must state that the order of the gradient rankings depends on calibration size, and list
these cells.

### Q2, calibration size of the default: **not** shown to matter

The rule: calibration size matters if the 512- or 1,024-image absolute-gradient mask is resolved ahead of the
64-image mask by 0.5 point or more on at least three checkpoints.
- Family: 328 contrasts. No checkpoint qualifies.
- One contrast is resolved: under-trained, 3e-3, 18.75%, 1,024 images, +0.44, below the 0.5-point margin.

In the 82 usable cells, the absolute-gradient mask built from n images instead of 64 gains:

| n | Median | Range | Resolved |
|---|---|---|---|
| 128 | +0.09 | −0.30 to +0.92 | 0 |
| 256 | +0.15 | −0.24 to +0.84 | 0 |
| 512 | +0.20 | −0.23 to +1.50 | 0 |
| 1,024 | +0.23 | −0.28 to +1.22 | 1 (below the margin) |

The plan's sentence for the paper applies: no checkpoint shows a resolved gain of 0.5 point from building the
absolute-gradient mask from more than 64 images. It should be accompanied by the pooled-seed figure in Section 3.4,
which is larger at 1e-2.

## 3. What the data show beyond the verdicts

### 3.1 With more images, the positive gradient leads at low BER and still collapses at higher BER

Usable cells of all nine checkpoints:

| BER | Images | Cells | Positive ahead (point estimate) | Median positive − absolute | Range | Resolved ahead / behind (Holm) |
|---|---|---|---|---|---|---|
| 1e-3 | 64 | 36 | 17 | −0.01 | −0.81 to +1.30 | (paper data) |
| 1e-3 | 512 | 36 | **36** | +0.65 | +0.23 to +1.32 | 8 / 0 |
| 1e-3 | 1,024 | 36 | **36** | +0.65 | +0.24 to +1.55 | 9 / 0 |
| 3e-3 | 64 | 29 | 3 | −4.29 | −8.10 to +1.46 | (paper data) |
| 3e-3 | 512 | 29 | 15 | +0.17 | −1.28 to +2.41 | 4 / 0 |
| 3e-3 | 1,024 | 29 | 20 | +0.42 | −0.91 to +2.46 | 5 / 0 |
| 1e-2 | 64 | 15 | 0 | −36.9 | −46.4 to −12.3 | (paper data) |
| 1e-2 | 512 | 15 | 0 | −10.2 | −22.6 to −0.5 | 0 / 13 |
| 1e-2 | 1,024 | 15 | 0 | −8.1 | −17.8 to −0.6 | 0 / 12 |
| 3e-2 | any | 2 | 0 | −52 to −58 | | 0 / 2 |

The pattern by fault rate:
- **1e-3:** once the positive gradient is built from 512 or more images, it is ahead of the absolute gradient in
  every usable cell of every checkpoint, by a fraction of a point to 1.5 points.
- **3e-3:** the two are roughly tied.
- **1e-2 and above:** the positive gradient is 8 to 18 points worse even at 1,024 images.

The language models already behave this way in the paper (Section 7, reversals): the positive gradient is best at
the lowest BERs and worse at 3e-3 and above.

### 3.2 The reversal is no longer specific to the under-trained checkpoint

With 1,024 images for both rankings, positive versus absolute in usable cells (Holm, Q1 family):

| Checkpoint | Resolved ahead | Resolved behind | Largest lead of the positive gradient |
|---|---|---|---|
| Pinned | 3 | 1 | +0.49 |
| Under-trained | 8 | 3 | +2.46 |
| Pruned 70% | 0 | 0 | +1.55 (not resolved) |
| Pruned 50% | 1 | 1 | +1.09 |
| R1 (7101) | 1 | 2 | +1.24 |
| R2 (7102) | 0 | 2 | +0.93 |
| R3 (7103) | 0 | 2 | +0.73 |
| R4 (7104) | 1 | 2 | +0.99 |
| R5 (7105) | 0 | 1 | +0.59 |

Five of nine checkpoints show resolved wins in both directions: pinned, under-trained, pruned 50%, R1 and R4.

The paper currently states (Section 7, reversals): "No pair of rankings shows resolved wins in both directions on
the pinned or pruned ResNet-8". With 64-image masks that statement is accurate. At 1,024 images it no longer holds
for the pinned and the pruned 50% checkpoints.

### 3.3 The positive score needs more images than the absolute score

Median gain from more images over the same ranking at 64 images, usable cells of all checkpoints:

| Ranking | BER | 128 | 256 | 512 | 1,024 |
|---|---|---|---|---|---|
| Absolute | 1e-3 | +0.01 | +0.07 | +0.11 | +0.10 |
| Absolute | 3e-3 | +0.13 | +0.16 | +0.22 | +0.24 |
| Absolute | 1e-2 | +0.47 | +0.60 | +0.56 | +0.81 |
| Positive | 1e-3 | +0.40 | +0.61 | +0.68 | +0.77 |
| Positive | 3e-3 | +2.61 | +4.11 | +4.46 | +4.68 |
| Positive | 1e-2 | +11.1 | +20.1 | +24.0 | +26.8 |
| Global | 1e-3 | +0.02 | −0.13 | −0.11 | +0.00 |
| Global | 3e-3 | +0.12 | +0.29 | +0.50 | +0.34 |
| Global | 1e-2 | −0.66 | −0.38 | +0.31 | −0.36 |

The masks themselves settle at different speeds. Overlap of the n-image mask with the 1,024-image mask, at 12.5%
protection, as a mean over replications:

| Checkpoint | Absolute 128 / 256 / 512 | Positive 128 / 256 / 512 | Global 128 / 256 / 512 |
|---|---|---|---|
| Pinned | 0.91 / 0.94 / 0.96 | 0.86 / 0.90 / 0.94 | 0.66 / 0.72 / 0.81 |
| Under-trained | 0.93 / 0.95 / 0.97 | 0.89 / 0.93 / 0.96 | 0.73 / 0.78 / 0.87 |
| R1 | 0.90 / 0.94 / 0.96 | 0.85 / 0.90 / 0.94 | 0.67 / 0.73 / 0.82 |
| Pruned 50% | 0.92 / 0.94 / 0.97 | 0.87 / 0.91 / 0.95 | 0.68 / 0.74 / 0.83 |

How each ranking behaves:
- **Absolute:** close to its final mask by 128 to 256 images.
- **Positive:** settles more slowly, and its accuracy keeps improving up to 1,024 images.
- **Global:** its mask keeps changing, but accuracy does not improve. The mean gradient over many images largely
  cancels, which matches the paper's explanation for why the global gradient trails.

The comparison in the paper, with every ranking built from 64 images, therefore understates the positive gradient
more than the absolute gradient.

### 3.4 Pooled over the five retrained seeds

Means over seeds, with Student-t 95% intervals across the five seeds, in points:

| Cell | Usable on | Positive − absolute, 64 | Positive − absolute, 512 | Positive − absolute, 1,024 | Absolute 1,024 − 64 | Absolute 1,024 − 512 |
|---|---|---|---|---|---|---|
| 1e-3, 6.25% | 5 | −0.24 [−0.69, +0.21] | +0.59 [+0.30, +0.88] | +0.68 [+0.43, +0.92] | +0.30 [+0.07, +0.54] | +0.09 [−0.11, +0.28] |
| 1e-3, 12.5% | 5 | −0.09 [−0.57, +0.40] | +0.66 [+0.34, +0.97] | +0.72 [+0.58, +0.86] | +0.14 [−0.14, +0.43] | −0.05 [−0.31, +0.21] |
| 1e-3, 25% | 5 | +0.34 [+0.08, +0.59] | +0.50 [+0.30, +0.70] | +0.51 [+0.31, +0.71] | +0.15 [−0.08, +0.37] | +0.03 [−0.16, +0.22] |
| 3e-3, 12.5% | 5 | −6.30 [−8.39, −4.22] | −0.39 [−0.67, −0.11] | −0.30 [−0.69, +0.10] | +0.46 [−0.08, +1.01] | −0.00 [−0.24, +0.24] |
| 3e-3, 25% | 5 | −3.21 [−4.05, −2.37] | +0.50 [+0.02, +0.99] | +0.80 [+0.40, +1.19] | +0.08 [−0.29, +0.45] | −0.02 [−0.26, +0.21] |
| 1e-2, 18.75% | 5 | −43.1 [−47.5, −38.7] | −18.5 [−22.7, −14.3] | −15.7 [−19.6, −11.8] | +0.95 [+0.74, +1.16] | +0.33 [+0.13, +0.52] |
| 1e-2, 25% | 5 | −35.6 [−40.0, −31.1] | −10.0 [−11.8, −8.1] | −7.7 [−10.2, −5.3] | +0.55 [+0.19, +0.90] | +0.04 [−0.11, +0.20] |

- **Absolute gradient at 1e-2:** pooled over the seeds, building it from 512 or 1,024 images gains about 0.5 to 1.0
  point at 18.75% and 25%. That is above the 0.5-point margin, although no single checkpoint resolves it under the
  frozen per-checkpoint rule.
- **Wording for the paper:** "within 0.5 point" would overstate the Q2 verdict; the honest sentence gives both
  numbers.

### 3.5 Above clean accuracy at 1e-3

At BER 1e-3 and 25% protection, the positive-gradient mask built from 512 images leaves:
- 7 of the 8 well-trained checkpoints at or above their clean accuracy (+0.11 to +0.64 points; R1 −0.06);
- the under-trained checkpoint at +1.0 (at 1,024 images and 12.5%, +1.4).

The absolute-gradient mask leaves most of them slightly below clean.

This is descriptive: 0.1 point is one evaluation image in 1,000, and it was not tested. It suggests that the
"faults acting like a training step" effect the paper describes for the under-trained checkpoint is present, more
weakly, on well-trained copies too.

### 3.6 Unusable cells

Where no ranking keeps the model within 5 points of clean, larger calibration sets help the absolute gradient more:
- median +2.4 points at 1,024 images, range −1.7 to +8.1, over 62 cells;
- mostly at 1e-2 with small budgets, and at 3e-2.

These cells are outside the region where the paper makes recommendations.

## 4. Reading

1. **The paper's measurements stand as measurements of 64-image masks.** What changes is how far they generalize.
   - The order "absolute ahead of positive" is partly a consequence of building every ranking from 64 images. The
     positive score is estimated less precisely from 64 images than the absolute score.
   - With 512 or more images, the positive gradient is slightly better at the lowest rate measured here (1e-3) on
     every ResNet-8 checkpoint, and much worse from 1e-2 on.
2. **The preferred ranking depends on the intended BER on most ResNet-8 checkpoints, not only the under-trained
   one.** This is the condition the paper already reports for the language models, now seen on a classifier with
   well-trained copies. It supports the advisor's framing: a single fixed ranking chosen without knowing the fault
   rate can be the wrong choice.
3. **The absolute gradient remains the safe choice.** In usable cells, at any calibration size measured:
   - the most it loses to the positive gradient is 1.55 points on the well-trained checkpoints and 2.5 on the
     under-trained one;
   - the positive gradient loses up to 18 points to it at 1e-2 with 1,024 images (23 with 512, 46 with 64), and up
     to 58 at 3e-2.

   "Default" should now be stated as the choice with the smallest worst-case loss, not as the best choice in every
   cell.
4. **Sixty-four images are enough for the absolute gradient by the frozen rule.** It levels off between 128 and 512
   images; 1,024 − 512 has a median of +0.06 points in usable cells. Pooled over seeds, at 1e-2, more images gain
   0.5 to 1.0 point.
5. **The under-trained checkpoint is the strongest instance of a general pattern, not a separate condition.** The
   paper's explanation (its loss is not at a minimum) is consistent with it having the largest effect. It should no
   longer be presented as the only ResNet-8 case.

## 5. Consequences for the paper (proposed; the researcher decides)

Sentences that need a calibration-size qualifier or a revision:

| Location | Current claim | Status after Run Q |
|---|---|---|
| Introduction, contribution 1 | "never resolved worse than another measured ranking on the classifiers … largest regret below 0.7 points … retrained copies" | True for 64-image masks. Add the condition; at 512+ images the positive gradient leads by up to 1.5 points at 1e-3 |
| Introduction, contribution 2 | reversal "on an under-trained checkpoint" | Generalize: with enough calibration images the preferred ranking depends on BER on five of nine ResNet-8 checkpoints |
| Section 7, retrained and pruned | "No ranking is better by a point in any decision-useful cell" | True at 64 images; at 1,024 images the positive gradient is resolved ahead by 1.24 (R1, 3e-3, 25%) |
| Section 7, reversals | "No pair of rankings shows resolved wins in both directions on the pinned or pruned ResNet-8" | Qualify with "with 64-image masks"; add the Run Q table |
| Section 7, harm mechanism | under-trained "not at a minimum" | Keep, but note the same direction of effect on well-trained copies at low BER |
| Discussion, designer points 2 and 4 | absolute gradient "a sound default"; avoid direction-selecting scores "for dense faults" | Point 2: smallest worst-case loss. Point 4 is reinforced: the positive gradient is good only when faults are sparse |
| Discussion, limits | — | Add: rankings compared at 64 calibration images; calibration size changes the positive gradient most |
| Conclusion | "holds across masks, inputs, pruned checkpoints and retrained copies" | Add the calibration-size condition |

**Smallest change that satisfies the frozen rule:**
- a short paragraph in Section 7 (reversals);
- one appendix table, combining the tables of Sections 3.1 and 3.2;
- the Q2 sentence with the pooled-seed number;
- the qualifiers above.

The story becomes stronger rather than weaker for the advisor's framing. A fixed ranking is a bad choice when it is
chosen without the intended fault rate: the ranking that is best at sparse faults fails at dense ones.

## 6. Not covered

- **BERs below 1e-3.** Only 1e-3 to 3e-2 were replayed. The lead of the positive gradient at 1e-4 and 3e-4 with large
  calibration sets is not measured.
- FKeras, DS-CNN, ToyCar and the transformers at larger calibration sizes.
- Calibration sets larger than 1,024 images.

None of these is claimed.
