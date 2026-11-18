# Run Q: calibration size of the gradient rankings (ResNet-8)

Status: written 11 November 2026, 23:00 CST, before any Run Q outcome. **Frozen as of this commit.** It needs the
researcher's go-ahead before launch.

## 1. Why

The paper builds every data-driven ranking from 64 calibration images. Run P found that the absolute-gradient mask
built from 512 images gains +0.1 to +1.0 accuracy points over the paper's mask in usable cells, and more at higher
rates (`RUN_P_PHASE2_SECONDARY_RESULTS_2026-11-11.md`, Section 4). The 512 images are a superset of the paper's 64,
and the code rebuilds the paper's mask exactly, so this gain is calibration size alone.

Two reviewer questions are open:

1. Do the other gradient rankings gain in the same way, so that the order the paper reports could change at larger
   calibration sizes?
2. How many images does the default need?

## 2. Design

**Checkpoints and replications.** The nine ResNet-8 checkpoints of Table 5: pinned (9 replications), five retrained
copies, pruned 50%, pruned 70% and under-trained (8 replications each). The saved fault schedules and evaluation
images of each replication are used, at BER 1e-3, 3e-3, 1e-2 and 3e-2:

| Checkpoint | Schedules per rate | Schedules per replication |
|---|---|---|
| Pinned | 4 | 16 |
| Each of the other eight | 2 | 8 |

**Masks.** Each ranking is built from nested calibration sets: training images [1000 r, 1000 r + n) for
replication r, with n = 128, 256, 512 and 1024. The paper's masks use the first 64 of the same images. Budgets are
6.25%, 12.5%, 18.75% and 25%, with the exact bit counts of the saved masks. Scores follow the paper's estimators:

| Ranking | Score |
|---|---|
| Absolute gradient | mean over images of \|g·Δ\| |
| Positive gradient | mean over images of max(0, g·Δ) |
| Global gradient | \|mean over images of g·Δ\| |

Ties are broken by storage order. The n = 1024 set overlaps the next replication's first 24 images; this is
accepted and reported.

**Reused outcomes.**
- n = 64: the saved outcomes of the paper's three masks.
- Absolute gradient at n = 512: Run P's replays on the same schedules. Every job checks that its rebuilt mask equals
  Run P's bit for bit.

**New evaluations.** 11 masks × 4 budgets on each schedule, plus one exact replay check per schedule:

| | Evaluations |
|---|---|
| Pinned | 9 × 16 × 45 = 6,480 |
| Other eight checkpoints | 64 × 8 × 45 = 23,040 |
| Total | 29,520 |

**Prerequisite checks**, all of which must pass:
- Identity, in `check_<checkpoint>.json`: the three rankings rebuilt from each replication's own 64 images must
  overlap the saved masks by at least 0.999 at every budget, on every checkpoint. Already measured on two
  checkpoints: pinned (minimum 0.99977) and under-trained (minimum 0.99999).
- Exact replay of the saved 12.5% absolute-gradient mask on every schedule of every job, or the job aborts.
- Equality with Run P's 512-image absolute-gradient masks.

## 3. Analysis, fixed now

Usable cells are taken from each checkpoint's measured table of the seven rankings (ResNet-8 rows). Tests are
pooled-field sign-flip tests with 100,000 random assignments (seed 20261114). Student-t intervals over replications
are reported alongside, and over the five seeds when they are pooled.

- **Q1, order.** Positive − absolute and global − absolute, at n = 512 and n = 1024, in usable cells of all nine
  checkpoints, in one Holm family. *The default holds* if neither competitor is resolved ahead of the absolute
  gradient by 0.5 point or more in a usable cell where it was not already resolved ahead at n = 64 (paper data).
  Otherwise the paper must state that the order depends on calibration size, with the cells.
- **Q2, size of the default.** Absolute gradient at n = 128, 256, 512 and 1024 minus at n = 64, in usable cells, in
  one Holm family. *Calibration size matters* if the 512- or 1024-image mask is resolved ahead of the 64-image mask
  by 0.5 point or more on at least three of the nine checkpoints. The paper then reports the gain and the size at
  which it levels off. Otherwise it states in one sentence that 64 images are within 0.5 point.
- **Reported:** absolute gradient at 1024 minus at 512 (does the gain level off); the same size curves for the
  positive and global gradients; unusable cells, separately.

## 4. Not covered

- FKeras at larger calibration sizes (Hessian eigenpairs, separate machinery).
- DS-CNN and the other models (no differentiable companion in this code).
- Retraining.

None of these is claimed.

## 5. Cost and outputs

| Item | Value |
|---|---|
| Jobs | 73 (one per checkpoint and replication) |
| Parallelism | 26 at a time, 2 TensorFlow threads each (52 of 80 cores, as in Run P) |
| Job time | about 40 minutes (75 for pinned), at 6.6 s per evaluation measured in Run P |
| Wall time | about 2.5 hours, plus about 10 minutes of identity checks |
| Memory | about 1.5 GB per job (per-example gradients for 1024 images: 634 MB) |
| Storage | under 400 MB |

- Code: `workflows/runQ_calibration.py`, branch `paper1-runP-mean-variance`, commit c442d09.
- Launcher: `workspace/launch/orchestrate_runQ.sh`, tmux `runQ`.
- Output: `workspace/runs/paper1_runQ_calibration_size_v1/`, write-once.
- Registry entry: `20_runQ`.

## 6. Launch record (11 November 2026, 22:42 CST, before any Run Q outcome)

- Go-ahead from the researcher: "Yes, launch Run Q".
- Launched in tmux `runQ` at runner commit c442d09; registered as `20_runQ`.
- Analysis script `workflows/runQ_analyze.py`, commit feb8db1, written after launch and before any outcome. It
  implements Section 3. It was dry-tested on the existing 64-image and Run P outcomes, with stand-in empty Run Q
  files in a scratch directory.

## 7. Amendment Q-A1 (22:46 CST, before any Run Q outcome)

The identity gate stopped the first attempt at 22:44, before any job ran. Its log is
`orchestrator_attempt1_gate_failed.log`.

**Finding.**
- Eight checkpoints passed, with a minimum overlap of 0.99977.
- Retrained seed 7102 failed, in one replication only (replication 4). Its rebuilt masks overlap the saved ones by:

  | Ranking | Overlap |
  |---|---|
  | Absolute gradient | 0.9987–0.9992 |
  | Positive gradient | 0.9983–0.9992 |
  | Global gradient | 0.9966–0.9972 |

  Replications 0–3 and 5–7 of the same checkpoint match.
- The calibration indices (4000–4063) and the sample hash are the saved ones.
- The bits that differ are not ties:
  - absolute gradient: their scores lie up to 1.3% from the cutoff;
  - global gradient: 0.71–1.33 times the cutoff, because it averages gradients that largely cancel.
- So the saved masks of that replication were built from gradients that differ by about 1% from those the code
  computes now. The cause is not known.

**Amendment.** Replication 4 of retrained seed 7102 is excluded from Run Q. That leaves 72 jobs, and seed 7102 has 7
replications. The gate is applied to every other replication, and the analysis rules are unchanged. Run Q was
relaunched at 22:46.

**Consequences outside Run Q.**
- The paper's outcomes for that replication are exact replays of the masks as saved, so they stay valid as
  measurements of those masks.
- The masks themselves are slightly noisier estimates of the rankings.
- In Run P, the abs512 − abs contrast for that replication includes this gradient difference in addition to
  calibration size.
- A follow-up check of how the Run K job for seed 7102 replication 4 computed its gradients is recorded as open.
