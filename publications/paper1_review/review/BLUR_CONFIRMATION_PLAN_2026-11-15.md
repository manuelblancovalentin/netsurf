# Confirmation of the "noise plus events" fault model: frozen plan (15 November 2026)

**Status:** frozen before any confirmation pass exists. The researcher asked for it ("Cool, let's do it then").

**Code:** branch `paper1-runR-v2-validated`, commit 5fd310c.
- `workflows/runR_blur_pilot.py`, unchanged since the pilot (7057778, sha256 `a653dddd…`), for the ResNet-8 checkpoints.
- `workflows/runG_blur_confirm.py` (sha256 `794d8468…`) for the jet, MNIST and VWW checkpoints. It imports the pilot's
  own `run` and `moments` functions, so the model is computed by the same code.
- `workflows/runG_blurconf_analyze.py` (sha256 `f6d3026f…`), tested on made-up data only
  (`tests/test_blurconf_analyze.py`, 8 tests).

## 1. Question

The exploratory pilot (`BLUR_PILOT_2026-11-13.md`) suggested that stored-bit faults under a mask can be described as:
- **background noise:** every weight gets random noise with the exact mean and spread of its faults in bits 0–5;
- **events:** the flips in the top two bits (6 and 7) are kept as real, discrete flips;
- the result is passed through the network itself. No gradient is used.

On the DS-CNN and the pinned ResNet-8 this reproduced measured accuracy to about 0.5 point, and both the ResNet-8
switch between rankings and the DS-CNN's lack of one. The pilot was designed after those results were known.

**Does the same model, with nothing changed, reproduce the measured outcomes on checkpoints it has never seen?**

## 2. Honest limits of this test

- The measured outcomes already exist and their summaries have been read (Stage 1, and the gradient-trust fault
  runs). What is new is the model's side. The model has no parameter fitted to any outcome, and every threshold
  below is fixed here, from the pilot, before any confirmation pass is run.
- The DS-CNN and the pinned ResNet-8 were the pilot's models. They are not counted.
- Timing smokes were run on replication 5 of four checkpoints, which is not in the analysis set. Only their run
  times and clean accuracies were read.
- The passes use the differentiable companion, not TFLite. Rule G0 checks that path separately.

## 3. Checkpoints, inputs and grid

| Family | Checkpoints | Replications | Measured target |
|---|---|---|---|
| ResNet-8 | under-trained, pruned 70%, pruned 50%, retrained seeds 7101–7105 (8) | 0, 1, 2, 3 | Stage 1 (`paper1_runR_stage1_crossover_v1`) |
| Others | jet-e1, e3, e10, e30, e100; mnist-e1, e3, e10, e30; vww (10) | 0, 1, 2, 3, 4 | gradient-trust fault runs (`paper1_gradtrust_faults_v1`) |

That is 18 checkpoints and 82 jobs. mnist-tcad stays exploratory and is not run.

- **Masks:** the masks of the measured runs. ResNet-8: the saved Stage 1 masks. Others: rebuilt from the locked
  Phase 1 statistics exactly as the fault runs built them.
- **Arms:** abs1024 and pos1024; also msb for the jet, MNIST and VWW checkpoints.
- **Budgets:** 12.5% and 25%.
- **Rates:** the pilot's six: 3e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2.
- **Inputs:** each replication's own evaluation window (1,000 inputs).
- **Per cell:** 8 Gaussian draws, 8 hybrid draws, and 8 replays of the measured runs' own fault fields. The same
  random numbers are used for every mask of a rate.

## 4. Variants

| Variant | What it is | Role |
|---|---|---|
| **hybrid** | top two bits as real flips, bits 0–5 as Gaussian noise | **primary** |
| gauss | all bits as Gaussian noise with the exact mean and spread | secondary |
| mean | mean shift only | secondary |
| real | the measured runs' actual flips through the companion | path check (G0) |

## 5. Rules (all in accuracy points)

A **cell** is one replication, arm, budget and rate. The model's value is the mean of its 8 draws; the measured value
is the mean of the 8 TFLite fields. A cell is **usable** if the better measured arm of abs1024 and pos1024 is within
5 points of that replication's measured clean accuracy. Rules use usable cells.

| Rule | Per checkpoint |
|---|---|
| **G0, path** | real against TFLite: mean absolute error ≤ 0.25 and worst cell ≤ 1.0. A checkpoint that fails G0 counts as failing C1 and C3, and is reported as a path failure, not a failure of the model. |
| **C1, accuracy** | hybrid against TFLite: mean absolute error ≤ 1.0. |
| **C2, order of masks** | Pairs pos1024 − abs1024 and msb − abs1024 with a measured gap of at least 0.5 point: the hybrid gap has the same sign. Pooled over each family, at least 85% must agree. |
| **C3, regime** | "Two regimes" if, at some budget and some rate ≤ 3e-3, pos1024 − abs1024 pooled over the replications (paired by field or by draw) has mean ≥ 0.2 and mean ≥ 2.5 standard errors. The same rule is applied to the measured runs and to the hybrid; they must agree. |

**Verdict:**
- **CONFIRMED:** G0 holds on at least 15 of 18; C1 on at least 15 of 18; C3 agrees on at least 15 of 18, with at least
  6 of the 8 ResNet-8 checkpoints and at least 8 of the 10 others; and C2 holds in both families.
- **FAILED:** C1 holds on fewer than 9 of 18, or C3 agrees on fewer than 12 of 18.
- **PARTIAL:** anything else. Reported with the rules that failed.

The analysis runs once, after every job has ended. Reported alongside, as description: the same statistics for the
gauss and mean variants, errors over all cells, and the per-family tables.

## 6. What each outcome allows

- **CONFIRMED:** the paper may say that, on these 18 checkpoints of four architectures, faults under a mask behave like
  Gaussian weight noise plus discrete top-bit events passed through the network, and that this model reproduces
  measured accuracy and which models show the switch. Not allowed: that masks *chosen* with the model beat static
  rankings (not tested; earlier searches did not), or anything about models outside this set.
- **PARTIAL or FAILED:** the model stays an exploratory appendix item, with the rules that failed stated.

## 7. Run

- **Where:** kona CPUs, `.venv-mlperf` (TensorFlow 2.14.1). No GPU, no TFLite fault run, no training.
- **Command:** one process per job, 4 threads each, 8 at a time:
  - `runR_blur_pilot.py blur --model <name> --rep <r> --out <out>` (ResNet-8, pilot defaults);
  - `runG_blur_confirm.py --id <id> --rep <r>` (others).
- **Output:** `workspace/runs/paper1_blurconf_v1/blur/<name>_rep<r>/` (write-once), logs in
  `paper1_blurconf_logs/`, analysis in `paper1_blurconf_analysis_v1/`.
- **Estimate**, from the timing smokes on an idle machine:

  | Jobs | Passes per job | Time per pass | Time per job |
  |---|---|---|---|
  | ResNet-8 (32) | 600 | 0.5 s (up to 5.6 s under load, as in the pilot) | 5–56 min |
  | VWW (5) | 900 | 1.3 s | about 20 min |
  | jet and MNIST (45) | 900 | 0.015 s | under 1 min |

  About 20 core-hours at most; 30 minutes to 4 hours of wall time with 8 jobs at once.

## 8. Not covered: DistilBERT, Pythia, Qwen

An exploratory two-moment prototype already exists for DistilBERT (`workflows/drift_diffusion_prototype.py`,
11 November): mean accuracy error 1.3 points over 78 cells, with a hybrid variant for the cells where sign bits are
exposed. A confirmation on the transformers needs its own plan: the prototype was run on the same sentences as the
measured grid, and Qwen needs the prototype adapted to its executable and a GPU.
