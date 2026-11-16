# Run T, stage T1 (TFLite models): frozen protocol (15 November 2026)

**Status:** frozen before any Run T outcome is read. Parent plan: `RUN_T_FULL_COMPARISON_PLAN_2026-11-15.md` with
amendment A1. The researcher's go-ahead: "Yes, build T1".

**Code:** branch `paper1-runR-v2-validated`, commit 89bf9b0.
- `netsurf/nonideal_protection.py` (sha256 `d94ad30f…`): the fault transforms.
- `workflows/runT_tflite.py` (`891fc8fe…`): the runner.
- `workflows/runT_analyze.py` (`c57aee3d…`): the analysis.
- Tests on made-up data only: `tests/test_nonideal_protection.py` (11), `tests/test_runT_analyze.py` (3).

## 1. What T1 is, and its two parts

| Part | Checkpoints | State |
|---|---|---|
| **T1a** | jet MLP (epochs 1, 3, 10, 30, 100), MNIST MLP (epochs 1, 3, 10, 30), VWW | launched with this protocol |
| **T1b** | ResNet-8 (9 checkpoints), DS-CNN, ToyCar; the FKeras arms for every T1 checkpoint | needs input cards and the Hessian code; same runner, same protocol |

The FKeras arms are added later with the same fault fields: every random stream is seeded by checkpoint,
replication, draw, rate and purpose, never by the list of arms.

## 2. Fault model and conditions

Every stored weight bit flips independently with probability p. So does every redundant copy and every check bit.
No protected bit is immune.

| Condition | Failure mode | Extra storage |
|---|---|---|
| `unprotected` | every flip applies | 0 |
| `tmr_<ranker>@<share>` | the top share of bits by the ranker is stored three times; a voted bit is wrong with probability 3p² − 2p³ | 2 × share |
| `secded_64 / 32 / 16 / 8` | one flipped bit per codeword (data or check) is repaired; two or more are left; miscorrection of three or more is not modelled, which favours the code | 12.5 / 21.9 / 37.5 / 62.5% |
| `abft_ideal` | two fault-free checksums per output channel; a channel with one faulty weight is repaired | 64 bits per channel |
| `abft_faulty` | the same decoder with flipping 32-bit checksums and wrong repairs | 64 bits per channel |

- **Rankers:** abs (per-example absolute gradient), msb (bit position; at a share of 12.5% this is FORTUNE-style
  sign-bit triplication), pos, global, jump, random. T1b adds FKeras as published (tier first) and the FKeras
  source's weighted variant.
- **Shares of bits:** 6.25, 12.5, 18.75, 25%.
- **Masks:** built from each replication's 1,024 calibration inputs (the locked gradient-trust statistics), in the
  same way as the gradient-trust fault runs.
- **Rates:** 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1.
- **Executable:** the TFLite reference kernels, on each replication's evaluation window of 1,000 inputs.
- **Draws:** 6 replications × 4 draws = 24 per checkpoint. All conditions of a draw share one raw fault field.

Note: for the tiny MLPs, 64 checksum bits per channel is not a small cost (25% for the jet MLP). The analysis reports
every condition with its own cost.

## 3. Analysis (run once per part, after every job has ended)

- **Unit:** one draw of one replication, paired across conditions.
- **Curves:** mean accuracy per condition and rate, and accuracy lost against clean.
- **Tolerated rate:** the highest grid rate up to which the mean accuracy stays within 5 points of clean, at that rate
  and every lower one.
- **Fixed contrasts,** left minus right, tested at every rate where the better of the two is within 5 points of clean:

  | Contrast | Left | Right |
  |---|---|---|
  | Ranked TMR against a code, 12.5% storage | `tmr_abs@6.25%` | `secded_64` |
  | Ranked TMR (25%) against a code (21.9%) | `tmr_abs@12.5%` | `secded_32` |
  | Ranked TMR against a code, 37.5% storage | `tmr_abs@18.75%` | `secded_16` |
  | Ranked TMR against FORTUNE, 25% storage | `tmr_abs@12.5%` | `tmr_msb@12.5%` |
  | abs against MSB-first, other shares | `tmr_abs` | `tmr_msb` at 6.25, 18.75, 25% |
  | abs against FKeras (T1b) | `tmr_abs` | `tmr_fkeras` at 6.25, 18.75%; weighted variant at 12.5% |
  | Ranked TMR against ABFT | `tmr_abs@6.25%` | `abft_ideal` |
  | ABFT, faulty against ideal checksums | `abft_faulty` | `abft_ideal` |
  | Ranked TMR against nothing | `tmr_abs@12.5%` | `unprotected` |

- **Test:** paired sign-flip, 100,000 assignments, Holm over all contrasts and usable rates of the checkpoint. A
  difference is **resolved** when it is at least 0.5 point with Holm p < 0.05.
- The other rankers (pos, global, jump, random) are reported as curves, without tests.

## 4. What the results may be used for

- The main comparison figure of each model, and the statements of which scheme tolerates the higher fault rate at
  equal extra storage.
- Whether the earlier ideal-protection conclusions still hold. They are re-derived from these results, not assumed.
- Not allowed: hardware cost claims beyond stored bits (voters, encoders, decoders, scrubbing), or physical fault
  rates.

## 5. Run (T1a)

- **Where:** kona CPUs, `.venv-mlperf`, one thread per job.
- **Command:** `runT_tflite.py --id <id> --rep <r> --draw <d>`, for r in 0–5 and d in 0–3.
- **Output:** `workspace/runs/paper1_runT_tflite_v1/<id>/rep<r>/draw<d>/` (write-once); logs in
  `paper1_runT_tflite_logs/`; analysis in `paper1_runT_t1a_analysis_v1/`.
- **Smokes:** replication 5, draw 0 of jet-e1 and mnist-e1 (two rates) and of vww (one rate), written to scratch.
  Only run times, clean accuracies and fault counts were read.
- **Estimate:** 31 conditions × 8 rates = 248 evaluations per job.

  | Jobs | Time per job |
  |---|---|
  | 216 MLP jobs | a few seconds |
  | 24 VWW jobs | set by the timing smoke, recorded in the run registry before launch |

## Addendum T1b (15 November 2026, frozen before any T1b outcome; code 09edb1a)

The researcher's go-ahead: "Yes, go for it."

**Code** (branch `paper1-runR-v2-validated`, commit 09edb1a):
- `workflows/runT_fkeras_masks.py` (sha256 `ab168521…`) and `netsurf/tflite_graph_continuous.py` (`ee36ed0c…`):
  FKeras for jet, MNIST and VWW. Hessian of the continuous float64 companion (no activation rounding), checked
  against finite differences in `tests/test_tflite_graph_continuous.py`. Same solver, options and seed as the
  engine's FKeras.
- `workflows/runT_engine.py` (`3089222c…`): ResNet-8 and DS-CNN through their engine adapters.

**Calibration:** FKeras uses the same 1,024 calibration inputs as the gradient rankings, so it is not handicapped by a
smaller set.

**Arms added:**
- For the T1a checkpoints: `tmr_fkeras@<share>` and `tmr_fkeras_weighted@<share>`, on the same fault fields, written
  as `results_fkeras.csv`.
- For ResNet-8 and DS-CNN, every arm: abs and pos (the saved 1,024-input masks), abs64, msb, fkeras,
  fkeras_weighted, jump and random.

**Cohort:** 6 replications × 4 draws per checkpoint.
- ResNet-8: pinned replications 1–6; the eight others 0–5, except retrained seed 7102, which uses 0, 1, 2, 3, 5, 6
  (its replication 4 was excluded in Run Q).
- DS-CNN: replications 0–5 (800-clip windows).

**Smokes,** read for run time and clean accuracy only:
- FKeras masks for jet-e1 and vww, replication 5 (eigen-solver converged in 25 products; 1 s and 3 min);
- DS-CNN replication 5 and retrained seed 7101 replication 5, one rate, written to scratch (3.6 s per evaluation
  under load).

**Analysis:** `runT_analyze.py` unchanged. For the T1a checkpoints, rerun once with `--suffix _fkeras` after every
FKeras job has ended; for ResNet-8 and DS-CNN, run once on their outputs. Both are written to new folders.

**Cost:**

| Jobs | Number | Time per job | Total |
|---|---|---|---|
| FKeras masks | 60 | under 4 min | — |
| FKeras arms on T1a | 240 | about 10 min for VWW, seconds otherwise | — |
| ResNet-8 and DS-CNN engine jobs | 240 | 39 conditions × 8 rates, 20–40 min each | about 120 core-hours |

All on kona, 24 at a time. ToyCar is still to be added (it needs AUC scoring).
