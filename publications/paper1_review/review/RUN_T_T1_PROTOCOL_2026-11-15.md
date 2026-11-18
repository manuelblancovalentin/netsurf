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

## Addendum T2 (15 November 2026, before any T2 outcome; code 7582f78)

The researcher's instruction: "Well build it! Let's run as much as possible overnight."

**Models:** DistilBERT, Pythia-160M and Qwen3-0.6B, with the exact W8A8 executable.
- Faults are XORed into the stored INT8 codes and removed after every evaluation.
- Code: `workflows/runT_w8a8.py` (sha256 `3c1f341424cb…`).
- Checks:
  - the clean codes are verified by hash after each job;
  - the frozen campaign's pinned kona-assets commit (1aea52b) has moved to a later commit that only added unrelated assets; the runner proceeds only if the two executable source files equal their pinned hashes, and records both commits.

**Conditions:** as in T1, with these differences:
- the ranked masks are the sealed masks of the frozen wide-BER campaign, the same masks as the main grid. These are six rankings: MSB-first in tier order with ties by global code, and five sealed rankings;
- ABFT works per output row;
- FKeras is not included yet. It needs a transformer Hessian and will be added later on the same fault fields.

**Draws:**
- 8 per BER for DistilBERT and Pythia;
- 4 for Qwen, because of its cost.

Each job is one (model, draw, BER).

**Metrics:**
- DistilBERT is judged by accuracy, as in T1.
- Pythia and Qwen are judged by mean token NLL. A cell is usable if the better of the compared arms is within 0.5 nats of clean.
- A difference is resolved if it is at least 0.05 nats with Holm p < 0.05.
- The language-model analysis applies the T1 contrasts to NLL. It will be written before any T2 output is read.

**Smoke:** DistilBERT, draw 9 (outside the analysis set), one BER, five conditions, written to scratch. About 20 s per evaluation and 80 s to load the model.

**Cost estimate:**

| Model | Jobs | Time per job |
|---|---|---|
| DistilBERT | 64 | about 12 min |
| Pythia | 64 | not yet timed |
| Qwen | 32 | not yet timed; probably 1–2 h |

Run on kona: 6 jobs of 4 threads now, plus 8 more when T1b has ended.

## Addendum T3: ToyCar (16 November 2026, 00:00, before any T3 outcome; code 25e5cb3)

**Runner:** `runT_engine.py --model toycar`. It uses the same conditions and fault model as T1b, with these differences:
- the metric is file-level ROC-AUC, written in the accuracy column;
- AUC points use the 0.05 usefulness margin of the main grid and a resolution threshold of 0.5 AUC points.

**Masks:**
- abs, pos, abs64 (here the same 16 files), FKeras and the tier-mixing variant are built from 16 normal training files per replication, as in the main grid;
- msb, jump and random are built as in T1.

**Cohort:** two disjoint windows of 62 balanced test files (evaluation half, offsets 0 and 62), 4 draws each: 8 draws in total.

**Analysis:** `runT_analyze.py --draws 8 --ids toycar`, run once at the end.

**Smoke:** replication 1, draw 9 (outside the analysis set), one BER, read for timing and clean AUC only. Clean AUC 0.877; about 3 min for FKeras; about 2 s per evaluation.

## Note T2-FK (16 November 2026): FKeras on the language models

**DistilBERT and Pythia:** \`runT_fkeras_w8a8.py\` on kona.
- The pinned PyTorch environment has no SciPy. The jobs import \`.venv-mlperf\`'s SciPy 1.17.1 through a symlink path; NumPy 1.26.4 still comes from the pinned environment. Neither environment was modified.
- DistilBERT's selection sentences are taken by source index and checked against the campaign's hash.

**Qwen3-0.6B:** \`runT_fkeras_qwen_gpu.py\` on a Quest A100 (80 GB SXM).
- **Loader:** standalone, because the kona loader requires the CPU build of torch. Every tensor is checked against the manifest hash.
- **Windows:** the frozen selection windows, shipped from kona with their hash.
- **GPU check (job 8948044):** the first-window NLL equals the kona CPU value to 1e-8.
- **First full job (8948045):** crashed with a segmentation fault inside ARPACK, whose 32-bit work arrays overflow at 596M dimensions × 20 vectors.
- **Replacement solver:** Lanczos with full reorthogonalization, with the same acceptance checks, ordering and sign rule. It agrees with ARPACK to 1e-10 on a test operator (\`tests/test_lanczos_qwen.py\`, commit ab9307e).
- **Resubmitted** as job 8951472.
- **Pythia's first mask build failed** the solver's Hessian symmetry check (5.3e-5 against 1e-5). The cause: the Hugging Face attention computes softmax in float32, and Qwen's RMSNorm also normalizes in float32. Both FKeras scripts now keep these operations in float64 (commit 3902c4d). This changes precision only; the function is the same.
- **Qwen** was resubmitted as Quest job 8955871. Pythia was restarted on kona.
- **Pythia's second kona build** crashed with a segmentation fault in ARPACK after 2.5 hours: 124M dimensions × 20 vectors also overflows its 32-bit indexing. Pythia now runs on Quest with the same GPU script and Lanczos solver as Qwen (job 8968279). Its frozen selection tokens match the campaign's hash, and a one-window check passed on kona's CPU (NLL 3.29).

## Note on conduct (16 November 2026, 11:15)

**The "exit" lines in the overnight chain logs are not job exit codes.** They have the form `echo "$(date …) … exit $?"`, so `$?` is the exit status of `date`. The finished markers (`DONE`, `DONE_fkeras`) are the record of what completed.

**What completed:**

| Part | Finished |
|---|---|
| T1a | 240 / 240 |
| T1b engine jobs | 240 / 240 |
| T1b FKeras arms | 240 / 240 |
| ToyCar | 8 / 8 |
| DistilBERT and Pythia FKeras arms | 64 / 64 each |
| T2 base jobs | 155 / 160 |
| Qwen FKeras arms | 18 / 32 |

**What was rerun:** the 19 missing jobs died with no error in their logs, most likely killed for memory while many language-model jobs ran at once. They were rerun from 11:15 with the same code and arguments (\`recover.sh\`, 6 at a time, exit codes now recorded).

**Stalls:** two stalls were caused by background waiters whose command lines matched the names they waited for. One is why the T1b analyses ran only at 11:06, from the same frozen script and unchanged inputs.

## Addendum T4 (2026-11-16): sixteen fault draws for the language models

Written before any added draw exists. **Reason:** with 8 (DistilBERT, Pythia) or 4 (Qwen3) draws the smallest
attainable sign-flip p is 1/128 or 1/8, so no contrast can pass Holm over the 51 to 76 contrasts of a model; the
T2 analysis resolved none, and the paper reported the language models descriptively. This addendum adds draws so
that the planned tests can resolve; it is a disclosed second look, with the stopping point fixed here.

- **Draws:** every language model goes to exactly 16 draws: DistilBERT and Pythia draws 8-15, Qwen3 draws 4-15, each
  at the 8 BERs, base conditions and the FKeras conditions (same mask files: `distilbert_sst2_int8/fkeras_masks.npz`,
  `<model>/fkeras_quest/fkeras_masks.npz`). No further draws will be added whatever the outcome.
- **Code:** `runT_w8a8.py` unchanged (draw index enters the seed, so new draws get new fields). Output:
  `workspace/runs/paper1_runT_w8a8_more_v1/<model>/draw<D>/ber<I>/`; the 8/4-draw outputs stay untouched.
- **Analysis:** `runT_analyze_lm_v2.py` (tests `tests/test_runT_analyze_lm_v2.py`), committed before the runs: the
  T2 rules unchanged, draws pooled from both directories, exactly draws 0-15 required, run once into
  `workspace/runs/paper1_runT_t4_lm_analysis_v1`. It replaces the T2 analysis as the language-model analysis of the paper;
  the T2 output is kept.
- **Compute:** 224 jobs on kona (64 DistilBERT, 64 Pythia, 96 Qwen3), each the base conditions then the FKeras
  conditions, 4 threads per job. Started at 10 jobs at a time; stopped after 3 minutes, before any output, because the Qwen3 jobs hold 35 to 45 GB each; restarted as two pools, at most 6 Qwen3 jobs and 5 DistilBERT/Pythia jobs at once (44 of 80 cores, about 330 GB peak). Measured job times:
  DistilBERT 33 min, Pythia 13 min, Qwen3 46 min (base plus FKeras), about 123 job-hours, roughly 12 to 14 hours of wall
  time. Progress log `paper1_runT_w8a8_logs/more.log`, registry entry 80.
- **Note (2026-11-16 19:27), early per-model run:** DistilBERT and Pythia finished all 16 draws (64/64 standard and
  FKeras jobs each) while Qwen3 was still running. At the owner's request, `runT_analyze_lm_v2.py` (unchanged, same seed)
  is run now with `--targets distilbert_sst2_int8 pythia160m_int8` into `workspace/runs/paper1_runT_t4_lm_analysis_dbpy_v1`.
  Each model is its own Holm family and the sign-flip matrix is reseeded per model, so these numbers are identical to
  those of the final three-model run, which still runs once after all 224 jobs; the final run is checked to reproduce them.
  Nothing about Qwen3 depends on these results: its draws were fixed at 16 in advance.
