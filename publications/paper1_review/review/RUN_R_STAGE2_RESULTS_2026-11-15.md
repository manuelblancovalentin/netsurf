# Run R, Stage 2 confirmatory results (15 November 2026)

**Model:** ResNet-8 (CIFAR-10), 9 checkpoints:
- the pinned MLPerf model;
- an under-trained model (100 epochs);
- two pruned models (70% and 50%);
- five retrained seeds.

**Protocol:** plan `RUN_R_RATE_AWARE_PLAN_2026-11-12.md`, Section 5 and its addendum, amendments R-A4 and R-A5, and
analysis note A2.

**How the analysis was run:** once, after every Stage 2 phase had ended (the chain finished 15 November, 00:39, with no
failed jobs).
- Script `workflows/runR_stage2_analyze.py`, frozen at 6c8b185 before any Stage 2 outcome was read. Its file is
  unchanged (sha256 `1b55b15d…`).
- The only later change to code it imports (764beb2) replaced fixed paths with one setting, with the same defaults on
  kona.
- The R-A5 choices file was checked against its frozen hash (`8c1a864f…`).
- Interpreter: `.venv-mlperf` (TensorFlow 2.14.1).

**Output:** `workspace/runs/paper1_runR_stage2_analysis_v1/` (`summary.json`, `cells.csv`, `cross_rate.csv`), locked.
SHA-256 sums and the code record are in `…_analysis_v1_logs/`.

## Decisions by the frozen rules

All tests are paired by fault field and compare accuracy points.
- A **violation** means an arm is ahead of the reference by at least 0.5 point, Holm p < 0.05 over the checkpoint's
  usable cells.
- A **usable cell** is one where the best arm is within 0.05 of clean accuracy.

| Rule | Question | Outcome |
|---|---|---|
| R2a | Is the analytic rate-aware mask (v1) never beaten by 0.5 point? | **Fails**: 17 violations on 6 of 9 checkpoints, by 0.54–1.16 points. The beating arm is the MC-corrected mask in 9, abs1024 or squared1024 on the pinned model at 5e-3 in 4, and pos1024 at 2e-3–3e-3 in 4. |
| R2b | Does each single static ranking fail (is it beaten) on at least 7 of 9 checkpoints? | **Holds**: abs1024 fails on 8 of 9; pos1024 fails on 9 of 9. |
| R2c | Does the MC correction help? | **Kept**: resolved ahead of v1 in at least one cell on all 9 checkpoints, behind in none. 54 of 450 cells are resolved ahead; the median difference is +0.07 point. |
| R2a′ (R-A5) | Is the selected rate-aware mask (v2) never beaten by 0.5 point by a static arm? | **Holds**: 0 violations on all 9 checkpoints. |

**Consequence under Section 9 of the plan:** R2a fails, so the v1 claim ("the rate-aware mask is the only method never
more than half a point behind") is **not allowed**, and the paper keeps the v14 claims for v1.

The v2 claim of R-A5 needs both R2a′ (ResNet-8) and R4d′ (DS-CNN).
- R2a′ holds.
- R4d′ failed narrowly in Stage 4: 3 violations against abs1024, of 0.50, 0.58 and 0.68 points.
- So the two-model v2 claim is **not allowed as written**.
- What may be said: on ResNet-8, v2 had no violation against any static mask; on DS-CNN it was at most 0.68 point
  behind abs1024 and at most 0.20 point behind every other static mask.
- Caveats:
  - v2 was designed after the v1 overfitting was seen on training-side data.
  - R-A5 was added without the researcher's explicit approval of the amendment itself.

## When a static ranking is a bad choice (R2b, by rate)

Violations against each static reference, counted over the 9 checkpoints:

- **abs1024 is beaten in 273 cells**, mostly at low rates: 42 at 3e-4, 83 at 1e-3, 79 at 2e-3, 44 at 3e-3, and 20 at
  5e-3 and above.
  - Median 0.87 point, largest 3.1.
  - It is beaten by the MC-corrected mask (113), the analytic mask (94) and pos1024 (66).
- **pos1024 is beaten in 228 cells**, mostly at high rates: 77 at 5e-3, 86 at 1e-2, 14 at 3e-2, and 51 at 3e-3 and
  below.
  - Median 4.1 points, largest 57.0 (under-trained model).
- **v2 is beaten in 0 cells.**

**Largest gap of each arm** to the best arm over usable cells, in points. This number is biased upward, because the best
arm is chosen on the same fields.

| Checkpoint | abs1024 | pos1024 | msb | fkeras | v1 | MC | v2 |
|---|---|---|---|---|---|---|---|
| pinned | 0.7 | 12.7 | 23.7 | 23.7 | 1.3 | 1.0 | 0.2 |
| under-trained | 3.1 | 57.0 | 52.3 | 27.1 | 1.2 | 0.1 | 0.8 |
| pruned 70% | 1.7 | 5.2 | 41.0 | 28.6 | 0.9 | 0.1 | 0.8 |
| pruned 50% | 1.5 | 6.7 | 32.6 | 27.1 | 0.6 | 0.3 | 0.4 |
| retrained (5 seeds) | 0.6–1.3 | 12.2–20.8 | 32.5–38.9 | 19.1–23.6 | 0.5–1.1 | 0.4–0.7 | 0.3–0.6 |

**Wrong-rate design** (descriptive, `cross_rate.csv`): a mask built for one rate and used at another can fail badly. On
the pinned model, the analytic mask built for 1e-4 is behind the best arm by:

| Tested at | Gap (points) |
|---|---|
| 1e-3 | 1.8 |
| 3e-3 | 20.3 |
| 1e-2 | 69.9 |

## Reading (one model family, ResNet-8)

- **No single fixed ranking is safe across the rate range on ResNet-8.**
  - pos1024 is fine at low rates and collapses at high rates, by up to 57 points.
  - abs1024 is safe at high rates but loses about a point (up to 3.1) at low rates.

  This is the advisor's question answered for this model: a static ranking is a bad choice when its built-in rate
  assumption is far from the operating rate.
- **Choosing among a few masks using the rate wins on ResNet-8.** Picking among v1, abs1024 and pos1024 with held-out
  training inputs (v2) is never beaten by half a point there.
- **Contrast with the DS-CNN** (Stage 4 and the partial-search work): abs1024 is hard to beat on that model across
  3e-3 to 3e-2. The two models differ in where the low-rate advantage of sign-aware or rate-aware masks matters.
- **Not supported:** the v1 rate-aware mask as the uniquely safe method (R2a fails); any physical fault-rate
  calibration; any claim about other models.
