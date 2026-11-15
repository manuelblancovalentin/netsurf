# Partial-search follow-up: pre-registered results (15 November 2026)

Pre-registration: `PARTIAL_SEARCH_FOLLOWUP_PREREG_2026-11-14.md` (49844d4), committed before any run existed.
Code: f478010 on branch `paper1-runR-v2-validated`. Requested by the researcher ("Run them").

**Setting:** DS-CNN at fault rate 1e-2, replications 1–5, 4 budgets.

## Runs

Quest jobs, all completed:

| Job | What | Tasks | Time per task |
|---|---|---|---|
| 8771990 | Greedy search, K = 32 | 5 | 1.8–2.6 h |
| 8771997 | Greedy search, K = 128 | 5 | 7.0–7.5 h |
| 8772000 | Swap search | 20 | 3–6 min |
| 8772001–8772003 | Stage 4 evaluation | 120 | — |

- All 120 evaluation tasks passed the engine check (0.85 = 0.85).
- Results were copied back over one connection; all files match the checksums taken on Quest.
- **Outputs**, locked under `workspace/runs/`:
  - `paper1_psearch_d_{k32,k128,swap}_v1` and `paper1_psearch_d_{k32,k128,swap}_eval_v1`;
  - logs and Quest checksums in `paper1_psearch_d_logs`.
- **Analysis:** `runR_psearch_followup_compare.py`, unchanged since f478010, run once with `.venv-mlperf`. Output in
  `paper1_psearch_d_compare_v1`.

**A flaw in the script's descriptive record:** the K = 32 and K = 128 search folders have the same names
(`rep<R>_mc_0.01_s4`), so in `compare.json` → `searches` the K = 128 records overwrite the K = 32 ones. The tests are
not affected. The per-arm records below were read straight from the locked search folders.

## Pre-registered tests

The table gives arm − comparator in accuracy points: the mean over 5 replications ± standard error, with Holm-adjusted
p and the 90% interval.

**Primary family:** each arm minus abs1024, 12 tests, Holm.

| Budget | K = 32 | K = 128 | Swap |
|---|---|---|---|
| 12.5% | −0.76 ± 0.43 (p 0.93) | −0.57 ± 0.51 (p 1.0) | +0.06 ± 0.08 (p 1.0) |
| 18.75% | −1.41 ± 0.25 (p 0.058) | −0.75 ± 0.21 (p 0.19) | 0.00 |
| 25% | −0.75 ± 0.14 (p 0.069) | −0.45 ± 0.21 (p 0.72) | 0.00 |
| 37.5% | −0.37 ± 0.08 (p 0.089) | −0.33 ± 0.07 (p 0.097) | 0.00 |

- **No cell is resolved**, ahead or behind.
- **K = 32:** all four point estimates are below abs1024. The unadjusted 90% intervals exclude zero at 18.75%, 25% and
  37.5%, but no cell passes after Holm with 5 replications.
- **K = 128:** within ±1 point at 25% and 37.5%. It is still below abs1024 in every budget.
- **Swap:** within ±1 point in all four budgets.

**Selection-noise family:** K = 128 minus K = 32, 4 tests, Holm.

| Budget | K = 128 − K = 32 |
|---|---|
| 12.5% | +0.19 ± 0.38 (p 1.0) |
| 18.75% | +0.66 ± 0.19 (p 0.097) |
| 25% | +0.30 ± 0.15 (p 0.33) |
| 37.5% | +0.05 ± 0.09 (p 1.0) |

No budget is resolved. All four point estimates favour K = 128.

## Descriptive

**Greedy searches.** Bits differing from abs1024 at 12.5%:

| Search | Differing bits |
|---|---|
| K = 32 | 3,838–4,336 |
| K = 128 | 2,831–3,481 |

With more fault samples, the search ends closer to abs1024.

**Saturation:** the round where the accepted gain first falls below its own standard error.

| Search | Round | Protected bits at that point |
|---|---|---|
| K = 32 | 47–122 | 17,000–42,000 |
| K = 128 | 77–154 | 33,000–52,000 |

**Swap search.**
- 64 swaps were proposed across the 20 searches.
- **Predicted gain** on the proposal fields: mean +0.15 points, range +0.04 to +0.29.
- **Gain on fresh fields** for the same swaps: mean −0.007 points, range −0.14 to +0.15; 38 of the 64 were negative.
- **Accepted:** 2 swaps, both at 12.5%, both moving 256 bits into the dense layer's sign bit:
  - pw2 bit 5 → dense bit 7 (replication 1);
  - conv1 bit 6 → dense bit 7 (replication 5).
- 18 of the 20 searches stopped after 3 rejections, so their result is exactly abs1024.

**Against the other masks**, at 12.5%:

| Comparison | Lead (points) |
|---|---|
| Greedy searches vs significance-first (msb) and FKeras | +4.1 to +4.3 |
| Swap vs msb and FKeras | +4.9 |
| Searches vs the analytic rate-aware mask, all budgets | +0.5 to +1.9 |

## Reading by the pre-registered rules

- **K = 128 does not beat K = 32 in any budget after Holm.** By the pre-registered rule, this counts against noise
  as the main cause, and greedy short-sightedness or abs1024 being near a local optimum lead.
  - Stated with the descriptive context: every point estimate favours K = 128 (largest +0.66 at 18.75%), and K = 128
    lands closer to abs1024.
  - So noise plausibly contributes, but 4 times more samples does not close the gap in a way 5 replications can
    resolve.
- **The swap search is the cleaner test, and its answer is clear.**
  - The swaps that look best on 32 fault fields lose all their predicted gain on fresh fields (+0.15 predicted, −0.007
    measured). This is the winner's curse, measured directly.
  - Once that is guarded against, almost no exchange of 256 bits between layer × bit groups improves abs1024 at 1e-2.
  - By the pre-registered rule: **abs1024 is locally hard to improve at 1e-2 on the DS-CNN.** This supports the
    paper's statement that abs1024 holds up at this rate.
- **Scope:** one model, one rate; the swap moves are 256-bit exchanges between groups in abs order. The replication 0
  conclusions (Phase A, C1) are unchanged.
- **For the paper:**
  - On the DS-CNN, searches judged by real network passes did not beat abs1024 at 1e-2, and the gains they appeared to
    find were largely selection noise.
  - On ResNet-8 (Stage 2), fixed rankings do fail across rates: abs1024 at low rates, pos1024 at high rates.
  - Together: whether a fixed ranking is a bad choice depends on the model and on how far the operating rate is from
    the rate its ranking implicitly assumes.
