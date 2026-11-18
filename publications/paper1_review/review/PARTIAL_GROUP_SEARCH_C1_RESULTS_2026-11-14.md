# Partial-group search, Phase C1 results (14 November 2026, EXPLORATORY)

Plan: `PARTIAL_GROUP_SEARCH_PLAN_2026-11-14.md` (commit 29600dd). Replication 0 had already been inspected, so C1
is descriptive. The comparison script, including the C2 analysis, was committed before any C1 outcome existed
(ee42ed0). Run during the researcher's 24-hour absence, under their standing permission.

## Runs

All runs are locked under `workspace/runs/`; the SHA-256 sums are in the `_logs/` folders.

- **Scores:** CPU array 8686563 → `paper1_partial_search_scores_v1`. For every replication 0–5, the abs scores
  rebuild the stored abs1024 masks exactly (0 differing bits at all four budgets).
  - The first smoke run (8685063) computed the scores on the GPU. It stopped at that check with 85–100 differing
    bits, because GPU summation order reorders near-tied scores. Scores have been computed on CPU since then
    (fafddfa).
- **Smoke:** 8686564 → `paper1_partial_search_smoke_v2`. One round, about 3 s with K = 2.
- **Search:** A100 array 8686569 → `paper1_partial_search_kws_v1`.
  - Rates 3e-3, 1e-2 and 3e-2; 4 slices per group; K = 32.
  - 186–193 rounds and 1.8 hours per path, at 16 ms per pass.
- **Evaluation:** array 8686570 → `paper1_partial_search_kws_eval_v1`. Stage 4 protocol, 8 fields; every engine
  check matched (0.85 = 0.85).
- **Comparison:** `paper1_partial_search_compare_c1_v1`.
- **Diagnostic:** 8701858 → `paper1_partial_search_insample_c1_v1` (4a39855).

## Result: finer units remove Phase A's large losses but still do not beat abs1024

The table gives search minus abs1024, in points, as the mean over 8 fields ± standard error. The test window has
800 clips.

| Budget | 3e-3 | 1e-2 | 3e-2 |
|---|---|---|---|
| 12.5% | −0.6 ± 0.7 | +0.1 ± 1.5 | 0.0 ± 3.4 |
| 18.75% | −1.2 ± 0.8 | **−1.3 ± 0.6** | −0.4 ± 1.5 |
| 25% | −0.2 ± 0.3 | **−1.4 ± 0.3** | **−0.9 ± 0.4** |
| 37.5% | −0.2 ± 0.1 | −0.3 ± 0.2 | −0.4 ± 0.6 |

The cells in bold are behind abs1024 by more than 2 standard errors. No cell is ahead.

- **Against Phase A:** at 1e-2 and 12.5%, the whole-group search was 3.3 points behind abs1024. With quarter
  slices it ties.
- **Against the other masks:** at 3e-2 the search beats pos1024 by 15–27 points, significance-first and FKeras by
  5–14, abs64 by 4–9, and the analytic rate-aware masks by 12–14. These are the fixed and rate-aware masks that
  fail at high rates. abs1024 does not.
- **Decision (plan rule):** 3 cells are behind abs1024 by 2 standard errors, and the rule stops at 2. **C2 does
  not run.** The optional S = 8 path does not run either: the 12.5% gap at 1e-2 closed fully.

## Why: the losses are not over-fitting to the search clips

The search masks differ from abs1024 in 3,700–15,000 bits. Compared with abs1024, they:
- put 1.5–2.2 times more protection into the four small depthwise layers (5,760–8,512 bits at 12.5%, against
  3,830);
- protect more of bits 4–5.

The diagnostic re-measured every search mask against abs1024 and significance-first:
- under 64 new real-flip fields per cell (seeds [20261216, …]);
- on the 512 clips the search used, and on the 512 calibration clips it never saw.

| Rate | Search clips: minus abs1024 | Unseen clips: minus abs1024 |
|---|---|---|
| 3e-3 | −0.15 to −0.27 (every budget ≥ 2.5 SE behind) | −0.14 to −0.42 |
| 1e-2 | −0.21 to −0.56 (every budget ≥ 1.9 SE behind) | −0.32 to −0.56 |
| 3e-2 | −0.08 to +0.38 (within noise) | −0.38 to +0.69 |

- **The search loses to abs1024 even on its own clips** at 3e-3 and 1e-2. The two clip sets agree closely, so
  this is not over-fitting to particular clips.
- **The allocation the greedy search builds is worse in itself.** The likely cause is selection noise, and the
  search's own records are consistent with it. Each round picks the largest of about 80 gains, each estimated
  from 32 fields, so the picked gain is biased upward.
  - The accepted gain falls below its own standard error by round 5 (3e-3) and round 7 (3e-2), at about 2,000
    protected bits. Most later choices are made in noise.
  - Small slices make this worse. A 144-bit depthwise slice with a lucky gain has a large gain per bit, which
    fits the depthwise excess.

## What this means

- **For the paper:** abs1024 is hard to beat on this model across 3e-3 to 3e-2. The two network-pass searches did
  not beat it:
  - Phase A, with whole groups;
  - C1, with slices in abs1024 order.

  They do beat the fixed masks that do fail at high rates (pos1024, significance-first, FKeras, abs64) and the
  analytic rate-aware masks, all by large margins. A defensible statement: *"when fixed rankings fail, it is the
  ones that ignore the gradient magnitude (significance-first) or its sign structure (pos1024), not abs1024."*
  One model, one replication, post hoc.
- **Not run, and needs the researcher's decision:**
  - K = 128 at 1e-2 (about 7 A100-hours), to test the selection-noise explanation directly;
  - a search that starts from abs1024 and only swaps slices.

  Both are fishing for a win against abs1024 on an already-inspected replication. They should be pre-registered
  for replications 1–5 if pursued.
