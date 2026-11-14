# Group search, Phase A results (14 November 2026, EXPLORATORY)

Plan: `GROUP_SEARCH_PLAN_v2_2026-11-14.md`, Phase A. The Stage 4 outcomes of replication 0 had already been read,
so everything here is descriptive.

## Runs

All runs are locked under `workspace/runs/`; the hashes are in the `_logs/` folders.

**Search: Quest array 8676915** → `paper1_group_search_kws_v1` (code 8b839b1)
- Six paths, all of which completed the 37.5% budget. No path hit the cost guard.
- Real flips (mc):
  - 1e-3 with K 128: 45 rounds, 87 min;
  - 3e-3 with K 32: 46 rounds, 23 min;
  - 1e-2 with K 32: 43 rounds, 22 min.
- Gaussian (gauss), K 16: 11–12 min each.
- About 15 ms per network pass on an A100.

**Evaluation: Quest array 8676916** → `paper1_group_search_kws_eval_v1`
- Uses the Stage 4 protocol: TFLite, 8 fields, 800 test clips.
- Each mask is evaluated at the rate it was built for.
- Every engine check matched the saved trial (0.85 = 0.85).

**Transfer:** the 42 files were copied in one connection, and every SHA-256 matched the sums computed on Quest.

**Comparison:** `paper1_group_search_compare_v1`, made with `workflows/runR_group_search_compare.py` (22a2399).

## Result: group search does not beat abs1024

The tables give the mean paired difference over 8 fields, search mask minus Stage 4 mask, in accuracy points. A
positive value means the search mask is better. Standard errors are in `compare.json`; they range from 0.1 to
2.4 points.

Real-flip search, full mask (partial last group filled at random):

| Rate | Budget | vs abs1024 | vs msb | vs ra_analytic |
|---|---|---|---|---|
| 1e-3 | 12.5% | −0.2 | −0.2 | +0.2 |
| 1e-3 | 25% | +0.2 | +0.3 | +0.6 |
| 3e-3 | 12.5% | −1.7 | −0.6 | −1.1 |
| 3e-3 | 25% | −0.6 | −0.6 | +0.5 |
| 1e-2 | 12.5% | −3.3 | +2.2 | −1.7 |
| 1e-2 | 18.75% | −2.5 | +0.1 | +0.3 |
| 1e-2 | 25% | −1.1 | +0.4 | +2.2 |
| 1e-2 | 37.5% | −0.8 | −0.5 | +1.7 |

Summary over each search's 12 budget-rate cells:

| Search | vs abs1024: ahead / behind by 2 SE | Mean | Worst | vs ra_analytic: ahead / behind by 2 SE |
|---|---|---|---|---|
| mc, full | 0 / 3 | −0.9 | −3.3 | 5 / 0 |
| mc, whole groups | 0 / 4 | −1.1 | −5.4 | 5 / 0 |
| gauss, full | 0 / 3 | −1.3 | −4.7 | 4 / 0 |
| gauss, whole groups | 0 / 4 | −1.7 | −5.4 | 3 / 0 |

- **No search mask is ahead of abs1024 by two standard errors in any cell.** All the cells where a search mask
  is behind are at 1e-2, or at 1e-2 and 3e-3.
- **The searches beat the analytic rate-aware masks** (ra_analytic) by about 2 points at 1e-2 for budgets of
  25–37.5%.
- **Real flips beat the Gaussian approximation** by about 0.4 points on average. At 1e-2 and 18.75% the gap is
  2.2 points.

## Why the search loses: groups are too coarse

Protected bits by position at the 12.5% budget (22,016 bits):

| Mask | Bit 7 (sign) | Bit 6 | Bit 5 | Bit 4 |
|---|---|---|---|---|
| abs1024 | 13,615 | 6,516 | 1,742 | 143 |
| msb / fkeras | 22,016 | 0 | 0 | 0 |
| mc 1e-2 | 17,920 | 4,096 | 0 | 0 |
| mc 1e-3 | 17,920 | 2,496 | 1,152 | 0 |
| gauss 1e-2 | 17,920 | 4,096 | 0 | 0 |

- **Every search leaves all 4,096 sign bits of the last pointwise layer (pw4) unprotected at 12.5%.** It spends
  that budget on whole bit-6 or bit-5 groups instead. This matches Phase 3: single sign-bit flips in pw4 lowered
  the loss on average.
- **abs1024 chooses weight by weight.** It skips the sign bits of weights whose gradient is small, in every
  layer, and protects bit 6 of the weights that matter. A group search can only take or leave a whole layer's bit
  position. At 1e-2 and 12.5%, that costs 3.3 points against abs1024. The search still beats protecting sign bits
  only (msb) by 2.2 points.
- **The selected group order is stable across rates and kinds:**
  1. sign bits of the four depthwise layers;
  2. their bit 6, the dense sign bits, and the first-convolution sign bits;
  3. the sign bits of pointwise layers 1–3;
  4. pw4's sign bits, at round 18 to 41, or not at all at 1e-3 (gauss).
- **Saturation:** the first round whose accepted gain is below its own standard error.

  | Path | Round | Protected bits |
  |---|---|---|
  | gauss 1e-3 | 6 | 7,552 |
  | gauss 3e-3 | 7 | 13,824 |
  | mc 3e-3 | 20 | 34,048 |
  | mc 1e-3 | 28 | 42,368 |
  | gauss 1e-2 | 31 | 47,616 |
  | mc 1e-2 | 34 | 58,368 |

  After that round, choices between groups are mostly noise at the K used. At 1e-3 and 3e-3, the differences
  from every Stage 4 mask are within noise anyway.
- **At 1e-2 and 12.5%, mc and gauss give identical results** (−3.28/−3.27 full, −5.42 whole). Both paths took the
  same groups up to that budget.

## What this changes

- The evaluation half of the plan works:
  - real-flip judging beats the analytic formula (ra_analytic) where that formula is weakest;
  - it beats the Gaussian displacement;
  - it costs about 15 ms per pass.
- The group granularity is the problem. A layer × bit-position group is too coarse to compete with a
  weight-by-weight ranking. The cheap per-weight gradient ordering carries information that whole-group decisions
  throw away.

  Phase 3 found that gradient scores rank isolated single-flip damage weakly. Here, under many faults, they still
  choose better *which weights within a group* to protect than taking the group whole. These two findings do not
  contradict each other: Phase 3 showed that isolated damage is the wrong target.
- A natural next step, not run and needing a go-ahead: let the search decide **how much** of each group to
  protect, in steps such as ¼ of a group, filled in abs1024 order within the group. Then compare that against
  abs1024 under the same protocol on a different replication. The comparison must be on a different replication
  because replication 0 has now been inspected.
- **For the paper:** a network-pass search over coarse groups does not beat the best fixed gradient ranking on
  this model. This is one more piece of evidence that abs1024 is a strong baseline across the whole rate range,
  and that the analytic rate-aware masks are the weakest of the "smart" options at 1e-2. One model, one
  replication, post hoc.
