# Partial-group protection search on the DS-CNN: plan (exploratory, then confirmatory; not run)

This plan follows `GROUP_SEARCH_PHASEA_RESULTS_2026-11-14.md`. Phase A searched over whole groups, where a group
is one layer at one bit position. It never beat abs1024 and lost by 2–5 points at 1e-2. The diagnosis was that a
whole group is too coarse: abs1024 decides weight by weight, while the search could only take or leave a group.

This plan keeps abs1024's order **inside** each group. It lets the network-pass search decide only **how much** of
each group to protect.

## 1. Question

At the same budget, does choosing how much of each layer × bit group to protect beat abs1024's own implicit choice?
- Protection is filled in abs1024 order within each group.
- Each choice is judged by network passes under real flips at the target rate.

This isolates a single difference. Both masks use the same per-weight order within a group, so the only thing
that differs is how protection is **allocated** across layers and bit positions.
- If the search wins, abs1024 allocates badly at that rate. That is a concrete case of a fixed ranking being the
  wrong choice.
- If it ties, abs1024's allocation is already about as good as a search with network passes can find at this
  granularity. A search that could have found something better and didn't is useful evidence for the paper.

## 2. Units and order

- **Groups:** the same 80 as in Phase A (10 layers × 8 bit positions).
- **Slices:** each group is split into S = 4 slices of nearly equal size, so sizes differ by at most one bit.
  - The slices are ordered by abs1024 score within the group, highest first. Ties go by storage order.
  - Slice sizes are 144 bits (depthwise), 192 (dense), 640 (first convolution) and 1,024 (pointwise).
- **Candidates in a round:** the next unprotected slice of every group, so at most 80. A group's second slice can
  only be taken after its first.
- **Scores:** recomputed from the replication's 1,024 calibration clips with `runQ_calibration.scores`, as in
  top-bit Phase 3.
  - The script rebuilds the stored abs1024 masks of that replication from these scores and stops unless 0 bits
    differ. The rebuild was exact on Quest with TF 2.14.1.
- **Partial fill at a budget:** the last accepted slice is filled in abs1024 order, not at random. Each budget
  therefore yields one mask. The Phase A "whole groups" variant is dropped.
- **Everything else is unchanged from Phase A:**
  - gain in accuracy per protected bit, paired by field;
  - ties broken by cross-entropy, then by group id;
  - new fault fields each round;
  - the path runs to the 37.5% budget, giving nested masks for 12.5%, 18.75%, 25% and 37.5%.

## 3. Settings, fixed before any run

| Setting | Value | Reason |
|---|---|---|
| Fault model | real flips (`mc`) only | they beat the Gaussian approximation in Phase A |
| Rates | 3e-3, 1e-2, 3e-2 | At 1e-3 every mask was within noise in Phase A (stated as a limit). 3e-2 is added because the masks differed most there in top-bit Phase 3. |
| K (fields per round) | 32 | Phase A setting at 3e-3 and 1e-2 |
| Clips per pass | first 512 calibration clips of the replication | Phase A setting |
| Slices per group | S = 4 | Phase C1 may add S = 8 at 1e-2 only (Section 5) |
| Seeds | search [20261215, rep, rate index, round, draw]; scores have no seed | new namespace, never used before |

## 4. Evaluation

The Stage 4 protocol is used for the replication being searched:
- engine check first;
- TFLite on test window `rep` (800 clips);
- fields [20261119, rep, f] for f = 0–7;
- each mask evaluated at the rate it was built for.

The search masks are paired field by field with the six existing Stage 4 masks of that replication. The script
also reports how far each search mask is from abs1024: the number of bits that differ, and the bits moved between
layers and bit positions. A search that gets back abs1024 shows up as 0 differing bits, not as a hidden tie.

## 5. Phases

### Phase 0: code and deterministic tests (no compute)

**`netsurf/group_search.py`:**
- add slices: a unit is (layer, bit, weight indices);
- `evaluate_round` takes units;
- `path_masks` fills the last unit in score order.

**`runR_group_search.py`:**
- `--rep`, `--slices`;
- the score recomputation and abs1024 rebuild check;
- evaluation and comparison for any replication.

**Tests:**
- slices partition each group exactly, sizes differ by at most 1, and score order is respected within a group;
- with S = 1 the new code reproduces Phase A's accepted order on the toy measure;
- a slice can only be taken after the slice before it;
- path masks are nested and have exact budget counts;
- full protection gives exactly the clean result;
- the rebuild check fails on a perturbed score array.

There is also a smoke check on Quest: one round with K = 2, logged, with a registry entry.

### Phase C1: development, replication 0 (one A100 per path, Quest)

**Cost estimate.** Phase A measured 15 ms per pass on 512 clips.
- One round is up to 81 × 32 ≈ 2,600 passes, about 40 s.
- Reaching 66,048 bits takes about 110–200 rounds. Small slices go first, so the count depends on the order.
- That is about **1.2–2.2 hours per path**.
- 3 paths run in parallel. The time limit is 4 hours, with a per-round guard of 120 s, three times the estimate.

**Evaluation:** 3 rates × 4 budgets = 12 masks × 8 fields, an 8-task CPU array of about 2 minutes per task.

**Optional within C1:** S = 8 at 1e-2. This is one more path of about 2–4 hours. It runs only if S = 4 at 1e-2
closes less than half of Phase A's gap to abs1024 at 12.5%, which is 3.3 points with the partial fill.

Replication 0 has already been inspected, so C1 is descriptive. Its role is to show whether finer units close
the gap before more GPU time is spent.

| C1 outcome | Next |
|---|---|
| Behind abs1024 by 2 SE in 2 or more cells | Stop. Within this family, network-pass allocation does not improve on abs1024. Report it, with C1 and Phase A. |
| Within 2 SE of abs1024 everywhere, or ahead somewhere | Phase C2 (separate go-ahead), with every setting frozen as run in C1 |

### Phase C2: confirmation, replications 1–5 (only with a separate go-ahead)

- **Runs:** 5 replications × 3 rates = 15 search paths, about 18–33 A100-hours in an array, plus 5 evaluation
  arrays.
  - Fallback if Quest time is short: replications 1–3, decided **before** any C2 run and stated in the job
    record.
- **Primary comparison, written now:** search minus abs1024, for each of the 12 rate × budget cells.
  - First average over the 8 fields within each replication, then over replications.
  - The standard error is taken across replications (n = 5, so 4 degrees of freedom). It therefore includes
    search randomness, calibration clips and test window.
  - Two-sided t-tests across the 12 cells, with Holm correction at 0.05.
- **Equivalence, written now:** in each cell, whether the 90% interval of the difference lies within ±1 point.
- **Secondary, descriptive:** search minus msb, minus FKeras, and minus ra_analytic. No "best fixed mask per
  cell" comparison is made, because choosing it after the fact favours the fixed side.
- Search uncertainty (the per-round standard errors), evaluation uncertainty (fields) and replication spread are
  reported separately.

## 6. How to read it

| Outcome | Meaning for the paper |
|---|---|
| Search ahead of abs1024 at 1e-2 or 3e-2 after correction | abs1024's allocation is wrong at high rates. Rate-dependent allocation judged by network passes fixes it. This is the "fixed ranking is a bad choice here" result, with a method. |
| Equivalent everywhere | abs1024 is near what this search can find at every tested rate. It is a strong default, and the search measures how much room there is. |
| Search behind | Greedy allocation under noisy gains loses to a fixed ranking. Report it, together with Phase A. |

## 7. Not claimed

- A greedy search is not optimal.
- Within-group order is abs1024's; the search does not choose it.
- One model (DS-CNN). ResNet-8 needs its own plan.
- Rates are logical bit-error rates, not physical upset rates.
- No area or power claims.
- C1 is descriptive. Only C2's primary comparison is confirmatory.
