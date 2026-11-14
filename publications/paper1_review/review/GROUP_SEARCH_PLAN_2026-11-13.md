# Group-level protection search on the DS-CNN: plan (exploratory, not run)

Status: PLAN ONLY. Nothing here has been implemented or run. Every run stage below needs its own go-ahead.
Classification: EXPLORATORY. The Stage 4 DS-CNN outcomes have already been read, so any comparison against
them is post hoc and is not a confirmatory result.

## 1. Question

Can protection be chosen in whole groups, where a group is one layer's bit position (all bit-7 cells of
one layer, for example)? The search would use:
- the two-moment Gaussian background as a cheap guide;
- real-flip Monte Carlo only to confirm the few decisions it makes in each round.

Would that give masks at least as good as the static rankings across the fault-rate range, without
simulating every bit?

The question has four parts:

- **Q1, guide accuracy.** Does the Gaussian guide order candidate groups the same way real flips do?
  The blur pilot showed it reproduces accuracy for whole masks. It has not been tested on the difference
  between two masks.
- **Q2, value.** At each tested rate and budget, how does the group search's held-out accuracy compare
  with the best static arm (significance-first, abs1024, pos1024, abs64, FKeras) and with the Stage 4
  analytic mask, on the same fault fields?
- **Q3, cost of the confirmation.** Does the Monte Carlo confirmation step change the result compared
  with the Gaussian guide alone?
- **Q4, does the search matter at all?** Does a fixed group order from the cheap linear-plus-curvature
  formula, applied per group instead of per bit, do as well as either search? The single-bit checks showed
  this formula fails for individual bits. Whether the failures average out over a whole group is untested.

The question also covers one descriptive output: at each rate, how many protected bits it takes before the
next group's measured gain is smaller than its own Monte Carlo uncertainty (the saturation point).

## 2. Groups and budgets

The DS-CNN has 10 weight layers × 8 bit positions, which makes **80 groups**. All counts below were
checked against the loaded model.

| Layer(s) | Weights | Bits per group |
|---|---|---|
| first convolution (kernel 17) | 2,560 | 2,560 |
| 4 depthwise (kernels 5, 8, 11, 14) | 576 each | 576 |
| 4 pointwise (kernels 18–21) | 4,096 each | 4,096 |
| dense (kernel 16) | 768 | 768 |
| **total** | **22,016** | **176,128 bits** |

The budgets are the Stage 4 budgets: 12.5%, 18.75%, 25% and 37.5% of all bits. 12.5% is exactly one bit
position across the whole network; significance-first at 12.5% protects every sign bit.

Groups are added one at a time, so a single search path gives nested masks for all four budgets. Each
budget's mask is read off the path at the point the budget is reached.

**Partial fill.** When the next chosen group does not fit in the remaining budget, the remaining bits are
taken from that group, in the order of the existing abs1024 calibration scores within the layer. The
masks then use exactly the same number of bits as the static arms. A "whole groups only" mask is also
saved and reported, with its actual bit count.

## 3. Method

### 3.1 Search (per replication and rate)

Notation for one round:
- M is the current mask.
- For each unprotected group G, M+G is M with G protected.
- Each candidate is scored by its gain per protected bit: (accuracy(M+G) − accuracy(M)) / |G|.

1. **Gaussian guide.** For every unprotected group, compute the accuracy under M+G and under M, using
   D_g = 8 Gaussian-background draws. Each weight's displacement is drawn as mu + sqrt(V)·eps, with the
   exact mean and variance over its unprotected bits (as in the blur pilot). The eps vectors are the same
   for all candidates in a round, so the differences are paired. To save time, the layer inputs of each
   draw are cached, and each candidate recomputes only from its own layer onward. This is exact, for the
   same reason as in the top-bit job.
2. **Monte Carlo confirmation.** Take the 3 groups with the highest guided gain per bit. For each one,
   compute the real-flip accuracy of M+G and of M with K = 32 draws. Every candidate uses the same uniform
   draws, so the fault field is identical and only the mask differs. Record the paired mean gain and its
   standard error.
3. **Accept** the confirmed candidate with the highest measured gain per bit, and record:
   - the guided and measured gains of all three candidates (this is the Q1 diagnostic);
   - the standard error;
   - the running bit count.
4. Repeat until the 37.5% budget is reached.

The saturation point is recorded, but it does not cut the path short: the first round in which the
accepted group's measured gain is below one standard error.

Two settings were fixed in advance as compute settings, not tuned: confirming 3 candidates per round, and
the draw counts D_g = 8 and K = 32. The search reports how often the Monte Carlo order differs from the
guided order. If it differs often, the guide is weak, and that is itself a finding for Q1.

### 3.2 Arms

| Arm | What it is | Extra search cost |
|---|---|---|
| `group_mc` | guide plus confirmation (3.1) | full |
| `group_gauss` | the same loop, accepting the top guided group with no Monte Carlo step (Q3) | guide only |
| `group_linear` | a fixed group order by gain per bit, from the old two-moment formula summed over the group: p·Σ ḡ_w Δ_wb + ½ p(1−p)·Σ F_w Δ_wb² (Q4); this is the "analyze the linear part" idea at group level | one backward pass |
| existing Stage 4 arms | msb (significance-first), abs1024, pos1024, abs64, fkeras, ra_analytic (built for that rate) | none (already built) |

`group_linear` is not rate-blind: the p-weighting changes the order.

### 3.3 Data and seeds (fixed before any group outcome is seen)

- **Search inputs:** the first 512 calibration clips of the replication, from the training "ranking"
  partition at offset 1,024·rep. These are the clips the blur pilot used, and they are disjoint from the
  test window.
- **Search seeds:** Gaussian eps [20261210, rep, rate index, round, draw]; Monte Carlo uniforms
  [20261211, rep, rate index, draw]. Both are new and distinct from every evaluation seed.
- **Evaluation:** the Stage 4 protocol unchanged. The TFLite engine replay, after the same engine check;
  test window rep (800 clips); fields 0–7 with seeds [20261119, rep, field]. The new arms are therefore
  paired, field by field, with the Stage 4 arms' existing outcomes. The existing Stage 4 rows are read,
  never rewritten.
- **Separation:** search noise (guide and confirmation draws, calibration clips) and evaluation noise
  (fields, test clips) are reported separately. No search draw is reused for evaluation.

## 4. Phases

### Phase 0: code and deterministic tests (no compute run)

A new script, `workflows/runR_group_search.py`, in the v2 worktree, with these tests:
- Group masks have exact bit counts, and the path masks are nested.
- With every group protected, the Gaussian guide and the Monte Carlo step both return the clean accuracy
  exactly.
- A candidate group that is already protected gives a paired gain of exactly 0.
- The cached downstream path equals the full companion forward under a background displacement. This is
  the same check as in the top-bit job, run eagerly.
- The partial-fill mask has exactly floor(f·176,128) bits.
- The evaluation reproduces the Stage 4 engine check.

### Phase A: pilot (exploratory), replication 0, three rates

- **Rates:** 1e-3, 3e-3 and 1e-2, from the Stage 4 grid. They span its range and were fixed here.
- **Search:** `group_mc` and `group_gauss` at each rate, which makes 6 search paths. `group_linear` has
  no search.
- **Evaluation:**
  - 3 new arms × 3 rates × 4 budgets = 36 evaluations per field;
  - 8 fields;
  - 288 TFLite evaluations in total.

**Outputs** (write-once, then locked, hashed and registered):
- `workspace/runs/paper1_group_search_kws_v1/` for the search paths: per-round rows, masks, and a DONE
  file per path;
- `workspace/runs/paper1_group_search_kws_eval_v1/` for the evaluation;
- `workspace/runs/paper1_group_search_kws_v1_logs/` for the progress logs. There is one line per round,
  with the round number, bits protected, accepted group, guided and measured gains, and elapsed and
  estimated remaining minutes. Every job gets a registry entry with a label of at most 48 characters.

**Resource estimate:**

| Item | kona CPU (4 threads) | Quest A100 |
|---|---|---|
| one forward pass, 512 clips | ~2.1 s (blur-pilot timing, unloaded machine) | to be measured by the top-bit job's check phase (expected tens of ms) |
| one round of the Gaussian guide (≤80 candidates × 8 draws, downstream-only) | ~10–12 min | ~0.5 min (if ~30 ms per pass) |
| Monte Carlo confirmation per round (3 × 32 draws, plus the current mask) | ~4 min | ~0.1 min |
| rounds per path (to 37.5%) | about 20–35 | same |
| **one `group_mc` path** | **~5–9 h** | **~15–25 min** |
| **Phase A search total (6 paths)** | **~30–50 CPU-job-hours** | **~1.5–2.5 GPU-hours** |
| Phase A evaluation (288 TFLite runs at ~5.7 s each) | ~30 min | (CPU job) |

**Where to run it.** Kona currently runs Stage 2 at about 85 load on 80 cores. Phase A on kona would slow
Stage 2 and take most of a day, so it should run either:
- on Quest, once p33112 is active (GPU for the search, CPU for the evaluation); or
- on kona, after the Stage 2 Monte Carlo and its evaluation finish.

The real GPU time per pass will be known from the top-bit job's check phase before Phase A is submitted.
If one Phase A round takes more than 3× the estimate, the job stops after that round and the plan is
revised.

**What Phase A reports** (descriptive only: one replication, post hoc):
- **Q1:** per round, the rank agreement between guided and measured gains over the 3 confirmed
  candidates, and how often the top guided group was not the top measured one.
- **Q2–Q4:** per rate and budget, held-out accuracy of every arm. Paired differences against the best
  static arm and against `ra_analytic`, with the spread across the 8 fields (evaluation uncertainty).
  Search uncertainty is reported separately, as the confirmation standard errors.
- **The saturation point** per rate.
- **The accepted group order** per rate: which layers and bit positions, and in what order. This shows
  whether the search simply rediscovers significance-first (all sign bits first), or does something else
  at low and high rates.

### Phase B: only with a separate go-ahead, after Phase A is read

All 6 replications, and a rate set and analysis rules written down before running: rates, budgets, primary
comparison, and how the paired tests are combined. A ResNet-8 version needs the downstream cache adapted
to skip connections.

## 5. How to read the outcomes

| Outcome in Phase A | What it would mean | Next step |
|---|---|---|
| `group_linear` ≈ searches ≈ or better than the best static arm | the cheap formula works at group level even though it fails per bit; no search needed | Phase B with `group_linear` as the main method; scales to large models with one backward pass |
| searches better than `group_linear` and than static, `group_gauss` ≈ `group_mc` | the Gaussian guide is enough; Monte Carlo confirmation is unnecessary | Phase B with `group_gauss` (cheaper) |
| `group_mc` better than `group_gauss` | the guide misorders some decisions; confirmation matters | Phase B with `group_mc`; report the guide's failure cases |
| no group arm beats significance-first anywhere | at group granularity, protecting sign bits first is already near the best; the gains of per-bit rankings come from within-layer choices | supports the "when does static ranking fail" framing; within-layer refinement becomes the question (top-bit data) |
| group arms worse than per-bit static arms at the same budget | whole-group granularity costs accuracy on this model | report as the price of hardware-granular protection |

## 6. Not in this plan

- **Removing protection, starting from full.** It needs about 50 rounds to come down to 37.5%, which is
  2–3× the cost. It will be added only if Phase A shows path effects.
- **Within-layer choice of individual bits.** This waits for the exhaustive top-bit measurement on Quest,
  which tests whether cheap scores rank bits correctly within a layer.
- **Claims.** The search is one-at-a-time, so its masks are not claimed to be optimal. A group's measured
  marginal gain is not the effect of swapping it with another group. Nothing here is a physical fault rate
  (all rates are logical bit-error rates), and there are no area or power claims.
