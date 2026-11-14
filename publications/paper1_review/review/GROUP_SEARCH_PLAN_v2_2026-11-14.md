# Group-level protection search on the DS-CNN, plan v2 (exploratory, not run)

Status: PLAN ONLY. Nothing here is implemented or run, and each phase needs its own go-ahead.

Classification: EXPLORATORY. The Stage 4 DS-CNN outcomes have already been read.

This plan replaces `GROUP_SEARCH_PLAN_2026-11-13.md`, which was never run. The main change is that every
decision is now judged by passes through the network. No step relies on a per-bit score.

## 0. Why v1 changed

Four results since v1 shaped this version:

| Result | Source | Consequence |
|---|---|---|
| Measured single-flip effects do not add up. With more than about 10–20 top bits faulty, the additive prediction overstates damage by 3 to 50 points. | top-bit Phase 2 | Any guide that sums per-bit effects is out. That includes v1's linear-plus-curvature group order. |
| Ranking top bits by measured single-flip damage loses up to 60 points to abs1024. | top-bit Phase 3 | Isolated effects are a poor guide to value under many faults. |
| Gradient scores rank measured damage weakly, and no better within layers than across them. | top-bit Phase 3, Q4 | v1's partial fill "within the layer by abs1024 score" has no support. |
| The formula's predicted swaps fail under real flips in 54–85% of cases. | Stage 2 MC search records | The direction of a step must be measured, not predicted. |

What has worked is putting the faults on the weights and running the network:
- the companion replays match TFLite within 0.08 points per cell;
- the blur pilot's Gaussian and hybrid passes are within 0.5–0.9 points.

On Quest's A100 a full pass over 1,312 clips took about 14 ms (9,216 passes in 133 s). That makes it affordable
to judge every candidate this way.

## 1. Questions

- **Q1, value.** At each tested rate and budget, how does a greedy group search compare with the six Stage 4
  masks? The search judges each candidate by real-flip passes through the network. The Stage 4 masks are
  abs1024, pos1024, abs64, significance-first, FKeras and the analytic mask, and the comparison uses the same
  TFLite fault fields.
- **Q2, a cheaper guide.** Does the same search with Gaussian weight noise in place of real flips choose the
  same groups? Does it reach the same held-out accuracy? Gaussian noise is smoother, so it needs fewer draws.
- **Q3, what the search does.** Which layers and bit positions does it protect first at each rate? Does it
  rediscover significance-first, or depart from it at low or high rates?
- **Q4, saturation.** At each rate, where does the measured gain of the next group drop below its own Monte
  Carlo uncertainty?

## 2. Groups, budgets, partial fill

There are 80 groups (10 weight layers × 8 bit positions), with 576–4,096 bits each and 176,128 bits in total.
The budgets are those of Stage 4: 12.5%, 18.75%, 25% and 37.5%. Groups are added one at a time, so one search
path gives nested masks for all budgets.

**Partial fill (changed from v1).** When the next accepted group does not fit in the remaining budget, the
remaining bits are taken from that group **uniformly at random**, with seed [20261213, rep, rate index]. No
per-bit score is used, which Q4 no longer supports. The "whole groups only" mask (below budget) is saved and
evaluated too.

## 3. Method

The search inputs are the first 512 calibration clips of replication 0, from the training partition. These
are disjoint from the test window.

**One round of `group_mc`.** Let M be the current mask.
1. Draw K fault fields of uniform numbers, with seeds [20261211, rep, rate index, round, draw]. They are shared
   by every candidate in the round.
2. For M and for each unprotected group G, apply the field's flips to the bits left unprotected by M + G, and
   run the network on the 512 clips. That gives each candidate's accuracy and mean cross-entropy per field.
3. Compute each candidate's gain per protected bit, (accuracy(M+G) − accuracy(M)) / |G|, averaged over the K
   fields and paired by field. Ties are broken by cross-entropy, then by group order.
4. Accept the best candidate, and record every candidate's gain and paired standard error.

New fields are drawn each round, so the search is not tuned to one fixed set of fault patterns. That
overfitting was a Stage 2 v1 concern.

**Number of fields per round.** K = 32 at 3e-3 and 1e-2, and K = 128 at 1e-3. At low rates few bits flip per
field, and the Stage 2 records showed 32 fields could not resolve gains there. This was fixed before any run.

**Q4 statistic.** The first round in which the accepted gain is below one paired standard error is recorded.
It does not stop the path, which runs to the 37.5% budget.

**`group_gauss`.** The same loop, but each field is replaced by a Gaussian displacement of the unprotected bits
of every position: mean p·ΣΔ, variance p(1−p)·ΣΔ². This is the blur pilot's "gauss" variant, with K = 16 at
every rate and seeds [20261214, ...].

**Removal direction (optional, Phase A2).** Start from full protection and remove the group whose removal costs
the least per bit, until 37.5% remains. That is about 50 rounds. It runs only if Phase A shows the adding search
departs from significance-first.

**Arms evaluated:**
- `group_mc` and `group_gauss`, at the four budgets plus the whole-groups-only versions;
- the six Stage 4 masks, whose outcomes already exist.

v1's `group_linear` arm is dropped, since Phase 2 already shows that sums of per-bit terms fail.

## 4. Evaluation

The Stage 4 protocol is used unchanged:
- TFLite replay after the engine check;
- test window of replication 0 (800 clips);
- fields [20261119, 0, f] for f = 0–7.

The new masks are paired field by field with the Stage 4 masks' existing outcomes. No search clip or search
field is reused.

Search uncertainty (the per-round standard errors) is reported separately from evaluation uncertainty (the
spread over the 8 fields).

## 5. Phases and cost

### Phase 0: code and deterministic tests (no compute)

The script is `workflows/runR_group_search.py`. Tests:
- group masks have exact bit counts, and the path masks are nested;
- the partial fill is reproducible from its seed;
- protecting every group gives exactly clean accuracy;
- an already protected candidate has a paired gain of exactly 0;
- one search round on a 2-group toy layer equals a brute-force evaluation;
- the evaluation reproduces the Stage 4 engine check.

There is also a CPU smoke check of one round with K = 2 on kona or Quest, with a log and a registry entry.

### Phase A: replication 0, rates 1e-3, 3e-3 and 1e-2, one A100 on Quest

Passes per round are (candidates + 1) × K, on 512 clips. The time per pass, about 6–8 ms, is estimated from the
14 ms measured on 1,312 clips.

| Path | Rounds to 37.5% (estimate) | Passes | GPU time |
|---|---|---|---|
| `group_mc`, K = 32 (3e-3 and 1e-2), each | 20–35 | ≈ 50,000–90,000 | ≈ 7–12 min |
| `group_mc`, K = 128 (1e-3) | 20–35 | ≈ 200,000–350,000 | ≈ 25–45 min |
| `group_gauss`, K = 16, ×3 rates | 20–35 each | ≈ 25,000–45,000 each | ≈ 4–6 min each |
| **total** | | | **≈ 1–1.5 GPU-hours** |

**Evaluation:** 2 arms × 3 rates × (4 budgets + 4 whole-group masks) × 8 fields = 384 TFLite evaluations. That
is an 8-task CPU array on Quest, about 5 minutes per task (measured in top-bit Phase 3).

If a round takes more than 3 times its estimate, the job stops after that round and the plan is revised.

**Outputs:**
- `paper1_group_search_kws_v1/`: per-round rows, masks, DONE;
- `paper1_group_search_kws_eval_v1/`.

Both are write-once, locked and hashed. Every job gets a monitor entry of kind "slurm" and a live log.

### Phase B: only with a separate go-ahead, after Phase A is read

All 6 replications, with the rates and analysis rules written down before running. The ResNet-8 version needs
skip connections in the pass code; it is a full forward pass, so this is straightforward.

## 6. How to read Phase A

| Outcome | Meaning | Next |
|---|---|---|
| `group_mc` ahead of every Stage 4 mask at some rates, never clearly behind | judging masks with network passes at the target rate works where per-bit rankings fail | Phase B; this is the method candidate for the paper or its follow-up |
| `group_mc` ties significance-first or abs1024 everywhere | on this model the cheap jump-size-driven rules are already near the best at group granularity | report it: it supports using them, and the evidence is a search that could have found better |
| `group_mc` behind at matched budget | group granularity costs accuracy, or the greedy search misses combinations | try the removal direction; report the price of hardware-granular protection |
| `group_gauss` ≈ `group_mc` | Gaussian passes are enough to judge masks (cheaper, smoother) | use them in Phase B |
| `group_gauss` clearly worse | top-bit events must be simulated as real flips | keep real flips, or the hybrid (real top bits, Gaussian low bits) |

## 7. Not claimed

- A greedy search is not optimal.
- A group's gain in one round is not the effect of swapping groups.
- There is one model and one replication in Phase A.
- Rates are logical bit-error rates.
- There are no area or power claims.
