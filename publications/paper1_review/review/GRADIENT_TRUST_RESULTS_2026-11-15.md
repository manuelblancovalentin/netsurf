# Gradient-trust test: results (15 November 2026)

**Plan:** `GRADIENT_TRUST_PLAN_2026-11-15.md`, with amendment A1.

**Predictions:** frozen before any fault run. `GRADIENT_TRUST_PREDICTIONS_2026-11-15.json`, sha256 `3c720998…`,
commit 7f4fedb.

**Fault runs:**
- Quest arrays 8855956–8855966: 11 checkpoints × 48 tasks, all completed.
- 1,584 files, all matching their Quest checksums.
- Output `paper1_gradtrust_faults_v1`, locked.

**Analysis:**
- `runG_gradtrust_analyze.py`, unchanged since 07e79ff, run once.
- Output `paper1_gradtrust_analysis_v1`, locked.

## Decision by the pre-registered rule: FAILED (1 of 9 predictions matched)

| Checkpoint | D1 | Predicted | Measured | Usable cells | pos ahead / abs ahead (≥0.2 point, Holm) |
|---|---|---|---|---|---|
| jet-e1 | 0.25 | one regime | one regime | 25 | 0 / 0 |
| jet-e3 | 0.51 | two regimes | one regime | 22 | 0 / 1 |
| jet-e10 | 0.42 | two regimes | one regime | 21 | 0 / 1 |
| jet-e30 | 0.51 | two regimes | one regime | 20 | 0 / 0 |
| jet-e100 | 0.40 | two regimes | one regime | 14 | 0 / 0 |
| mnist-e1 | 0.55 | two regimes | one regime | 26 | 0 / 0 |
| mnist-e3 | 0.61 | two regimes | one regime | 26 | 0 / 0 |
| mnist-e10 | 0.32 | no prediction | one regime | 26 | 0 / 0 |
| mnist-e30 | 0.42 | two regimes | one regime | 26 | 0 / 1 |
| vww | 0.94 | two regimes | one regime | 26 | 0 / 0 |
| mnist-tcad (exploratory) | 0.56 | two regimes | two regimes | 26 | 1 / 0 |

**The signed ranking (pos1024) was never usefully ahead of abs1024 on any test checkpoint.** This includes VWW, whose
average gradient is the most reliable measured in this project (D1 0.94).
- Across the low rates (≤ 3e-3), the mean of pos − abs is between −0.03 and +0.06 point on every checkpoint.
- The largest single gap is +0.38 point (jet-e30, 3e-3, 12.5%), with Holm p 0.58.
- Where the two rankings separate, abs1024 is ahead: up to 1.0 point on the jet models at 1.5e-2, and 2.6 points on
  mnist-e30 at 3e-2.

The agreement check therefore **does not predict** whether the signed ranking helps. A high agreement did not
produce a two-regime model in 7 of 7 cases.

## Why: little room left to win (descriptive)

At 12.5% of bits protected, both gradient rankings leave most of these models almost untouched:

| Model | abs1024 accuracy at 1e-3 / 1e-2 / 3e-2 | Significance-first (msb) | Clean |
|---|---|---|---|
| VWW | 83.9 / 84.0 / 83.9 | 81.9 / 70.8 / 60.0 | 84.0 |
| mnist-e10 | 95.7 / 95.5 / 95.1 | 95.5 / 93.4 / 87.3 | 95.7 |
| jet-e30 | 75.3 / 64.3 / 42.1 | 74.5 / 62.5 / 46.0 | 76.1 |

- **VWW and MNIST:** with either gradient mask, the faults cost a few tenths of a point at most. A signed ranking
  cannot beat abs1024 by 0.2 point when the faults cost less than that.
- **Jet:** the faults do cost accuracy, but only at 1e-2 and above. That is the high-rate side, where abs1024 is
  expected to win anyway. At the low rates, where the signed ranking should help, the jet models lose under 1.5
  points in total.
- **ResNet-8 was the exception.** It is much more fragile, which left room for the signed ranking to win by 0.2–0.6
  point at low rates (Stage 1).

So the plan's design, with two budgets of 12.5% and 25% and a minimum gain of 0.2 point, could only show a signed
advantage on models where low-rate faults still hurt after protection. Most of the new models are not like that. This
explains the failure; it does not change the verdict.

**Exploratory only:** mnist-tcad, whose calibration data overlaps its training data, is the only checkpoint where
pos1024 was ahead: +0.21 point at 3e-2, 12.5%.

## What this means for the paper

- **Not allowed:** "gradient agreement predicts when the signed ranking can be trusted." It failed its pre-registered
  test.
- **Supported, and simpler:** abs1024 is never worse than the signed ranking by a usable margin on any of the 10 new
  checkpoints, across five architectures. Where the two differ, abs1024 is ahead. Together with the DS-CNN (Stage 4,
  the search follow-ups), this makes **abs1024 the safe default**.
- **ResNet-8's low-rate advantage for the signed ranking** (0.2–0.6 point, Stage 1) remains real, but it is specific
  to a fragile model. It did not appear on the other four architectures.
- **For the advisor's question, "when is a static ranking a bad choice?":**
  - Rankings that ignore gradient size (significance-first, FKeras) fail badly at high rates on every model; on VWW,
    msb is 24 points behind abs1024 at 3e-2.
  - Between the two gradient rankings, choosing by rate mattered only on ResNet-8.
- **The exploratory agreement analysis** (`GRADIENT_AGREEMENT_EXPLORATION_2026-11-15.md`) stays as a description of
  gradient noise. It is no longer a candidate trust check.

## Next

- **The DS-CNN calibration-size test** (pre-registered, 1be5878) still runs as approved. Its result now mainly asks
  whether more calibration data changes the DS-CNN's abs/pos gap at all.
- **A possible follow-up, not run:** the same comparison at small budgets (1–6%), where faults still hurt after
  protection. It would need its own pre-registration, stating this explanation in advance.

## Correction (15 November 2026, while drafting the paper section)

The "Why" section above says the jet models lose under 1.5 points at the low rates. That holds at 1e-3 but not at
3e-3. With abs1024 at 12.5%, the accuracy lost at 3e-3 is (from `data/tecs_rate_choice/summary.json`, computed from
the locked fault files):

| jet-e1 | jet-e3 | jet-e10 | jet-e30 | jet-e100 |
|---|---|---|---|---|
| 0.3 | 1.0 | 2.8 | 3.0 | 7.2 |

So "little room left to win" explains VWW and MNIST, not the jet models: there, faults still cost 1-3 points in usable
cells at 3e-3, and the signed ranking still gained nothing. What makes ResNet-8 different stays open. The verdict and
every pre-registered number are unchanged.
