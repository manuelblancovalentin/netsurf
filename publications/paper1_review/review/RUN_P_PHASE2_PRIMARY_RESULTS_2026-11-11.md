# Run P, Phases 1 and 2: primary checkpoints and Block B

Status, 11 November 2026, 21:00 CST: the three primary checkpoints (under-trained, pruned 70%, pinned ResNet-8) and
Block B are complete. The secondary checkpoints (pruned 50%, five retrained seeds) are still running. Nothing here
is in the paper.

- Plans: `RUN_P_MEAN_VARIANCE_DRAFT_PLAN_2026-11-11.md` (decision rules, Section 6) and
  `RUN_P_MEAN_VARIANCE_PLAN_V2_2026-11-11.md` (Sections 4 and 4a, amendment A3).
- Analysis: `workflows/runP_phase2_analyze.py`, branch `paper1-runP-mean-variance`, commit d409b1f.
- Primary snapshot: `workspace/runs/paper1_runP_phase12_v1/phase2_analysis_primary_2026-11-11.json`, read-only,
  sha256 `aedadcfa9542097f29a9ed85043a535cd7cacb658285cc5575d23a9d3534bdc7`.
- Block B: `workspace/runs/paper1_runP_phase12_v1/blockb/blockb.csv`, sha256
  `f3ca1be00cfd82d4f9fdaf3580a32925cd3f1c4b83413a3fafd8701de42c6b2f`.

Arms:

| Name | What it is |
|---|---|
| abs | The paper's absolute-gradient mask, built from 64 images |
| abs512 | The absolute-gradient mask built from 512 images |
| squared | The variance term only |
| range W | Mean plus variance, averaged over 8 rates in [3e-4, 3e-2] |
| single | Mean plus variance at the test rate (the method knows p) |

## 1. Integrity

- All 25 primary jobs and Block B exited with code 0. Before evaluating any new mask, each job replayed the saved
  absolute-gradient mask on every one of its schedules and had to reproduce the saved accuracy and cross-entropy
  exactly, or abort. None aborted.
- In Block B, the paper's mask reproduced Run N's accuracy on all 288 evaluations.
- The analysis code had two faults, both fixed before the result was read as final:
  1. The pinned bundle also contains random-fill partial-tier policies, which are not among the seven rankings.
     They are now skipped; before the fix the run crashed.
  2. The pinned measured table also contains the DS-CNN rows, and these overwrote ResNet-8's usable-cell flags.
     This was visible as a clean accuracy of 92.1%; ResNet-8's is 87.0%. The table is now filtered to ResNet-8.
     The fix changes which pinned cells count as usable but not the verdict.

## 2. Frozen decision rule

The Phase 1 check passed on all three checkpoints. The median held-out reduction of the range-W objective,
relative to abs512, was:

| Checkpoint | Median reduction (nats) |
|---|---|
| Under-trained | 0.017 |
| Pruned 70% | 0.39 |
| Pinned | 0.005 |

| Reference | Helps | Harmful | No gain |
|---|---|---|---|
| Paper's absolute gradient (64 images) | yes | no | no |
| Absolute gradient from 512 images | yes | no | no |

**By the frozen rule, the method helps.** The rule is met only on the under-trained checkpoint and only at
p = 1e-2. The cells that meet it, as accuracy differences in points (Holm-adjusted p; intervals are Student-t over
the 8 replications):

| Cell (under-trained) | Range W − abs | Range W − abs512 | Range W − squared (unadjusted p) |
|---|---|---|---|
| 1e-2, 12.5% | +2.5 (0.008) [+1.8, +3.2] | +1.5 (0.006) [+0.9, +2.1] | +1.4 (<0.001) |
| 1e-2, 18.75% | +1.4 (0.017) [+1.1, +1.6] | +1.0 (0.022) [+0.7, +1.3] | +1.0 (0.001) |
| 1e-2, 25% | +1.1 (0.12) [+0.6, +1.5] | +1.2 (0.019) [+0.6, +1.8] | +1.1 (<0.001) |

No usable cell on any primary checkpoint has range W resolved worse than either reference.

## 3. What the numbers say

1. **Under-trained checkpoint, high fault rate.** At 1e-2, range W beats both absolute-gradient masks by 1.0–2.5
   points. It also beats the variance-only mask by about the same amount, so the gain comes from the mean term:
   it accounts for the average displacement of the faulty weights. That is the effect the researcher's
   balancing idea targets. At 1e-3 and 3e-3 the positive-gradient ranking (the reversal case in the paper) is
   still 0.5–1.4 points ahead of range W.
2. **Pinned checkpoint.** Range W is the same as abs512: +0.0 to +0.3 points, none resolved. Most of its gain over
   the paper's mask comes from using more calibration images. abs512 beats the 64-image mask by +0.4 to +0.8
   points in usable cells at 3e-3 and 1e-2 (unadjusted p 0.008–0.03; exploratory, no correction). In two cells
   where the seven rankings leave the pinned model unusable, abs512, squared and range W all make it usable:

   | Cell (pinned) | abs | abs512 | Range W |
   |---|---|---|---|
   | 3e-3, 6.25% | 80.7 | 81.7 | 82.0 |
   | 1e-2, 18.75% | 81.5 | 83.0 | 83.3 |

   Clean accuracy is 87.0.
3. **Pruned 70%.** No difference is resolved. Range W's point estimates at 3e-3 are −0.2 to −1.4 points, with
   intervals down to −3.6. The held-out objective dropped most on this checkpoint (0.39 nats), yet accuracy did
   not follow. The second-order objective does not predict accuracy well here; the many pruned zero codes are a
   possible cause, untested.
4. **Knowing the rate.** The single-rate mask beats range W by:
   - 1.3–2.3 points on the under-trained checkpoint at 1e-3 and 3e-3;
   - 1.0–2.0 points on pruned 70% at the same rates;
   - nothing on the pinned checkpoint.

   On the under-trained checkpoint it also beats the positive-gradient ranking, and its faulty accuracy exceeds
   the clean accuracy (84.1 vs 81.5). The mask leaves unprotected the bits whose average displacement lowers the
   loss of an under-trained model: the faults act as a small training step. That is a property of an
   under-trained model, not protection a designer can rely on.

### Block B: the Run N protocol (pinned, 16 fault fields per cell)

Accuracy difference from the paper's mask, in points; * marks unadjusted sign-flip p < 0.05. This block is
exploratory. Usable cells for the paper's mask are 3e-3 at both budgets and 1e-2 at 25%.

| Cell | Paper's mask | abs512 | Range N | Single | Relaxed | Relaxed range | Group CEM | MC search | Thales | HarDNN |
|---|---|---|---|---|---|---|---|---|---|---|
| 3e-3, 12.5% | 83.0 | +0.4 | +0.8* | +0.9* | −5.8* | −5.2* | −0.7* | +0.1 | −21.7* | −14.8* |
| 3e-3, 25% | 85.0 | −0.0 | +0.2 | +0.7* | −4.6* | −3.7* | −0.4 | −0.1 | −24.0* | −11.8* |
| 1e-2, 12.5% | 76.8 | +0.6 | +1.1 | +0.1 | −12.7* | −22.6* | −1.3 | +0.7 | −58.4* | −47.9* |
| 1e-2, 25% | 83.7 | +0.3 | +0.9* | +0.7* | −8.2* | −12.3* | −1.0* | +0.3 | −66.4* | −41.0* |
| 3e-2, 12.5% | 54.1 | +1.9 | −0.6 | +2.0 | −12.4* | −33.2* | −3.0 | +4.0* | −43.5* | −41.2* |
| 3e-2, 25% | 79.7 | +1.0 | −1.8* | +0.7 | −7.3* | −35.2* | −1.9* | +0.5 | −68.8* | −64.2* |

The mean-variance masks are the only gradient-based optimizers here that do not fall behind the paper's mask:
they gain up to 1.1 points. The relaxed per-bit optimizers lose 4–35 points.

## 4. What this supports and what it does not

Supports:
- On the one checkpoint where the preferred ranking reverses (under-trained), a mask that accounts for the
  average displacement of faulty weights beats both absolute-gradient masks by 1.0–2.5 points at 1e-2.
- It does not lose to them on the pinned or pruned-70% checkpoints in any resolved cell.

Does not support:
- A general improvement. On the pinned checkpoint the gain is calibration size; on pruned 70% there is none,
  with negative point estimates.
- Replacing the absolute gradient as the default.
- Any claim about DS-CNN or the other models (not run; amendment A3).
- Any claim before the five retrained seeds and pruned 50% are analyzed.

Side finding (exploratory, not corrected): building the paper's absolute-gradient mask from 512 images instead of
64 gains up to about 0.8 points in usable cells of the pinned checkpoint at 3e-3 and 1e-2.

## 5. Next

- Analyze the secondary checkpoints when they finish, with the same code, as a separate snapshot.
- Paper: nothing added yet. Decide with the researcher after the secondary results.
- Extending to the other models needs a differentiable companion for each and a new plan with its own go-ahead.
