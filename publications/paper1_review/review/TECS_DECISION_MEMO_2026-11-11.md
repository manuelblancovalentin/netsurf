# Paper 1 decision memo after the offline audit (11 November 2026)

Scope: offline re-analysis of the six finalized main-grid studies and the 32-pool KWS study, from saved policy-field records, masks, code and manifests. No model inference was run. Nothing finalized was modified. Every number below is reproducible with the commands at the end; none is a publication number until the revised manuscript's provenance file binds it.

All statements are conditional on **one frozen checkpoint, one evaluation cohort and one constructed mask per deterministic method and budget**. Nothing here generalizes across trained checkpoints.

## 1. What the TECS review got right, and wrong

| Review claim | Audit result |
|---|---|
| Featured reversals sit where the model is non-functional | **Confirmed for the six cells in the old Table 4.** ResNet-8 at BER 1%, 12.5%: accuracy 15% unprotected, 41% MSB, 72% absolute gradient (clean 88%). But the grid also contains many useful cells; the old paper did not show them. |
| TFLite cross-entropy is dominated by a floor | **Confirmed in code.** Loss is `-ln(max((int8_out+128)/256, 1e-12))`, no renormalisation: per-example loss is at most 5.5 nats or exactly 27.6. Accuracy must be the classifier task metric. |
| Absolute gradient looks like a robust default | **Confirmed for the three classifiers, weaker for the two language models** (section 2). |
| KWS three-contrast arithmetic is inconsistent | **Not an error.** Each contrast is a separately path-conditioned paired-swap estimate on the same master fields; estimands are transitive, finite estimates are not. Mean gap 1.2e-4 nats, 15 of 32 pools exactly transitive. |
| KWS conditional test is uninformative by design | **Confirmed, and stronger than stated** (section 3). |
| Partial-tier FKeras/MSB contrasts are tie-break confounded | **Confirmed from saved mask composition** (section 4). |
| Positive-gradient harm may be a sign bug | **Refuted.** The code computes `E max(0, g*delta)` exactly from the two saved moments; stored-bit transitions are exact XOR; fields are paired; zero failed units; applied counts are consistent. The harm is real and has a measured mechanism (section 5). |
| Holm p-values look like resolution floors | **Confirmed.** Raw p = 1/10001 in a 10^4-draw sign-flip test; Holm = that times 112 (TFLite per-BER family) or 160 (W8A8 family). |
| Embeddings and biases are excluded | **Confirmed** from the quantization manifests: Linear `qweight` codes only. Qwen's `lm_head` is a separately stored quantized copy of the tied embedding. |

## 2. Central claim that survives

> Among the measured static rankings, the per-example absolute-gradient ranking is a strong default wherever the model remains useful, and ranking choice moves the usefulness boundary by up to about a decade of BER.

Usefulness was fixed before computing regret: accuracy or file AUC within 5 points of clean; language-model NLL within 0.5 nats/token of clean with no failed units. A (BER, budget) cell is decision-useful if at least one measured ranking is useful there.

| Model | Decision-useful cells | Absolute-gradient worst regret there | Holm-resolved losses to any other ranking |
|---|---:|---|---:|
| ResNet-8 | 19 of 32 | 0.06 pp accuracy | 0 |
| DS-CNN | 22 of 32 | 0.16 pp accuracy | 0 |
| DistilBERT | 30 of 32 | 0.14 pp accuracy | 0 |
| Pythia-160M | 17 of 32 | 0.157 nats/token (global gradient better) | 7 cells to global gradient, 3 to positive, 1 to MSB |
| Qwen3-0.6B | 21 of 32 | 0.035 nats/token | 9 cells to positive gradient, 1 to global |
| ToyCar | 4 of 20 | nothing resolved | 0 |

Highest measured BER at which each ranking keeps the model useful at 12.5% protection (grid spacing is half a decade):

| Model | Unprotected | MSB-first | Absolute gradient |
|---|---|---|---|
| ResNet-8 | 1e-4 | 3e-4 | 3e-3 |
| DS-CNN | 1e-3 | 3e-3 | 3e-3 |
| DistilBERT | 1e-3 | 1e-2 | 3e-2 |
| Pythia-160M | 1e-4 | 3e-4 | 1e-3 |
| Qwen3-0.6B | 3e-4 | 1e-3 | 3e-3 |

What does **not** survive: "preferences change with checkpoint" (one checkpoint each); "reversals across models, budgets and BER" as the headline; any implication that no robust default exists. Holm-resolved preference reversals inside the useful region exist only on the two language models, and they are dominated by the positive-gradient ranking (best by 0.01 to 0.03 nats/token at the lowest BERs, then worse by 0.2 to 8 nats/token).

## 3. KWS conditional versus isolated selection

- Background risk at BER 5% is 7.1 nats/clip (range 4.4 to 9.0) with a field SD of 3.4. The model is non-functional there.
- The whole within-pool range over the 15 candidate masks is a median 0.013 nats/clip (max 0.086) on the selection tables: under 1.3% of background.
- In 27 of 32 pools the selected mask beats its runner-up by less than one paired standard error of the eight selection fields; in all 32 by less than two. The "27 of 32 masks changed" count is selection noise, not evidence that conditioning changes preferences.
- In-sample gain of selected over isolated is 0.0053; held-out it is 0.00008. That is the winner's curse.

Verdict: the experiment cannot support a practical claim in either direction. Keep it as a short negative note in the appendix, with its effect ceiling.

## 4. Partial-tier MSB is a layer-order artefact

At 6.25%, canonical tie-breaking gives ResNet-8 MSB-first every sign bit of eight tensors and none in the largest tensor (48% of all bits). On DS-CNN it covers seven tensors fully and two not at all. FKeras spreads its half-tier across tensors. So "FKeras minus MSB" at partial tiers compares Hessian scores with an arbitrary tensor order, in both models. It cannot be read as Hessian information helping or hurting until a randomized within-tier control exists.

## 5. Protection that is worse than none

Positive-gradient protection is resolved-worse than no protection in 17 of 32 DistilBERT cells on loss (6 on accuracy, up to +6.0 pp error) and 8 of 32 Pythia cells (up to +5.8 nats/token). MSB-first is worse than unprotected on DistilBERT loss in 4 cells at BER 3% to 10%, but never on accuracy: the model becomes confidently wrong, not more often wrong.

Bounded offline check on DistilBERT (saved gradients, masks and stored codes): for every other mask the residual first-order effect is balanced (positive and negative parts cancel to within 0.2%). Under the positive-gradient mask it is one-sided: at 12.5% the residual is +6.7 against -20.6. The mask removes the cancelling half of the perturbation, so the remaining faults push coherently in one direction. Residual jump energy is similar to the absolute-gradient mask, so the difference is direction, not amount. This was checked on DistilBERT only.

## 6. What can be fixed without new inference (done in this revision)

- Task metrics on every figure: accuracy, file AUC, NLL with failed fraction.
- Full-grid regret and win/loss/unresolved tables replacing the six hand-picked contrasts.
- Usefulness region stated and shaded.
- Exact loss definitions, Holm families and resolution floors.
- Mask anatomy and the tie-break caveat.
- Model, tensor, quantization and excluded-parameter identities.
- Reliability-importance attribution for the conditioning identity.
- Title and abstract rewritten around what is measured.

Not fixable offline: cap frequency and finite-token NLL for the main language-model grids were never recorded (only the 144-row Qwen follow-up has them).

## 7. Proposed production runs (frozen plans; none launched)

Machine state when written: 80 cores, 471 GiB RAM, load about 2, one unrelated tmux session. No NetSuRF campaign running.

### Run A: TFLite confirmation with mask-construction variance and tie-break control

- **Estimand.** (i) Paired mean error-rate regret of absolute gradient against each other ranking, in the decision-useful cells already fixed by the frozen criterion. (ii) Paired difference between FKeras within-tier selection and the distribution of randomized within-tier MSB masks at 6.25% and 18.75%.
- **Models.** ResNet-8 and DS-CNN, same frozen checkpoints and backend.
- **Cohort.** A new 1,000-input evaluation window per model, disjoint from every index in the exposure ledgers; ranking inputs drawn as five disjoint 64-input calibration sets from the existing ranking pool.
- **Fault regime.** The complete existing BER grid (all eight values) and all four budgets. No BER subset is chosen from outcomes.
- **Masks.** Absolute, global and positive gradient and FKeras rebuilt on each of the five calibration sets; numerical jump and canonical MSB once; randomized within-tier MSB with eight seeds.
- **Seeds.** New field seeds, new mask seeds, recorded before execution; 32 fields per BER.
- **Uncertainty.** Field is the unit; calibration draw is a second, separately reported factor; Student-t paired intervals; sign-flip tests with 10^5 draws; Holm within model over the stated family.
- **Requests.** About 31 policies x 4 budgets x 8 BERs x 32 fields = 31,744 policy-field evaluations per model.
- **Cost.** The finalized main studies took 14.1 h (ResNet-8) and 6.3 h (DS-CNN) of coordinator time for 7,680 evaluations each, so roughly 58 h and 26 h at the same parallelism (worker count was not recorded, so this is an upper-end planning figure to be replaced by a 64-evaluation timing smoke). Memory about 8 GiB coordinator; storage under 5 GB per model, dominated by masks and schedules.
- **Decision rule.** If the upper 95% bound of absolute-gradient regret stays below 1 pp accuracy in every decision-useful cell, across calibration draws, the default claim is confirmatory for these two checkpoints. Otherwise report where it fails.

### Run B: language-model replay with finite-token statistics

- **Estimand.** For every existing main-grid cell of Pythia and Qwen: finite-token mean NLL, fraction of tokens at the 30-nat cap, failed fraction. No new effect estimate.
- **Design.** Deterministic replay of the frozen plans (same fields, seeds, masks, cohort), with the follow-up recorder that saves cap counts. A bitwise match of `mean_nll` against the finalized records is the acceptance gate.
- **Requests and cost.** 4,800 requests per model. Measured service was 15.7 h (Pythia) and 49.4 h (Qwen) summed over shards; about 4 h and 12.5 h wall on four shards. Peak memory was not recorded in the finalized manifests and must be measured in a one-field smoke first.
- **Value.** Closes the one requested diagnostic that cannot be produced offline.

### Deferred, needs separate approval

Independent trained checkpoints (retraining is excluded by the repository rules without explicit approval). Whole-network or group-level conditional allocation. These are the only routes to a checkpoint-level claim and to an informative test of conditional selection.

## 8. Is TECS still realistic?

Yes, as a narrower paper: a paired, task-level evaluation showing where ranking choice matters, that one cheap ranking is a strong default among those measured, and that a plausible-looking ranking can be worse than no protection, with a measured reason. It is a major-revision-grade submission until Run A exists, because the default claim currently rests on single masks and previously exposed cohorts. Without Run A the honest framing is "exploratory, single-checkpoint evidence", and reviewers may still ask for hardware cost and checkpoint replication.

## 9. Reproduction commands

```bash
PY=<netsurf-env>/bin/python
cd <repo>/workspace/implementation-worktrees/paper1-tecs-revision-v1
$PY publications/paper1_review/scripts/tecs_grid_audit.py --out workspace/paper1_tecs_revision/NEW_DIR
$PY publications/paper1_review/scripts/tecs_kws_conditional_audit.py workspace/paper1_tecs_revision/NEW_FILE.json
$PY publications/paper1_review/scripts/tecs_residual_drift_check.py workspace/paper1_tecs_revision/NEW_FILE.json
$PY publications/paper1_review/scripts/build_tecs_figures.py --audit workspace/paper1_tecs_revision/grid_audit_v2 --out publications/paper1_review/figures
```
