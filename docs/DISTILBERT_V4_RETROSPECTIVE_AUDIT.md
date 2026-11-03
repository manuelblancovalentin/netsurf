# DistilBERT v4: read-only adversarial results audit

This note audits the finalized `distilbert_sst2_classifier1024_v4_run1` evidence. It is retrospective. It does not change the frozen plan, selected masks, final records, or original analysis. The fault population is exactly the first 32 FP32 `classifier.weight` words, or 1,024 encoded bits. It is not full-model coverage.

## Provenance and completeness

| Item | Verified identity |
|---|---|
| Frozen plan object | `735ad6e8531628059bf891075e37885026b220a586bc2d0dc640e507238dc7d7` |
| Frozen READY snapshot | `124de29e1f918bc3acc0153124dbfdf111f3b79984135a21652f7e89e222fc22` |
| Selection seal file | `87870b552a2e759987f93fd91c750664eee43d60c413b491336f0bef5276848b` |
| Saved analysis file | `5fdd2bf99543190d947b099cc607fc016fb6cc35a9e100dc9d237e90e8bb9707` |
| Complete ZIP | `9df099fa58b125315aa68ec528cb3ee71bbe00c7f7c33744e3f1f274849259a4` |
| Compact ZIP | `f677ce2cf0f0da5eaa33cbc14aaa1def52cca24055106da55520e51d80a72529` |

The plan digest recomputes exactly. All three selection members match the seal's file hashes and their own payload digests. There are exactly 32 final fields for each of the three nonzero BERs, or 96 fields and 768 policy evaluations. Every final payload digest matches its content; each field has eight policies, a distinct prescribed seed, one shared event schedule, 256 evaluated clips per policy, and proposed = applied + suppressed. All 96 Bernoulli schedules, 768 mask digests and policy fault counts, and full-protection clean controls were reproduced offline from the frozen plan. Recomputed mean cross-entropies match every saved analysis row exactly. The observed master schedules contained 25, 116, and 488 proposed events across the 32 fields at BER 1/1024, 4/1024, and 16/1024. Across all eight policy evaluations, records report 5,032 proposed, 3,664 applied, and 1,368 suppressed events. All twelve saved paired bootstrap intervals, sign-flip p-values, and Holm-adjusted p-values were reproduced exactly from the field records. The two ZIPs pass CRC checks. Archived source hashes and the original complete ZIP were independently checked during export recovery.

The original clean-reference validation pass covered all 872 SST-2 validation records. That clean pass is disclosed; it is not fault-outcome exposure. Recorded NetSuRF fault outcomes used validation indices 0–31 in engineering and 32–287 in v4 final. A local run-directory audit found no other NetSuRF DistilBERT fault runs; unshared external exposure cannot be ruled out by this audit.

## Protection decisions

Bit numbering is FP32 bit 31 sign, bits 30–23 exponent, and bits 22–0 significand. The sealed masks yield:

| Budget | Method | Sign | Exponent | Significand | Distinct words |
|---:|---|---:|---:|---:|---:|
| 128 | Significance | 32 | 96 | 0 | 32 |
| 128 | Exact isolated | 20 | 99 | 9 | 30 |
| 128 | Random | 4 | 30 | 94 | 31 |
| 256 | Significance | 32 | 224 | 0 | 32 |
| 256 | Exact isolated | 22 | 141 | 93 | 31 |
| 256 | Random | 11 | 63 | 182 | 32 |

Significance and isolated masks overlap in 64 bits at budget 128 and 147 at budget 256. At budget 128, significance alone covers 12 additional sign bits and 52 upper-exponent bits (positions 30, 29, 28); isolated alone substitutes lower exponent positions 27–23 and nine significand bits. At budget 256, significance alone includes ten additional sign bits and 99 exponent bits, whereas isolated alone includes 16 lowest-exponent bits and 93 significand bits. These are exact selected-bit differences, not an explanation of why the held-out contrast arose. Isolated signed scores include 436 beneficial-to-protect, 405 harmful-to-protect, and 183 zero coordinates; their range is approximately −0.000874 to 17.81213 nats. The saved selection artifact contains aggregate bit scores, not per-example score vectors, so score uncertainty and ranking stability cannot be reconstructed from it.

## Original held-out result

The original contrast is baseline minus exact isolated, so negative means significance had lower cross-entropy. At BER 16/1024:

| Budget | Significance CE | Exact isolated CE | Baseline minus isolated | Original 95% percentile bootstrap interval | Original Holm p, family of 12 | Fields favoring isolated / significance / tied |
|---:|---:|---:|---:|---|---:|---:|
| 128 | 0.467338 | 0.925460 | −0.458122 | [−0.730823, −0.208421] | 0.00119988 | 3 / 25 / 4 |
| 256 | 0.461172 | 0.925510 | −0.464337 | [−0.727536, −0.217719] | 0.00119988 | 2 / 26 / 4 |

Accuracies were 0.898560 versus 0.897705 at budget 128 and 0.898438 versus 0.897705 at budget 256. Each field is a paired stochastic replicate, and the same 256 clips recur across fields; clip observations are not independent field replicates. The saved sign-flip p-values assume an exchangeable or symmetric paired-null distribution and use 10,000 random sign flips. The original 12-comparison multiplicity correction remains authoritative for this v4 analysis. The result is specifically about the declared classifier bits, task metric, failure cap, cohort, and fault law.

## Retrospective sensitivity and competing explanations

Recomputing per-example cross-entropy from saved full logits and pinned public labels reproduces the saved 30-nat capped means. The isolated policy has 144 capped *finite-logit* examples in the same nine of 32 high-BER fields at both budgets; significance has none. The largest raw finite loss is about `4.27e37` nats. No nonfinite-inference failure is needed for these capped outcomes. They account for much of the high-BER mean gap.

| Retrospective per-example cap | Mean significance minus isolated, budget 128 | Budget 256 |
|---:|---:|---:|
| 5 nats | −0.025998 | −0.027158 |
| 10 nats | −0.106560 | −0.112775 |
| 20 nats | −0.282341 | −0.288556 |
| Prespecified 30 nats | −0.458122 | −0.464337 |

Removing the nine capped fields after observing them yields −0.001609 and −0.010248 nats on the remaining 23 fields. This is a diagnostic, **not** a replacement estimate or a valid confirmatory subset. The three largest absolute field differences contribute roughly 37–38% of the total absolute paired difference. An exact two-sided sign test on non-tied high-BER fields gives about `2.74e-5` (25 versus 3) at budget 128 and `3.03e-6` (26 versus 2) at budget 256. That tests the direction frequency, not the capped mean estimand; its retrospective use does not revise the original multiplicity family. At lower BERs, many fields tie exactly and the contrasts are small: mean significance minus isolated is −0.0000815/−0.0000428 at 1/1024 and −0.0000501/−0.0025505 at 4/1024 for budgets 128/256.

The nine capped fields all include at least one exponent bit 30 protected by significance and left exposed by isolated selection. This is an association within multi-fault fields; it does not isolate that bit's effect. At least one field with such a bit did not cap. A matched-field intervention that toggles one disputed protection coordinate while holding the complete outside fault pattern and input cohort fixed is necessary for specific-bit or interaction claims. That counterfactual has not been run.

There are three unresolved explanations: (1) isolated score error from just 32 task-mismatched training examples, (2) train-to-validation cohort shift, and (3) genuine context-dependent reversal under multiple faults. The current archive cannot separate them. In particular, the visible mask differences and catastrophic outcomes establish neither a higher-order interaction nor a causal mechanism. A larger validation-derived calibration cohort and paired conditional bit responses are the proposed falsification tests.

## Frozen independent replication

`config/experiments/distilbert_sst2_validation_replication_v1.json` is a separate, hash-bound plan. It retains the same 1,024 bits, methods, budgets, BER grid, 30-nat failure rule, and 32 paired final fields per BER. Isolated scores use 256 previously unexposed validation examples, original indices 288–543; final uses disjoint indices 544–799. Indices 800–871 remain unused by NetSuRF fault trials. The selection seed is 57101; final seed base is 57301. A two-sided Holm family across the two high-BER significance-versus-isolated budget contrasts is the primary replication test, with all twelve v4-style contrasts retained as secondary. The plan was frozen before any replication fault outcomes. The READY snapshot, row/label digests, source hashes, bit population, inputs, seeds, and costs are bound. Local exposure audit is recorded in the plan; external fault-outcome exposure must be disclosed before a run.

Exact planned work: 1,025 selection model requests on 256 clips (262,400 clip inferences), then 769 final requests on 256 clips (196,864 clip inferences): 1,794 requests and 459,264 clip inferences total, with a conservative 11.72-hour estimate from earlier engineering throughput and an observed engineering peak of 823 MiB. Selection is sealed before final. A calibrated gradient arm is excluded because the required task-specific gradient ranking has not passed a conformance gate; substituting an unvalidated gradient score would change the question. Direct outside-field-conditioned bit-response measurements also need a separately frozen diagnostic plan and are not present in this replication.

This plan is **frozen but unexecuted**. Neither selection nor final inference has been run under it.

## Researcher foreground replication command

The following is the complete prospective one-shot command. It validates the frozen source and READY snapshot, runs selection, seals all masks, runs final paired fields, saves a concise analysis summary with detailed JSON in the bundle, exports complete/compact ZIPs, and prints exact Mac copy commands. It is documented here and **has not been executed**.

```sh
/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration/scripts/run_distilbert_replication_once.sh \
  /projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration/config/experiments/distilbert_sst2_validation_replication_v1.json \
  /projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration/workspace/plans/distilbert_sst2_validation_replication_v1/NETSURF_ASSET_READINESS.json \
  /projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration/workspace/runs/distilbert_sst2_validation_replication_v1_run1 \
  kona
```

After successful finalization, the generated `mac_scp_commands.txt` under the export directory gives the exact host path. Its intended complete-evidence command is:

```sh
scp kona:/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration/workspace/runs/distilbert_sst2_validation_replication_v1_run1_export/complete.zip ~/Downloads/complete.zip
```
