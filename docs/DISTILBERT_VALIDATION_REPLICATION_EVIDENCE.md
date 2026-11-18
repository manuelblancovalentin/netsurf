# DistilBERT validation-cohort replication: finalized evidence

## Scope and provenance

The independently frozen `distilbert_sst2_validation_replication_v1` campaign has completed. Its object-hash-bound plan is `config/experiments/distilbert_sst2_validation_replication_v1.json`, SHA-256 `a07ebf6849203e56613a11166f1ba651175da9814d2df594500dd41017d1bbc6`. The frozen READY snapshot is SHA-256 `3192d3ce57b7a11089af4c1e0f802e851ebc8b179fd823a793fecf333313db35`; the relevant DistilBERT consumer record is `23b96d38a0670a52e0eb82bbbd41577b549feb1a247691d11ada73a64d3abf87`. The live asset and source checks passed both before execution and after export. The source plan, seeds, bit population and analysis family were not changed after observing outcomes. The original v4 complete ZIP remains SHA-256 `9df099fa58b125315aa68ec528cb3ee71bbe00c7f7c33744e3f1f274849259a4` and its selection seal remains `87870b552a2e759987f93fd91c750664eee43d60c413b491336f0bef5276848b`.

The eligible population is exactly 1,024 FP32 stored bits in the first 32 classifier weight words; all other model weights remain at their clean values. The protection budgets are 0, 128, 256 and 1,024 bits, with Bernoulli probabilities 0, 1/1,024, 4/1,024 and 16/1,024. At budget 128 the planned expected applied counts per field are 0.875, 3.5 and 14; at budget 256 they are 0.75, 3 and 12. Selection used original SST-2 validation indices 288–543 (256 labeled examples), and final used disjoint indices 544–799 (256). The local pre-run ledger showed NetSuRF fault-outcome exposure at engineering indices 0–31 and v4 final indices 32–287, with no known fault-outcome overlap in either new window. Kona's earlier clean-reference inference on all 872 validation examples is disclosed. The ledger cannot rule out unrecorded manual or external exposure.

The exact isolated-bit scores were reconstructed on the larger, task-matched selection cohort. Their protected masks overlap the original v4 isolated masks at only 15/128 and 38/256 bits. For the new 128-bit mask, the sign/exponent/significand counts are 4/92/32 across 30 words; for the 256-bit mask they are 8/120/128 across 31 words. The larger cohort and a new final cohort, fault schedules and seeds all differ from v4, so their separate causal effects are not identified.

Selection made 1,025 authoritative model requests and 262,400 clip inferences; all masks were sealed before final. Final made 769 requests and 196,864 clip inferences, giving 1,794 requests and 459,264 clip inferences total. The conservative prespecified time envelope was 11.72 hours, with an earlier engineering peak of 823 MiB. Actual recorded selection and final elapsed intervals were approximately 5,863 and 3,782 seconds, about 2 hours 41 minutes in total. These timestamps include stage overhead and are not isolated per-call benchmarks. One foreground run completed; no selection or final field was repeated. The 96 final master schedules contain 694 proposed fault events in total: 37, 126 and 531 across the three nonzero BERs. Eight policy records share each master schedule, and their aggregate event accounting is 5,552 proposed, 4,081 applied and 1,471 suppressed events. Full protection always matched clean risk and had zero applied events. The 96 final seeds are unique. The saved field receipts, 768 policy records, schedule checksums, selection seal, analysis digest and ZIP CRC were rechecked independently.

## Prespecified comparison

The primary family contains two two-sided, paired comparisons at BER 16/1,024: encoded bit significance versus task-matched exact isolated at budgets 128 and 256. Sign-flip p-values receive Holm adjustment across these two. The twelve significance/random-versus-isolated comparisons across all nonzero BERs and both budgets form a separate secondary Holm family. Confidence intervals are percentile bootstraps over the 32 independent fault fields per BER. The fixed 256 clips are shared across fields and methods, so they are not independent statistical replicates. A positive difference below means **significance minus isolated** cross-entropy, and favors isolated allocation.

| BER, protected bits | Significance NLL | Isolated NLL | Paired NLL difference, nats | 95% field interval | Holm primary p | Favorable / adverse / tied fields | Accuracy, both | Failed clips, both |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| 16/1,024, 128 | 0.4899803 | 0.4868021 | +0.0031783 | [0.0018532, 0.0047650] | 0.000200 | 30 / 1 / 1 | 0.890625 | 0 |
| 16/1,024, 256 | 0.4886417 | 0.4868047 | +0.0018370 | [0.0012044, 0.0025498] | 0.000200 | 32 / 0 / 0 | 0.890625 | 0 |

Clean final NLL was 0.4886414 and clean accuracy was 0.890625. The primary policies had the same accuracy and no recorded failed clips despite their paired cross-entropy differences. The primary contrasts also have secondary-family Holm p=0.001200. At BER 1/1,024, significance minus isolated was +0.0001011 and +0.0000949 nats for budgets 128 and 256; both have secondary-family Holm p=0.09719 and are unresolved under that family. At BER 4/1,024, the differences were +0.0016693 and +0.0011113, with secondary-family Holm p=0.001200 each. The complete 12-comparison table and all 24 risk rows are in the immutable analysis.

The primary field medians are +0.0011571 and +0.0009831 nats, and their largest absolute field differences are 0.0156348 and 0.0074487. The separately recomputed 10% trimmed means are +0.0023190 and +0.0014799; omitting the single largest absolute field gives +0.0027765 and +0.0016560. These checks show that one extreme field does not alone reverse the new primary means, while the effects remain small in absolute NLL and absent in accuracy. Failed-clip fractions are zero in all saved risk rows. The 30-nat numerical-failure policy remained fixed.

The original v4 high-BER significance-minus-isolated differences were about **−0.4581 and −0.4643 nats** at the same two budgets. That direction does not replicate here. The earlier result was sensitive to catastrophic finite-logit fields under the capped loss, whereas this new result is a much smaller contrast on a different final cohort. This campaign cannot establish that the larger calibration cohort alone caused the reversal, cannot infer across new SST-2 cohorts from field-only uncertainty, and cannot support full-network or physical upset-rate claims.

## Immutable receipt and foreground reproducibility

Run root: `workspace/runs/distilbert_sst2_validation_replication_v1_run1`. Selection seal SHA-256 `2f4c890f9b87557066824e2f4a4a092ca0eb5e9a61fb841fad962ab3c52c4fd1`; saved analysis SHA-256 `154b002eb9e99febcd522f14290f9be5b171411e63f1486b4aa6bde89c2bf989`. The complete evidence ZIP contains 123 verified members and has SHA-256 `fbb13e175c0cbd8e83b76a6207463ad21d2beece92d5a2fd5aa5890bd4f423ea`; the compact ZIP has SHA-256 `18aba39a0dc6468706de768ab64ba65fac23bd7685a14d0e2865f6b2b6bc9477`. The export directory includes `checksums.json`, a concise `summary.json`, tables, plots and exact host copy commands; detailed analysis remains in the evidence bundle.

The exact foreground command used was:

```sh
/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration/scripts/run_distilbert_replication_once.sh \
  /projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration/config/experiments/distilbert_sst2_validation_replication_v1.json \
  /projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration/workspace/plans/distilbert_sst2_validation_replication_v1/NETSURF_ASSET_READINESS.json \
  /projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration/workspace/runs/distilbert_sst2_validation_replication_v1_run1 \
  kona
```

This command is an exact receipt for reproduction on a **new, separately authorized run identity**; the finalized root is write-once and must not be rerun. The existing complete ZIP can be retrieved on a Mac with a unique destination name:

```sh
scp kona:/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration/workspace/runs/distilbert_sst2_validation_replication_v1_run1_export/complete.zip ~/Downloads/distilbert_sst2_validation_replication_v1_complete.zip
```
