# DistilBERT W8A8 whole-qweight exposure pilot v2

The unrun v1 draft was superseded during preflight because its resource gate
avoided only an active job's last CPU, rather than that job's complete pinned
affinity. No pilot field or reserved final example was evaluated under v1.
The v1 plan remains in the workspace as a nonexecutable audit trail; v2 binds
the corrected gate and otherwise retains the same scientific design and seeds.

## Purpose and evidence boundary

The finalized DistilBERT W8A8 v2 reliability campaign remains immutable. Its
highest expected exposure, 256 proposed faults per field, produced a small and
statistically unresolved unprotected loss increase. This separately frozen,
exploratory pilot locates a useful **unprotected** degradation range. It makes
no protection-method effectiveness comparison and creates no new M12 database.

The population is all 344,469,504 eligible bits in 43,058,688 stored signed
INT8 `qweight` codes across 38 quantized modules. The CPU executable consumes
those stored codes through the pinned integer matmul path. Scales, embeddings,
normalization, biases and other non-qweight values are outside the declared
fault universe. All faults are independent Bernoulli draws over this complete
declared universe, held fixed for one evaluation of the entire labeled cohort.
The geometric-gap sampler implements the Bernoulli law exactly. Logical BER is
not a physical failure rate.

## Frozen design

| Expected proposed faults λ | BER = λ / 344,469,504 | Independent unprotected fields | Expected total proposed faults |
| ---: | ---: | ---: | ---: |
| 1,024 | 2.972686952282429e-6 | 16 | 16,384 |
| 4,096 | 1.1890747809129717e-5 | 16 | 65,536 |
| 16,384 | 4.756299123651887e-5 | 16 | 262,144 |
| 65,536 | 1.9025196494607547e-4 | 16 | 1,048,576 |

There are 64 independent whole-model fields, two clean evaluations, and one
full-protection control using the first frozen field at each BER. Thus the
pilot has exactly 70 model requests and 8,960 labeled-example inference
requests. The full-protection controls are repeated schedules, not additional
independent fields. Actual per-field event counts and exact schedule digests
are generated and bound at plan freeze. Each unprotected event must be applied;
each full-protection event must be suppressed.

The pilot uses original SST-2 validation indices 544–671 (128 labeled clips),
with an ordered text-and-label digest in the plan. Indices 672–871 (200 clips)
are reserved for a later W8A8 final campaign and must not be used by this
pilot or engineering probes. The prior W8A8 v2 final window was 32–543. Kona
already performed clean reference inference on all 872 validation examples;
earlier FP32 NetSuRF studies faulted some of the proposed pilot and reserved
examples. We disclose this cross-precision exposure and make no virgin-sample
claim. The new W8A8 pilot and later W8A8 final cohorts are disjoint. The
engineering probe below used previously exposed indices 0–7 only.

The primary descriptive outcome is mean SST-2 cross-entropy in nats, with
30-nat loss and an incorrect prediction for each numerical or runtime-failed
example. Accuracy and failure fraction are separate outcomes. Field means,
standard deviations, a two-sided 95% t interval over 16 independent fields,
median, one-field-per-tail trimmed mean, largest effect and leave-largest-out
mean describe variability and tail sensitivity. Clips are fixed inputs within
fields and are not independent fault replicates. This is exploratory
calibration, not a confirmatory ranking-method hypothesis test.

The decision rule is frozen before faulted pilot evaluation: choose the
smallest grid λ with mean unprotected cross-entropy increase at least 0.02
nats and at most 10% failed examples, then report it and its immediately lower
grid neighbor (or the first two arms if the first qualifies). If no arm meets
the rule, report unresolved and propose a separate prespecified follow-up.
The 0.02-nat threshold is a practical screening threshold, not a power
calculation or a publication claim. No ranking-method result can enter this
decision.

After the pilot, a **new** six-method BER × protection-budget study may be
frozen around the diagnosed range. It must use all six existing method
definitions, the same proposed fields across methods and budgets, new
independent seeds and fields, the reserved 672–871 cohort, and a distinct run
directory. Exact-isolated and FKeras/PrioriFI gaps cannot be silently filled
with substitutes. The new final plan and its cost must be reviewed separately;
the pilot does not authorize that run.

## Feasibility and resource envelope

On 2026-11-04, a bounded **engineering-only** high-exposure check used eight
previously exposed validation examples, four separate physical CPU cores,
and seed 9700001. It sampled 65,875 proposed faults at λ 65,536, applied all
65,875, suppressed zero, and restored the exact clean output. Its schedule
digest is `484960d34e5f472b25d6b4bca00f48e9b9605473471b5606f2a108b01812d28d`.
Loading took 7.95 seconds; the clean eight-example request took 0.19 seconds;
the faulted request including mutation and restoration took 3.43 seconds;
the restored clean request took 0.20 seconds. Peak resident memory was
1,860,236 KiB (1.78 GiB). This one short check validates basic feasibility,
not the throughput or outcome distribution of the 128-example pilot.

The bounded execution envelope is **four CPU threads, 8 GiB planned maximum
RSS, at least 5 GiB free output disk, and 10–60 minutes wall time**. The range
allows for field generation, model load, full-cohort inference, exact mutation
and restoration, hash verification and export. The admission script measures
host CPU/load, active model jobs, available RAM and disk, then selects four
distinct physical cores away from currently active model cores. It does not
alter existing jobs. The one-shot run reports actual wall time and peak RSS.
The 8 GiB figure is an admission envelope, not a hard cgroup limit.

On the same date, live headroom was about 417 GiB available RAM and 17 TiB
free output disk, while Pythia W8A8 and two M12 model studies were active.
These transient measurements do not authorize a production pilot by
themselves; the foreground command measures admission again immediately
before execution. A failed or interrupted run retains its write-once partial
evidence and cannot be silently restarted under another seed.

## Foreground execution and export

The plan is at
`workspace/plans/distilbert_sst2_w8a8_exposure_pilot_v2/` and the unique
output is `workspace/runs/distilbert_sst2_w8a8_exposure_pilot_v2_run1/` in the
isolated transformer worktree. The pinned interpreter is
`<assets>/envs/torch-2.9.1-cpu-py311/bin/python`.
The researcher can run this one foreground command on Kona after approving
the resource envelope:

```bash
<repo>/workspace/implementation-worktrees/transformer-integration/scripts/run_distilbert_w8a8_exposure_pilot_v2.sh
```

The command validates the frozen plan and target-specific READY record, runs
measured admission, executes all fields, verifies clean and full-protection
controls and byte-exact restoration, aggregates offline, and creates compact
and complete CRC- and SHA-256-checked ZIP archives. It prints and saves exact
Mac commands in the sibling export directory:

```bash
scp kona:<repo>/workspace/implementation-worktrees/transformer-integration/workspace/runs/distilbert_sst2_w8a8_exposure_pilot_v2_run1_export/compact.zip ~/Downloads/distilbert_w8a8_exposure_pilot_v2_compact.zip
scp kona:<repo>/workspace/implementation-worktrees/transformer-integration/workspace/runs/distilbert_sst2_w8a8_exposure_pilot_v2_run1_export/complete.zip ~/Downloads/distilbert_w8a8_exposure_pilot_v2_complete.zip
```

The complete bundle contains the frozen plan and READY/quantization snapshots,
resource-admission receipt, 64 intent records and 64 field records with
schedule digests, all four full-protection controls, clean checks, field CSV,
analysis, and member checksum inventory. The compact bundle retains the plan,
analysis, field CSV, clean checks and checksum inventory. Neither archive
exists until the pilot is actually run and finalized.
