# M12 reliability-matrix gate (derived evidence, not new protection results)

This worktree adds a new, write-once v3 database product. The v1/v2 archives and
all finalized NetSuRF runs remain unchanged. The default primary table remains
the three hash-pinned whole-kernel INT8 studies: ResNet-8, KWS DS-CNN and ToyCar.
The separate KWS 128-clip FKeras cohort and restricted FP32/BF16 studies must
not fill missing cells in those primary cohorts.

## READY W8A8 assets versus NetSuRF readiness

Kona-assets commit `1aea52b18b28af9e6078316f7948d92282393f83` and
`NETSURF_ASSET_READINESS.json` SHA-256
`7128399521cef0d3e08f5d70bcb4f5b360fe2c85406c8ea8ac495e85be77b1b0`
are the source of this gate. The v3 builder checks every declared model and
dataset file and the clean and exact-mutation receipts before populating the
`asset_readiness` table. No transformer is loaded or faulted by that builder.

| Target | Eligible stored bits | Clean source result | Asset | NetSuRF full-universe adapter | Protection result |
|---|---:|---|---|---|---|
| DistilBERT/SST-2 W8A8 | 344,469,504 | 872 validation examples; accuracy 0.9082568807, CE 0.3923907734 nats | READY | NOT VALIDATED | NOT MEASURED |
| Pythia-160M/WikiText-2 W8A8 | 988,545,024 | 288,729 scored tokens; NLL 3.2682748707 nats | READY | NOT VALIDATED | NOT MEASURED |
| Qwen3-0.6B/WikiText-2 W8A8 | 4,767,875,072 | 299,077 scored tokens; NLL 2.5229869003 nats | READY | NOT VALIDATED | NOT MEASURED |

These are derived executables, not integer-only networks: eligible `qweight`
codes are INT8; dynamic per-token activations and `torch._int_mm` use INT8/INT32
within Linear layers; embeddings, normalization, attention, nonlinearities,
biases and final losses remain floating point. There is no validated NetSuRF
whole-universe fault adapter, ranking mask, paired field plan or effectiveness
observation for these three assets. The existing transformer-integration worktree
has FP32/BF16 targeted adapters. At this audit its separate branch
`codex/transformer-integration` was at
`c066921953421ac5bebc62249b93db157b23f1f2` with untracked W8A8 adapter,
fault, ranking, surrogate and test files under active development. Those files
are not yet committed or validated and must not be treated as implementing
the new W8A8 contracts. That worktree is not modified by this work. The
integration gate for its eventual handoff is: hash-bound asset load, all
eligible INT8 code identities and per-channel scales, exact XOR/restore,
clean-reference agreement, calibration/final partition separation, frozen
method masks, matched proposed fields, bounded task-risk smoke, then a
versioned source receipt for ingestion into this database. No receipt or
effectiveness row is fabricated in advance.

## Method feasibility and missing common cells

All six established static methods (random, MSB-first, encoded numerical jump,
global first-order, per-example absolute and per-example positive) have measured
primary-cohort cells at 12.5% and 25%, with complete no/full controls. The
6.25% and 18.75% cells are absent in all three primary studies. The existing
frozen, unexecuted matched plans in the v2 database product cover these cells;
they are not evidence. The v3 product regenerates them as distinct unexecuted
receipts from the unchanged source configurations.

Published-form FKeras has only a distinct 128-clip KWS result. At 12.5% and
25% homogeneous INT8 budgets, its strict significance-tier-first masks equal
MSB-first masks; 6.25% and 18.75% are the distinguishing budgets. The KWS
continuous Hessian implementation passed the existing finite-difference test
and a new 16-calibration-clip preflight: 22,016 weights, two eigenpairs,
25 HVPs, normalized residuals 3.25e-16 and 6.71e-13, 1.06 s after adapter
construction, 28.28 s total process wall time, 513 MiB peak RSS. A second
bounded preflight used the **planned 64 ranking clips**: 25 HVPs, residuals
3.21e-16 and 3.88e-16, 1.18 s after construction, 29.66 s whole process,
765 MiB peak RSS. This establishes bounded eigensolver feasibility, not the
full matched-protection runtime or effectiveness. The implementation is KWS-only.
ResNet-8's straight-through surrogate is
not a validated continuous Hessian of the integer reference; ToyCar's file AUC
is not the differentiable reconstruction-MSE surrogate. Neither is presently
FKeras-ready. Do not configure a fabricated FKeras mask for them.

PrioriFI's pinned source commit is
`6e7637b92315e5df75bb6ce258972bdf1131bc20`. It orders Hessian-ranked
parameters into significance-tier bit lists, probes one isolated bit per tier,
then selects the next tier from recent observed clean-vs-faulty class-change
metrics (median of the last five when the tier has at least five bits remaining;
otherwise mean of retained observations). It probes all bits in the source
campaign. `netsurf.priorifi_feedback.feedback_probe_order` reproduces this
scheduling rule with a supplied, state-restoring isolated probe callback.
PrioriFI is a fault-injection prioritizer, **not** a published fixed-budget
protection allocator. `derived_allocation` is a clearly separate NetSuRF
adaptation that refuses an incomplete eligible-bit calibration set and ranks
fully observed change fractions. No effectiveness cell is measured. KWS and
ResNet-8 would require complete calibration probing plus independently held-out
final fields. ToyCar is incompatible with the source class-change metric unless
a separately named, justified AUC-specific adaptation is prespecified.
The full ASPLOS paper was not available from this environment; source behavior
is verified, but exact paper/source correspondence remains an audit limitation.
The source-style complete calibration requires at least 176,128 isolated KWS
bit evaluations or 618,880 ResNet-8 evaluations, before any held-out fault
fields. Naively multiplying the historical coordinator-time/evaluation ratios
gives approximately 37 h and 24 days respectively; these are rough capacity
warnings, not validated throughput predictions, and source-model differences
may dominate. A truncated prioritization trace is **not** a full-universe
protection ranking under the documented adaptation.

## Resource and execution boundary

The pinned historical manifests give rough coordinator time of 12,019 s for
3,584 ResNet-8 policy evaluations, 2,701 s for 3,584 KWS evaluations and
1,288 s for 448 ToyCar evaluations. These are whole-run observations including
setup, with 2/4/2 workers respectively, not a per-evaluation performance model.
The current validated matched follow-up plans contain 6,656 ResNet-8, 7,680
KWS and 832 ToyCar policy-budget evaluations. Simple source-ratio estimates
are roughly 6.2 h, 1.6 h and 0.7 h, excluding new FKeras ranking,
contention and storage variation. The matching source-bundle disk-ratio
estimates are roughly 4.5 GiB, 1.5 GiB and 7.1 GiB. New jobs need a fresh
host-resource check and researcher approval while the two M11 runs are active.
No new publication campaign was launched by this M12 work. At the M12 check,
the two M11 workers were each consuming approximately one CPU core and
0.7–0.8 GiB RSS; the host reported 425 GiB available RAM and 17 TiB free on
`/projects`. Storage capacity is not the immediate limit. The 64-clip FKeras
eigensolve itself is measured above; simultaneous CPU contention, repeated
worker setup and complete matched-run elapsed time remain unmeasured.

Run the v3 read-only builder with the existing canonical interpreter:

```bash
cd <repo>/workspace/implementation-worktrees/reliability-db
<netsurf-env>/bin/python -m workflows.build_reliability_database_v3 \
  --sources config/reliability_sources_v1.json \
  --assets <assets>/NETSURF_ASSET_READINESS.json \
  --output workspace/reliability_database_v3_m12_release
```

The output contains `primary_conditions.csv`, `asset_readiness.csv`, the
method/missing-cell ledgers (including `M12_METHOD_MATRIX.md`), measured-only
and full-target plots, SQLite,
checksums and a finalization marker. A new output path is required on rebuild.
`primary_condition_context` / `primary_condition_context.csv` add the exact
eligible-bit population, cohort identities and expected applied count
`(eligible_bits − protected_bits) × BER` to each measured condition; these are
not new task measurements.

After checking the product, create a separate compact/full ZIP export without
changing it:

```bash
<netsurf-env>/bin/python -m workflows.export_reliability_database_v3 \
  workspace/reliability_database_v3_m12_release \
  workspace/reliability_database_v3_m12_release_export
cat workspace/reliability_database_v3_m12_release_export/mac_scp_commands.txt
```

For a frozen, unexecuted matched plan, use a foreground dry-run first:

```bash
<netsurf-env>/bin/python -m workflows.run_reliability_followup \
  --receipt workspace/reliability_database_v2_primary_final/plans/plan_receipt.json \
  --plan kws_partial_tier_fkeras_matched_v1 \
  --output workspace/m12_kws_followup
```

Only after researcher approval of the live host-resource and planned campaign
cost, repeat with `--execute` in tmux. That command verifies plan hashes and
assets, runs the canonical CLI in the foreground, verifies its finalized
bundle, makes a new source receipt and v3 database, then exports ZIPs and
Mac download commands. It never overwrites a prior finalized run. No
production run has been invoked here.

The next authorized experimental step is **not** to run all three plans at
once. The KWS 64-clip FKeras eigensolver preflight has converged; next approve
a bounded, paired partial-tier
campaign with distinct final field seeds. Transformer plans must wait for
NetSuRF adapter conformance, exact eligible-bit maps, restoration, task-risk
and resource gates. PrioriFI must wait for an explicit feasible full-universe
calibration budget and separate held-out allocation protocol.
