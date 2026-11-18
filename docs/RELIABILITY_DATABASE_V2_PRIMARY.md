# Primary reliability database, schema 1.1.0

The v1 database archive is immutable. This release is a separate, read-only
derivation of the same checksum-verified finalized evidence. The source receipt
remains `config/reliability_sources_v1.json` (source schema 1.0.0); the new
SQLite product uses schema 1.1.0. No inference, mask construction or fault
sampling occurs during database building.

## Comparative boundary

`primary_conditions` is a hash-bound SQL view and `primary_conditions.csv` is
its default comparative export. Exactly three whole-kernel studies qualify:

| Cohort | Eligible INT8 kernel bits | Evaluation units | BER values | Measured budgets | Conditions |
|---|---:|---:|---:|---|---:|
| ResNet-8 / CIFAR-10 | 618,880 | 1,000 images | 8 | 0, 12.5, 25, 100% | 112 |
| DS-CNN / Speech Commands | 176,128 | 1,000 clips | 8 | 0, 12.5, 25, 100% | 112 |
| ToyCar autoencoder | 2,113,536 | 64 complete files | 4 | 0, 12.5, 25, 100% | 56 |

The view joins `primary_studies` to `reliability_grid`, not a benchmark-name
substring. The three source scientific-payload SHA-256 values are pinned in the
builder. A missing source or changed payload fails closed. The separate
128-clip KWS FKeras cohort, 19 engineering studies, and targeted FP32/BF16
studies have separate CSVs. `all_conditions.csv` is explicitly archival and
must not be treated as the main comparison. The raw paired field IDs and exact
mask artifacts remain queryable via `trial_observations`, `protection_masks`,
`source_artifacts` and the original checksummed bundles.

The four-panel main plots show 6.25%, 12.5%, 18.75% and 25% protection.
Missing partial-tier measurements are labeled unavailable. Each panel contains
its actual unprotected and full-protection controls; no control-only panels or
interpolated policy points are generated. Classification degradation is
cross-entropy in nats; ToyCar degradation is `1 − file-level ROC-AUC` relative
to its clean cohort. Intervals characterize fault-field variation conditional
on the frozen model, inputs and masks, not uncertainty over model training or
independent mask draws.

## Frozen, unexecuted gap-filling plans

The product's `plans/` directory contains three TOMLs derived from the
checksum-verified source `config.toml` files. Each plan retains the source
checkpoint, asset lock, ranking/evaluation cohort, estimator definitions and
BER grid. It adds both missing partial-tier budgets and new independent fault
seeds while retaining 12.5%/25% and zero/full controls so comparisons within
the new campaign remain paired. They reuse exposed evaluation inputs and are
exploratory, not untouched-cohort confirmation.

| Plan | BERs | Fields/BER | Policies | Policy/budget evaluations | Rough elapsed time | Rough output |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-8 matched grid | 8 | 32 | 6 | 6,656 | 6.20 h | 4.46 GiB |
| KWS matched grid + published FKeras | 8 | 32 | 7 | 7,680 | 1.61 h plus Hessian construction | 1.43 GiB |
| ToyCar matched grid | 4 | 8 | 6 | 832 | 0.66 h | 7.06 GiB |

Each operation count is `BERs × paired fields × (2 controls + policies × 4
interior budgets)`. Elapsed and storage estimates scale the source-run
coordinator duration and bundle size by this operation ratio; they are rough
planning proxies, not runtime guarantees. They omit changed ranking costs,
especially published-form FKeras eigenpair construction, and contention with
the two ongoing M11 jobs. A bounded FKeras timing preflight and host-capacity
approval are required before running the KWS production grid. No new campaign
was launched by this database delivery.

PrioriFI-derived allocation is blocked pending a faithful feedback
prioritizer and a separately frozen, leakage-free ranking-to-mask contract. It
must not be relabeled as another Hessian ranker. Quantized DistilBERT, Pythia
and Qwen full-model grids require the assets team to publish hash-bound
clean-quality and exact-stored-bit fault gates; the existing restricted
FP32/BF16 pilots do not satisfy that prerequisite.

## Build, verify and inspect

Use a new output directory, never the v1 product path:

```bash
cd /projects/netsurf/manuelbv/workspace/implementation-worktrees/reliability-db
/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python -m workflows.build_reliability_database \
  --sources config/reliability_sources_v1.json \
  --output workspace/reliability_database_v2_primary_new
/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python -m workflows.build_reliability_database \
  --sources config/reliability_sources_v1.json \
  --output workspace/reliability_database_v2_primary_new --verify
```

The product contains `PRIMARY_README.md`, the method × model dashboard,
`primary_coverage.csv`, complete source-separated exports, plots, frozen plans,
checksums and `FINALIZED`. The pinned MLPerf interpreter can validate a plan
without executing it:

```bash
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf validate \
  workspace/reliability_database_v2_primary_new/plans/kws_partial_tier_fkeras_matched_v1.toml
```

The researcher should run a plan only after checking current M11 resource use
and accepting the preflight estimates. Future INT8 transformer planning should
be coordinated with Kona Assets after separate quantized-executable gates pass.
