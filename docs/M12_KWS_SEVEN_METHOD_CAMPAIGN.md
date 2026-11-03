# KWS seven-method matched reliability campaign

This is a frozen, **unexecuted** follow-up to the finalized 1,000-clip DS-CNN
KWS study. It is a distinct study, not a modification of the historical v3
database or its source bundles. The production command below must be started
only after the researcher authorizes the resource receipt.

## Frozen comparison

The configuration is
`workspace/reliability_database_v3_m12_release/plans/kws_partial_tier_fkeras_matched_v1.toml`
(SHA-256 `abc218c5df197afc56fa991d9fccdb2c14d122e0c15033aff1f71bb07fc753a1`).
It specifies the pinned 176,128 stored INT8 KWS kernel bits and TensorFlow Lite
2.14.1 `BUILTIN_REF`, with 64 training-split calibration clips and the same
1,000 test-split evaluation clips as the finalized source study. The source
scientific payload is
`b7b725fd1c196f2d5896b67aba34dfe8aa864b8308875df098d48be6135975e0`.

The seven methods are seeded random, corrected word-relative MSB, exact
numerical bit jump, global first-order gradient, per-example absolute
gradient, per-example positive gradient, and **published-form FKeras only**.
FKeras uses two largest-magnitude signed continuous-companion Hessian
eigenpairs, at most 300 HVPs, and strict bit-significance tiers. Its
continuous Hessian is not a Hessian of exact integer TFLite inference.
The budgets are 6.25%, 12.5%, 18.75%, and 25% of eligible bits. The eight
logical BERs are `1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1`.
Four replications of eight independent persistent fault trials give 32
proposed fields per BER and 256 proposed fields total. Each field is shared
across seven methods × four budgets and the unprotected/full-protection
controls: **7,680 policy/budget evaluations**, or approximately 7.68 million
clip inferences before calibration and clean-inference overhead. The random
ranking seed changes separately from the fault-field seed across
replications. These are not independently trained networks or new test cohorts.

The engine constructs all ranking scores and protection masks before it
generates the first proposed fault schedule in each replication. It does not
write a separate pre-evaluation mask file; instead the exact masks and their
digests are included in the immutable finalized bundle. The foreground
runner verifies the finalized masks and every paired `(BER, replication,
trial)` schedule before database ingestion. At 12.5% and 25%, published
FKeras and MSB necessarily protect complete significance tiers. The runner
requires their saved bit sets and every paired trial outcome to be exactly
identical, then writes a derived equivalence receipt. Those rows are duplicate
allocations, **not independent competing policies**. Only partial-tier
budgets can distinguish their allocations.

## Asset and source-revision gate

The pinned checkpoint SHA-256 is
`aeea436800704fce17b17292e4412630ad856e9d777c044c64ef748a880bd0ae`;
the official evaluation-index SHA-256 is
`6cb3709762d53fe64eb00eaf4a75796248bb3c1386f1d1f93ccdb07b071e9698`.
The current Kona lock has canonical digest
`sha256:d87daf0dbc24bbc334589dae83e200504a9624008f647d461c6c4721ce1bcf6f`,
whereas the historical source bundle records
`sha256:9de0816f7973b31febc84d04667068b3ad583ab2cb5d17c0f44da1a2c4ddd3d2`.
These are **not** represented as the same lock artifact. The `/data` and
`/mnt/raid0/ml` model paths resolve to the same inode and bytes; the adapter
accepts only that exact same-file alias and still checks the pinned model
SHA-256. The gate rederives all current selected clips with the pinned
preprocessing and compares their label-bound sample hashes to the finalized
source. It also records the current encoded-input SHA-256 for each role.
The old lock file is not available for a field-by-field historical diff, so
the lock-lineage discrepancy remains an explicit provenance limitation.

The foreground runner checks that tracked source is clean and records its
exact Git commit. Resuming a run requires the same commit and the same model,
lock, cohort and source-payload identities. No runtime metadata is included
in the scientific payload comparison.

## Bounded preflight and execution

The measured 64-clip published-form FKeras preflight produced 176,128 finite
scores with two converged eigenpairs, 25 HVPs, normalized residuals
`3.2054e-16` and `3.8826e-16`, orthogonality error `1.6419e-16`, and
symmetry error `1.6586e-17`. The score computation took 1.38 seconds after
adapter/sample setup in one check; the whole process took 7.17 seconds and
peaked at 806,316 KiB RSS. This validates bounded eigensolving, not the
full reliability campaign's runtime.

From this implementation worktree, inspect the frozen plan without running
fault trials:

```bash
cd /projects/netsurf/manuelbv/workspace/implementation-worktrees/reliability-db
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m workflows.run_reliability_followup \
  --receipt workspace/reliability_database_v3_m12_release/plans/plan_receipt.json \
  --plan kws_partial_tier_fkeras_matched_v1 \
  --sources config/reliability_sources_v1.json \
  --assets /projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json \
  --output workspace/m12_kws_seven_method_followup
```

Only after explicit researcher authorization, run the same single foreground
command in tmux with `--execute` appended. No `nohup` or background wrapper
is needed. It validates the plan, current assets, source revision, KWS cohort
and bounded Hessian; calls the canonical NetSuRF CLI; verifies the finalized
scientific bundle and complete-tier mask identity; appends the run as a
distinct follow-up study; creates a new derived database and measured plots;
then makes compact/full ZIPs for both the database and the source bundle.
`export/mac_scp_commands.txt` contains exact Mac transfer commands. The
compact source ZIP explicitly omits the large per-bit estimator-prediction
CSV and is not a substitute for the complete archive.

The source-ratio estimate is approximately 1.61 hours and 1.43 GiB for the
source run alone, excluding Hessian setup, simultaneous workloads, derived
database and archives. Runtime and output-space planning must be refreshed
immediately before the researcher starts the job. The two active M11 studies
are independent and must not be interrupted.

PrioriFI remains only a source-audited feedback scheduler. No validated
PrioriFI-derived protection mask is included in this seven-method campaign.
