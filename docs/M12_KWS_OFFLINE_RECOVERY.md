# KWS matched-campaign offline recovery

The frozen `kws_partial_tier_fkeras_matched_v1` scientific run completed under
`bb33fc39e53dae8553c5f8058fdbf8d883469017`. Its finalized scientific
payload is `9fbf8f905051a9b79a04232475e65dfc5a4a7b78c395bebf02bfe1d4bce85b4b`.
The foreground workflow then failed **after** all evaluation because it used
the pinned MLPerf inference Python for database plotting. That interpreter
does not have Matplotlib. The canonical analysis Python does. No package was
installed and no model inference, ranking, or fault schedule was repeated.

The source bundle's 21 checksummed artifacts, `manifest.status=complete`, and
`FINALIZED` marker verify. Saved `trials.csv` has 256 distinct paired proposed
fault fields: eight BERs, four replications, eight trials each. For each field,
the two controls and seven ranking methods at four interior budgets give 30
policy/budget observations, or 7,680 observations total. The 1,000-clip
evaluation cohort, frozen checkpoint, eligible 176,128 kernel bits, source
revision and postflight identities agree. At 12.5% and 25%, the exact stored
FKeras-paper and MSB protected-bit sets are identical in all four replications;
their paired fault outcomes are also identical. They are not independent
allocation policies at those budgets.

The failed `workspace/m12_kws_seven_method_followup/derived_database/` remains
unchanged. It has a valid intermediate SQLite database and nine CSV exports but
no plots, checksums manifest or finalization marker. The recovery makes a new
versioned product and archive set, using temporary staging and write-once
publication. A failed recovery retains its temporary stage for inspection.
`recovery_audit_v2.json` records the SHA-256 of each of the ten preserved
partial artifacts and the final product/archive identities.

Run the postprocessing-only entry point with the analysis interpreter:

```bash
cd /projects/netsurf/manuelbv/workspace/implementation-worktrees/reliability-db
/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python \
  -m workflows.recover_reliability_followup \
  --output workspace/m12_kws_seven_method_followup \
  --sources config/reliability_sources_v1.json \
  --assets /projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json \
  --baseline workspace/reliability_database_v3_m12_release
```

This command can only read saved experiment evidence and construct derived
artifacts. Its call path has no benchmark adapter execution. The foreground
campaign runner now routes postprocessing through this separate interpreter;
the pinned MLPerf interpreter remains responsible only for the scientific
experiment. Do not call the original `--execute` path to recover this run.

The new database keeps the 280 historical primary rows byte-identical to the
v3 release (`primary_conditions.csv` SHA-256
`b3493022c8c2de31e30c269e90c4b63fd3085f43ecfe280d2fe185ad688faa06`).
The KWS follow-up is a **distinct** 240-condition study, not an overwrite of
the existing primary KWS cohort. Each condition has 32 fault fields across four
replications. The clean 1,000-clip result is 0.901 accuracy and 0.37939147
categorical cross-entropy (nats).

Example measured means (32 fields per BER; full uncertainty and all conditions
are in `all_conditions.csv` and SQLite):

| Method | BER 0.001, 12.5%: loss / accuracy | BER 0.01, 12.5%: loss / accuracy |
|---|---:|---:|
| Unprotected | 0.52076 / 0.87753 | 6.71446 / 0.43319 |
| Random | 0.49565 / 0.88175 | 5.51160 / 0.48906 |
| MSB-first | 0.40916 / 0.89934 | 1.10787 / 0.78922 |
| Numerical bit jump | 0.43553 / 0.89403 | 2.21615 / 0.69291 |
| Global first-order | 0.42231 / 0.89675 | 1.08623 / 0.77878 |
| Per-example absolute gradient | 0.41114 / 0.89941 | 0.75465 / 0.83625 |
| Per-example positive gradient | 0.42147 / 0.89719 | 1.72686 / 0.73425 |
| FKeras, published form | 0.40916 / 0.89934 | 1.10787 / 0.78922 |
| Full protection | 0.37939 / 0.90100 | 0.37939 / 0.90100 |

These two BER slices are descriptive, not a claim of general method
superiority. The FKeras/MSB equality at 12.5% is exact mask degeneracy. The
6.25% and 18.75% partial-tier budgets provide genuine FKeras/MSB comparisons.
Logical BER and ideal event suppression are not physical radiation or ECC data.

The recovered product contains a checksummed SQLite database, CSV tables,
plots, frozen-plan copy and a `FINALIZED` marker. In addition to legacy generic
plots, `plots_kws_matched/kws_loss_vs_ber.png` and
`plots_kws_matched/kws_loss_vs_budget.png` present four interior budgets,
controls in each panel and field-level confidence intervals. The plot inputs
come solely from the saved measurements. The compact export omits the source
estimator-prediction table and is labeled as non-complete; the full source ZIP
contains all finalized source artifacts. Archive and `scp` paths are listed in
`export_recovered_v2/mac_scp_commands.txt`.
