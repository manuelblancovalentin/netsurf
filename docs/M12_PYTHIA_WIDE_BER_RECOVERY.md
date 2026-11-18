# Pythia W8A8 wide-BER recovery audit

The frozen 24-field-per-BER run at
`workspace/implementation-worktrees/w8a8-ber-stress/workspace/runs/w8a8_wide_ber_matched_v1/pythia160m_int8`
saved all 192 proposed fields and 4,992 policy observations (4,800 model
evaluations and 192 derived full-protection controls), but **did not finalize**.
The foreground runner exited after shard 0 raised
`pinned model did not restore to exact clean output after shard`.
The source directory, masks, fields, observations, and plan were not edited.

## Failure and forensic evidence

- Shards 1–3 have identical clean start and end records: mean token NLL
  `3.529556153342128`, with the same logit-statistics digest. This also matches
  the earlier finalized Pythia W8A8 clean record.
- Shard 0 started at `3.530695146881044` with a different logit-statistics
  digest, and it has no end-clean marker. Thus the original finalizer must
  reject the run. Its 48 full-protection records reproduce its own anomalous
  start value; the other 144 full-protection records reproduce the consensus.
- A fresh pinned process repeated clean inference twice and returned the
  consensus exact record both times, with unchanged stored INT8 qweight hashes.
- Read-only replay of saved unprotected fields `(BER arm 0, field 0)` and
  `(BER arm 0, field 1)` matched their saved complete risk records exactly.
  Read-only replay of shard-0 protected fields `(arm 5, field 0,
  per-example absolute at 25%)` and `(arm 7, field 20, MSB-first at 6.25%)`
  also matched saved complete risk records exactly, and clean inference after
  replay matched the consensus. These bounded replays do **not** prove that
  every shard-0 measurement is sound.
- The precise cause of the one anomalous initial clean evaluation is still
  unresolved. Per-field code checks restore qweight bytes, but a one-off
  numerical or non-qweight-state issue cannot be ruled out from the saved
  records alone. Do not replace shard-0's clean record or weaken the gate.

## Derived, clean-conformant subset

The separate write-once bundle
`workspace/implementation-worktrees/pythia-recovery/workspace/runs/pythia_wide_ber_clean_shards_v2`
contains only shards 1–3, selected by their exact clean-conformance status,
not by their method outcomes. It has 18 independently proposed paired fields
per BER, 144 fields total, 3,744 accepted policy observations, 208 condition
summaries, 160 within-BER paired comparisons, CSVs, and BER/budget plots.
The original 1,248 shard-0 policy records remain in their source directory.

The derived script verifies the frozen plan and READY asset identities, all
sealed protection masks, each accepted packed fault-field SHA-256 and bit
count, every accepted observation digest and identity, event accounting,
full-protection equality, and exact clean start/end equality. It records
source-member hashes and finalizes its **derived subset** with a separate
manifest. It never loads a model or regenerates a field.

This is an exploratory post-failure subset, **not** the complete frozen
24-field study. It must not be pooled with the earlier narrow-BER Pythia run
or cited as if all 24 fields passed clean conformance. The smaller field count
and the failed original gate are limitations even when a paired contrast has
a narrow interval.

The loss figures show a clear degradation transition in this subset: the
unprotected mean token NLL is around 3.55 at BER `1e-5`, 5.70 at `1e-3`,
22.46 at `1e-2`, and 29.91 at `1e-1`, against clean 3.52956. The frozen task
risk caps even finite per-token NLL at 30 nats. The accepted unprotected
fields reported zero nonfinite failed units at every BER; near-30 mean NLL
therefore indicates saturation under the loss cap, not recorded nonfinite
failures. It is not a physical radiation-reliability measure. Inspect the
saved paired intervals before making any policy-superiority claim.

## Reproduction and preservation

The analysis Python is
`/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python`.
To create a **new, nonexisting** derived destination, run
`scripts/recover_pythia_wide_ber_subset.py` with the source run, new output
directory and `/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json`.
Set `PYTHONPATH` to this isolated worktree. The script rejects an existing
destination and never edits the source.

The next scientific decision is whether to retain the explicitly qualified
18-field subset for exploratory analysis or to commission an independent
fresh confirmation campaign after diagnosing the shard-0 clean drift. Do not
silently retrofit the failed original run into a successful 24-field result.
