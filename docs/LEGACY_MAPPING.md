# Legacy-to-v2 mapping and preservation record

## Handoff identity

- Historical base: `master` at `a95ae571fadf482eb53d4e43314759fc203f99f4`.
- Original reconstructed branch: `codex/netsurf-v2` (its commit identifiers are preserved in finalized manifests, but its local Git objects were deleted before migration).
- Recovered canonical branch: `v2`, recreated from remote `master` at the exact historical base.
- Tracked dirty diff at preservation: SHA-256 `769f4980517fdce7f157e707efba8952bf021795b297814e3c58ffe374b9a214`.
- Audit: `NETSURF_AUDIT_2026-10-21.md`, SHA-256 `1726191735374bfbef527e2bd1c3c788d5a79362941cf411cc484eb95958dbde`.
- Manuscript: `TCAD-2026-0117.R1_Proof_hi.pdf`, SHA-256 `41b60dae3654e0ec855746e48bebfbeeacc0316e5cd929b73dd2625bed3a26c0`.
- External QKeras remains unmodified and is reachable through the ignored link `DEPRECATED/external_links/qkeras -> /home/manuelbv/qkeras`; exact repository provenance is unknown.

The original modified/untracked inventory was preserved during reconstruction.
The remote `master` history was reattached without checking out or overwriting
the working tree. The empty replacement `.git` directory created after deletion
is retained at `DEPRECATED/recovery/git-metadata-fresh-init-2026-10-22`.
Historical source is now under `DEPRECATED/legacy_source/netsurf`; its tracked
dirty changes remain represented in the canonical migration commit.

## Behavioral mapping

| Legacy behavior | V2 status |
|---|---|
| `core.quantization.QuantizationScheme` q<6,2,1> delta | Independent `BitCodec`, exhaustive test |
| Reversed `bitwise_msb` alias | Replaced by tested `MSBEstimator`; legacy configuration untouched |
| Gradient score | Explicit gradient-magnitude and first-order gradient×delta estimators |
| qPolarGrad `abs(P*gradient)` | Preserved as `legacy_qpolargrad`, with activation-source metadata |
| qPolar global attacked activations | First-call behavior reproduced without cross-call state contamination |
| Ranker-imposed MSB tiers | Moved into separate `position_first` policy |
| Batch-dependent TensorFlow injector | Not reused; deterministic event schedules are external |
| Perfect protected-bit suppression | Explicit current mitigation model, not equated with physical TMR |
| Mutable CSV appends | Write-once JSON payloads and finalized digest |
| Legacy checkpoint loader | Isolated adapter loading preserved source under `_netsurf_legacy`; known `strip_pruning` wrapper defect bypassed with its custom-object map |

## Canonical first artifact

The milestone uses checkpoint SHA-256 `ba48aee8adbe7d04092d74f41958ec99112ae3e5f88ed56552ea9d1c4dba3006` and MNIST cache SHA-256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`. Their canonical Kona paths are recorded in `STORAGE_MIGRATION.md`.
