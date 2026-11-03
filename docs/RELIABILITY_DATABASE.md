# NetSuRF reliability database v1.0.0

This is a read-only derived data product. It does **not** execute a model, redraw a fault field, construct a mask, or alter a finalized bundle. Build it with the pinned Python 3.11 `netsurf` interpreter:

```bash
cd /projects/netsurf/manuelbv/workspace/implementation-worktrees/reliability-db
/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python -m workflows.build_reliability_database \
  --sources config/reliability_sources_v1.json \
  --output workspace/reliability_database_v1_new
```

The output directory must be new. The builder checks the source receipt, every listed checksum, the `FINALIZED` marker, the canonical scientific-payload digest, matching schedules and `proposed = applied + suppressed`. The FP32/BF16 targeted studies additionally check their frozen plan digest, analysis digest and each final field's bound digest. No source artifact is rewritten. Absolute source paths in the receipt are Kona locations; porting requires a new explicit path receipt with the same verified hashes.

## Tables and units

| Table | Key and meaning |
|---|---|
| `metadata` | Database schema version and source-receipt SHA-256. |
| `models` | One exact benchmark/protocol identity per full-kernel study. Checkpoint, dataset, backend, eligible-bit universe, clean loss/quality, and full calibration/evaluation partition maps are retained. A distinct 128-clip KWS cohort is a distinct model record. |
| `studies` | Source bundle path, finalization and payload/checksum identities. `study_kind` distinguishes whole-kernel from distinct-cohort evidence. |
| `methods` | Mathematical ranking description, ordering rule, calibration requirement, static/control/source-only status. The archived FKeras repository variant is source-only and excluded from the primary comparison matrix and plots. |
| `method_implementations` | Per-study estimator implementation version and source Git commit. Control methods have no estimator implementation row. |
| `source_artifacts` | Every checksum-verified relative artifact path, SHA-256 and byte size. |
| `protection_masks` | `(study, replication, method, coverage)` with exact selected-bit count, per-replication mask hash, original bit-identity artifact path, construction configuration and ranking runtime. Exact identities remain in the large checksum-verified source JSON, not duplicated into the compact SQLite file. |
| `trial_observations` | One row per `(study, method, coverage, BER, replication, trial)`; source schedule hash, proposed/applied/suppressed events, task loss, quality and optional auxiliary loss. BER is probability per eligible stored bit, not physical SEU rate. `failure` and timestamps are nullable when the original trial schema did not record them. |
| `aggregated_conditions` | One row per measured condition. `mean_loss`, sample `loss_sd`, `loss_se`, and two-sided Student-t 95% mean interval are in task-loss units. `loss_degradation = mean_loss − clean_task_loss`. `paired_unprotected_delta_*` is the matched-field policy-minus-unprotected loss contrast; its interval is computed from paired differences, not marginal interval subtraction. `field_count` counts independent proposed fields, whereas `replication_count` and `ranking_mask_count` are separate design factors. |
| `coverage_matrix` | Explicit desired full-kernel `(study, method, coverage, BER)` cells, including missing and incompatible cases. |
| `restricted_studies`, `restricted_trials`, `restricted_conditions`, `restricted_coverage_matrix` | Separate targeted FP32/BF16 universes; never join to the whole-kernel tables on the basis of the same percentage. The primary loss is capped classification NLL or penalized token NLL. `failed_fraction` is failed tokens/clips. `finite_loss` is the algebraic mean over nonfailed units after subtracting the frozen 30-nat failure penalty; finite units can still be capped at 30. These are not whole-model reliability claims. |
| `planned_model_methods` | Method-by-target readiness ledger for additional full-model precision targets whose eligible population or BER grid is not yet frozen. These rows deliberately have no fabricated coverage/BER coordinates. |
| `excluded_studies` | Checksum-verified finalized evidence intentionally excluded from the kernel-only comparison because its eligible population differs. |
| `reliability_grid`, `restricted_grid` | Query views joining condition rows to exact model, representation, method, universe and task-risk identity. These are the exported long-form CSVs. They cannot silently mix whole-kernel and targeted universes. |

ToyCar's `task_loss` is exactly `1 − file-level ROC-AUC`; `auxiliary_loss` is reconstruction MSE and must not be substituted for the task risk. Classification `task_loss` is categorical cross-entropy in nats and `task_quality` is accuracy. Restricted language-model loss is token NLL in nats with a 30-nat nonfinite-failure penalty. Never average loss units across architectures or tasks.

Status vocabulary: `FINALIZED_MEASURED`, `FINALIZED_RESTRICTED`, `READY_TO_RUN`, `RUNNING`, `NOT_MEASURED`, `NOT_IMPLEMENTED`, `INCOMPATIBLE`, `ASSET_BLOCKED`, `VALIDATION_FAILED`. Present output currently uses measured, restricted, missing, not-implemented and incompatible. The coverage matrices contain the **complete** method × requested budget × measured-BER cross product; non-applicable control/boundary combinations are explicitly `INCOMPATIBLE` with an explanation rather than omitted. `READY_TO_RUN` must not be inferred merely from installed code; a frozen executable plan and assets are required.

The confidence intervals describe fault-field variation conditional on each frozen model, cohort, and the observed mask(s). They do not cover checkpoint variation, unseen inputs or independent random-mask construction unless those were separately sampled. The raw field identities are retained for paired analysis. Clean risk is deterministic for its frozen cohort; the interval for degradation is the mean-loss interval translated by that fixed clean value. These are descriptive intervals, not claims of radiation reliability.

## Current measured inventory

| Study | Eligible bits | Evaluation cohort | BER values | Budgets | Trial rows | Source commit |
|---|---:|---:|---:|---:|---:|---|
| ResNet-8/CIFAR-10 INT8 kernels | 618,880 | 1,000 images | 8 | 0, 12.5, 25, 100% | 3,584 | `1a7e374ed90b8ff4ce4fbe780262ff452f46d790` |
| DS-CNN/Speech Commands INT8 kernels | 176,128 | 1,000 clips | 8 | 0, 12.5, 25, 100% | 3,584 | `e65a5f2895dbc7445b61ab6655ec80e05287a2fa` |
| ToyCar INT8 kernels | 2,113,536 | 64 complete files / 12,544 windows | 4 | 0, 12.5, 25, 100% | 448 | `2e255dddf956a474781a7b727f5a8af5113f72bc` |
| DS-CNN published-form FKeras, separate cohort | 176,128 | 128 clips | 3 | 0, 6.25, 12.5, 18.75, 25, 100% | 2,496 | `060667ef1a2d34841d20548d0b65febba716b744` |

The database also contains 19 separately identified, checksum-verified whole-kernel engineering pilots/smokes with their raw trials and condition rows. They have smaller or different evaluation cohorts and are **not pooled** into the principal curves. Together, the 23 whole-kernel/distinct-cohort studies contribute 15,764 trial rows and 809 measured condition rows. Several engineering reruns have identical scientific payload hashes; they remain separate execution records and must not be counted as independent scientific replications. The separate ResNet-8 kernel-plus-bias pilot has 629,952 eligible bits; it is verified and listed in `excluded_studies`, not silently merged with the 618,880-bit kernel-only universe.

The 12.5% and 25% published FKeras and MSB masks in the separate KWS study have identical verified hashes: these are complete INT8 significance tiers. The 6.25% and 18.75% masks differ. Do not present the common-tier cells as independent allocation policies.

Five restricted studies are also ingested separately: the original and independent DistilBERT FP32 1,024-bit classifier-row studies, Pythia FP32-storage/FP32-compute and BF16-storage/FP32-compute 96-bit studies, and Qwen BF16-storage/BF16-compute 96-bit study. Their 3,072 final field-policy observations form 96 condition rows. The first 32 DistilBERT classifier weights are on one output-logit row. The two Pythia representations share sampled word ordinals and should not be counted as independent architectural confirmations.

## Missing cells, in scientific priority order

1. **Partial-tier full-kernel INT8 budgets (6.25%, 18.75%)** on the same frozen ResNet-8, KWS and ToyCar cohorts. These are the most direct new cells for a meaningful budget curve. New proposed fields must remain paired across methods; a new campaign cannot borrow a historical control unless its exact fields and cohort are reused under an explicit sealed plan.
2. **Published-form FKeras on the full 1,000-clip KWS cohort**, then bounded ResNet-8 and ToyCar feasibility probes. The 128-clip FKeras study cannot fill the 1,000-clip KWS matrix. At full-tier budgets its mask degeneracy with MSB is expected.
3. **PrioriFI-derived allocation** needs a pinned faithful feedback-prioritization implementation and a separately specified, leakage-free score-to-fixed-mask conversion. It is not an existing measured protection method. The public [PrioriFI repository](https://github.com/KastnerRG/priorifi) describes QKeras Smart Pixel examples; the [ASPLOS authors' description](https://kastner.ucsd.edu/priorifi-asplos26/) identifies a fault-injection prioritizer. The [conference paper](https://doi.org/10.1145/3779212.3790204) was not accessible in full from this environment. No Hessian-only substitute is called PrioriFI.
4. **Exact empirical isolated full-universe masks** are not supported by the existing stratified candidate samples. A whole-universe mask requires exhaustive scoring or a prespecified approximating policy with a different name.
5. **Representation-aware exponent-first controls and quantized large-model assets** remain separate precision-specific tasks. Current FP32/BF16 targeted evidence is secondary; no full-model INT8/INT4 LLM cell is measured.

Gradient methods are measured on all three full-kernel INT8 studies and on the separate FKeras KWS cohort. The restricted transformer campaigns have no matching validated gradient or Hessian mask; their matrix says `NOT_IMPLEMENTED`. PrioriFI-derived allocation is unimplemented for every model. Published-form FKeras is measured only on the separate 128-clip KWS cohort. This is an honest method-by-model coverage report, not a plot legend filtered to successful methods.

## Bounded resource estimate, not authorization to run

The source manifests report coordinator elapsed time of approximately 12,019 s for 3,584 ResNet-8 policy evaluations (2 workers), 2,701 s for 3,584 KWS evaluations (4 workers), and 1,288 s for 448 ToyCar evaluations (2 workers, 64 files). Artifact directories occupy approximately 2.4 GiB, 685 MiB and 3.8 GiB respectively. These include model/ranking/setup, so per-evaluation divisions are rough upper-order engineering proxies, not performance guarantees. A new two-partial-tier, six-policy grid at the existing BERs and 32 fields would add 3,072 ResNet-8 and 3,072 KWS policy evaluations; using the observed ratios gives roughly 2.9 h and 0.64 h respectively before contention. ToyCar with four BERs and 32 fields would add 1,536 evaluations; the pilot ratio gives roughly 1.2 h, with particularly high storage risk (a naive scaling of the source bundle is ~13 GiB). FKeras eigensolving, full-cohort rank construction and any PrioriFI feedback calibration are **not included** in these estimates. Final cost must be recalculated from a frozen plan and current host load.

Two M11 processes were active during this database build. No new production campaign was started. The requested future orchestration script should be implemented only after the missing protocol cells, source assets, and resource/host-contention plan have been approved by the researcher.

## Query examples

```sql
SELECT study_id, method_id, coverage, ber, mean_loss, loss_ci_low, loss_ci_high,
       loss_degradation, field_count
FROM aggregated_conditions
WHERE study_id='kws_speechcommands_int8_full'
ORDER BY coverage, ber, method_id;

SELECT method_id, coverage, ber, status, reason
FROM coverage_matrix
WHERE study_id='resnet8_cifar10_int8_full' AND status!='FINALIZED_MEASURED';

SELECT study_id, method, coverage, ber, mean_loss, failed_fraction
FROM restricted_conditions ORDER BY study_id, ber, coverage;
```

`conditions_long.csv`, `coverage_matrix.csv`, `restricted_conditions_long.csv`, `restricted_coverage_matrix.csv`, and `planned_model_methods.csv` export these queryable rows. Plots use the saved raw trials and condition statistics only; they do not rerun inference. Each plotted point is measured, with no interpolation. The full-kernel budget figure focuses on BER values 1e-4, 1e-3, 1e-2 and 1e-1 where available; the complete measured BER grid is retained in CSV and SQLite.
