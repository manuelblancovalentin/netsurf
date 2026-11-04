# M12 published-form FKeras: frozen matched-study plans

These four schema-3.6 TOMLs are **prepared, validated, and unrun**. They use
the pinned TensorFlow 2.14.1 `BUILTIN_REF` executable and exact kernel-only
INT8 XOR faults. No prior result, asset lock, checkpoint, dataset or frozen
configuration is changed. Execute only after researcher authorization and
fresh host admission; Qwen W8A8 was running when these plans were prepared.

## Source evidence and unchanged cohorts

The read-only planning source is the finalized ResNet six-method campaign
(`7bc87303976b2b082902947e03d45b6cfa87e3652b52df42d3324e249fee8012`)
and the finalized ToyCar six-method campaign
(`428519770a9b9c211bf23d48f42adbc7d295f9393a6835b9b2e748c7ed1fe18b`).
The new campaigns use **new fault seeds**, but retain the same previously
exposed cohorts: ResNet ranking CIFAR train indices 0–63, SHA-256
`f0f34d666631ab202850b59f833dcdc20c34053cea0cd8be78bb29858b592250`,
evaluation CIFAR test indices 0–999, SHA-256
`a3d025b12e92c6d8138689aeec27331a826734f510ffec65a977cd9bb2a1d997`;
ToyCar ranking training files 0–15, SHA-256
`4bbf2f3c29475bfe50d6c2bd3ea89b0553f36daa368eb1ea4305b97de4ff2831`,
evaluation 64 complete files, SHA-256
`95858ad6700174e7168367768176f43d7f6af3adb4128ff92240fa399dfa77cf`.
The pinned adapters reproduced these hashes for the new plans. These are
prospective **fault-field** comparisons conditional on fixed, already exposed
inputs, one checkpoint per model and the recorded masks. They are not new
input-cohort or independently trained-model confirmations.

## Mathematical and numerical gate

`fkeras_paper` is the published weight-Hessian projection
`abs(sum_i lambda_i (v_i^T theta) v_i)`, used only within descending INT8
bit-significance tiers. No finite-bit jump or diagonal-Hessian substitution is
inserted. ResNet's continuous Hessian loss is classification cross-entropy;
ToyCar's is reconstruction MSE, while exact task evaluation is complete-file
`1 - ROC-AUC`. At 12.5% and 25%, homogeneous INT8 tier-first FKeras and
corrected MSB necessarily have identical masks. Only 6.25% and 18.75% are
primary FKeras-versus-MSB decision comparisons.

Using the **actual planned ranking cohorts**, the pinned bounded eigensolver
returned two eigenpairs in 25 HVP calls each: ResNet 64 images, maximum
normalized residual `5.92e-16`, 18.51 seconds process wall, 2,060,368 KiB
peak RSS; ToyCar 16 complete training files/3,136 windows, maximum residual
`5.06e-16`, 20.11 seconds, 1,504,912 KiB peak RSS. These are direct bounded
HVP/eigensolve checks, not full ranking or evaluation timings. The earlier
one-field engineering bundles verify full-bit mapping, mask cardinalities,
exact restoration and task-level execution. They do not estimate effectiveness.

## Four separate studies

| TOML suffix | BERs | Budgets | Proposed fields | Policy evaluations | Scheduled fault work |
|---|---|---|---:|---:|---:|
| `resnet8_fkeras_matched_grid_v1` | 8 historical values, `1e-5`–`0.1` | 6.25, 12.5, 18.75, 25% | 256, 32/BER | 7,680 | 7,936 |
| `toycar_fkeras_matched_grid_v1` | `1e-5`, `1e-4`, `1e-3`, `0.003`, `1e-2` | same four | 160, 32/BER | 4,800 | 4,960 |
| `resnet8_fkeras_paired_precision_v1` | `0.003`, `0.01` | 6.25, 18.75% | 640, 320/BER | 5,120 | 5,760 |
| `toycar_fkeras_paired_precision_v1` | `0.001`, `0.01` | 6.25, 18.75% | 768, 384/BER | 6,144 | 6,912 |

ToyCar's additional `0.003` is a prespecified diagnostic between the measured
`0.001` and `0.01` regimes, whose historical unprotected AUC was nonmonotonic;
it is not an interpolated result or a new primary contrast. The grid includes
random, corrected MSB, numerical jump, global first-order,
per-example absolute, per-example positive and published-form FKeras, plus
shared no/full controls. Four grid replications use eight independent fields
each per BER and varied estimator/ranking seeds; the resulting mask variability
must remain separate from fault-field variability. The higher-precision plans
include only MSB, per-example absolute and FKeras, plus controls. They hold
each ranking mask fixed (`replications.count = 1`) across all new fields.
No grid result is pooled with an older six-method or separate focused study.

## Prespecified paired inference

For each model, the focused study's **primary metric** is ResNet categorical
cross-entropy in nats or ToyCar complete-file risk `1 - ROC-AUC`. Define
`D = risk(FKeras) - risk(comparator)` on the *same* `(BER, trial,
schedule_sha256)` field. Negative `D` favors FKeras. The exact eight-member
comparison family per model is FKeras versus each of MSB and per-example
absolute ranking, crossed with two BERs and the two partial-tier budgets.
Report signed paired means, standard deviations, 95% descriptive intervals,
and simultaneous Bonferroni 99.375% two-sided intervals. Use a two-sided
paired sign-flip permutation test with Holm correction across these eight
tests; call a contrast statistically resolved only if its adjusted `p < 0.05`
and its direction and interval are reported. Accuracy for ResNet and window
MSE for ToyCar are secondary, with no transferred primary significance claim.
Do not use marginal-policy intervals, individual file windows or multiple
policies on one field as independent observations. The engine's automatically
reported broader pairwise family remains supplementary; the prespecified
eight contrasts must be recalculated directly from saved trial rows after
finalization. No optional stopping or post-hoc BER/budget selection is allowed.

Historical 32-field ResNet S5-minus-MSB paired CE standard deviation at
BER `0.01`, coverage 6.25%, was `5.08082` nats; historical eight-field ToyCar
S5-minus-MSB paired AUC standard deviation at BER `0.001`, coverage 6.25%,
was `0.298`. These are **proxies**, not FKeras variance estimates. With a
conservative per-contrast Bonferroni critical value `z = 2.734`, 320/384
fields imply proxy simultaneous half-widths about `0.78` nats / `0.042` AUC
and approximate 80%-power detectable differences about `1.02` nats / `0.054`
AUC. This is conditional on those proxy variances and an approximate normal
calculation; heavy tails, ties, lower effects or different FKeras variance can
leave results unresolved. Statistical significance cannot be guaranteed by a
large trial count. No ranking-seed or input-population uncertainty is inferred
from the fixed-mask focused studies.

## Resource and launch gate

The authoritative historical manifests report ResNet `43,960.4` seconds for
6,656 policy evaluations over 1,000 images (full-run peak `5,088` MiB) and
ToyCar `3,383.0` seconds for 832 policy evaluations over 64 files (peak
`10,755` MiB). Simple evaluation-count scaling gives rough ResNet grid/focused
guides `14.09`/`9.39` hours and ToyCar `5.42`/`6.94` hours, about `35.8`
hours total. Setup, scoring, different BER mix, report serialization, machine
contention and archive creation make these *capacity estimates*, not ETAs.
Source bundle sizes were 2.5/4.0 GiB; plan at least 25 GiB free for four
new bundles and their later exports, and allow materially more. Check live CPU,
RAM, swap and disk again; do not run alongside the active Qwen campaign without
explicit resource admission. Run one study at a time in foreground `tmux`.

The foreground runner validates a clean worktree, the exact expected Git
commit, schema and pinned assets, then prints the operation count and the
historical-ratio estimate before execution. The handoff records the final
source commit; use it literally as the second argument. The script also
rejects any TOML whose bytes differ from these four frozen SHA-256 digests:

| Plan | SHA-256 |
|---|---|
| ResNet grid | `237f4697d69d9778cfd2f08510cc5ec4099ce1d1693b854c77906b83e901496a` |
| ResNet focused | `00ab9e7e7697142c3205a4476b64c3ce7efe1a3f17f9ee68c79fefa344a67e46` |
| ToyCar grid | `f8b59c0b3674f84ffd126fda82b6c03b3f441216b63e846aa95f591fc446d806` |
| ToyCar focused | `967f4048e939a04b98f0ff81a600e8a450b6ed5bb2ba65ac65b18c7c72614057` |

Never rerun a finalized configuration into its previous output. Verify each
result's `FINALIZED`, manifest, checksums, sample hashes, shared schedules,
exact masks, proposed/applied/suppressed accounting and full-protection
equality before interpreting effects. All four are separate studies in any
later database ingestion. No run or derived output is a physical ECC/radiation
measurement.
