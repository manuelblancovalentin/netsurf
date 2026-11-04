# Published-form FKeras on ResNet-8 and ToyCar: bounded implementation gate

This extension starts at canonical `v2` commit `e3209f7f67ef8f30cebb0eed791bf817164d39c0`
in an isolated worktree. It changes no finalized experiment, asset lock,
stored-bit fault law, authoritative backend, S4–S6 definition, or KWS FKeras
formula. It does **not** establish protection superiority. The prior M97
FKeras paper/source audit and its pinned citation remain in
`docs/M97_FKERAS_REPRODUCTION.md`.

## Exact adaptation and separate objectives

The only new ranking method is published-form `fkeras_paper`, whose signed
parameter projection is

\[
H' = \sum_{i=1}^{k}\lambda_i(v_i^\top\theta)v_i.
\]

The ordering uses `|H'_j|` *within* each descending INT8 bit-significance
tier, followed by stable encoded-bit identity. It does not multiply by the
finite XOR jump, predict signed bit damage, or estimate conditional protection
benefit. The pinned public repository's different sum score remains available
for historical KWS schema 3.5 only and is **not** extended as a second paper
baseline.

| Benchmark | Differentiated decoded variables | Continuous Hessian loss | Exact final task evaluation | Eligible kernel bits |
|---|---|---|---|---:|
| ResNet-8 / CIFAR-10 | 77,360 INT8 kernel words, 10 tensors | mean float64 continuous-logit sparse CE | TF 2.14.1 `BUILTIN_REF` classification CE and accuracy | 618,880 |
| ToyCar / anomaly detection | 264,192 INT8 kernel words, 10 tensors | mean float64 per-window reconstruction MSE | TF 2.14.1 `BUILTIN_REF` complete-file ROC-AUC; authoritative risk `1-AUC` | 2,113,536 |

Every differentiated variable is initialized from its exact affine-decoded
stored INT8 code. Fixed decoded INT32 biases participate in forward inference
but are neither differentiated nor eligible. Float64 continuous companions
omit activation rounding/clipping and remain separate from the established
float32 STE companions. ReLU kinks limit finite-difference neighborhoods.
Neither continuous Hessian is a Hessian of the exact integer TFLite program.
ToyCar's MSE Hessian is a surrogate for, not a derivative of, file AUC.

The shared M97 eigensolver retains signed largest-magnitude eigenpairs, checks
HVP symmetry, normalized residuals and orthogonality, and rejects numerical or
work-limit failures. The ten tensor names, shapes, flat offsets and affine
metadata map the projection back to exactly one scalar score per stored
kernel word. All eight encoded bits of that word receive the same within-tier
score; an explicit population gate checks unique complete bit identities.

## Bounded numerical receipts

The pinned TF 2.14.1 interpreter and source asset locks passed. ResNet used
one previously exposed CIFAR training image for the HVP preflight; ToyCar used
two windows of one training file. Both used two eigenpairs, tolerance `1e-5`,
seeded initialization and a cap of 150 HVP products.

| Gate | ResNet-8 | ToyCar |
|---|---:|---:|
| Dimension | 77,360 | 264,192 |
| HVP products | 25 | 25 |
| Largest normalized residual | 2.20e-15 | 3.45e-16 |
| Symmetry error | 1.90e-18 | 2.27e-19 |
| Orthogonality error | 6.76e-16 | 1.11e-15 |
| Eigensolve wall time after adapter construction | 1.10 s | 0.74 s |
| Directional finite-difference relative HVP error at step `1e-7` | 1.07e-9 | 5.68e-9 |

The finite-difference check tests local smoothness around the frozen decoded
weights, not a Taylor approximation for large INT8 sign-bit jumps. The ToyCar
complete 196-window calibration-file HVP was 0.356 s on its first traced call
and 0.016 s on a repeat in the bounded process. These numbers are not a
portable full-campaign runtime prediction.

Forward discrepancy was measured rather than assumed away. On 64 already
exposed official ResNet images, continuous versus reference classification
agreed on 61/64 classes, with probability MAE `0.00603`, maximum probability
error `0.33188`, reference CE `0.91815` and continuous CE `0.62556` nats.
This is substantial loss disagreement despite high class agreement: the
continuous Hessian is a baseline surrogate, not a validated damage predictor.
On two previously exposed complete ToyCar files (392 windows), decoded output
MAE was `0.14789`, maximum absolute output error `4.62531`, reference window
MSE `10.97316` and continuous MSE `10.94825`. A two-file MSE match says
nothing about file-AUC ranking over a larger cohort. Exact empirical bit-damage
validation and held-out protection comparisons remain necessary.

## Exact one-field engineering bundles

Both smokes used previously exposed inputs and one proposed field. They are
implementation checks, not estimates of policy effectiveness or new
confirmation data. Neither used a new checkpoint or altered an asset lock.

| Bundle | Scientific payload SHA-256 | FKeras score time | Mask symmetric difference at 6.25% / 12.5% / 18.75% / 25% |
|---|---|---:|---|
| `m12_resnet8_fkeras_smoke` | `64f2da6e9c781baa70d507e1f7d32f7825f4b397d3fd4c2dba7b18f1c842c048` | 3.23 s | 13,754 / 0 / 13,754 / 0 bits |
| `m12_toycar_fkeras_smoke` | `2a81a96363f7abd8f973bf129276372c591315ad3bf14ceb0156b244e265cdcd` | 7.79 s | 90,630 / 0 / 90,630 / 0 bits |

The saved masks protect respectively 38,680 / 77,360 / 116,040 / 154,720
ResNet bits and 132,096 / 264,192 / 396,288 / 528,384 ToyCar bits. At
12.5% and 25%, FKeras and corrected MSB are the *same sets*, so paired
outcomes on the same proposed field must coincide. They are not independent
method observations. Both bundles passed `netsurf inspect` checksum checks,
partition-disjointness, exact quantized-state and restoration checks. ToyCar
evaluated all 392 windows of two complete, differently labeled files under
one persistent fault state and kept `roc_auc` separate from
`reconstruction_mse`. Two files yield only an extreme 0/1 AUC and cannot
support a statistical claim.

The two engineering bundles occupy about 179 MB and 597 MB. Their complete
foreground process durations were approximately one minute and three minutes,
respectively; reported 3.23/7.79 seconds count only FKeras construction and
must not be used as end-to-end campaign estimates. Full-population score CSVs
and masks dominate output size. Peak ToyCar smoke RSS observed during the
process was approximately 1.5 GiB. The new schema-3.6 result payloads are
distinct from historical schema-3.5 KWS results.

## Configuration and future campaign gate

Schema `3.6.0` permits the already pinned KWS `fkeras_paper` and, for the
two models here, only the following new estimator combinations:

```toml
[[estimators]]
name = "fkeras_paper"
implementation_version = "fkeras-paper-continuous-resnet8/v1" # or fkeras-paper-continuous-toycar/v1
seed = 1208102
hessian = {eigenpairs = 2, max_iterations = 100, max_products = 150, tolerance = 0.00001, ordering = "largest_magnitude_signed"}

[[policies]]
name = "fkeras_paper"
estimator = "fkeras_paper"
priority = ["bit_significance_desc", "estimator_score_desc", "stable_identity"]
prespecified = true
```

The new fields are the existing versioned Hessian options; the benchmark
adapter, checkpoint/dataset/asset lock, kernel-only population, task metric,
sample partitions, BER, coverage, ranking seed and fault seed remain explicit.
Schema 3.5 continues to accept FKeras only for KWS. Schema 3.6 rejects the
repository variant for ResNet/ToyCar, wrong implementation versions, bias-inclusive
storage, non-tier-first FKeras ordering, unknown Hessian options and unsupported
numerical orderings.

Engineering templates are
`config/examples/m12_resnet8_fkeras_smoke.toml` and
`config/examples/m12_toycar_fkeras_smoke.toml`. The optional foreground
launcher `scripts/run_m12_fkeras_study_once.sh CONFIG.toml` validates and
prints the operation plan before running a researcher-frozen configuration;
it is **not** a production authorization. Do not rerun the one-field smokes.

For a matched effectiveness study, use the same eligible population, calibration
and final cohorts, exact mask cardinality and proposed fault fields as all
other methods. Compare FKeras against MSB at partial tiers 6.25% and 18.75%.
Before a production run, freeze BERs, primary paired contrasts, desired CI
precision and independent field count; then measure resource cost of the
actual cohort in a separate pilot. Existing whole-grid receipts provide only a
rough initial capacity guide: 6,656 ResNet policy/budget observations over
1,000 images took about 6.2 hours by a historical source-ratio estimate, while
832 ToyCar observations over 64 complete files took about 0.7 hours by the
same method. Adding FKeras as one extra policy at four budgets and the same
32/8 fields per BER would add 1,024/128 policy-budget observations, respectively,
before any added fields or rerun of comparator policies. These estimates are
not validated throughput measurements for the new configurations; extra
independent fields and ranking-calibration changes scale cost further.
The above tiny smokes cannot determine a
scientifically powered trial count. Do not add BER points for plot smoothness
or infer that MSE-based FKeras ranks file-AUC vulnerability accurately.
