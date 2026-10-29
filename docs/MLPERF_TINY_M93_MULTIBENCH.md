# M9.3 MLPerf Tiny multi-benchmark gate

This document records the M9.3 implementation and bounded validation performed
on 2026-10-29. The adapters consume immutable Kona Assets locks and never
rewrite models or datasets. TensorFlow Lite 2.14.1 `BUILTIN_REF`, with no
delegates and one thread, is authoritative for clean and faulted inference.
Differentiable companions supply ranking derivatives only.

## Readiness matrix

| Benchmark | Reference inference | Exact kernel XOR | Companion | S4--S6 | Empirical bit audit | Protection-ready |
|---|---|---|---|---|---|---|
| ResNet-8 image classification | VERIFIED (M7.1/M9) | VERIFIED | VERIFIED surrogate | VERIFIED | VERIFIED | VERIFIED |
| DS-CNN keyword spotting | VERIFIED: 1,000-example output digest and 0.901 accuracy | VERIFIED, including depthwise and pointwise kernels | PARTIAL: topology/mapping verified; not integer-program equivalent | VERIFIED, bounded | VERIFIED, bounded | VERIFIED for kernel-only persistent trials |
| ToyCar anomaly detection | VERIFIED: 248-file score digest and 0.862037037 ROC-AUC | VERIFIED | PARTIAL: topology/mapping verified; not integer-program equivalent | VERIFIED, bounded for reconstruction MSE | VERIFIED, bounded | VERIFIED for kernel-only persistent trials with task-specific metrics |
| MobileNetV1-0.25 VWW candidate | BLOCKED: official evaluation-pixel identity unresolved | NOT STARTED | NOT STARTED | NOT STARTED | NOT STARTED | BLOCKED |

`PARTIAL` for a companion is deliberate: an STE graph cannot be claimed as the
derivative of the exact integer TFLite program. Exact isolated-bit and
protection outcomes always use the verified reference interpreter.

## Immutable identities and eligible storage

| Identity | Model SHA-256 | Kernel tensors | INT8 words | Eligible bits |
|---|---|---:|---:|---:|
| `mlperf_tiny_ic_resnet8_int8_v1` | `3c002613...05415` | 10 | 77,360 | 618,880 |
| `mlperf_tiny_kws_dscnn_int8_v1` | `aeea4368...bd0ae` | 10 | 22,016 | 176,128 |
| `mlperf_tiny_ad_toycar_int8_v1` | `87cf2419...ba24` | 10 | 264,192 | 2,113,536 |

The M9.3 populations contain every allocated INT8 kernel word, including
encoded zero. INT32 biases, activations, affine metadata, and operator metadata
are excluded. Bit identities are `(TFLite tensor, flat word index, LSB-zero bit
index)`. Multiple events in one word compose by XOR. Protection is ideal event
suppression, not ECC, TMR, or a physical-radiation guarantee.

The older blocked 49-logit QKeras keyword-spotting candidate and 320-input
historical autoencoder remain separate identities. The new adapters do not
retroactively validate them.

## Reference and companion conformance

### DS-CNN KWS

The official pinned MFCC preparation is reproduced from the TFDS shard order:
round-robin blocks of 16, waveform maximum normalization, length 16,000,
480/320 STFT, 40 mel bins, and 10 MFCC coefficients. No approximate
spectrogram pipeline is substituted.

The complete pinned 1,000-example subset produced accuracy `0.901` and output
SHA-256 `a62256c3f761c18a7ce7f39c44ff4134df5052d96e95b5867e7f6e8db0120b96`,
exactly matching the asset lock. On a bounded eight-sample companion check,
90.625% of output codes matched, 62.5% of complete vectors were exact, encoded
MAE was 0.197917, and the maximum difference was five INT8 codes. These
discrepancies require empirical estimator validation; they do not affect exact
fault evaluation.

Ordinary adapter loading also replays a source-controlled 16-example output
digest derived from this full gate. A preprocessing, dataset, runtime, or model
change therefore fails before a study begins; the full 1,000-example gate
remains available for an explicit audit.

### ToyCar anomaly detection

The official pinned pipeline uses 128-bin librosa mel power, FFT 1024, hop 512,
reflect padding, frames 50:250, five-frame concatenation, and 196 windows per
file. Its float64 target identity additionally pins the validated NumPy
OpenBLAS build's two-thread reduction mode; spawned workers enforce this only
during feature extraction while TFLite remains single-threaded. The complete
248-file run produced ROC-AUC `0.862037037037037` and
file-score SHA-256
`f5174517de344da8126ea1875baae36dfd53bf37151ed18e058bd5a57e0fb4b5`,
exactly matching the lock.

The parallel-initialization root cause, exact intermediate digests, and
diagnostic replay are documented in `docs/TOYCAR_PARALLEL_CONFORMANCE.md`.

Ordinary adapter loading replays a two-file, 392-window file-score digest
derived from the full gate. The full 248-file conformance remains an explicit
audit rather than being repeated for every experiment replication.

Ranking differentiates per-window float-target reconstruction MSE. Final
evaluation uses file-level anomaly scores and ROC-AUC. They are related but not
identical objectives; ROC-AUC is never differentiated or serialized as
classification accuracy. A bounded two-window companion check matched 61.02%
of output codes, with encoded MAE 0.3945 and maximum error two codes. Exact and
surrogate reconstruction MSE were 10.3588 and 10.4188 respectively.

Ranking uses development normal-train files. Empirical validation and final
evaluation use deterministic, disjoint, label-containing halves of the
official development test. The complete 248-file set is used only by the clean
conformance method, never simultaneously for estimator selection and final
protection evaluation.

## Estimator and experiment contract

Schema `3.4.0` adds task-named primary/loss fields while preserving readers for
schema 3.0--3.3 bundles. It accepts:

- adapters `mlperf_tiny_kws_int8_v1` and `mlperf_tiny_ad_int8_v1`;
- benchmark identities listed above;
- `quantization = "tflite_affine_int8"` and
  `quantization_backend = "tflite_affine_integer_v1"`;
- `injectable_parameters = ["kernels"]` only;
- `inference_backend = "tensorflow_lite_builtin_ref_v1"`;
- KWS split `official_subset_1000` and ToyCar split `official_248_files`;
- persistent-trial independent Bernoulli logical faults;
- random, corrected word-relative MSB (S2), exact numerical jump (S3), global
  first order (S4), per-example absolute first order (S5), and per-example
  positive first order (S6), with their existing implementation versions;
- uniform or stratified exact-bit candidate selection, multiple coverages and
  logical BERs, independent ranking/fault seeds, and replications.

ToyCar requires schema 3.4, and at least two files in each empirical/final
partition so both labels are present. `trials.csv`, `protection_results.csv`,
paired comparisons, summaries, reports, and plots retain explicit
`roc_auc`/`reconstruction_mse` names. Replication pooling matches
`(replication, logical BER, trial, schedule SHA-256)` and also writes
replication-level summaries.

Researcher-authored TOMLs must use these exact benchmark fields (paths may be
replaced only by assets that pass the same lock and content hashes):

| Field | KWS | ToyCar |
|---|---|---|
| `benchmark.adapter` | `mlperf_tiny_kws_int8_v1` | `mlperf_tiny_ad_int8_v1` |
| `benchmark.id` | `mlperf_tiny_kws_dscnn_int8_v1` | `mlperf_tiny_ad_toycar_int8_v1` |
| `benchmark.checkpoint` | `/ml/models/models--mlcommons--tiny/snapshots/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite` | `/ml/models/models--mlcommons--tiny/snapshots/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/anomaly_detection/trained_models/ad01_int8.tflite` |
| `benchmark.dataset` | `/ml/datasets/speech_commands/0.0.3` | `/ml/datasets/dcase2020-toycar` |
| `benchmark.asset_lock` | `/projects/kona-assets/manuelbv/locks/mlperf-tiny/kws.lock.json` | `/projects/kona-assets/manuelbv/locks/mlperf-tiny/ad.lock.json` |
| `benchmark.split` | `official_subset_1000` | `official_248_files` |

Both also require:

```toml
schema_version = "3.4.0"

[benchmark]
inference_backend = "tensorflow_lite_builtin_ref_v1"
quantization = "tflite_affine_int8"
quantization_backend = "tflite_affine_integer_v1"
injectable_parameters = ["kernels"]

[fault]
target = "weight"
location = "logical_trainable_parameter_bits"
persistence = "persistent_trial"
distribution = "independent_bernoulli"

[protection]
mitigation_model = "perfect_event_suppression"
```

The remaining required schema-3.4 tables are `[experiment]`, `[samples]`,
`[candidates]`, one or more `[[estimators]]`, `policies`, `[execution]`,
`[replications]`, and `[output]`. Validation rejects unsupported estimators,
storage populations, split names, and fault persistence before model loading.

The exact estimator versions remain:

| Method | Configuration name | Implementation version |
|---|---|---|
| S2 | `msb_first` | `relative-word-bit-index/v1` |
| S3 | `numerical_bit_jump` | `absolute-affine-xor-jump/v1` |
| S4 | `first_order_gradient_delta` | `signed-gradient-times-exact-delta/v2` |
| S5 | `per_sample_first_order_magnitude` | `mean-absolute-per-example-gradient-times-exact-delta/v1` |
| S6 | `per_sample_positive_damage` | `mean-positive-per-example-gradient-times-exact-delta/v1` |

## Bounded cost observations

These are engineering measurements on Kona, not portable performance claims.

| Operation | Bounded observation |
|---|---:|
| KWS gated adapter load (includes 16-example digest) | 2.79 s |
| KWS 1,000-example conformance, including preprocessing | 37.7 s; 471,544 KiB peak RSS |
| KWS S3 over all 176,128 bits | 0.231 s |
| KWS S4 over eight calibration examples | 0.379 s |
| KWS shared S5/S6 over eight calibration examples | 3.63 s |
| KWS one-fault evaluation over 32 prepared examples | 0.090 s (356 examples/s) |
| ToyCar gated adapter load (includes two-file digest) | 1.97 s |
| ToyCar 248-file conformance, including preprocessing | 17.5 s; 1,550,656 KiB peak RSS |
| ToyCar S3 over 1,024 selected identities | 0.026 s |
| ToyCar S4 over two windows/1,024 selected identities | 0.124 s |
| ToyCar shared S5/S6 over two windows/1,024 identities | 0.073 s |
| ToyCar one-fault evaluation over two prepared files (392 windows) | 0.038 s (10,352 windows/s) |

Per-example S5/S6 gradient cost grows approximately with calibration-window
count. Exact empirical cost grows with candidate count times authoritative
evaluation cost. ToyCar's 2.11-million-bit Python identity/score maps are a
material memory limit: the bounded estimator process peaked near 880 MiB, and
the full clean audio conformance peaked near 1.48 GiB. Publication studies must
budget memory and artifact size explicitly. NetSuRF does not compress away
identities or fault events needed for replay.

## Environment and reproducibility commands

The canonical NetSuRF environment can run KWS. ToyCar preprocessing additionally
requires a compatible librosa installation; the validated reference was the
existing pinned Kona Assets TensorFlow 2.14.1 environment. The environment
owner, not NetSuRF, controls installation.

The canonical `netsurf` environment currently lacks librosa. If ToyCar is to
run there, the exact validated addition for the environment owner is:

```bash
python -m pip install "librosa==0.10.2.post1"
```

That package is required only for the pinned waveform-to-log-mel preprocessing;
KWS and existing core tests are not blocked by its absence.

```bash
proj activate netsurf
netsurf doctor
netsurf benchmarks info mlperf_tiny_kws_dscnn_int8_v1
netsurf benchmarks info mlperf_tiny_ad_toycar_int8_v1
netsurf benchmarks identity mlperf_tiny_kws_dscnn_int8_v1
netsurf benchmarks identity mlperf_tiny_ad_toycar_int8_v1
netsurf validate /path/to/researcher-authored-schema-3.4.toml --resolved
```

For an installation without `proj`, invoke the validated interpreter
explicitly after arranging equivalent pinned dependencies:

```bash
/absolute/path/to/python -m netsurf doctor
/absolute/path/to/python -m netsurf validate /path/to/study.toml --resolved
```

M9.3 deliberately does not add publication experiment TOMLs or launch research
sweeps. VWW remains blocked until an authoritative image archive or pixel-level
identity record is obtained and checked.
