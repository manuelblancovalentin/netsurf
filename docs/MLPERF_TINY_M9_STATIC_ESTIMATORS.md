# M9: MLPerf Tiny INT8 static susceptibility estimators

## Scope and authority

M9 adds static kernel-bit estimators to the pinned MLPerf Tiny ResNet-8 image
classifier. The checkpoint remains the immutable TFLite model with SHA-256
`3c002613d1b2475eb51dd78dfb85a546c8ae658dee71cf6ade43b022fe205415`.
TensorFlow 2.14.1 `tf.lite.Interpreter` with `BUILTIN_REF`, no delegates and
one thread remains the only authoritative clean and faulted inference backend.

The differentiable companion is not a replacement model. It is an
architecture-specific straight-through graph whose trainable variables are
the dequantized values of the ten actual TFLite INT8 kernel buffers. INT32
biases are dequantized constants but are not fault eligible in M9. No model is
trained and no checkpoint is modified.

The official model topology and conversion provenance are documented by the
[MLPerf Tiny image-classification source](https://github.com/mlcommons/tiny/tree/master/benchmark/training/image_classification).

## Parameter correspondence

The companion consumes TFLite OHWI convolution kernels and the OI dense
kernel directly. The official HDF5 checkpoint (SHA-256
`5f938a8eea605438cc7360aa13c987bc6d7bd0c8d931b2917361ae7ae4eeda2f`)
is used only for a mapping diagnostic. Keras convolution kernels are changed
from HWIO to OHWI; the seven BatchNormalization layers are folded as
`kernel * gamma / sqrt(moving_variance + epsilon)`. Projection and dense
kernels require no BN fold.

| TFLite tensor | Keras layer | Scalars | Relative L2 error | Correlation |
|---:|---|---:|---:|---:|
| 8 | conv2d | 432 | 0.005027 | 0.9999875 |
| 9 | conv2d_1 | 2,304 | 0.007418 | 0.9999725 |
| 10 | conv2d_2 | 2,304 | 0.007901 | 0.9999688 |
| 11 | conv2d_3 | 4,608 | 0.007082 | 0.9999748 |
| 12 | conv2d_4 | 9,216 | 0.007291 | 0.9999734 |
| 13 | conv2d_5 | 512 | 0.005212 | 0.9999863 |
| 14 | conv2d_6 | 18,432 | 0.007369 | 0.9999727 |
| 15 | conv2d_7 | 36,864 | 0.007838 | 0.9999692 |
| 16 | conv2d_8 | 2,048 | 0.005838 | 0.9999823 |
| 7 | dense | 640 | 0.007386 | 0.9999724 |

These residuals are consistent with per-channel INT8 quantization. The HDF5
values do not participate in gradient evaluation after this diagnostic.

## Straight-through graph and acceptance gate

Every TFLite activation scale and zero point is copied into an explicit
requantization boundary. The forward value is rounded and clipped to INT8;
the backward rule is the identity through rounding and clipping:

`x_ste = x + stop_gradient(requantize(x) - x)`.

TFLite fixed-point multiplier rounding is not reproduced. Gradient loss is
sparse cross-entropy on the requantized pre-softmax logits, whereas empirical
damage uses NetSuRF's registered loss on the authoritative INT8 softmax
output. This difference is intentional and recorded in each manifest.

On the official 200-image subset, the gate measured:

- class agreement: 0.985;
- exact output-code fraction: 0.8505;
- exact ten-code-vector fraction: 0.50;
- probability mean absolute error: 0.00423047;
- maximum probability error: 0.308594;
- authoritative accuracy: 0.875;
- companion accuracy: 0.870.

The provisional engineering criteria are class agreement at least 0.95,
probability MAE at most 0.01, and accuracy difference at most 0.02. Passing
them permits bounded estimator validation only. The large worst-case output
error means the companion must not be described as bit-exact. These thresholds
require scientific approval before they become a publication protocol.

## Estimator definitions

For encoded kernel bit `b`, let `delta_b` be the exact per-channel affine
decoded jump produced by XOR of that stored bit.

- **S2 `msb_first`**: relative bit position in the eight-bit word; sign bit
  first. It has no task sensitivity.
- **S3 `numerical_bit_jump`**: `abs(delta_b)`. It is not a signed loss
  predictor.
- **S4 `first_order_gradient_delta`**: signed prediction `g_b delta_b` and
  ranking score `abs(g_b delta_b)`, where `g_b` is the gradient of mean
  companion loss.
- **S5 `per_sample_first_order_magnitude`**:
  `mean_x(abs(g_b(x) delta_b))`.
- **S6 `per_sample_positive_damage`**:
  `mean_x(max(0, g_b(x) delta_b))`.
- **S9 empirical reference**: exact TFLite isolated-bit signed loss and
  accuracy changes on sampled candidates.

All 256 INT8 codes and all eight XOR positions are regression-tested against
the scalar affine transition. Gradients preserve the original TFLite tensor
identity and C-order flat index. Per-example gradients use explicit examples,
not TensorFlow `pfor`, avoiding retracing and batch-size-dependent semantics.

## Bounded evidence

The finalized estimator-validation run used eight training images for
ranking, a disjoint 16-image training partition for empirical damage, 16
uniformly sampled kernel bits, and 16 held-out official test images for clean
evaluation. Its payload is
`c97ad027303fcd102028e6095d6356e16cce990c9ee5a44fe352385ad129b233`.

| Estimator | Spearman positive damage | Kendall tau-b | Top-10 recall | Runtime (s) |
|---|---:|---:|---:|---:|
| S2 MSB | 0.7217 | 0.5744 | 0.80 | 0.000 |
| S3 numerical jump | 0.5588 | 0.4129 | 0.80 | 0.841 |
| S4 gradient × delta | 0.7000 | 0.5780 | 0.90 | 1.317 |
| S5 per-example absolute | 0.7000 | 0.5780 | 0.90 | 12.321 shared |
| S6 per-example positive | 0.3101 | 0.2110 | 0.80 | 12.321 shared |
| Random control | -0.0921 | -0.0642 | 0.50 | 0.000 |

These are sample-limited diagnostics with only 16 candidates. They neither
establish estimator superiority nor estimate whole-network top-k recall.
Schema 3.2 records expected recall under randomized selection at a tied score
boundary and the `k/N` random expectation.

The two-trial protection smoke finalized with payload
`3bd0b19bc9ba054145223ba7b2229d582063d996fb4319eb81914df5bbf3a785`.
It used eight held-out samples, BER `1e-5`, 1% coverage and matched schedules.
The two schedules proposed eight and five flips; no 1%-mask happened to
suppress an event, so all non-full policies were identical. This validates
execution and restoration only and contains no protection-effect evidence.

## Runtime and storage scaling

The validation took 2:56.9, peaked at 2.23 GB RSS and produced a 464 MB
bundle. The smoke took 3:30.2, peaked at 2.34 GB RSS and produced a 489 MB
bundle. Full-score CSV serialization and pairwise full-ranking diagnostics
dominate both runs. Exact empirical evaluation averaged 0.1997 seconds per bit
for 16 samples; naive exhaustive evaluation of 618,880 bits would therefore
take about 34.3 hours before reporting overhead and is not an approved M9
command.

## Reproduction

Use the pinned TensorFlow 2.14 environment for scientific execution:

```bash
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf validate \
  config/experiments/m9_resnet8_static_estimator_validation_v1.toml --resolved

/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf run \
  config/experiments/m9_resnet8_static_estimator_validation_v1.toml

/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf run \
  config/experiments/m9_resnet8_static_protection_smoke_v1.toml
```

The reference environment lacks Matplotlib, so the maintained M9 TOMLs leave
`output.plots = false`. Non-mutating derived plots can be regenerated from a
finalized bundle in the canonical NetSuRF environment:

```bash
netsurf report <run-directory> --output workspace/derived-analysis/<name> --plots
```

No TensorFlow inference is rerun by the reporting command.
