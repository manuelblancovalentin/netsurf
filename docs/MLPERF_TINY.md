# MLPerf Tiny modern benchmark boundary

## Accepted identity

`mlperf_tiny_ic_resnet8_int8_v1` is the official MLPerf Tiny image-
classification ResNet-8 graph at MLCommons Tiny commit
`4addd0fa08d216e20637637874e084895f289da4` and protocol
`mlperf-tiny/4addd0fa08d2/ic01`. The immutable INT8 TFLite file is
`benchmark/training/image_classification/trained_models/pretrainedResnet_quant.tflite`,
SHA-256
`3c002613d1b2475eb51dd78dfb85a546c8ae658dee71cf6ade43b022fe205415`.
Kona Assets owns retrieval and path resolution; NetSuRF consumes the verified
path read-only.

The model is locally present in the Kona Assets model cache. The required
official CIFAR-10 Python batches are not present as of 2026-10-27. Therefore
the TFLite graph, storage mapping, deterministic synthetic inference and exact
fault path are validated, while CIFAR-10 accuracy, cross-entropy and the
official 0.85 quality target are **not measured**. A synthetic input is not a
substitute for task validation.

## Representation and eligible population

The graph has one subgraph, 38 tensors and 16 operators. NetSuRF includes only
the immutable parameter buffers used by convolution, depthwise-convolution or
fully-connected operators:

| Role | Tensors | Storage | Scalars | Bits |
|---|---:|---|---:|---:|
| kernels | 10 | signed INT8 | 77,360 | 618,880 |
| fused biases | 10 | signed INT32 | 346 | 11,072 |
| total | 20 | mixed width | 77,706 | 629,952 |

All 811 zero-valued allocated words remain eligible. Tensor scales, zero
points, activations, non-parameter constants and graph metadata are excluded.
Faults XOR the actual little-endian signed storage bytes in a private model
copy. Dequantization uses each tensor's per-tensor or per-channel affine
metadata. The shared TFLite artifact is never modified.

Because kernels and biases have different widths, the accepted MSB baseline
uses word-relative tiers: each word's sign bit ties at score zero, its next bit
has score -1, and so on. Schema-2 uses `global_ranking`. Schema-3 policies must
use `estimator_score_desc`; raw `bit_significance_desc` is rejected for this
adapter because raw indices 31 and 7 are not comparable.

## Runtime and ranking boundary

The adapter prefers `ai_edge_litert` when installed and otherwise uses the
installed `tensorflow.lite.Interpreter` fallback. The fallback is deprecated
upstream in TensorFlow 2.21 but remains functional; NetSuRF does not hide that
warning or install its replacement.

Immediately supported after the dataset is resolved:

- deterministic clean TFLite inference;
- random and word-relative MSB rankings;
- persistent-trial and per-inference exact stored-bit faults;
- paired ideal `perfect_event_suppression` masks;
- bounded exact empirical single-bit measurements.

Blocked pending a separate equivalence gate:

- gradient magnitude and gradient times exact delta;
- per-example magnitude and positive-damage estimators;
- Hessian-derived estimators;
- every qPolarGrad variant.

The official float Keras checkpoint exists (SHA-256
`5f938a8eea605438cc7360aa13c987bc6d7bd0c8d931b2917361ae7ae4eeda2f`),
but its tensors have not been mapped to the fused, per-channel-quantized TFLite
constants. Clean output agreement and representative exact-bit fault agreement
must be demonstrated before its gradients can score TFLite bits.

## Configuration fields

No maintained research TOML is supplied. A researcher-authored schema-2 static
smoke uses:

```toml
schema_version = "2.0.0"
scientific_protocol = "quantized-weight-fault/v2"

[benchmark]
adapter = "mlperf_tiny_ic_int8_v1"
id = "mlperf_tiny_ic_resnet8_int8_v1"
checkpoint = "/verified/path/pretrainedResnet_quant.tflite"
dataset = "/verified/path/cifar-10-python.tar.gz"
split = "test"
evaluation_samples = 32
ranking_samples = 32
quantization = "tflite_affine_int8"
quantization_backend = "tflite_affine_integer_v1"
injectable_parameters = ["kernels", "biases"]

[[estimators]]
name = "msb_first"
implementation_version = "relative-word-bit-index/v1"
policy = "global_ranking"
result_name = "msb_relative_word"
```

The remaining schema-2 fields are unchanged: protection coverage,
`perfect_event_suppression`, logical BER, persistence, independent Bernoulli
distribution, trial count, confidence, ranking/fault seeds, and output flags.
Use a logical BER of zero and one bounded trial for a clean-path validation;
that still records the clean baseline and is not an official quality run.

Schema-3 empirical validation uses protocol
`single-bit-susceptibility/v1`, a small `uniform` candidate population, the
same benchmark fields, and only `msb_first` and/or `random`. A protection study
uses `multi-coverage-protection/v1`, `protection.coverages`, optional
`protection.probabilities`, and policies whose priorities start with
`estimator_score_desc`. Full-network empirical sweeps are not bounded smoke
tests.

```bash
proj activate netsurf
netsurf benchmarks info mlperf_tiny_ic_resnet8_int8_v1
netsurf benchmarks identity mlperf_tiny_ic_resnet8_int8_v1
netsurf validate /path/to/research-authored.toml --resolved
netsurf run /path/to/research-authored.toml --workers 1 --threads-per-worker 1
```

Set `NETSURF_MLPERF_TINY_IC_MODEL` and `NETSURF_CIFAR10_PYTHON` to verified
resolved paths for benchmark readiness diagnostics. These variables do not
download or alter assets.

## Keyword spotting readiness

The pinned MLPerf Tiny DS-CNN INT8 model and Speech Commands data are present
in Kona Assets, and the 1,000 evaluation labels were verified there. The exact
TFDS interleaving and TensorFlow MFCC pipeline (normalization, STFT, mel
projection and coefficient selection) have not yet been reproduced and
numerically checked in NetSuRF. Keyword spotting is therefore an M7.B
candidate, not a registered executable adapter in this milestone.

## General metrics boundary

Adapters may expose explicitly named task metrics. Current result schemas
specialize classification accuracy and categorical cross-entropy. Future
reconstruction MSE, anomaly ROC-AUC, negative log-likelihood and perplexity
must retain their names and receive a reviewed result schema; they are never
relabeled as classification metrics.

