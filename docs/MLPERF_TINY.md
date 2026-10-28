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

The model and official CIFAR-10 Python batches are locally present and verified
by the neutral Kona Assets lock `mlperf-tiny/ic`, digest
`sha256:f953370a8b98f7a016e6907e4b001f3b099bbe216c3202d0306fbbca00702687`.
NetSuRF independently verifies the lock digest plus every recorded file size
and SHA-256 without importing `kona_assets`. The archive SHA-256 is
`6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`.

## Representation and eligible population

The graph has one subgraph, 38 tensors and 16 operators. NetSuRF includes only
the immutable parameter buffers used by convolution, depthwise-convolution or
fully-connected operators:

| Role | Tensors | Storage | Scalars | Bits |
|---|---:|---|---:|---:|
| kernels | 10 | signed INT8 | 77,360 | 618,880 |
| fused biases | 10 | signed INT32 | 346 | 11,072 |
| total | 20 | mixed width | 77,706 | 629,952 |

Experiments explicitly select either `injectable_parameters = ["kernels"]`
(618,880 bits) or `injectable_parameters = ["kernels", "biases"]` (629,952
bits). Equal logical BER across INT8 kernels and INT32 biases is a software
fault-model assumption, not evidence of equal physical upset exposure.

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

The registered numerical identity is TensorFlow Lite 2.14.1
`BUILTIN_REF`, no delegates, one thread. It exactly reproduces the official
200-output digest
`f045f8f11e923bb37c467d1969e3e53c5b966061f1134688bd69b3e7f01338b9`
and accuracy 0.875, and the 10,000-output digest
`2cd2747d949fe123a144160c6e2b8c45277c438812803058c49cb4a123b9cfd1`
with accuracy 0.8707.

TensorFlow 2.21 is **not byte-equivalent**, even with `BUILTIN_REF`: 5 of the
200 output arrays change (13 integer elements, maximum absolute difference 8),
although predictions and 0.875 accuracy are preserved. The optimized built-in
path changes 87 arrays and 3 predictions (maximum 54); AUTO/XNNPACK changes 98
arrays and 3 predictions (maximum 79). NetSuRF therefore records the runtime,
resolver, delegates and thread count. Reference scientific execution requires
the recorded 2.14.1 backend or a newly reviewed output-equivalence gate.
Normal lock-backed experiment adapters enforce the registered 200-output
digest during loading and fail before scientific execution on TensorFlow 2.21.
The non-enforcing constructor switch exists only for explicit backend
diagnostics such as the M7.1 comparison and is not a TOML experiment option.

Supported after M9:

- deterministic clean TFLite inference;
- random and word-relative MSB rankings;
- persistent-trial and per-inference exact stored-bit faults;
- paired ideal `perfect_event_suppression` masks;
- bounded exact empirical single-bit measurements.
- exact numerical affine-XOR jump ranking;
- kernel-only straight-through gradient magnitude, gradient-times-delta,
  per-example magnitude and positive-damage estimators.

Still blocked pending separate gates:

- gradient ranking of INT32 biases;
- Hessian-derived estimators;
- every qPolarGrad variant.

The official float Keras checkpoint exists (SHA-256
`5f938a8eea605438cc7360aa13c987bc6d7bd0c8d931b2917361ae7ae4eeda2f`),
was mapped diagnostically to the fused, per-channel-quantized TFLite kernels in
M9. The implemented gradient graph is derived from the TFLite constants and is
an explicitly recorded straight-through surrogate, not exact TFLite inference.
See `docs/MLPERF_TINY_M9_STATIC_ESTIMATORS.md` for correspondence results and
remaining limitations.

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
asset_lock = "/verified/path/ic.lock.json"
inference_backend = "tensorflow_lite_builtin_ref_v1"
split = "official_subset_200" # or "test" for the complete 10,000 images
evaluation_samples = 32
ranking_samples = 32
quantization = "tflite_affine_int8"
quantization_backend = "tflite_affine_integer_v1"
injectable_parameters = ["kernels", "biases"] # alternatively ["kernels"]

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
