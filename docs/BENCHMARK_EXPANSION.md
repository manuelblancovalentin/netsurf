# Benchmark expansion gate

This document records the inference-only benchmark gate performed on 2026-10-27.
It is evidence for new NetSuRF v2 benchmark identities, not an attempt to
reproduce the rejected TCAD manuscript. Source assets were read but not
modified or repackaged.

## Readiness decision

| Benchmark identity | Ready for inference? | Exact fault support? | Ranking support? | Blocking condition |
|---|---:|---:|---:|---|
| `mnist_lite_exact_kb_v1` | Yes, bounded validation | Yes, 47,760 kernel/bias bits | Static gradient, per-example, MSB, random, and stochastic diagonal-Hessian controls | Historical TensorFlow/QKeras compatibility environment is required to load the source checkpoint; alpha/beta are excluded float auxiliaries |
| `keyword_spotting_candidate.v1` | No | No accepted population | No | The preserved checkpoint has 49 logits, while the available Speech Commands dataset has 12 labels; no evidenced 49-class vocabulary/preprocessing contract was found |
| `toyadmos_autoencoder_candidate.v1` | Traceable diagnostic only | No accepted population | No | Classification-specific result schemas, 992 unresolved trainable auxiliaries, and no reviewed anomaly-evaluation protocol |
| `mnist_quantized` | Unchanged | Yes | Existing v2 support | None within its documented software scope |

Machine-readable evidence is stored under
`netsurf/benchmarks/identities/`. Candidate identity documents do not register
blocked candidates as executable adapters.

## Accepted MNIST-lite identity

The immutable source checkpoint has SHA-256
`ba48aee8adbe7d04092d74f41958ec99112ae3e5f88ed56552ea9d1c4dba3006`.
The v2 adapter deterministically rounds and saturates only the four kernel/bias
tensors to `q<6,2,1>` using `netsurf_bitcodec_v1`. It leaves the 20 learned
alpha/beta scalars as uncorrupted float32 auxiliaries. The source checkpoint is
never rewritten.

The converted full-state digest is
`4d0d6ee025d8603592de1e67d4d0164ea3f2d9591fde75f4fdc230ac12ca5e32`.
The digest covers sorted tensor names, dtype, shape, and C-order bytes. The
fault population contains all 7,960 allocated kernel/bias scalars—including
zero-valued words—and exactly 47,760 six-bit identities. Alpha/beta are neither
encoded nor injectable. The model therefore has 7,980 trainable scalars in
total but only 7,960 eligible encoded scalars.

On the complete 10,000-example MNIST test split, exact-kernel/bias inference
gave accuracy `0.8568` and mean categorical cross-entropy
`0.47477197647094727`. The output-array SHA-256 was
`c267e559958630913c37aecdf980ebbbbe6b5e1a3640ea8b696788ed6f97b82c`,
and a repeated inference was bitwise equal. These values identify the new v2
conversion; they are not historical manuscript targets.

The explicit analysis partitions are:

- ranking/calibration: `mnist_train[50000:60000]`, identity
  `df50cd78a93c052dfcae39b60b78eeb90cdce9bf5cc660a4abe8318785de40ad`;
- empirical validation: `mnist_train[40000:50000]`;
- held-out evaluation: `mnist_test[0:10000]`, identity
  `db1bc86b55fcf21af55231c17571b5e5e8ac6b33c4124ef4022709185ccd222b`.

The adapter supports clean/faulted inference, gradients, per-example gradients,
Hessian-vector products, encoded identities, persistent/per-inference schedules,
perfect-mask event suppression, and exact state restoration. Dense32-specific
incremental search, topology responses, and Kronecker formulas remain rejected
for this topology. Use full inference for fault evaluation.

## Blocked keyword-spotting candidate

The preserved DSConv checkpoint SHA-256 is
`5fb74ea4fbb4add8eb85db555fb9067fe57b07b48c95ac8796293e92b3a9e48c`.
It consumes `(49, 10, 1)` inputs and produces 49 logits. The available TFDS
Speech Commands 0.0.3 asset exposes 12 labels. No immutable mapping from the 49
outputs to those labels, nor the checkpoint's exact feature-preparation
contract, was established. Relabeling the dataset or truncating outputs would
be scientifically invalid.

Minimum next step: provide the immutable 49-class vocabulary and matching
feature/calibration artifacts, or prepare a separately versioned 12-class
checkpoint in a future training milestone. A future DSConv adapter must define
convolution/depthwise tensor layouts and the representation of alpha/beta and
BatchNorm state before exposing any XOR population.

## Traceable autoencoder candidate

The preserved checkpoint SHA-256 is
`8991de3f87e47e4754f674ab3b3dc3b7002fe78966f98943208b33c613708b8c`.
Its topology is `320 -> 10 -> 10 -> 10 -> 10 -> 8 -> 10 -> 10 -> 10 -> 10 ->
320`, with 8,560 trainable scalars: 7,568 kernel/bias values and 992 trainable
alpha/beta/BatchNorm auxiliaries.

The recovered loader normalizes inputs to `[-4, 3.875]` using per-feature
normal-train extrema but retains raw, unnormalized log-mel reconstruction
targets. On a bounded 256-example anomalous slice, the float-shadow state gave
MSE `162.91279416680706`; the q<6,2,1> kernel/bias plus float-auxiliary state
gave MSE `161.31930625948345`. This is reconstruction error, not anomaly
accuracy.

Minimum next step: introduce a reviewed task-metric schema for per-sample
reconstruction error and MSE, define normal/anomalous partitions and ROC-AUC or
an independently calibrated threshold, then resolve or explicitly exclude all
992 auxiliary representations. The current classification report path must not
rename MSE as accuracy or cross-entropy.

## Research-authored configuration contract

No maintained research TOML is provided by this gate. A researcher-authored
static protection experiment may use experiment schema `2.0.0` and must set:

```toml
schema_version = "2.0.0"
scientific_protocol = "quantized-weight-fault/v2"

[benchmark]
adapter = "mnist_lite_exact_kb_v1"
id = "mnist_lite_exact_kb_v1"
checkpoint = "/ml/experiments/netsurf/legacy/benchmarks/mnist_fnn_lite/q6_2_1/pruned_0.125_159569_qflatten_dense10u_qapplyalpha_quantized_relu_dense10u_qsoftmax/models/pruned_0.125_159569_qflatten_dense10u_qapplyalpha_quantized_relu_dense10u_qsoftmax.20250414_152629.keras"
dataset = "/ml/datasets/mnist/mnist.npz"
split = "v2_identity_gate_disjoint_partitions"
evaluation_samples = 256
ranking_samples = 256
quantization = "q<6,2,1>"
quantization_backend = "netsurf_bitcodec_v1"
injectable_parameters = ["kernels", "biases"]
```

The remaining required schema-2 sections are `experiment`, `estimators`,
`protection`, `fault`, `execution`, and `output`; copy their structure from a
current schema-2 template, not from a historical legacy configuration. The
validator rejects any exact-MNIST request that includes alpha/beta or
`all_trainable` in the injectable population.

Estimator studies may instead use schema `3.0.0` or `3.1.0` with the same
benchmark fields plus explicit `[samples]`, `[candidates]`, `[[estimators]]`,
`policies`, protection, fault, execution, and output sections. Supported
estimators are `random`, `msb_first`, `gradient_magnitude`,
`first_order_gradient_delta`, `per_sample_first_order_magnitude`,
`per_sample_positive_damage`, and `hutchinson_diagonal_second_order` with their
versions already enforced by `netsurf.study_config`. Candidate selection may
be `all`, `uniform`, or `stratified` subject to the existing schema-version
rules. `samples.require_disjoint = true` is required for confirmatory use.

Schema-4 conditional/incremental search does not accept this adapter because
its topology-specific exact-update and factorized formulas have not been
validated. This is a deliberate rejection, not missing registry wiring.

Validate a researcher-authored file without executing it:

```bash
proj activate netsurf
netsurf benchmarks identity mnist_lite_exact_kb_v1
netsurf benchmarks info mnist_lite_exact_kb_v1
netsurf validate /path/to/research-authored-experiment.toml --resolved
```

Run only after inspecting the resolved population, partitions, trial count,
logical BER, and output destination. New results remain under
`workspace/runs/` and are write-once after finalization.
