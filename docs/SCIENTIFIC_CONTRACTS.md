# Scientific contracts

## Quantization and bit identity

For historical `q<6,2,1>`, the recovered convention is six encoded bits, two integer magnitude bits, one sign bit, three fractional bits, step `0.125`, and range `[-4, 3.875]`. Values are rounded to nearest with ties to even and saturated before encoding. Encoded bit index `0` is the LSB and index `5` is the two's-complement sign/MSB.

For value `w` and bit `k`:

```text
delta(w, k) = decode(encode(w) XOR (1 << k)) - decode(encode(w))
```

Fault injection adds this exact delta to the checkpoint value. For modern
adapters the checkpoint is already exactly quantized, so encoded XOR, decode,
and inference state use the same representation. The exhaustive
64-code/384-flip codec test is mandatory; the NumPy reference additionally
exhausts every parameter bit through model inference.

## Estimator versus policy

An estimator maps bit identities to scalar scores. It does not choose protected bits. A policy combines scores, ordering constraints, costs, and a budget to produce an immutable mask.

- `first_order_gradient_delta`: `abs(dL/dw * delta(w,k))`.
- Its signed prediction is retained separately as `dL/dw * delta(w,k)`.
- `per_sample_first_order_magnitude` is
  `mean_x(abs(dell_x/dw * delta(w,k)))`; it is not the absolute value of the
  mean gradient.
- `exact_diagonal_second_order` and `hutchinson_diagonal_second_order` rank
  the magnitude of `g*delta + 0.5*H_bb*delta^2`. The latter estimates the
  diagonal with seeded Rademacher HVP probes and reports probe diagnostics.
- `legacy_qpolargrad`: `abs(P(w,k) * dL/dw)`, where dense-kernel `P` is mean input activation times bit delta. In the recovered runtime, downstream layer inputs come from a global all-bit attack. The v2 implementation reproduces the first-call attacked activation source but eliminates accidental persistent `*_N` accumulation.
- `clean_activation_qpolargrad` and `uncancelled_activation_drift` are distinct
  research candidates, not silent corrections of the historical method.
- `msb_first`: encoded bit index descending. At exactly one bit per scalar, a position-first policy must produce the same mask as correct MSB-first.

No current estimator is claimed globally optimal.

M3 ordering is lexicographic. A later criterion cannot cross a primary tier.
`msb+ordered`, `msb+reversed`, seeded `msb+random`, input/output layer order,
and `msb+gradient` all prioritize encoded bit significance. Therefore their
masks are exactly equal at complete tier boundaries such as 1/6 coverage for
six-bit parameters, while partial-tier masks may differ.

## Fault model

Modern experiments use independent Bernoulli logical weight-bit events either
sampled once per Monte Carlo trial and held persistently for its evaluation
window, or sampled separately for each indexed inference. The latter is
independent of software batch size. Protected events are perfectly suppressed.
Base model state is restored after every policy/trial or per-inference
evaluation. Policies in a trial consume the same schedule hash.

This probability is a logical BER parameter, not fluence, cross-section, FIT, SEU rate, or mission-time reliability.

## Statistics

Trial results are independent across explicitly recorded seeds. Policy comparisons are paired by schedule. Reports include trial count, sample mean, sample standard deviation, standard error, and the configured two-sided Student-t interval. Accuracy intervals are clipped to its physical `[0,1]` range. M3 paired comparisons use a two-sided sign-flip permutation test and Holm correction within each reported metric family. Ranking-mask random seeds and fault-schedule seeds are separate factors. These summaries characterize the bounded simulation only.

Confirmatory M3 studies use explicitly disjoint training, ranking/calibration,
empirical-validation, and final-evaluation roles. Exact indices and label-bound
sample hashes are recorded. A configuration requiring independence fails when
source split/index pairs overlap.

## Result immutability

A finalized run contains a manifest, scientific payload, trial records, exact ranking scores, exact protection-mask identities, exact fault events, a human-readable report, and `FINALIZED` digest. Files are created exclusively and made read-only. Hashes of the exact artifacts are embedded in the scientific payload. A repeated run must produce the same canonical scientific-payload SHA-256; timestamps and run paths are excluded from that digest.

## Modern benchmark scope

The MNIST source architecture is `Flatten(28x28) -> Dense(32, ReLU) ->
Dense(10 logits)`. Injectable state is exactly the two kernels and two bias
arrays. Optimizer state, training-time float shadows, and learned quantizer
parameters are not fault targets. Ranking examples are drawn from the reserved
training tail; evaluation examples are drawn from the test split.

The current backend identifier is `netsurf_bitcodec_v1`. HGQ2 and hls4ml
availability does not imply C-simulation, RTL-equivalence, or synthesis
validation. `perfect_event_suppression` is an ideal logical mask, not ECC, TMR,
or another hardware mitigation guarantee.

The modern MNIST numerical execution contract is CPU TensorFlow eager mode with
oneDNN graph rewrites disabled. This is enforced before framework import because
the enabled rewrite path exhibited cross-process last-bit drift in gradients
and cross-entropy on Kona. A process that initialized TensorFlow under a
conflicting environment must be restarted; the adapter does not silently hash
numerically unstable results as equivalent.
