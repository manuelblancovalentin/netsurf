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
- `per_sample_positive_damage` is
  `mean_x(max(0, dell_x/dw * delta(w,k)))`. It is a positive-part,
  per-example first-order approximation, not an exact aggregate loss change.
  When both per-example estimators are requested, they share one gradient
  Jacobian. Their reported shared runtime must not be summed.
- `exact_diagonal_second_order` and `hutchinson_diagonal_second_order` rank
  the magnitude of `g*delta + 0.5*H_bb*delta^2`. The latter estimates the
  diagonal with seeded Rademacher HVP probes and reports probe diagnostics.
- `legacy_qpolargrad`: `abs(P(w,k) * dL/dw)`, where dense-kernel `P` is mean input activation times bit delta. In the recovered runtime, downstream layer inputs come from a global all-bit attack. The v2 implementation reproduces the first-call attacked activation source but eliminates accidental persistent `*_N` accumulation.
- `clean_activation_qpolargrad` and `uncancelled_activation_drift` are distinct
  research candidates, not silent corrections of the historical method.
- `msb_first`: encoded bit index descending. At exactly one bit per scalar, a position-first policy must produce the same mask as correct MSB-first.

No current estimator is claimed globally optimal.

## Conditional protection search

Schema-4 masks have fixed logical-bit cardinality. For a swap `a` (protected)
to `b` (unprotected), NetSuRF samples backgrounds excluding both bits and uses
`p_a[L(Ga)-L(G)] - p_b[L(Gb)-L(G)]`. The background and both candidate states
share inputs and context. Each network evaluation is numerically exact within
the documented dtype tolerance; its Monte Carlo mean is not an exact expected
risk. Search-sample estimates, proposal-surrogate error, and optimizer error
remain distinct.

The finite polar interaction compares additive logit responses with exact
four-state loss interaction. It can bias proposals but cannot accept a mask by
itself. Simulated annealing is explicitly a heuristic with noisy risk estimates
and is not represented as Metropolis–Hastings sampling. Held-out final inputs
and mask-independent full fault states are never used for acceptance decisions.

The incremental evaluator supports only explicit Dense→ReLU→Dense networks.
It retains `ΔW·δh`, recomputes ReLU gates exactly, operates on current encoded
words, and falls back to full inference if its activation cache exceeds the
configured limit. NumPy float64 reference tests require `rtol=atol=1e-12`;
the float32 MNIST bounded validation records observed errors in its bundle.
The inference-work proxy includes each distinct background-cache construction
and every candidate update. Identical `FaultState` backgrounds may share an
exact cache; cache hits, misses, fallback count, wall time, and dense MAC proxy
remain separate measurements.

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

Schema-3.1 replications record candidate-sampling, Hutchinson-probe,
sample-partition, and fault-schedule randomization separately. Dispersion over
candidate samples is not probe uncertainty and neither is fault-trial
uncertainty. Uniform-without-replacement candidate results remain
sample-limited unless the full eligible population is selected. Stratified
candidate results additionally require their recorded conditional inclusion
probabilities and cannot be described as unbiased whole-network top-k recall.

Confirmatory M3 studies use explicitly disjoint training, ranking/calibration,
empirical-validation, and final-evaluation roles. Exact indices and label-bound
sample hashes are recorded. A configuration requiring independence fails when
source split/index pairs overlap.

Replicated protection summaries pool all recorded fault realizations only after
matching their complete replication, logical-BER, trial, and schedule identity.
Pooled confidence intervals describe fault randomness conditional on the
recorded model, data, and masks. Replication-level summaries are reported
separately; identical masks across replications are not independent ranking-mask
samples. Presentation-only reanalysis of a finalized bundle is written to a
separate destination and cannot alter its scientific payload hash.

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

## M4.1 independent final comparison

Equal-budget masks are connected by pairing sorted members of `S0 - S1` with
sorted members of `S1 - S0`. This deterministic legal path does not imply
path-independent finite-sample variance. Every independent mask-agnostic field
uses backgrounds excluding the current protected set and both swapped bits.
Step contributions are summed before uncertainty is calculated.

The primary endpoint is target-minus-initial expected cross-entropy; zero-one
risk is secondary. Background fields—not inputs or path steps—are sampling
units. Intervals are conditional on the fixed recorded input set. Natural
Bernoulli trials are a separate endpoint. Final data and seeds are unavailable
to search. Candidate pools constrain proposals, never the fault universe.

Pruning is deferred. A zero-valued stored word remains faultable; a permanent
logical mask needs explicit semantics; structured physical removal changes
topology, checkpoint identity, eligible bits, and hardware cost. Zeroing a
weight does not remove its upset vulnerability.

## M4.2 topology and curvature levels

For the explicit Dense--ReLU--Dense adapter, topology responses decode the
current background word and calculate exact finite hidden/output logit changes.
Same-hidden-neuron and connected cross-layer pairs retain their exact ReLU or
bilinear residual. Other structurally additive logit pairs can still have a
nonzero cross-entropy interaction.

Finite additive-logit polarity, topology-corrected finite interaction,
input-dependent local GGN curvature, conventional Kronecker factors, and
stratified Kronecker factors are distinct recorded quantities. The latter two
are local curvature approximations and are not exact finite sign-bit models.
Sparse mask-dependent proposal scores omit unmeasured edges; omission is never
represented as evidence that the physical interaction is zero. Every proposed
swap remains subject to exact matched conditional evaluation, and final claims
use independent M4.1 held-out fields.
