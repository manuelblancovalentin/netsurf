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

The recovered `legacy_mnist_lite` checkpoint is a special, explicitly named
exception: it stores off-lattice float32 shadows. Its recovered historical
path adds the encoded delta to that shadow. This is not exact stored-state XOR.
The identity-gate exact path first converts allocated kernel/bias tensors to
q<6,2,1>, includes zero-valued words, and excludes learned alpha/beta until a
separate storage or folding format is approved. Historical and exact-converted
populations must never share an unlabeled result.

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

Multiprocess execution preserves the single-worker scientific contract. Logical
task identities, configured seeds, fault fields, masks, and reductions are
independent of worker assignment and completion order. Schema-3 replications
and schema-4 BER/budget/replication units may execute in spawned processes; an
individual search trajectory, training run, and iterative pruning trajectory
remain sequential. Worker/thread counts and timing are manifest provenance and
are excluded from the scientific payload digest.

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

`mnist_lite_exact_kb_v1` is a separate v2 identity derived from one recovered
checkpoint. Its fault population is exactly the 7,960 allocated kernel/bias
words after deterministic `q<6,2,1>` conversion, including stored zeros. Its
20 learned alpha/beta scalars remain float32, uncorrupted, and outside the
47,760-bit eligible population. This hybrid contract is scientifically scoped
and hashed; it is neither the historical float-shadow experiment nor a claim
to reproduce a manuscript table. See `docs/BENCHMARK_EXPANSION.md`.

Task metrics are part of benchmark identity. Classification accuracy and
cross-entropy cannot be reused as labels for reconstruction MSE or anomaly
detection. A non-classification benchmark requires a reviewed metric/result
schema before adapter registration.

`mlperf_tiny_ic_resnet8_int8_v1` uses the official affine-quantized TFLite
constant buffers, not `q<6,2,1>`. Kernels are INT8 and fused biases are INT32;
both include allocated zeros. Experiments select kernels alone or kernels plus
biases explicitly; equal logical BER across those storage classes is a model
assumption, not a physical exposure claim. Exact XOR occurs in a private
flatbuffer copy.
Scales, zero points, activations and graph metadata are excluded. Random,
word-relative MSB, numerical-jump and kernel-only straight-through first-order
rankings are accepted. The companion is not authoritative inference; INT32
bias gradients, curvature and qPolarGrad remain blocked. The numerical
backend is part of benchmark identity: the registered TensorFlow Lite 2.14.1
`BUILTIN_REF` outputs are not byte-equivalent to TensorFlow 2.21 even when the
same resolver name is selected. See `docs/MLPERF_TINY.md`.

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

## M4.3 mask-aware screening and aggregate guidance

Reusable removal and addition marginals are proposal-screening quantities.
Their difference `C-D` is not an exact swap effect because it omits the pair
correction. Candidate rows share common background fields, and bundles retain
per-context contributions so their covariance is recoverable. Marginal-only,
corrected-shortlist, factorized, and hybrid searches are separately named.

The full-population Kronecker aggregate is exact only as a matrix identity
within its stated local GGN surrogate. It excludes only the queried encoded bit
from its vulnerable sum, supports rank-one accepted-swap updates, and records
population coverage and reconstruction drift. Stratified and conventional
aggregates remain different approximations. Restricted-pool aggregation is a
negative control, not a full-network quantity.

Neither a factorized score nor a polar pair term accepts a swap. Exact matched
three-state conditional evaluation remains the acceptance gate, and independent
held-out natural and conditional fields remain the scientific endpoints.

## M4.4 controlled proposals and cached factors

Controlled proposal comparisons share their encoded-bit candidate universe,
initial mask, search backgrounds, iteration count, shortlist sizes, and exact
verification budget. A proposal model may reorder or reweight those candidates
but cannot change the full logical fault population. Conditional greedy is a
quality reference rather than an equal-cost controlled method.

Sign-bit enrichment is an explicit sampling ablation, not a susceptibility
estimator. Shuffled Kronecker guidance preserves either the global aggregate
score distribution or each layer/bit-position distribution while destroying
the corresponding bit identities. It is a negative control and cannot establish
that the unshuffled factor is correct.

Cached factors depend only on fully hashed calibration inputs. Protected-mask
state and logical BER remain uncached method-local aggregate state. Cache reuse
may change wall time but must not change proposal scores, masks, fault states,
held-out measurements, or the scientific payload digest.

## M5 structured pruning

Physical hidden-neuron pruning removes the incoming kernel column, hidden bias,
and outgoing kernel row as one dependency group. It re-enumerates bit identities
over the smaller tensors; a zero-valued or masked source tensor is never called
pruned. `grouped_taylor` is an architecture-specific adaptation of the Vinck
et al. workflow, not a reproduction of their CNN, quantizer, accelerator, or
single-transient fault experiment. `group_magnitude` and independently trained
widths are controls.

`netsurf_reliability_aware` and `netsurf_fault_augmented` are explicitly new
NetSuRF proposals. They are not RAP: the accessible RAP source lacks the exact
criterion and fine-tuning equations needed for faithful reproduction. Clean
preservation, stored-bit susceptibility, resource cost, and fault-augmented
training weights remain separately recorded and ablatable.

Post-pruning topology records exact parameter and encoded-bit counts, dense MAC
count, and packed encoded storage bytes. MAC reduction is not a FIT reduction
without an approved physical exposure model. Each pruned model is evaluated
relative to its own clean baseline.
## MLPerf Tiny differentiable companion

The MLPerf Tiny ResNet-8 differentiable companion is a static susceptibility
surrogate, never an inference authority. Its variables are dequantized values
of the exact frozen TFLite INT8 kernel buffers. TFLite 2.14.1 `BUILTIN_REF`
remains authoritative for every clean and faulted measurement. Straight-
through gradients, exact affine XOR jumps, and empirical TFLite damage must be
identified separately in manifests and reports. M9 excludes INT32 biases from
gradient-based ranking until they pass an independent mapping gate.

## M9.5.4 contextual restricted protection

A schema-6.2 contextual experiment may exchange protected and unprotected bits
inside a prespecified candidate pool only after verifying the finalized source
mask, source scientific payload, artifact checksum, checkpoint, eligible-bit
population, uniqueness, and cardinality. If `P_out` is the verified source
mask outside that pool, every candidate allocation is `P_out union Q`, with
`|Q|` fixed to the source mask's protected count inside the pool. Candidate
faults and already protected outside coordinates are excluded from the sampled
outside Bernoulli universe. Proposed, applied, and suppressed events retain
their existing exact meanings.

For each independently sampled outside field, all `2^K` exact candidate-state
responses are sufficient to integrate every eligible fixed-budget mask under
the stated independent Bernoulli candidate law. Derived marginal and pairwise
mask statistics reuse that table and require no additional TFLite evaluation;
therefore they retain every interaction order represented in the table. This
identity does not make a naturally unflipped bit equivalent to a protected bit
under correlated, fixed-count, adversarial, or otherwise different laws.

Mask construction uses model-selection data only. Final-evaluation fields are
paired across frozen comparators and cannot select a post-hoc optimum. The
previous K=6 engineering smoke exposed KWS test indices 3048--3053, so schema
6.2 excludes them from any claim of previously untouched confirmation. Its
final-evaluation offsets begin at six within the fixed final window.

## M9.5.5 frozen-mask paired-swap evaluation

Schema 6.3 keeps exhaustive `2^K` response tables for model-selection fields
but evaluates frozen masks on additional final fields through the exact legal-
swap identity. For a current mask `P`, removal `a`, addition `b`, and target
`P'`, the contribution is `p E[R(G union {a}) - R(G union {b})]`, where `G`
is Bernoulli over every other unprotected eligible coordinate. Multiple swaps
follow a deterministic sorted path and are summed before statistical
aggregation. The independent master field—not a path step or inference—is the
replication unit.

Original S5, exact-isolated, and restricted-selected masks are frozen from
source/model-selection evidence before paired final fields are generated.
Final outcomes cannot choose a mask, path, field count, or seed. Exact full-
enumeration final fields remain a diagnostic numerical reference; their post-
hoc best mask remains selection-biased. Schema 6.3 excludes final-window
relative indices 0--11 because the M9.5.3 and M9.5.4 engineering smokes have
already exposed them.

Model-selection replicate seeds and paired-final seeds are explicit and
independent. Selection stability over finite model-selection fields is
descriptive; it is not uncertainty over final fault fields, input sampling, or
independently trained models. Negative contrasts and beneficial faults remain
valid observations.

## M9.5.6 publication preflight

Schema 6.4 selection-only execution never requests final-window inputs, scans
final-window labels, generates final fields, or reports final performance.
It finalizes exact model-selection tables and frozen equal-budget masks.
Final-only execution verifies that artifact and its complete construction and
prespecified final-plan identities before accessing final inputs. It cannot
rerun selection or choose a post-hoc final optimum. The earlier joint workflow
remains available under its historical schemas.

The audited M953–M955 smokes exposed original KWS indices 3048–3065, including
the M955 six-clip validation. The new final template reserves 3066–3193;
confirmation mode rejects known ledger overlaps and prior finalized schema-6.4
final exposures under its output root. Aggregate clean-reference inference
and earlier label audits mean these data are not inference-naive. External,
failed and concurrent executions still require researcher exposure accounting.

The historical eight-bit pool is exploratory and is not a random population
sample. Independent final inputs/fields do not remove pool-selection bias.
Selection stability describes finite outside-field sensitivity, whereas final
Student-t intervals describe whole-path master-field uncertainty conditional
on frozen inputs, model and masks. The primary selected-minus-S5 contrast and
two secondary contrasts are frozen; additional Bonferroni intervals cover the
three-contrast family. No population-wide optimality or physical reliability
claim follows. See `docs/M956_PUBLICATION_PREFLIGHT.md`.

## M9.5.7 independent frozen-mask replication

Schema 6.5 separately freezes a new final-input/field plan while retaining the
original M956 artifact hashes, candidate pool, selected `(2,6)`, isolated
`(2,7)`, original S5 `(6,7)`, model, kernel universe, context and BER. The
schema-6.4 final-only contract is unchanged. Replication never accesses
model-selection inputs, ranks candidates or reselects masks.

The primary selected-minus-S5 mean cross-entropy contrast uses whole master
fields, two-sided 95% Student-t inference and an unadjusted primary p-value.
Two prespecified secondary hypotheses use Holm-adjusted p-values and separate
Bonferroni secondary-family intervals. Tail counts, quantiles and influence
diagnostics do not replace mean-risk inference or discard adverse fields.
Student-t inference is approximate under heavy tails, not an exact finite-
sample coverage guarantee. Optional per-input contributions come from the
same authoritative calls without changing scalar reductions; input×field
cells are not independent replications.

The expanded exposure ledger includes original final indices 3066–3193 and
real engineering records. The prespecified 3194–3449 replication window has
no overlap in that inventory; unknown manual/failed/concurrent exposure
remains a disclosure obligation. The exploratory eight-bit pool cannot support
population-wide protection-superiority claims. See
`docs/M957_FROZEN_MASK_REPLICATION.md` for hashes, costs and execution gates.
