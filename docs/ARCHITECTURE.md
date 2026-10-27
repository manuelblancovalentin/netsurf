# NetSuRF v2 architecture

NetSuRF 2 is the canonical `netsurf` package. The historical package remains
unchanged under `DEPRECATED/legacy_source/netsurf` and is accessed only through
explicit adapters. Importing canonical `netsurf` does not initialize Nodus,
TensorFlow, a database, or user configuration.

```text
Benchmark artifact + held-out split
              |
              v
       Framework adapter --------> immutable parameter identities
              |                               |
              v                               v
          gradients                    exact BitCodec deltas
              |                               |
              +----------> SusceptibilityEstimator
                                      |
                                      v
                              ProtectionPolicy
                                      |
FaultModel -> deterministic FaultSchedule -> InjectionEngine
                                      |
                                      v
                              application evaluation
                                      |
                                      v
                     typed measurements + statistics
                                      |
                                      v
                     atomic immutable result bundle
                         /             |             \
                      CLI          Python API       report/plots
```

## Implemented modules

- `netsurf.quantization`: signed/unsigned fixed-point contract, saturating ties-to-even encoding, exact two's-complement XOR deltas.
- `netsurf.identities`: immutable tensor and bit identities. Bit index zero is always the encoded LSB.
- `netsurf.ranking`: score-only estimators for random, correct MSB, gradient magnitude, elementary first order, and recovered legacy qPolarGrad.
- `netsurf.estimators_m3`: signed first/second-order predictions, per-example
  magnitude, seeded Hutchinson curvature, and separately named qPolarGrad variants.
- `netsurf.ordering`: deterministic lexicographic composition of bit
  significance, layer direction, estimator score, canonical, reversed, and
  seeded-random criteria.
- `netsurf.empirical` and `netsurf.analysis_m3`: exact isolated-bit effects,
  stratified candidate selection, tied-rank metrics, mask composition, paired
  permutation tests, and multiplicity correction.
- `netsurf.samples`: explicit sample indices, hashes, provenance, and
  disjointness enforcement.
- `netsurf.protection`: global, position-first, and greedy nonuniform-cost policies. The greedy policy explicitly makes no optimality claim.
- `netsurf.faults`: explicit deterministic events, persistent and
  per-inference schedule generation, perfect-mask suppression, and exact
  accounting. Per-inference windows are indexed by sample, not software batch.
- `netsurf.adapters.legacy_mnist`: isolated TensorFlow/legacy checkpoint bridge with exact state restoration.
- `netsurf.benchmarks.registry`: static benchmark contracts and readiness
  diagnostics with no eager ML imports.
- `netsurf.adapters.numpy_reference`: fixed 2-3-2 quantized reference network
  used for exhaustive bit and gradient validation.
- `netsurf.adapters.mnist_quantized`: explicit Keras-trained Dense-32 MNIST
  architecture whose faulted inference consumes exact fixed-point values. Its
  per-example Jacobian is one lazily compiled, shape-stable TensorFlow graph;
  parameter values remain explicit inputs so cached tracing cannot retain a
  stale clean or faulted model state.
- `netsurf.doctor`: environment, dataset, workspace, and adapter diagnostics
  using package metadata rather than importing TensorFlow.
- `netsurf.statistics`: sample statistics and Student-t confidence intervals.
- `netsurf.experiment_config`: strict versioned TOML loading and executable validation.
- `netsurf.adapters.protocol`: framework-neutral benchmark/model adapter contract; the registry imports TensorFlow only when explicitly requested.
- `netsurf.experiment_engine`: benchmark-neutral estimator, policy, schedule, trial, and aggregation orchestration.
- `netsurf.study_config`, `netsurf.study_engine`, and `netsurf.study_bundle`:
  strict schema-3 empirical/protection protocols and applicable immutable tables.
- `netsurf.incremental`: immutable XOR fault states, cached Dense→ReLU→Dense
  forward state, exact downstream updates, full-forward reference/fallback, and
  the matched three-state conditional swap estimator.
- `netsurf.search`: existing-ranking proposal adapters and fixed-budget static,
  greedy, uniform, guided, polar-guided, and annealed search strategies.
- `netsurf.polar`: bounded additive-logit polar, local quadratic, exact
  four-state interaction, error-bound, gate, and cancellation diagnostics.
- `netsurf.search_config`, `netsurf.search_engine`, `netsurf.search_bundle`, and
  `netsurf.search_reporting`: schema-4 orchestration, immutable outputs, and
  saved-data-only report/plot regeneration.
- `netsurf.result_bundle`: atomic staging, write-once finalization, checksums, typed reading, and corruption detection.
- `netsurf.derived_analysis`: presentation-only pooled and replication-level
  protection statistics reconstructed from immutable trial rows using exact
  replication/BER/trial/schedule identities.
- `netsurf.reporting` and `netsurf.plotting`: presentation derived only from saved measurements.
- `netsurf.progress`: presentation-only measured-work tracking, a single Rich
  Live queue/current-experiment renderer, persistent per-task runtimes,
  phase-aware ETA estimation, and ANSI-free redirected output. Batch status
  never enters result serialization.
- `netsurf.api` and `netsurf.__main__`: one public engine exposed to scripts, notebooks, and the CLI.
- `netsurf.milestone`: retained historical compatibility workflow; new work uses versioned experiment configurations.

## Storage boundary

`netsurf.paths` resolves shared datasets from environment variables and new
results under `workspace/runs/` (override with `NETSURF_WORKSPACE`). Benchmark
definitions belong under `config/`; generated data are ignored. Historical
external archives are not migrated or modified.

## Version boundaries

Experiment, manifest, summary, and measurement schemas are independently
versioned under `netsurf/schemas/`. They are distinct from package version,
scientific protocol, estimator implementation versions, run ID, and stable
scientific payload SHA-256. Unsupported versions fail rather than being
reinterpreted. Presentation timestamps and absolute output paths are excluded
from the payload digest.

## Framework boundaries

The engine depends only on the adapter protocol. An adapter explicitly owns
model construction, dataset roles, fixed parameter names, exact current state,
evaluation, gradients, and temporary-state restoration. NetSuRF does not clone
arbitrary user models; adding an architecture means implementing and testing a
new adapter.

Keras prepares the modern MNIST source model. The inference checkpoint contains
already quantized arrays, and TensorFlow operations consume those exact decoded
arrays in clean and faulted inference. HGQ2/hls4ml are installed investigation
tools, not part of this benchmark's numerical claim.

## Deferred interfaces

Activation injection, an accepted new qPolarGrad theory, mitigation
implementations, HLS/RTL export, logical-to-physical mapping, and Wolf hardware
execution remain deferred. Exact Hessian-diagonal validation is implemented
only for the tractable NumPy reference; MNIST uses configurable stochastic HVP
probes and explicitly sampled empirical candidates, not a 152,700-bit exhaustive claim.

Schema 4.1 adds `conditional_risk.py` without replacing the optimizer. It owns
deterministic mask paths and whole-path contributions. `search_engine.py` fixes
masks before creating separate natural and conditional fields. Bundling and
reporting consume saved cluster-level measurements without rerunning inference.

Schema 4.2 adds two architecture-specific analytical layers without changing
the benchmark-neutral fault codec. `topology.py` owns exact on-demand
Dense--ReLU--Dense response signatures and structural pair residuals over an
incremental forward cache. `curvature.py` owns input-dependent local GGN,
Kronecker, and confidence-stratified Kronecker factors. Both consume exact
`BitIdentity` and `BitCodec` state; neither is a general model converter.

Protection remains globally budgeted. Per-layer values limit sparse interaction
discovery only. The search layer converts recorded edges into mask-dependent
proposal adjustments, while exact matched conditional evaluation remains the
sole acceptance path. Saved topology and proposal-influence tables regenerate
reports and plots without TensorFlow.

Schema 4.3 adds `conditional_screening.py`, which computes reusable candidate
marginals from common random fields while retaining paired context
contributions, and `factorized.py`, which represents a full vulnerable-
population Kronecker interaction sum with dynamic rank-one updates.
`search.py` composes those proposal stages with the existing exact conditional
acceptance gate. `timing.py` defines non-overlapping timing intervals; bundle
and reporting code label inclusive method and cumulative trajectory timing so
they cannot be mistaken for additive phase measurements.
