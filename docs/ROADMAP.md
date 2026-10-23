# Roadmap

## M0 — preservation and foundation

- [x] Preserve dirty-state provenance, recover remote `master`, and create neutral branch `v2`.
- [x] Inspect and register `proj`; inspect Wolf without modifying it.
- [x] Add exact codec, bit identities, estimators, policies, deterministic schedules, state-safe adapter, statistics, result store, and dependency-free tests.
- [x] Promote the reconstruction to canonical `netsurf`, package it, and externalize datasets and generated artifacts without merging conflicting collections.

## M1 — deterministic MNIST-lite experiment

Acceptance criteria: exact checkpoint and held-out split; deterministic ranking/mask hashes; matched schedules; exact restoration; correct confidence summaries; two identical payload hashes; explicit legacy qPolarGrad label.

## M1.5 — self-service experiment framework

- [x] Versioned executable configuration, manifest, summary, and measurement schemas.
- [x] Adapter-driven engine shared by CLI and Python/notebook API.
- [x] Atomic immutable bundles under `workspace/runs/`, with checksums and stable scientific digest.
- [x] Reports and plots regenerated solely from finalized data.
- [x] Dependency-free core regression suite plus bounded TensorFlow integration validation.

## M2 — modern benchmarks and self-service fault injection

- [x] Benchmark registry, environment doctor, and explicit readiness states.
- [x] NumPy reference network with exhaustive bit injection and gradient checks.
- [x] Fresh Keras Dense-32 MNIST preparation and exact fixed-point checkpoint.
- [x] Single- and multi-ranker runs with matched schedules and no/full controls.
- [x] Persistent and sample-indexed per-inference replay with restoration.
- [x] Schema-2 bundles, paired reports, plots, and repeated payload digests.

## M3 — estimator and composable-policy validation

- [x] Exhaustive exact injection of all 102 NumPy-reference bits.
- [x] Bounded stratified MNIST exact-bit measurements with explicit sample identities.
- [x] Composable MSB-primary tie-breaking and complete-tier equivalence tests.
- [x] Signed first-order, per-example magnitude, exact/Hutchinson diagonal curvature, and separately versioned qPolarGrad variants.
- [x] Tied-rank quality metrics, paired permutation tests, Holm correction, mask composition, reports, and plots.

Resolve qPolarGrad theory only after the M3 data are reviewed. A sampled MNIST
study is not a whole-network top-k result, and isolated-bit ranking is not
automatically an optimal multi-fault protection policy.

## M3.1 — researcher-driven estimator validation

- [x] Add positive per-example first-order damage with shared Jacobian work.
- [x] Add signed/nonnegative activation-cancellation controls for qPolarGrad.
- [x] Add uniform candidate sampling and independently controlled replication
  factors, including aliased Hutchinson probe-count series.
- [x] Add schema-3.1 bundles, aggregate reports, and multi-logical-BER studies
  while preserving schema-3.0 payload compatibility.

The next activity is researcher-run configuration studies and scientific
review, not another estimator redesign.

## M4 — hardware equivalence and mapping

Freeze one checkpoint and verify NetSuRF checkpoint → folded QKeras → HLS C simulation → RTL fixed-vector equivalence. Produce a machine-readable logical-bit-to-register map. Parse, hash, and validate zero/full plus selected intermediate Genus reports. At this boundary, create a project-owned Wolf flow copy/registry manifest and validate its Cadence compatibility path.

## M5 — mitigation sequence

1. Reconstruct register-storage TMR and voter guarantees.
2. Add packed-word parity with an explicit detected-error action.
3. Add shortened packed-word SECDED with SEC/DED/DUE accounting.
4. Add Ranger on calibrated activation sites with synthesized clamp logic.
5. Add architecture-specific ABFT last, because recovery/re-execution requirements are the least portable.

Every method must share stated physical targets, matched fault conditions where meaningful, and measured area/power/latency/timing—not a forced per-bit abstraction.
