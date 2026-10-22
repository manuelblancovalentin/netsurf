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
- `netsurf.protection`: global, position-first, and greedy nonuniform-cost policies. The greedy policy explicitly makes no optimality claim.
- `netsurf.faults`: explicit deterministic events, persistent and per-inference schedule generation, perfect-mask suppression, and exact accounting. The first milestone executes persistent-trial weight faults only.
- `netsurf.adapters.legacy_mnist`: isolated TensorFlow/legacy checkpoint bridge with exact state restoration.
- `netsurf.statistics`: sample statistics and Student-t confidence intervals.
- `netsurf.experiment_config`: strict versioned TOML loading and executable validation.
- `netsurf.adapters.protocol`: framework-neutral benchmark/model adapter contract; the registry imports TensorFlow only when explicitly requested.
- `netsurf.experiment_engine`: benchmark-neutral estimator, policy, schedule, trial, and aggregation orchestration.
- `netsurf.result_bundle`: atomic staging, write-once finalization, checksums, typed reading, and corruption detection.
- `netsurf.reporting` and `netsurf.plotting`: presentation derived only from saved measurements.
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

## Deferred interfaces

Activation injection, Hessian estimation, empirical single-bit validation, mitigation implementations, HLS/RTL export, logical-to-physical mapping, and Wolf hardware execution are intentionally deferred until the first software milestone passes review.
