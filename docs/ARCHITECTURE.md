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
                     statistics + write-once ResultStore
```

## Implemented modules

- `netsurf.quantization`: signed/unsigned fixed-point contract, saturating ties-to-even encoding, exact two's-complement XOR deltas.
- `netsurf.identities`: immutable tensor and bit identities. Bit index zero is always the encoded LSB.
- `netsurf.ranking`: score-only estimators for random, correct MSB, gradient magnitude, elementary first order, and recovered legacy qPolarGrad.
- `netsurf.protection`: global, position-first, and greedy nonuniform-cost policies. The greedy policy explicitly makes no optimality claim.
- `netsurf.faults`: explicit deterministic events, persistent and per-inference schedule generation, perfect-mask suppression, and exact accounting. The first milestone executes persistent-trial weight faults only.
- `netsurf.adapters.legacy_mnist`: isolated TensorFlow/legacy checkpoint bridge with exact state restoration.
- `netsurf.statistics`: sample statistics and Student-t confidence intervals.
- `netsurf.result_store`: exclusive-create, read-only finalized artifacts.
- `netsurf.milestone`: bounded paired MNIST-lite experiment and twice-run reproducibility check.

## Storage boundary

`netsurf.paths` resolves datasets and persistent outputs from environment
variables. Benchmark definitions belong under `config/`; generated checkpoints,
rankings, runs, HLS, RTL, and synthesis workspaces remain outside Git. Root-level
compatibility symlinks are ignored and are not part of the package contract.

## Deferred interfaces

Activation injection, Hessian estimation, empirical single-bit validation, mitigation implementations, HLS/RTL export, logical-to-physical mapping, and Wolf hardware execution are intentionally deferred until the first software milestone passes review.
