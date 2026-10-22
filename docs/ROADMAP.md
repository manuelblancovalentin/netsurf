# Roadmap

## M0 — preservation and foundation

- [x] Preserve dirty-state provenance, recover remote `master`, and create neutral branch `v2`.
- [x] Inspect and register `proj`; inspect Wolf without modifying it.
- [x] Add exact codec, bit identities, estimators, policies, deterministic schedules, state-safe adapter, statistics, result store, and dependency-free tests.
- [x] Promote the reconstruction to canonical `netsurf`, package it, and externalize datasets and generated artifacts without merging conflicting collections.

## M1 — deterministic MNIST-lite experiment

Acceptance criteria: exact checkpoint and held-out split; deterministic ranking/mask hashes; matched schedules; exact restoration; correct confidence summaries; two identical payload hashes; explicit legacy qPolarGrad label.

## M2 — estimator validation

On a deliberately small network/subset, exhaustively inject individual bits. Compare rank correlations, top-k overlap, loss prediction calibration, and protection outcomes for gradient magnitude, first order, clean-activation qPolarGrad, recovered attacked-activation qPolarGrad, Hessian, and random. Resolve qPolarGrad theory only after these data exist.

## M3 — hardware equivalence and mapping

Freeze one checkpoint and verify NetSuRF checkpoint → folded QKeras → HLS C simulation → RTL fixed-vector equivalence. Produce a machine-readable logical-bit-to-register map. Parse, hash, and validate zero/full plus selected intermediate Genus reports. At this boundary, create a project-owned Wolf flow copy/registry manifest and validate its Cadence compatibility path.

## M4 — mitigation sequence

1. Reconstruct register-storage TMR and voter guarantees.
2. Add packed-word parity with an explicit detected-error action.
3. Add shortened packed-word SECDED with SEC/DED/DUE accounting.
4. Add Ranger on calibrated activation sites with synthesized clamp logic.
5. Add architecture-specific ABFT last, because recovery/re-execution requirements are the least portable.

Every method must share stated physical targets, matched fault conditions where meaningful, and measured area/power/latency/timing—not a forced per-bit abstraction.
