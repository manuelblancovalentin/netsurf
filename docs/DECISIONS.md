# Decisions

## Accepted engineering decisions

1. V2 is the canonical side-effect-free package (`netsurf`); historical code is preserved under `DEPRECATED/legacy_source/netsurf` and loaded only by explicit adapters.
2. Bit identity is LSB-zero and fixed-point notation matches recovered QKeras semantics.
3. Estimators, policies, schedules, injection, mitigation, and hardware cost are separate concepts.
4. The first milestone uses held-out MNIST test data, persistent-trial logical weight faults, matched schedules, and write-once outputs.
5. Historical qPolarGrad is retained by name and warning; it is not silently promoted as the accepted estimator.
6. `proj` project `netsurf` selects the dedicated Python 3.11 environment. The canonical core is installed editable with NumPy 1.26.4. The TensorFlow 2.11 milestone remains isolated in the preserved Python 3.10 `netsurf-legacy` environment until equivalence is validated on Python 3.11.
7. Wolf is reserved for hardware stages; its Cadence backend is currently a compatibility path and must be validated against NetSuRF's historical scripts before use.
8. Pinned MLPerf Tiny DS-CNN and ToyCar TFLite models are accepted as distinct
   modern benchmark identities. Their exact inference backend is TensorFlow
   Lite 2.14.1 `BUILTIN_REF`; architecture-specific STE companions provide
   ranking derivatives only.
9. ToyCar ranking uses reconstruction MSE while final task evaluation uses
   file-level ROC-AUC. The objectives and their uncertainty are reported
   separately under schema 3.4.
10. VWW remains blocked until official evaluation-image pixel provenance is
    independently verified.

## Not yet approved scientifically

- Whether the accepted qPolarGrad formulation should use clean activations, attacked activations, an additional delta, or be replaced by another estimator.
- Global versus position-first protection as the canonical policy.
- Whether learned alpha/beta and biases are physical protected storage targets or folded implementation parameters.
- Canonical fault window and physical exposure normalization.
- Canonical hardware memory architecture and mitigation recovery semantics.
- Publication direction and benchmark suite.
