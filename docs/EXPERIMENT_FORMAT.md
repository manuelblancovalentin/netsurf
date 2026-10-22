# Versioned experiments and result bundles

Experiment TOML schema `1.0.0` is the user-authored contract. The loader is
strict: unknown fields, unsupported estimators/policies/fault semantics, and
unsupported schema or protocol versions fail before model loading. Templates
are in `config/experiments/`; packaged copies make `netsurf init-experiment`
work outside the checkout.

Each run is staged as a hidden `.incomplete` directory. Only after every
requested measurement, artifact, report, plot, manifest, and checksum has been
written is it atomically renamed to its unique run ID and marked `FINALIZED`.
A failed run remains visibly incomplete and is never accepted by the reader.

```text
workspace/runs/<experiment>/<run-id>/
├── config.toml
├── resolved_config.json
├── manifest.json
├── summary.json
├── scientific_payload.json
├── measurements/
│   ├── trials.csv
│   ├── estimator_scores.csv
│   └── protection_results.csv
├── artifacts/
│   ├── protection_masks.json
│   └── fault_schedules.json
├── plots/                         # only when requested and meaningful
├── report.md                      # when requested
├── checksums.json
└── FINALIZED
```

The measurement schema documents names, types, units, null representation,
and the independent sampling unit. `trials.csv` contains paired policy rows:
the independent unit is the persistent fault schedule/trial, not an inference
sample. Schema evolution must add a new reader or an explicit non-mutating
migration; finalized bundles are never rewritten in place.
