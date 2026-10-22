# Versioned experiments and result bundles

Experiment TOML schemas `1.0.0` and `2.0.0` are supported user-authored
contracts. Version 1 preserves the historical adapter; version 2 adds explicit
benchmark, quantization-backend, injectable-parameter, and persistent or
per-inference fault semantics. The loader is strict: unknown fields,
unsupported combinations, and unsupported schema or protocol versions fail
before model loading.

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
│   ├── estimator_runtime.csv             # modern experiments
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
the independent unit is the fault schedule/trial, not an inference sample.
Schema-2 runs include no- and full-protection controls. Estimator wall times are
execution metadata and excluded from the stable scientific payload. Schema
evolution must add a reader or explicit non-mutating migration; finalized
bundles are never rewritten in place.
