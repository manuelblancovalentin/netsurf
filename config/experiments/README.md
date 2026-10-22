# Experiment configurations

`template.toml` and `smoke.toml` preserve the schema-1 historical compatibility
experiment. New work uses schema `2.0.0` and protocol
`quantized-weight-fault/v2`:

- `numpy_reference.toml`: dependency-light exact encoded-bit reference.
- `mnist_msb.toml`: one MSB-first ranker plus no/full protection controls.
- `mnist_ranker_comparison.toml`: matched MSB-first, random, and
  first-order gradient-times-exact-delta comparison.

All examples default to unique immutable bundles under `workspace/runs/`.
Delete estimator tables to select a single existing ranker; unsupported names
fail during validation rather than being ignored.
