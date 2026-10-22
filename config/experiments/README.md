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

Schema-3 M3 studies are intentionally separate protocols:

- `numpy_single_bit.toml`: exhaustive 102-bit reference and estimator checks.
- `mnist_single_bit.toml`: bounded 96-candidate stratified empirical study.
- `mnist_msb_policies.toml`: partial/full-tier MSB tie-breaking comparison.
- `mnist_estimator_comparison.toml`: larger empirical estimator study.
- `mnist_multi_coverage.toml`: bounded matched protection curves.

Ranking/calibration, empirical-validation, and final-evaluation partitions are
explicit and disjoint. Multiple `msb+random` policy entries need distinct names
and seeds; their ranking-mask variability is not interchangeable with repeated
fault schedules using one fixed mask.
