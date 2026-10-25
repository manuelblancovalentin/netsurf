# Experiment configurations

M4 conditional-search configurations:

- `numpy_pgcs_exact.toml`: bounded exact-enumeration and all-strategy reference.
- `mnist_incremental_validation.toml`: prepared-checkpoint E01/E02 smoke run.
- `mnist_static_guided_search.toml`: researcher-run static/nonpolar comparison.
- `mnist_polar_guided_search.toml`: researcher-run polar/nonpolar comparison.

The last two are intentionally not executed by the regression suite. Adjust
budgets only through TOML and retain disjoint ranking/search/final partitions.

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
- `numpy_qpolar_cancellation.toml`: exhaustive signed-activation cancellation
  control; switch its documented dataset path for the matched nonnegative
  control.
- `mnist_estimator_uniform.toml`: uniform-without-replacement comparison of
  mean-gradient, absolute per-example, positive per-example, and Hessian
  estimators.
- `mnist_estimator_replicated.toml`: independent candidate/sample replications
  and aliased 2/4/8-probe Hutchinson variants.
- `mnist_multi_coverage_ber.toml`: existing policies across several protection
  fractions and logical-BER conditions.

Ranking/calibration, empirical-validation, and final-evaluation partitions are
explicit and disjoint. Multiple `msb+random` policy entries need distinct names
and seeds; their ranking-mask variability is not interchangeable with repeated
fault schedules using one fixed mask.

The four M3.1 templates use schema `3.1.0`. The researcher can change candidate
`count`/`seed`, estimator `probes`/`seed`, sample counts, replication strides,
protection `coverages`/`probabilities`, and execution `trials` without editing
NetSuRF source. Unsupported estimator, adapter, partition, and schema
combinations fail before result creation.

Schema-4.1 maintained configurations are:

- `numpy_pgcs_conditional_validation.toml`: exhaustive small reference;
- `mnist_pgcs_matched_smoke.toml`: bounded prepared-checkpoint integration;
- `mnist_pgcs_matched_research.toml`: five-replication, two-BER researcher
  study, not executed by ordinary tests.
