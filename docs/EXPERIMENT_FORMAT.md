# Versioned experiments and result bundles

Experiment TOML schemas `1.0.0`, `2.0.0`, `3.0.0`, and `3.1.0` are supported user-authored
contracts. Version 1 preserves the historical adapter; version 2 adds explicit
benchmark, quantization-backend, injectable-parameter, and persistent or
per-inference fault semantics. Version 3 adds explicit sample partitions,
empirical candidate selection, versioned numerical estimators, composable
ordering policies, multiple coverages, and separate ranking/fault seeds. The loader is strict: unknown fields,
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

Schema-3 bundles generate only applicable tables. Empirical studies use
`empirical_bits.csv`, `estimator_predictions.csv`, and `ranking_metrics.csv`;
protection studies use `trials.csv`, `protection_results.csv`,
`paired_comparisons.csv`, and `mask_composition.csv`. Every study records
`artifacts/sample_partitions.json`; sampled empirical populations also record
the inclusion rule and conditional stratum probability. Column contracts and
null semantics are defined in `netsurf/schemas/measurements/v3.schema.json`
and `v3_1.schema.json` for their respective versions.

Schema `3.1.0` is additive. It adds:

- `candidates.strategy = "uniform"` for uniform sampling without replacement;
- estimator `result_name`, allowing independently recorded probe-count
  variants of the same estimator definition;
- `per_sample_positive_damage`;
- `[replications]` with `count`, `candidate_seed_stride`,
  `hutchinson_seed_stride`, `sample_offset_stride`, and `fault_seed_stride`;
- optional `protection.probabilities` for multiple logical-BER conditions.

The base candidate, estimator, and fault seeds are advanced by their respective
stride times the zero-based replication index. The sample offset is advanced
the same way. A benchmark rejects offsets its declared partitions cannot
support. Each selected identity, inclusion rule, sampling probability, probe
seed/count, partition index/hash, and fault schedule is serialized. Replicated
tables add a `replication` column and aggregate rows use a configured
replication—not an inference—as the statistical unit. Runtime is reportable
execution metadata and never enters the scientific payload hash.

## Schema 4.0 conditional search

Schema `4.0.0` uses protocol `conditional-protection-search/v1`. Its mandatory
tables are `[samples]`, `[[estimators]]`, `[[guidance]]`, `[initial_mask]`,
`[search]`, `[polar]`, `[fault]`, `[final_evaluation]`, `[replications]`, and
`[output]`. Guidance references existing estimator results and declares a
lexicographic ordering plus rank-mixture weight. Search declares strategy,
incremental/full evaluator, fixed budgets, explicit candidate-pool restriction,
nonzero uniform exploration, cache memory, and annealing controls.

Schema-4 bundles add `final_trials.csv`, `search_steps.csv`, `convergence.csv`,
`proposal_evaluations.csv`, `incremental_validation.csv`,
`estimator_runtime.csv`, `mask_composition.csv`, `mask_overlap.csv`,
and, when applicable, `polar_interactions.csv` and
`exact_mask_enumeration.csv`. Exact masks, rankings, candidate pools, partitions,
and matched held-out schedules are stored under `artifacts/`. Table units and
identities are in `netsurf/schemas/measurements/v4.schema.json`.

`search_results.csv` records the iteration and accumulated measured search time
at which each strategy first reached its final best search-risk estimate.
`mask_overlap.csv` records pairwise intersection, union, Jaccard overlap, and
fraction-of-budget overlap for the final best masks within each replication.
These are derived diagnostics and do not enter the scientific payload hash.

## Schema 4.1 controlled conditional comparison

Schema `4.1.0` uses protocol `conditional-protection-search/v2`. It adds named
`[[methods]]`, multiple logical BER conditions, and separate natural-Bernoulli
and conditional held-out counts and seeds. Schema 4.0 remains readable with its
original scientific payload.

Methods share the checkpoint, codec, initial mask, eligible universe, candidate
pool, data partitions, and base search backgrounds within a BER/replication.
Candidate pools restrict search moves only; faults remain sampled over the full
eligible-bit population.

Final masks are fixed before held-out seeds are used. Natural evaluation writes
`final_trials.csv`. Conditional evaluation follows
`sorted_removed_then_sorted_added/v1`; one base field contributes one complete
path sum. New tables are `conditional_context_contributions.csv`,
`conditional_mask_comparisons.csv`, `conditional_swap_paths.csv`, and
`proposal_diagnostics.csv`, defined by `measurements/v4_1.schema.json`.

## Schema 4.2 topology-aware search

Schema `4.2.0` uses protocol `topology-aware-conditional-search/v1`. It adds a
strict `[topology]` table, per-method interaction-model/strength/shuffle
ablations, optional per-method iteration budgets, and one or more global
logical-bit budgets through `initial_mask.budget_bits_list`. Protection budget
is a reporting and pairing condition and is never pooled across budgets.

`topology_interactions.csv` separates exact finite polarity,
topology-corrected interaction, local GGN, conventional Kronecker, and
stratified Kronecker values. `proposal_influence.csv` records rank changes,
top-k overlap, total variation, and Jensen--Shannon divergence before versus
after interaction guidance. Sparse missing edges are unavailable observations,
not numerical zeros.

## Schema 4.4 controlled factorized-guidance ablation

Schema `4.4.0` uses protocol `factorized-guidance-ablation/v1`. It preserves
all schema-4.3 fields and requires `[proposal_control]` and `[factor_cache]`.
Each method explicitly names `proposal_model`, `comparison_role`,
`aggregate_shuffle_mode`, and `sign_bit_weight`. Exact accepted values and
validation rules are documented in `docs/M4_4_AUDIT.md` and encoded by
`netsurf.search_config.load_search`.

New `controlled_proposals.csv` rows identify every removal/addition candidate
that reaches the configured shortlist, its normalized proposal probability,
rank, selection and exact-verification status, proposal model, and shuffle
mode. `factorized_guidance.csv` adds bit-table/factor cache keys, hit/miss
status, build attribution, batched-query time, aggregate-build time, update
time, conditional-screening time, and exact-verification time. These execution
times remain outside the stable scientific payload.
