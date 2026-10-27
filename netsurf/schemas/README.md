# NetSuRF serialized schemas

The experiment configuration, resolved execution manifest, result summary, and
tabular measurement contracts are independently versioned. Readers reject
unknown major/minor schema versions; a package release number is not a schema
version. Finalized bundles are never rewritten in place.

Experiment and measurement version `1.0.0` support the bounded legacy MNIST
persistent-weight protocol. Experiment `2.0.0` adds explicit quantization
backend, injectable parameter classes, and modern benchmark adapters;
measurement `2.0.0` adds the corresponding control/policy rows and estimator
runtime records. Manifest and summary remain compatible at `1.0.0`.

Experiment and measurement `3.0.0` add explicit role-specific sample
partitions, empirical candidate inclusion metadata, estimator implementation
versions, composable ordering policies, multiple coverages, signed
predictions, exact isolated-bit effects, tied-rank metrics, mask composition,
and paired multiplicity-adjusted comparisons. Manifest and summary remain at
`1.0.0`; schema-3-specific fields are additive and identified by the resolved
experiment version.

Experiment and measurement `3.1.0` add uniform candidate sampling, aliased
estimator instances, positive per-example first-order damage, explicit
independent-replication seed/offset strides, replication-labelled tables and
aggregates, pairwise estimator-order diagnostics, and multiple logical-BER
conditions. The schema-3.0 reader and serialization path are retained; its
validated scientific digest is unchanged.

The Python validators enforce the executable subset and load these documents as
the machine-readable field contract. Breaking changes require a new schema file
and a new reader. Experiment and measurement `4.0.0` add the conditional-search
protocol, matched held-out fault schedules, exact incremental diagnostics,
sparse polar interactions, fixed-budget search traces, and restricted exact
enumeration. Its executable validator is `netsurf.search_config.load_search`.
and reader; migrations must write a new bundle.

Experiment and measurement `4.1.0` add named matched methods, multi-BER
conditions, independent natural and telescoping-conditional held-out fields,
and whole-path cluster-level uncertainty. Schema 4.0 remains readable unchanged.

Experiment and measurement `4.2.0` add exact Dense--ReLU--Dense topology
responses, structural pair classes, layer-budgeted sparse discovery,
mask-dependent proposal guidance, and conventional or confidence-stratified
Kronecker GGN diagnostics. Missing sparse edges are recorded as unmeasured,
not assumed physically zero. Schemas 4.0 and 4.1 retain their original readers.

Experiment and measurement `4.3.0` add common-background conditional
candidate marginals, exact shortlist verification, full-population factorized
Kronecker aggregates, rank-one mask updates, and exclusive timing semantics.

Experiment and measurement `4.4.0` add explicit controlled proposal models,
fixed candidate and exact-verification budgets, shared search backgrounds,
global or layer/bit-position-stratified shuffled Kronecker controls, run-local
factor calibration caching, batched factorized queries, and per-candidate
proposal records. Schema 4.3 retains its historical scalar numerical path.

Pruning preparation schema `5.0.0` is a separate checkpoint-generation
contract, not an experiment-result schema. It defines the immutable source,
calibration partition, physical hidden-neuron pruning method/schedule,
separately weighted reliability components, fine-tuning mode, and workspace
checkpoint destination. Its executable validator is
`netsurf.pruning_config.load_pruning`. Existing experiment, study, search,
manifest, summary, and measurement readers remain unchanged.
