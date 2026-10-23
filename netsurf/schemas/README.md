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
and reader; migrations must write a new bundle.
