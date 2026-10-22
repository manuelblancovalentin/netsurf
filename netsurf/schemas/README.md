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

The Python validators enforce the executable subset and load these documents as
the machine-readable field contract. Breaking changes require a new schema file
and reader; migrations must write a new bundle.
