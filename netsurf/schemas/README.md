# NetSuRF serialized schemas

The experiment configuration, resolved execution manifest, result summary, and
tabular measurement contracts are independently versioned. Readers reject
unknown major/minor schema versions; a package release number is not a schema
version. Finalized bundles are never rewritten in place.

Version `1.0.0` supports the bounded legacy MNIST persistent-weight protocol.
The Python validators enforce the executable subset and load these documents as
the machine-readable field contract. Breaking changes require a new schema file
and reader; migrations must write a new bundle.
