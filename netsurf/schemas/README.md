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
The MLPerf Tiny adapter's additive `asset_lock` and `inference_backend` fields
bind its neutral asset identity and numerical TFLite resolver/runtime contract;
other adapters reject those fields.

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

Experiment and measurement `3.2.0` add the affine-TFLite numerical-bit-jump
baseline and tie-aware top-k boundary metrics used by the M9 ResNet-8 gate.

Experiment and measurement `3.3.0` separate independently randomized
estimator/ranking-mask seeds from Hutchinson probe seeds across replications.
This prevents ranking-mask variability, curvature-probe variability and fault
schedule variability from being conflated. Schema 3.0--3.2 execution and
serialized scientific payloads retain their historical behavior.

Experiment and measurement `3.4.0` add explicit task-level primary and loss
metric names. Classification retains the established accuracy and categorical
cross-entropy columns. Anomaly detection uses file-level ROC-AUC and
float-domain reconstruction MSE and never aliases them as classification
metrics.

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

Exact conditional-damage schema `6.0.0` is a separate bounded evidence
contract. It requires an explicit candidate-bit list, disjoint calibration and
held-out samples, independently seeded Bernoulli background fields, and exact
paired encoded-XOR measurements of `D_b(G) = R(G ∪ {b}) - R(G)` with the
candidate excluded from `G`. Its executable validator is
`netsurf.conditional_config.load_conditional`. It does not select a global
protection mask or redefine an existing susceptibility estimator.

Same-distribution conditional-protection schema `6.1.0` adds three explicit,
nonoverlapping KWS test-population windows, a model-selection-only exact
restricted-subcube mask, and independently seeded final-evaluation outside
backgrounds. It records every `2^K` response, exact candidate and outside-bit
identities, the isolated-bit comparator mask, paired held-out risk differences,
and post-hoc restricted regret with an explicit selection-bias warning. The
restricted optimum is never described as a whole-network optimum.

Contextual conditional-protection schema `6.2.0` retains the schema-6.1
restricted control and optionally verifies a finalized whole-network static
mask by scientific-payload hash, artifact checksum, checkpoint identity,
eligible population, and cardinality. A restricted candidate mask replaces
only the source-mask members inside the candidate pool; the source mask
outside the pool stays fixed. Every comparator therefore has the same global
protection cardinality. Exact response-table integration derives all
fixed-budget candidate-mask risks, conditional marginals, and paired mask
differences without additional inference. Its target law remains independent
Bernoulli; no importance or adversarial sampling semantics are implied.

Paired-swap held-out schema `6.3.0` preserves exact response-table selection
while adding efficient comparison of already frozen equal-cardinality masks.
Model-selection fields may be grouped under explicit independent replicate
seeds. Final master Bernoulli fields use a separate seed and contribute one
whole telescoping-path observation each. Step records retain exact backgrounds,
swapped identities, schedules, risks, event counts and pairing keys. The schema
also retains at least one optional full-enumeration final field as a numerical
reference; it does not use final data for mask or path selection.

Frozen-selection schema `6.4.0` separates `selection_only` from `final_only`.
The former finalizes exact selection tables without accessing final inputs;
the latter loads an explicitly hash-pinned frozen selection and never reruns
selection. A recorded exposure ledger rejects known reused confirmation
samples, and artifact contracts freeze the pool, context, budget, sample and
seed plans and comparison family. Joint schemas 6.0–6.3 retain their readers.
See `docs/M956_PUBLICATION_PREFLIGHT.md` for executable fields and boundaries.

Independent replication schema `6.5.0` adds only `replication_only` for the
hash-pinned original M956 selection. It freezes a separate final sample/field
plan without relaxing schema 6.4 or selecting new masks. The optional workflow
`record_per_input` diagnostic preserves scalar reductions and crossed
input/field identities. Replication-plan and statistical artifacts independently
identify version `1.0.0`. See `docs/M957_FROZEN_MASK_REPLICATION.md` for exact
keys, primary/secondary inference and exposure limitations.
