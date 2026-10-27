# M4.4 factorized-guidance ablation and optimization contract

## Gate A: M4.3 fairness audit

M4.3 methods used the same checkpoint, encoded-bit population, initial mask,
outer candidate pool, ranking and search samples, and held-out protocol. They
did **not** constitute a controlled equal-cost comparison:

- method search seeds were offset by method index, so search backgrounds were
  not identical;
- marginal screening verified one swap while corrected/factorized methods
  verified the configured shortlist;
- per-method iteration overrides could differ; and
- factor construction was repeated per factorized method and condition.

Schema `4.4.0` corrects these controllable asymmetries without rewriting the
optimizer. Methods whose `comparison_role` is `controlled` share the same
iteration-indexed search-background seed, fixed candidate universe, initial
mask, iteration count when enforcement is enabled, shortlist sizes, and exact
verification budget. They also share the same base guidance, preventing hidden
bit-significance or layer-priority access from varying between controlled
methods. Their proposal model remains the experimental variable.
The restricted candidate universe constrains protection moves only; logical
fault backgrounds continue to cover the full eligible-bit population.

`conditional_greedy` remains a separately labelled `reference`, not an
equal-cost member of the screening comparison. Static gradient is a zero-search
baseline. Exact conditional marginal screening is exact for the candidates it
screens, but its configured shortlist is not an exhaustive whole-network scan.

## Controlled proposal models

Each `[[methods]]` entry declares one of `gradient`,
`gradient_sign_enriched`, `conditional_marginal`, `kronecker`,
`stratified_kronecker`, `shuffled_kronecker`, `uniform`, or
`quality_reference`.

For `shuffled_kronecker`, `aggregate_shuffle_mode = "global"` preserves the
global score multiset. `"layer_bit_position"` independently preserves the
score multiset within every `(layer, encoded bit position)` stratum. Neither
mode changes or expands the legal candidates. Proposal identities,
probabilities, selected status, and exact-verification status are saved in
`measurements/controlled_proposals.csv`.

## Exact configuration additions

Schema `4.4.0` uses protocol `factorized-guidance-ablation/v1` and adds:

```toml
[proposal_control]
candidate_universe = "shared_candidate_pool"
verification_budget = 4
shared_backgrounds = true
enforce_equal_iterations = true

[factor_cache]
enabled = true
scope = "run"
vectorized_queries = true
```

Every schema-4.4 method additionally declares:

```toml
[[methods]]
name = "dynamic_kronecker"
strategy = "factorized_screening"
guidance = ["gradient_global"]
interaction_model = "factorized_kronecker"
proposal_model = "kronecker"
comparison_role = "controlled"
aggregate_shuffle_mode = "none"
sign_bit_weight = 1.0
aggregate_population = "full"
dynamic_aggregate = true
```

`comparison_role` is `controlled`, `static`, or `reference`. The sign weight
must be at least one. Shuffling is accepted only for
`proposal_model = "shuffled_kronecker"`; stratified shuffling uses
`aggregate_shuffle_mode = "layer_bit_position"`. A controlled verification
budget cannot exceed `removal_shortlist × addition_shortlist`. Unsupported
combinations fail validation.

The repository deliberately contains no maintained M4.4 research TOML. A
researcher should copy a schema-4.3 configuration outside the maintained set,
change the schema/protocol, add these tables and explicit method fields,
validate it, inspect the resolved configuration, and only then run it:

```bash
proj activate netsurf
netsurf validate /path/to/research-authored-m4_4.toml --resolved
netsurf run /path/to/research-authored-m4_4.toml
```

## Gate B: cache and vectorization semantics

`FactorCalibrationCache` is run-local. It separately stores immutable encoded
bit metadata and calibration-dependent conventional/stratified factors.
Mask-dependent vulnerable-bit moments, proposal scores, and rank-one updates
are never shared between methods or BER conditions.

The bit-table key covers checkpoint digest, benchmark/architecture, eligible
bit identities, quantization interpretation, numerical backend, and NumPy
version. The factor key additionally covers dataset digest, exact calibration
partition indices/hash, factor definition, stratification configuration,
response-cache configuration, and deterministic search seed. A mismatch is a
cache miss; stale factors are not substituted.

Vectorized construction uses indexed accumulation into the same layerwise
moment matrices. Batched queries gather factorized terms for all candidates and
perform exact encoded-bit self exclusion. Schema 4.3 continues to use its
historical scalar construction and query order.

Calibration and bit-table build costs are charged only to the first method
that creates an exact key; cache hits receive zero shared-build charge.
Aggregate construction, candidate queries, mask updates, conditional screening,
and exact verification remain method-specific. `timing_phases.csv` remains the
only additive whole-run timing table.

## Scientific interpretation

The primary endpoint remains independently held-out target-minus-initial
conditional risk. Search iterations or contexts are not independent research
replications. Shuffled controls diagnose identity-aligned interaction
information beyond layer/bit-position score distributions; they do not
validate Kronecker curvature as a physical fault model. A component break-even
estimate is diagnostic only. An end-to-end claim requires complete workloads
at matched candidate, verification, and held-out budgets.

## Bounded performance evidence

On Kona on 2026-10-27, the internal 102-bit schema-4.4 fixture was warmed once
and measured five times per path. It used eleven methods, two iterations, four
search backgrounds, and two exact verifications per controlled method and
iteration. No MNIST campaign was launched.

| Path | Factor + bit-table mean (s) | Candidate-query mean (s) | Search-phase mean (s) | Whole-call mean (s) |
|---|---:|---:|---:|---:|
| cache disabled, scalar queries | 0.019267 | 0.006873 | 0.393108 | 0.477003 |
| run cache, batched queries | 0.008365 | 0.002350 | 0.388484 | 0.518144 |

The corresponding query component was about 2.93× faster and charged factor
preparation was about 2.30× lower. The complete search phase was about 1.2%
shorter. The whole API call was not faster in this tiny benchmark; non-search
phases dominate and the two paths showed different out-of-search overhead.
Therefore these measurements validate component acceleration and accounting,
not an end-to-end scaling claim or break-even point. A matched prepared-MNIST
study is still required for an end-to-end conclusion.
