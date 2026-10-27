# M4.3 reconstruction and audit

## M4.2 25%-coverage failure mode

The finalized M4.2 bundle
`mnist_topology_polar_research/20261027T011431.539017Z-94dcc7c2` is preserved
unchanged. At 38,175 protected bits, finite-polar, topology-corrected,
conventional-Kronecker, and stratified-Kronecker proposal distributions changed
in 0 of 120 recorded iterations per strategy (mean total variation at numerical
noise scale). The shuffled-polar negative control changed all 120 distributions
and had mean total variation 0.01605.

This was not caused by globally zero interactions. In the same 25%-coverage
replications, nonzero recorded interactions were present for 6,475/9,216
finite-polar, 6,485/9,216 topology-corrected, 5,418/7,680 conventional
Kronecker, and 5,393/7,680 stratified-Kronecker rows. The supported diagnosis
is sparse edge/value alignment: the measured pair graph did not cover the
candidate-to-current-vulnerable neighborhoods used by the proposal at that
budget. Shuffling nonzero values over the existing keys restored a distribution
change, but is only a negative-control diagnostic.

Schema 4.3 addresses missing coverage with a full-vulnerable-population
factorized aggregate. It does not reinterpret missing sparse edges as zero
physical interactions and does not claim that aggregate guidance improves the
final mask.

## Exact swap decomposition

For protected removal `a`, unprotected addition `b`, and common background `G`,
candidate screening records reusable marginal terms `C_a` and `D_b`. Their
difference omits the pair term. The corrected three-state swap estimate is
evaluated independently before acceptance. Tests enumerate all states of a
small universe, including unequal bit probabilities and a constructed
higher-order loss, and verify the corrected identity against direct expected
risk.

Candidate marginals are evaluated on common random fields. The bundle retains
both marginal summaries and every candidate/context weighted contribution, so
paired covariance and alternative summaries can be reconstructed without
rerunning the model.

## Factorized aggregate contract

`FactorizedInteractionAggregate` sums the selected Kronecker-GGN surrogate over
the current vulnerable population. Conventional and confidence-stratified
forms are distinct. Exact self exclusion uses encoded bit identity; other bits
in the same scalar word remain separate events. Accepted swaps update the
aggregate by rank-one changes, and each iteration records reconstruction drift.

Full-population coverage is the main method. Candidate-pool-only aggregation is
an explicit restricted negative control. The aggregate is a local curvature
surrogate, not exact finite-fault interaction or accepted protection theory.
All accepted swaps still pass exact matched conditional verification.

## Timing audit

Schema 4.3 records mutually exclusive engine phases in `timing_phases.csv`.
The `whole_process` row contains elapsed time, the sum of exclusive phases, and
uninstrumented remainder. `search_results.csv` contains inclusive per-method
time, while `convergence.csv` contains cumulative trajectory time. Summing
either of those tables with phase timing double-counts work; this explains the
historical discrepancy between whole-process time and naive sums of method,
iteration, and convergence columns.

Factorized setup time is attributed once to the first decision row; cumulative
query/update time is attributed once to the last row. Timing columns are
operational evidence and remain outside the scientific payload hash.
