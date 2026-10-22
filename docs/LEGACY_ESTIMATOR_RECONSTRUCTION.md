# Legacy Hessian and qPolarGrad reconstruction

This document records source archaeology. Historical code is evidence, not a
validated estimator specification.

## Hessian-related implementations

`DEPRECATED/legacy_source/netsurf/core/ranking.py:1213` defines
`HessianWeightRanker`. Its `top_eigenvalues` routine at approximately lines
1458–1522 computes leading Hessian eigenpairs by power iteration using
Hessian-vector products. It is **not** a diagonal-Hessian estimator. Parameter
scores are assembled from terms proportional to
`abs(lambda_k * v_k * parameter_value)` and summed over retained eigenpairs.
The historical `max` option also takes a sum, a confirmed implementation
defect. The default constructor path requests eight eigenpairs and up to 1,000
iterations; the legacy experiment alias in `config/defs.py:66` requests three
and 1,000.

The implementation has additional reproducibility and interpretation risks:

- random initial vectors are not seeded;
- normalization by sample count appears both around the HVP/eigenvalue path
  and around losses that may already be means;
- simple Gram–Schmidt deflation is used without robust eigensolver diagnostics;
- convergence and numerical error are not propagated into scores;
- the score is a leading-eigenspace heuristic, not the Taylor term
  `0.5 * H_bb * delta_b^2`.

A second path at approximately lines 1619–1856 imports the external
`fkeras.metrics.HessianMetrics`, uses `top_k=8`, `max_iter=500`, and a batch of
480, omits bias parameters, and remaps eigenpair scores to stored parameters.
The exact FKeras package/revision and its cited method were not preserved, so a
faithful independent reproduction is blocked on provenance. It remains a
historical accelerated Hessian-inspired method, not a validated diagonal
Hessian baseline. `HessianDeltaWeightRanker` at line 1886 additionally replaces
weights with bit deltas before eigenanalysis, which changes the Hessian
evaluation point rather than merely evaluating a finite perturbation.

One useful idea is preserved independently: the later legacy Laplace code
contains Hutchinson diagonal estimation through the identity
`diag(H)=E[v * H v]`. M3 implements a seeded Rademacher-HVP estimator with an
explicit probe count and standard-error diagnostics. It also implements the
exact diagonal on the 17-parameter NumPy reference. These are separately
versioned methods and are not presented as faithful FKeras reproduction.

## qPolar and qPolarGrad

The legacy rankers are registered around `ranking.py:564–567`.
`QPolarWeightRanker` begins at line 1962 and `QPolarGradWeightRanker` at line
2132. Dense-layer drift is computed in
`DEPRECATED/legacy_source/netsurf/dnn/layers.py:968` and following. For a dense
kernel the recovered score is approximately

`abs(mean_x(attacked_layer_input_i * delta_b) * dL/dw_i)`.

Bias impact is the bit perturbation itself. Signed input/drift values are
averaged before the final absolute value, so cancellation across examples can
produce zero scores. The historical global all-bit attack is applied through
mutable `*_N` state. Downstream activations are consequently taken from a
globally corrupted first call, and repeated calls can contaminate later state.

The gradient already includes the corresponding layer-input factor. Multiplying
it by an averaged activation-drift factor introduces another input-related
factor, and the result does not have a settled loss-change interpretation. The
manuscript Equation (7), Table I, and code are also not mutually unambiguous
about whether the bit perturbation is counted once or twice.

M3 therefore exposes three explicit, stateless versions:

1. `legacy_qpolargrad`, implementation
   `recovered-dense-first-call/v1`: the recovered dense expression with the
   first-call globally attacked activation semantics, but no persistent state.
2. `clean_activation_qpolargrad`, implementation
   `clean-dense-mean-cancelled/v1`: the same operations using clean activations.
3. `uncancelled_activation_drift`, implementation
   `clean-dense-mean-absolute/v1`: aggregates per-input drift magnitude before
   cancellation.

Only the first is the recovered historical method. The other two are research
candidates, not corrections accepted by definition. All three must be judged
against exact finite-bit measurements and must not be silently renamed
`qPolarGrad`.

