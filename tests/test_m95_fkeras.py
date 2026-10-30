import numpy as np

from netsurf.fkeras_baseline import (
    fkeras_msb_tier_order, leading_hessian_eigenpairs, principal_weight_hessian_score,
)
from netsurf.identities import BitIdentity


def test_hessian_power_iteration_and_paper_principal_score_on_quadratic():
    hessian = np.diag([4.0, 2.0])
    vectors, diagnostics = leading_hessian_eigenpairs(
        lambda value: hessian @ value, 2, count=2, seed=95,
        max_iterations=200, tolerance=1e-10,
    )
    np.testing.assert_allclose(sorted(diagnostics.eigenvalues, reverse=True), [4.0, 2.0], atol=1e-8)
    assert max(diagnostics.residual_norms) < 1e-8
    score = principal_weight_hessian_score(
        np.asarray([1.0, 2.0]), vectors, np.asarray(diagnostics.eigenvalues)
    )
    np.testing.assert_allclose(score, [4.0, 4.0], atol=1e-7)


def test_fkeras_order_never_leaks_word_score_across_msb_tiers():
    identities = tuple(BitIdentity("k", word, bit) for word in range(2) for bit in range(2))
    ordered = fkeras_msb_tier_order(identities, {("k", 0): 1.0, ("k", 1): 100.0})
    assert [value.bit_index for value in ordered] == [1, 1, 0, 0]
    assert [value.flat_index for value in ordered] == [1, 0, 1, 0]
