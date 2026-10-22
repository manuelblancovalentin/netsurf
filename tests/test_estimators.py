import numpy as np

from netsurf.identities import BitIdentity
from netsurf.ranking import FirstOrderEstimator, LegacyQPolarGradEstimator


def test_first_order_is_gradient_times_exact_delta():
    identities = (BitIdentity("w", 0, 0), BitIdentity("w", 0, 1))
    gradients = {"w": np.array([3.0])}
    deltas = {"w": np.array([[0.25, -2.0]])}
    ranking = FirstOrderEstimator().estimate(identities, gradients, deltas)
    assert ranking.scores[identities[0]] == 0.75
    assert ranking.scores[identities[1]] == 6.0


def test_legacy_qpolargrad_does_not_add_a_second_delta():
    identity = BitIdentity("w", 0, 0)
    ranking = LegacyQPolarGradEstimator().estimate(
        (identity,),
        {"w": np.array([3.0])},
        {"w": np.array([[0.25]])},
        activation_source="clean",
    )
    assert ranking.scores[identity] == 0.75
    assert ranking.metadata["expression"] == "abs(P_bit * dL/dw)"
