from __future__ import annotations

import itertools

import pytest

from netsurf.paired_swap import (
    deterministic_swap_path, exact_bernoulli_mask_risk,
    exact_paired_swap_contrast, paired_swap_field_contrast,
)
from netsurf.restricted_subcube import (
    integrate_response_table, subcube_states, swap_contrast_from_response_table,
)


@pytest.mark.parametrize(
    "risk",
    (
        lambda z: 2.0 * (0 in z) + 0.5 * (1 in z) - 0.75 * (3 in z),
        lambda z: 3.0 * ((0 in z) and (2 in z)) - 1.5 * ((1 in z) and (3 in z)),
        lambda z: 7.0 * all(index in z for index in (0, 1, 2)),
        lambda z: -4.0 * all(index in z for index in (0, 1, 2, 3)),
    ),
)
def test_exact_paired_swap_matches_full_bernoulli_risk_for_all_interaction_orders(risk):
    baseline, target = (0, 1), (2, 3)
    for probability in (0.0, 0.07, 0.5, 1.0):
        expected = (
            exact_bernoulli_mask_risk(risk, 5, probability, target)
            - exact_bernoulli_mask_risk(risk, 5, probability, baseline)
        )
        observed = exact_paired_swap_contrast(
            risk, 5, probability, baseline, target
        )
        assert observed == pytest.approx(expected, abs=1e-12)


def test_swap_path_orientation_roles_budget_and_zero_ber():
    path = deterministic_swap_path((3, 0), (4, 2))
    assert [(row.removed, row.added) for row in path] == [(0, 2), (3, 4)]
    risk = lambda z: 5.0 * (0 in z) - 2.0 * (2 in z)
    forward = exact_paired_swap_contrast(risk, 5, 0.2, (0, 3), (2, 4))
    reverse = exact_paired_swap_contrast(risk, 5, 0.2, (2, 4), (0, 3))
    assert forward == pytest.approx(-reverse)
    assert exact_paired_swap_contrast(risk, 5, 0.0, (0, 3), (2, 4)) == 0.0
    with pytest.raises(ValueError, match="equal-cardinality"):
        deterministic_swap_path((0,), (1, 2))


def test_master_field_contribution_is_whole_path_and_respects_context_suppression():
    # Bit 5 models fixed P_out. It is present in the master draw but must never
    # reach the authoritative risk callback because it is outside the exchange.
    seen = []

    def evaluate(faults):
        seen.append(faults)
        assert 5 not in faults
        return 2.0 * (0 in faults) + 3.0 * (2 in faults) + 9.0 * (
            (1 in faults) and (3 in faults)
        )

    master = frozenset({1, 3, 5})
    result = paired_swap_field_contrast(
        evaluate, master, (0, 1), (2, 3), probability=0.1,
        fixed_protected=(5,),
    )
    assert len(result["steps"]) == 2
    assert result["authoritative_evaluations"] == 4
    assert all(len(row["current_mask"]) == 2 for row in result["steps"])
    # One summed path value is the replication unit, not its two steps.
    assert result["target_minus_baseline"] == pytest.approx(sum(
        row["target_minus_baseline_contribution"] for row in result["steps"]
    ))


def test_response_table_swap_identity_matches_direct_mask_integration():
    probability = 0.23
    table = {
        state: (
            1.5 * (0 in state) - 0.4 * (1 in state)
            + 5.0 * all(index in state for index in (0, 2, 3))
        ) for state in subcube_states(4)
    }
    result = swap_contrast_from_response_table(
        table, 4, probability, (0, 1), (2, 3)
    )
    direct = (
        integrate_response_table(table, 4, probability, protected=(2, 3))
        - integrate_response_table(table, 4, probability, protected=(0, 1))
    )
    assert result["target_minus_baseline"] == pytest.approx(direct, abs=1e-12)
    assert result["absolute_equivalence_error"] < 1e-12


def test_field_estimator_average_matches_exact_contrast_over_complete_master_space():
    probability = 0.5
    risk = lambda z: (
        2.0 * (0 in z) - 0.7 * (2 in z)
        + 4.0 * all(index in z for index in (0, 1, 3))
    )
    observations = []
    for enabled in itertools.product((0, 1), repeat=5):
        master = frozenset(index for index, present in enumerate(enabled) if present)
        result = paired_swap_field_contrast(
            risk, master, (0, 1), (2, 3), probability=probability
        )
        observations.append(result["target_minus_baseline"])
    expected = exact_paired_swap_contrast(risk, 5, probability, (0, 1), (2, 3))
    assert sum(observations) / len(observations) == pytest.approx(expected, abs=1e-12)
