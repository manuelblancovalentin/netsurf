import itertools

import pytest

from netsurf.restricted_subcube import (
    analyze_restricted_response_tables, conditional_marginal_from_table,
    fixed_budget_masks, integrate_response_table, response_table_diagnostics,
    select_restricted_subcube_mask, subcube_states,
)


def test_restricted_subcube_selects_only_on_calibration_and_reports_held_out_separately():
    def calibration(faults):
        return 10.0 * (0 in faults) + 1.0 * (1 in faults) + 0.1 * (2 in faults)

    def held_out(faults):
        return 1.0 * (0 in faults) + 9.0 * (1 in faults) + 0.1 * (2 in faults)

    result = select_restricted_subcube_mask(
        [calibration], [held_out], candidate_count=3, budget=1, probability=0.2
    )
    assert result.selected_mask == (0,)
    assert result.held_out_risks[(1,)] < result.held_out_risks[result.selected_mask]


def test_restricted_subcube_is_exact_and_stably_tie_broken():
    result = select_restricted_subcube_mask(
        [lambda faults: float(len(faults))], [lambda faults: float(len(faults))],
        candidate_count=3, budget=1, probability=0.25,
    )
    assert result.selected_mask == (0,)
    assert result.calibration_risks[(0,)] == 0.5
    assert result.held_out_risks[(0,)] == 0.5


def _table(candidate_count, risk):
    return {state: risk(set(state)) for state in subcube_states(candidate_count)}


def test_response_table_exactly_integrates_additive_pairwise_and_third_order_terms():
    probability = 0.2
    additive = _table(3, lambda z: 2.0 * (0 in z) + 3.0 * (1 in z) + 5.0 * (2 in z))
    pairwise = _table(3, lambda z: 7.0 * ((0 in z) and (1 in z)))
    third = _table(3, lambda z: 11.0 * all(index in z for index in range(3)))
    assert integrate_response_table(additive, 3, probability) == pytest.approx(10.0 * probability)
    assert integrate_response_table(pairwise, 3, probability) == pytest.approx(7.0 * probability**2)
    assert integrate_response_table(third, 3, probability) == pytest.approx(11.0 * probability**3)
    assert integrate_response_table(third, 3, probability, protected=(0,)) == 0.0


def test_response_table_zero_ber_mask_enumeration_ties_and_validation():
    table = _table(3, lambda z: float(len(z)))
    assert integrate_response_table(table, 3, 0.0) == table[()]
    assert fixed_budget_masks(3, 2) == ((0, 1), (0, 2), (1, 2))
    analysis = analyze_restricted_response_tables(
        [table], [table], table, candidate_count=3, budget=1, probability=0.25
    )
    assert analysis["selected_mask"] == (0,)
    with pytest.raises(ValueError, match="incomplete"):
        integrate_response_table({(): 0.0}, 3, 0.2)
    with pytest.raises(ValueError, match="unique"):
        conditional_marginal_from_table(table, 3, 0.2, 0, protected=(1, 1))


def test_probability_mass_normalizes_for_every_fixed_budget_mask():
    table = _table(4, lambda _z: 1.0)
    for budget in range(5):
        for mask in itertools.combinations(range(4), budget):
            assert integrate_response_table(table, 4, 0.37, protected=mask) == pytest.approx(1.0)


def test_table_oracle_agrees_with_independent_callable_enumeration_on_same_fields():
    tables = [
        _table(3, lambda z, offset=offset: offset + 2.0 * (0 in z) + 5.0 * ((1 in z) and (2 in z)))
        for offset in (0.25, 1.5)
    ]
    callables = [lambda faults, table=table: table[tuple(sorted(faults))] for table in tables]
    direct = select_restricted_subcube_mask(
        callables, callables, candidate_count=3, budget=1, probability=0.2
    )
    table_result = analyze_restricted_response_tables(
        tables, tables, tables[0], candidate_count=3, budget=1, probability=0.2
    )
    assert table_result["selected_mask"] == direct.selected_mask
    for mask, expected in direct.calibration_risks.items():
        observed = sum(table_result["calibration_field_risks"][mask]) / len(tables)
        assert observed == pytest.approx(expected)


@pytest.mark.parametrize(
    "risk",
    (
        lambda z: 2.0 * (0 in z) - 0.5 * (1 in z),
        lambda z: 3.0 * ((0 in z) and (1 in z)) - 2.0 * ((1 in z) and (2 in z)),
        lambda z: 7.0 * all(index in z for index in (0, 1, 2)),
    ),
)
def test_response_reuse_matches_direct_equal_budget_risks_for_all_interaction_orders(risk):
    table = _table(3, risk)
    diagnostics = response_table_diagnostics(table, 3, 1, 0.2)
    for mask, observed in diagnostics["mask_risks"].items():
        expected = select_restricted_subcube_mask(
            [lambda faults, table=table: table[tuple(sorted(faults))]],
            [lambda faults, table=table: table[tuple(sorted(faults))]],
            candidate_count=3, budget=1, probability=0.2,
        ).calibration_risks[mask]
        assert observed == pytest.approx(expected)
    for (left, right), difference in diagnostics["pairwise_mask_differences"].items():
        assert difference == pytest.approx(
            diagnostics["mask_risks"][left] - diagnostics["mask_risks"][right]
        )
    for candidate in range(3):
        assert diagnostics["conditional_marginals"][candidate] == pytest.approx(
            conditional_marginal_from_table(table, 3, 0.2, candidate)
        )
