"""Small deterministic pseudo-Boolean search tests; no model assets used."""

import pytest

from netsurf.equal_compute_search import evaluate_held_out, search_allocation


def test_equal_queries_budget_and_deterministic_reproduction():
    universe = tuple(range(24))
    initial = tuple(range(8))
    calls = []

    def risk(mask):
        calls.append(mask)
        # Signed additive effects plus a genuine pair interaction.
        return 1.0 + sum((0.02 if bit < 8 else -0.01) for bit in mask) + \
            (0.25 if {8, 9} <= mask else 0.0)

    kwargs = dict(query_budget=31, seed=701, selection_identity="selection-fields-v1",
                  calls_per_query=12, clip_inferences_per_query=12 * 16,
                  evaluate=risk)
    random = search_allocation(universe, initial, method="random_search", **kwargs)
    greedy = search_allocation(universe, initial, method="greedy_swap", **kwargs)
    assert len(calls) == 62
    assert random.model_requests == greedy.model_requests == 31 * 12
    assert random.clip_inferences == greedy.clip_inferences == 31 * 12 * 16
    assert all(len(mask) == 8 and mask <= set(universe) for mask in calls)
    assert random.observations[0].candidate == greedy.observations[0].candidate == initial
    assert random.observations[0].measured_risk == greedy.observations[0].measured_risk
    assert random.observations[0].accepted and greedy.observations[0].accepted
    assert min(x.measured_risk for x in random.observations) <= random.observations[0].measured_risk
    assert min(x.measured_risk for x in greedy.observations) <= greedy.observations[0].measured_risk
    repeated = search_allocation(universe, initial, method="greedy_swap", **kwargs)
    assert repeated.observations == greedy.observations
    assert repeated.selected_mask == greedy.selected_mask


def test_held_out_gate_and_no_final_feedback():
    universe = tuple(range(12))
    initial = tuple(range(4))
    selection_queries = []
    final_queries = []
    def selection(mask):
        selection_queries.append(mask)
        return 1.0 - 0.1 * sum(bit >= 4 for bit in mask)
    def final(mask):
        final_queries.append(mask)
        return 1.0 + 0.05 * sum(bit >= 4 for bit in mask)
    common = dict(query_budget=9, seed=23, selection_identity="selection-a",
                  calls_per_query=2, clip_inferences_per_query=16,
                  evaluate=selection)
    random = search_allocation(universe, initial, method="random_search", **common)
    greedy = search_allocation(universe, initial, method="greedy_swap", **common)
    assert len(selection_queries) == 18 and not final_queries
    outcome = evaluate_held_out((random, greedy), {"initial": initial},
        final_identity="final-b", evaluate=final)
    assert len(outcome) == len(final_queries) == 3
    assert outcome["initial"] == 1.0
    assert all(item.query_budget == 9 for item in (random, greedy))
    with pytest.raises(ValueError, match="independent"):
        evaluate_held_out((random, greedy), {}, final_identity="selection-a", evaluate=final)
    with pytest.raises(ValueError, match="share"):
        evaluate_held_out((random, search_allocation(universe, (4, 5, 6, 7),
            method="greedy_swap", **common)), {}, final_identity="final-b", evaluate=final)
