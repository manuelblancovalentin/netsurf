"""Frozen-pool and response-table checks; no KWS inference or asset mutation."""
import copy
import itertools
import json
import math

import pytest

from netsurf.equal_compute_m96 import SOURCE, SOURCE_SHA256, _validated_table, retrospective_search
from netsurf.equal_compute_search import evaluate_held_out, search_allocation
from netsurf.manifests import sha256_file


def _small_table():
    return {"budget": 2, "candidate_count": 6, "eligible_masks": [
        {"indices": list(pair), "field_risks": [float(sum(pair)) + i / 100 for i in range(32)]}
        for pair in itertools.combinations(range(6), 2)]}


def test_exact_table_requires_all_15_masks_and_finite_fields():
    table = _small_table()
    assert len(_validated_table(table)) == 15
    broken = copy.deepcopy(table)
    broken["eligible_masks"].pop()
    with pytest.raises(ValueError, match="15"):
        _validated_table(broken)
    broken = copy.deepcopy(table)
    broken["eligible_masks"][0]["field_risks"][0] = float("nan")
    with pytest.raises(ValueError, match="risks"):
        _validated_table(broken)
    broken = copy.deepcopy(table)
    broken["eligible_masks"][1]["indices"] = [0, 1]
    with pytest.raises(ValueError, match="duplicate"):
        _validated_table(broken)


def test_frozen_m96_pool_identity_and_retrospective_query_fairness():
    if not SOURCE.is_file():
        pytest.skip("historical M9.6 worktree not available")
    assert sha256_file(SOURCE) == SOURCE_SHA256
    source = json.loads(SOURCE.read_text())
    pools = source["pools"]
    assert len(pools) == 12
    ids = [row["identity"] for pool in pools for row in pool["candidates"]]
    assert len(ids) == len(set(ids)) == 72
    initial = [row["identity"] for pool in pools for row in pool["candidates"]
               if row["source_protected"]]
    assert len(initial) == 24
    result = retrospective_search()
    assert len(result["pools"]) == 12
    for pool in result["pools"]:
        runs = pool["searches"]
        assert len(runs) == 6 and len({x["selection_identity"] for x in runs}) == 1
        assert len({tuple(x["initial_mask"]) for x in runs}) == 1
        assert all(x["budget"] == 2 and x["query_budget"] == 10 for x in runs)
        assert all(x["model_requests"] == x["clip_inferences"] == 0 for x in runs)
        assert all(x["saved_table_lookups"] == 10 for x in runs)
        assert pool["selection_identity"] != pool["heldout_identity"]


def test_72_bit_synthetic_scaling_keeps_global_budget_and_holdout_independent():
    if not SOURCE.is_file():
        pytest.skip("historical M9.6 worktree not available")
    source = json.loads(SOURCE.read_text())
    universe = tuple(row["identity"] for pool in source["pools"]
                     for row in pool["candidates"])
    initial = tuple(row["identity"] for pool in source["pools"]
                    for row in pool["candidates"] if row["source_protected"])
    assert len(universe) == 72 and len(initial) == 24
    index = {bit: i for i, bit in enumerate(universe)}
    selection_calls, final_calls = [], []

    def pseudo_boolean(mask, *, shift):
        chosen = {index[bit] for bit in mask}
        additive = sum((0.008 + (i * 17 % 11) / 1000) for i in range(72) if i not in chosen)
        interactions = sum(0.02 for i in range(0, 70, 7) if i not in chosen and i + 1 not in chosen)
        return 0.2 + additive + interactions + shift

    def selection(mask):
        selection_calls.append(mask)
        return pseudo_boolean(mask, shift=0.0)

    def final(mask):
        final_calls.append(mask)
        return pseudo_boolean(mask, shift=0.003)

    searches = [search_allocation(universe, initial, method=method,
        query_budget=64, seed=seed, selection_identity="synthetic-selection-fields",
        calls_per_query=16, clip_inferences_per_query=16 * 64, evaluate=selection)
        for method in ("random_search", "greedy_swap") for seed in (901, 902, 903)]
    assert len(selection_calls) == 6 * 64 and not final_calls
    assert all(len(mask) == 24 for mask in selection_calls)
    assert all(x.model_requests == 64 * 16 and x.clip_inferences == 64 * 16 * 64
               for x in searches)
    risks = evaluate_held_out(searches, {"initial_s5": initial},
        final_identity="synthetic-final-fields", evaluate=final)
    assert len(risks) == len(final_calls) == 7
    assert math.isfinite(risks["initial_s5"])
