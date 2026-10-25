from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from netsurf.adapters.registry import create_adapter
from netsurf.incremental import DenseReluDenseIncrementalEvaluator
from netsurf.quantization import BitCodec, FixedPointFormat
from netsurf.search import (
    PriorComponent,
    ProposalPrior,
    RankingPrior,
    conditional_search,
    exhaustive_optimum,
)
from netsurf.study_config import load_study


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def search_fixture():
    configuration = load_study(ROOT / "config/experiments/numpy_qpolar_cancellation.toml")
    codec = BitCodec(FixedPointFormat.parse(configuration.benchmark.quantization))
    adapter = create_adapter(configuration, codec); adapter.load()
    inputs, labels, _ = adapter.sample_partition("evaluation", 4)
    # Cross both layers while keeping exact enumeration deliberately tiny.
    identities = tuple(adapter.bit_identities())
    universe = (identities[0], identities[1], identities[5], identities[-1], identities[-2], identities[-6])
    scores = {bit: float(bit.bit_index) for bit in universe}
    prior = RankingPrior.from_scores("msb_test", scores, "encoded-bit-index/v1")
    proposal = ProposalPrior(universe, (PriorComponent(prior, 1.0),), uniform_exploration=0.2)
    evaluator = DenseReluDenseIncrementalEvaluator(adapter, codec)
    return evaluator, inputs, labels, universe, proposal


def test_proposal_prior_has_full_support_and_seeded_sampling(search_fixture):
    _, _, _, universe, proposal = search_fixture
    probabilities = proposal.probabilities(universe)
    assert np.all(probabilities > 0)
    assert probabilities.sum() == pytest.approx(1.0)
    first = proposal.sample(universe, generator=np.random.default_rng(7))[0]
    replay = proposal.sample(universe, generator=np.random.default_rng(7))[0]
    assert first == replay


@pytest.mark.parametrize("strategy", ["uniform_swap", "static_guided_swap", "conditional_greedy", "simulated_annealing"])
def test_search_strategies_preserve_budget_and_are_reproducible(search_fixture, strategy):
    evaluator, inputs, labels, universe, proposal = search_fixture
    kwargs = dict(
        strategy=strategy, proposal=proposal, probability=0.1, iterations=3,
        background_contexts=3, seed=301, final_contexts=4, final_seed=901,
        greedy_add_candidates=2, greedy_remove_candidates=2,
    )
    first = conditional_search(evaluator, inputs, labels, universe, universe[:2], **kwargs)
    replay = conditional_search(evaluator, inputs, labels, universe, universe[:2], **kwargs)
    assert len(first.final_mask) == len(first.best_mask) == 2
    assert first.final_mask == replay.final_mask
    assert first.best_mask == replay.best_mask
    def scientific_steps(result):
        rows = []
        for step in result.steps:
            row = step.to_dict(); row.pop("wall_seconds")
            rows.append(row)
        return rows
    assert scientific_steps(first) == scientific_steps(replay)
    assert first.proposals_evaluated >= len(first.steps)


def test_static_baseline_performs_no_search_proposals(search_fixture):
    evaluator, inputs, labels, universe, proposal = search_fixture
    result = conditional_search(
        evaluator, inputs, labels, universe, universe[:2], strategy="static",
        proposal=proposal, probability=0.1, iterations=100, background_contexts=3,
        seed=1, final_contexts=3, final_seed=2,
    )
    assert not result.steps
    assert result.final_mask == result.initial_mask == result.best_mask


def test_exact_small_optimum_enumerates_every_mask_and_fault_state(search_fixture):
    evaluator, inputs, labels, universe, _ = search_fixture
    mask, risk, rows = exhaustive_optimum(evaluator, inputs, labels, universe, budget=2, probability=0.1)
    assert len(rows) == 15
    assert len(mask) == 2
    assert risk == min(row["risk"] for row in rows)


def test_polar_guidance_is_proposal_only_and_exactly_verified(search_fixture):
    evaluator, inputs, labels, universe, proposal = search_fixture
    interactions = {(universe[0], bit): index * 0.01 for index, bit in enumerate(universe[2:], 1)}
    result = conditional_search(
        evaluator, inputs, labels, universe, universe[:2], strategy="polar_guided_swap",
        proposal=proposal, probability=0.1, iterations=2, background_contexts=3,
        seed=22, final_contexts=3, final_seed=23, polar_interactions=interactions,
    )
    assert result.proposals_evaluated == 2
    assert result.exact_evaluations == 18  # 3 matched states × 3 contexts × 2 swaps
