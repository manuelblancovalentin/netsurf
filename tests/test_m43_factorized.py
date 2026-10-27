from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from netsurf.adapters.registry import create_adapter
from netsurf.curvature import DenseReluDenseCurvature
from netsurf.factorized import FactorizedInteractionAggregate
from netsurf.incremental import DenseReluDenseIncrementalEvaluator, FaultState
from netsurf.quantization import BitCodec, FixedPointFormat
from netsurf.study_config import load_study
from netsurf.topology import DenseReluDenseTopologyEvaluator


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture():
    config = load_study(ROOT / "config/experiments/numpy_qpolar_cancellation.toml")
    codec = BitCodec(FixedPointFormat.parse(config.benchmark.quantization))
    adapter = create_adapter(config, codec); adapter.load()
    inputs, _, _ = adapter.sample_partition("evaluation", 4)
    evaluator = DenseReluDenseIncrementalEvaluator(adapter, codec)
    topology = DenseReluDenseTopologyEvaluator(evaluator, inputs, FaultState.from_bits())
    return adapter, DenseReluDenseCurvature(topology)


@pytest.mark.parametrize("stratified", [False, True])
def test_factorized_aggregate_matches_explicit_full_population_sum(fixture, stratified):
    adapter, curvature = fixture; universe = tuple(adapter.bit_identities())
    protected = frozenset(universe[::7])
    probabilities = {bit: 0.01 + 0.001 * (index % 5) for index, bit in enumerate(universe)}
    aggregate = FactorizedInteractionAggregate(curvature, universe, protected, probabilities=probabilities,
                                               stratified=stratified, strata=2, minimum_samples=2)
    for bit in universe[::11]:
        assert aggregate.query(bit) == pytest.approx(aggregate.explicit(bit), abs=1e-12)
    assert aggregate.population_coverage == 1.0


def test_rank_one_updates_match_full_reconstruction_across_same_and_cross_layer_swaps(fixture):
    adapter, curvature = fixture; universe = tuple(adapter.bit_identities())
    hidden = [bit for bit in universe if bit.parameter.startswith("dense0")]
    output = [bit for bit in universe if bit.parameter.startswith("dense1")]
    protected = frozenset((hidden[0], hidden[1], output[0], output[1]))
    probabilities = {bit: 0.02 + 0.001 * (index % 3) for index, bit in enumerate(universe)}
    aggregate = FactorizedInteractionAggregate(curvature, universe, protected, probabilities=probabilities)
    swaps = ((hidden[0], hidden[3]), (output[0], output[3]), (hidden[1], output[4]))
    current = set(protected)
    for removed, added in swaps:
        aggregate.swap(removed, added); current.remove(removed); current.add(added)
        rebuilt = FactorizedInteractionAggregate(curvature, universe, current, probabilities=probabilities)
        for bit in universe[::13]:
            assert aggregate.query(bit) == pytest.approx(rebuilt.query(bit), abs=1e-12)
        assert aggregate.reconstruction_error() < 1e-12


def test_self_exclusion_is_by_bit_identity_not_parameter_word(fixture):
    adapter, curvature = fixture; universe = tuple(adapter.bit_identities())
    same_word = [bit for bit in universe if bit.parameter == "dense0/kernel" and bit.flat_index == 0]
    candidate = same_word[0]
    aggregate = FactorizedInteractionAggregate(curvature, universe, (), probabilities=0.1)
    assert aggregate.query(candidate) == pytest.approx(aggregate.explicit(candidate), abs=1e-12)
    contributions = [0.1 * curvature.kron(candidate, other) for other in same_word if other != candidate]
    assert any(abs(value) > 0 for value in contributions)


def test_restricted_pool_coverage_is_explicit_and_changes_score(fixture):
    adapter, curvature = fixture; universe = tuple(adapter.bit_identities())
    restricted = universe[:20]
    full = FactorizedInteractionAggregate(curvature, universe, (), probabilities=.1)
    partial = FactorizedInteractionAggregate(curvature, universe, (), probabilities=.1, population=restricted)
    candidate = restricted[0]
    assert partial.population_coverage == pytest.approx(20 / len(universe))
    assert full.query(candidate) != pytest.approx(partial.query(candidate))
