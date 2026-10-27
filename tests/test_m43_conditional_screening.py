from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from netsurf.adapters.registry import create_adapter
from netsurf.conditional_screening import enumerate_exact_swap_decomposition, screen_conditional_candidates
from netsurf.identities import BitIdentity
from netsurf.incremental import DenseReluDenseIncrementalEvaluator, FaultState, evaluate_conditional_swap, metrics_from_logits
from netsurf.quantization import BitCodec, FixedPointFormat
from netsurf.study_config import load_study


ROOT = Path(__file__).resolve().parents[1]


def test_exact_conditional_decomposition_with_unequal_probabilities_and_higher_interactions():
    bits = tuple(BitIdentity("p", index, index % 2) for index in range(5))
    protected = {bits[0], bits[4]}; removed, added = bits[0], bits[1]
    probabilities = {bit: value for bit, value in zip(bits, (0.2, 0.35, 0.1, 0.6, 0.05))}
    def loss(state):
        indexes = {bit.flat_index for bit in state}
        return (sum((index + 1) * 0.3 for index in indexes)
                + (1.7 if {0, 1} <= indexes else 0)
                - (0.8 if {0, 2, 3} <= indexes else 0))
    result = enumerate_exact_swap_decomposition(bits, protected, removed, added, probabilities, loss)
    assert result.corrected_swap_difference == pytest.approx(result.direct_swap_difference, abs=1e-14)
    assert result.pair_interaction != 0
    assert result.marginal_swap_difference != pytest.approx(result.direct_swap_difference)


def test_exhaustive_model_decomposition_matches_direct_risk_for_sign_cross_layer_and_same_word_bits():
    config = load_study(ROOT / "config/experiments/numpy_qpolar_cancellation.toml")
    codec = BitCodec(FixedPointFormat.parse(config.benchmark.quantization))
    adapter = create_adapter(config, codec); adapter.load()
    inputs, labels, _ = adapter.sample_partition("evaluation", 3)
    evaluator = DenseReluDenseIncrementalEvaluator(adapter, codec)
    all_bits = tuple(adapter.bit_identities())
    selected = (
        next(bit for bit in all_bits if bit.parameter == "dense0/kernel" and bit.bit_index == 5),
        next(bit for bit in all_bits if bit.parameter == "dense1/kernel" and bit.bit_index == 5),
        next(bit for bit in all_bits if bit.parameter == "dense0/kernel" and bit.flat_index == 0 and bit.bit_index == 0),
        next(bit for bit in all_bits if bit.parameter == "dense0/kernel" and bit.flat_index == 0 and bit.bit_index == 1),
    )
    probabilities = {bit: p for bit, p in zip(selected, (0.2, 0.3, 0.4, 0.1))}
    protected = frozenset((selected[0],))
    def loss(state):
        return metrics_from_logits(evaluator.full_forward(inputs, FaultState(state)).logits, labels).categorical_crossentropy
    result = enumerate_exact_swap_decomposition(selected, protected, selected[0], selected[1], probabilities, loss)
    assert result.corrected_swap_difference == pytest.approx(result.direct_swap_difference, abs=1e-12)


def test_topology_screening_reuses_contexts_and_matches_direct_single_bit_effects():
    config = load_study(ROOT / "config/experiments/numpy_qpolar_cancellation.toml")
    codec = BitCodec(FixedPointFormat.parse(config.benchmark.quantization))
    adapter = create_adapter(config, codec); adapter.load()
    inputs, labels, _ = adapter.sample_partition("evaluation", 3)
    evaluator = DenseReluDenseIncrementalEvaluator(adapter, codec)
    bits = tuple(adapter.bit_identities())
    protected = frozenset(bits[:3]); removals = bits[:2]; additions = bits[3:6]
    fields = (FaultState.from_bits(), FaultState.from_bits((bits[7],)), FaultState.from_bits((bits[8], bits[9])))
    result = screen_conditional_candidates(evaluator, inputs, labels, fields, protected, removals, additions, probabilities=0.2)
    assert result.paired_weighted_contributions.shape == (3, 5)
    assert result.background_cache_hits > 0
    for column, bit in enumerate(result.candidate_order):
        values = []
        for field in fields:
            state = FaultState(field.bits - protected)
            if bit in additions: state = state.without_bit(bit)
            base = metrics_from_logits(evaluator.full_forward(inputs, state).logits, labels).categorical_crossentropy
            faulted = metrics_from_logits(evaluator.full_forward(inputs, state.with_bit(bit)).logits, labels).categorical_crossentropy
            values.append(0.2 * (faulted - base))
        assert result.paired_weighted_contributions[:, column] == pytest.approx(values, abs=1e-12)


def test_finite_sample_direct_swap_matches_existing_three_state_evaluator():
    config = load_study(ROOT / "config/experiments/numpy_qpolar_cancellation.toml")
    codec = BitCodec(FixedPointFormat.parse(config.benchmark.quantization))
    adapter = create_adapter(config, codec); adapter.load()
    inputs, labels, _ = adapter.sample_partition("evaluation", 3)
    evaluator = DenseReluDenseIncrementalEvaluator(adapter, codec)
    removed, added = tuple(adapter.bit_identities())[:2]
    backgrounds = (FaultState.from_bits(), FaultState.from_bits(tuple(adapter.bit_identities())[5:8]))
    result = evaluate_conditional_swap(evaluator, inputs, labels, backgrounds, removed, added,
                                       probability_removed=.2, probability_added=.35)
    expected=[]
    for state in backgrounds:
        base=metrics_from_logits(evaluator.full_forward(inputs,state).logits,labels).categorical_crossentropy
        left=metrics_from_logits(evaluator.full_forward(inputs,state.with_bit(removed)).logits,labels).categorical_crossentropy
        right=metrics_from_logits(evaluator.full_forward(inputs,state.with_bit(added)).logits,labels).categorical_crossentropy
        expected.append(.2*(left-base)-.35*(right-base))
    assert result.risk_difference["mean"] == pytest.approx(np.mean(expected), abs=1e-12)
