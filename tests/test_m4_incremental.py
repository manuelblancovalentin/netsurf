from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from netsurf.adapters.registry import create_adapter
from netsurf.incremental import (
    DenseReluDenseIncrementalEvaluator,
    FaultState,
    evaluate_conditional_swap,
    metrics_from_logits,
    sample_conditional_backgrounds,
)
from netsurf.quantization import BitCodec, FixedPointFormat
from netsurf.study_config import load_study


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/experiments/numpy_qpolar_cancellation.toml"


@pytest.fixture
def reference():
    configuration = load_study(CONFIG)
    codec = BitCodec(FixedPointFormat.parse(configuration.benchmark.quantization))
    adapter = create_adapter(configuration, codec)
    adapter.load()
    inputs, labels, _ = adapter.sample_partition("evaluation", 4)
    return adapter, codec, inputs, labels, DenseReluDenseIncrementalEvaluator(adapter, codec)


def _assert_exact(evaluator, inputs, state, cache_state=FaultState.from_bits()):
    cache = evaluator.cache(inputs, cache_state)
    incremental = evaluator.incremental_forward(cache, inputs, state)
    full = evaluator.full_forward(inputs, state)
    np.testing.assert_allclose(incremental.logits, full.logits, rtol=1e-12, atol=1e-12)
    return incremental


def test_incremental_matches_full_across_required_fault_classes(reference):
    adapter, codec, inputs, labels, evaluator = reference
    identities = tuple(adapter.bit_identities())
    dense0 = [bit for bit in identities if bit.parameter == "dense0/kernel"]
    dense0_bias = [bit for bit in identities if bit.parameter == "dense0/bias"]
    dense1 = [bit for bit in identities if bit.parameter == "dense1/kernel"]
    dense1_bias = [bit for bit in identities if bit.parameter == "dense1/bias"]
    cases = (
        FaultState.from_bits(),
        FaultState.from_bits((dense1[0],)),
        FaultState.from_bits((next(bit for bit in dense1 if bit.bit_index == 5),)),
        FaultState.from_bits((dense0[0], replace(dense0[0], bit_index=1))),
        FaultState.from_bits((dense0[0], dense0_bias[0], dense0[12])),
        FaultState.from_bits((dense1[0], dense1_bias[0])),
        FaultState.from_bits((dense0[1], dense1[7], dense1_bias[2])),
    )
    for state in cases:
        result = _assert_exact(evaluator, inputs, state)
        full_metrics = metrics_from_logits(evaluator.full_forward(inputs, state).logits, labels)
        incremental_metrics = metrics_from_logits(result.logits, labels)
        assert incremental_metrics.accuracy == full_metrics.accuracy
        assert incremental_metrics.categorical_crossentropy == pytest.approx(full_metrics.categorical_crossentropy, abs=1e-14)


def test_fault_addition_removal_and_current_encoded_word_semantics(reference):
    adapter, codec, inputs, _, evaluator = reference
    bit0 = next(bit for bit in adapter.bit_identities() if bit.parameter == "dense0/kernel" and bit.bit_index == 0)
    bit1 = replace(bit0, bit_index=1)
    background = FaultState.from_bits((bit0, bit1))
    target = background.without_bit(bit0)
    _assert_exact(evaluator, inputs, target, cache_state=background)
    original = evaluator.base_encoded[bit0.parameter][bit0.flat_index]
    expected = original ^ np.uint64(1 << bit1.bit_index)
    actual = codec.encode(evaluator.values_for_state(target)[bit0.parameter]).reshape(-1)[bit0.flat_index]
    assert actual == expected


def test_relu_gate_changes_are_recorded_without_approximating(reference):
    adapter, _, inputs, _, evaluator = reference
    gate_changing = None
    for bit in adapter.bit_identities():
        if bit.parameter.startswith("dense0/"):
            result = evaluator.incremental_forward(evaluator.cache(inputs), inputs, FaultState.from_bits((bit,)))
            if result.diagnostics.relu_gate_changes:
                gate_changing = bit
                np.testing.assert_allclose(result.logits, evaluator.full_forward(inputs, FaultState.from_bits((bit,))).logits, rtol=1e-12, atol=1e-12)
                break
    assert gate_changing is not None


def test_final_layer_reuses_hidden_activation_and_cache_fallback_is_exact(reference):
    adapter, codec, inputs, _, evaluator = reference
    final_bit = next(bit for bit in adapter.bit_identities() if bit.parameter == "dense1/kernel")
    result = _assert_exact(evaluator, inputs, FaultState.from_bits((final_bit,)))
    assert result.diagnostics.earliest_affected_layer == 1
    assert result.diagnostics.reused_activation_layers == 1
    fallback = DenseReluDenseIncrementalEvaluator(adapter, codec, cache_memory_limit_bytes=1)
    assert fallback.cache(inputs) is None
    result = fallback.incremental_forward(None, inputs, FaultState.from_bits((final_bit,)))
    assert result.diagnostics.fallback
    np.testing.assert_allclose(result.logits, fallback.full_forward(inputs, FaultState.from_bits((final_bit,))).logits)


def test_three_state_swap_matches_full_reference_and_exact_identity(reference):
    adapter, _, inputs, labels, evaluator = reference
    removed = next(bit for bit in adapter.bit_identities() if bit.parameter == "dense0/kernel" and bit.bit_index == 5)
    added = next(bit for bit in adapter.bit_identities() if bit.parameter == "dense1/kernel" and bit.bit_index == 4)
    universe = tuple(adapter.bit_identities())[:24] + (removed, added)
    backgrounds = sample_conditional_backgrounds(
        universe, {removed}, {removed, added}, probability=0.15, count=8, seed=73,
    )
    incremental = evaluate_conditional_swap(
        evaluator, inputs, labels, backgrounds, removed, added,
        probability_removed=0.2, probability_added=0.05, mode="incremental",
    )
    full = evaluate_conditional_swap(
        evaluator, inputs, labels, backgrounds, removed, added,
        probability_removed=0.2, probability_added=0.05, mode="full",
    )
    assert incremental.risk_difference["mean"] == pytest.approx(full.risk_difference["mean"], abs=1e-14)
    assert incremental.removed_fault_effect["mean"] == pytest.approx(full.removed_fault_effect["mean"], abs=1e-14)
    assert incremental.added_fault_effect["mean"] == pytest.approx(full.added_fault_effect["mean"], abs=1e-14)
    assert incremental.incremental_evaluations == 16
    assert full.full_evaluations == 16


def test_conditional_background_rejects_swapped_bits(reference):
    adapter, _, inputs, labels, evaluator = reference
    removed, added = adapter.bit_identities()[:2]
    with pytest.raises(ValueError, match="exclude"):
        evaluate_conditional_swap(
            evaluator, inputs, labels, (FaultState.from_bits((removed,)),), removed, added,
            probability_removed=0.1, probability_added=0.1,
        )


def test_three_state_identity_matches_exhaustive_fault_distribution(reference):
    adapter, _, inputs, labels, evaluator = reference
    universe = tuple(adapter.bit_identities())[:4]
    removed, added = universe[0], universe[3]
    probability = 0.2
    others = tuple(bit for bit in universe if bit not in {removed, added})
    backgrounds, weights = [], []
    for count in range(len(others) + 1):
        for selected in combinations(others, count):
            backgrounds.append(FaultState.from_bits(selected))
            weights.append(probability ** count * (1 - probability) ** (len(others) - count))
    conditional = 0.0
    for background, weight in zip(backgrounds, weights):
        estimate = evaluate_conditional_swap(
            evaluator, inputs, labels, (background,), removed, added,
            probability_removed=probability, probability_added=probability,
        )
        conditional += weight * estimate.risk_difference["mean"]

    def exact_risk(protected):
        risk = 0.0
        for count in range(len(universe) + 1):
            for selected in combinations(universe, count):
                weight = probability ** count * (1 - probability) ** (len(universe) - count)
                active = FaultState.from_bits(set(selected) - set(protected))
                loss = metrics_from_logits(evaluator.full_forward(inputs, active).logits, labels).categorical_crossentropy
                risk += weight * loss
        return risk

    current, swapped = {removed}, {added}
    assert conditional == pytest.approx(exact_risk(swapped) - exact_risk(current), abs=1e-14)
