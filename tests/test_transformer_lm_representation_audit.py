"""Deterministic offline checks; no model loading or fault inference."""

import pytest

from netsurf.transformer_lm_representation_audit import (
    bit_parts, risk_decomposition, structural_mask, _mask_digest,
)


@pytest.mark.parametrize("width,positions", [(32, (31, 30, 22)), (16, (15, 14, 6))])
def test_encoded_classes_and_data_free_exponent_first(width, positions):
    keys = [f"layer.weight[{word}]#b{position}" for word in (10, 2)
            for position in positions]
    assert [bit_parts(key, width)[3] for key in keys[:3]] == [
        "sign", "exponent", "significand"]
    assert structural_mask(keys, width, 2) == [
        f"layer.weight[2]#b{positions[1]}",
        f"layer.weight[10]#b{positions[1]}"]
    assert structural_mask(list(reversed(keys)), width, 2) == structural_mask(keys, width, 2)
    assert set(structural_mask(keys, width, 2)).isdisjoint(
        {key for key in keys if key.endswith(f"#b{positions[0]}")})
    with pytest.raises(ValueError): structural_mask(keys + keys[:1], width, 2)
    with pytest.raises(ValueError): bit_parts(keys[0], 8)


def test_mask_digest_uses_numeric_word_identity_not_display_order():
    assert _mask_digest(["x[10]#b14", "x[2]#b14"], 16) == _mask_digest(
        ["x[2]#b14", "x[10]#b14"], 16)


def test_failure_penalty_finite_cap_and_partial_nonfinite_are_separate():
    risk = {"total_units": 4, "failed_units": 1,
            "logit_record_kind": "target_logit_and_logsumexp",
            "logits": [None, [0.0, 5.0], [0.0, 200.0], [1.0, 3.0]],
            "failure_kinds": ["nonfinite_logits", None, None, None],
            "mean_nll": (30 + 5 + 30 + 2) / 4}
    result = risk_decomposition(risk, 30.0)
    assert result["state"] == "partially_nonfinite_or_failed"
    assert result["failed_tokens"] == result["nonfinite_tokens"] == 1
    assert result["finite_capped_tokens"] == 1
    assert result["failed_penalty_mean_nll"] == result["finite_cap_mean_nll"] == 7.5
    assert result["ordinary_finite_mean_nll"] == 1.75
    broken = {**risk, "mean_nll": 1.0}
    with pytest.raises(ValueError, match="disagrees"):
        risk_decomposition(broken, 30.0)


def test_full_nonfinite_and_finite_catastrophic_are_distinct():
    base = {"total_units": 2, "logit_record_kind": "target_logit_and_logsumexp"}
    all_failed = {**base, "failed_units": 2, "logits": [None, None],
                  "failure_kinds": ["nonfinite_logits", "inference_failure"], "mean_nll": 30.0}
    finite_cap = {**base, "failed_units": 0, "logits": [[0.0, 31.0], [0.0, 1.0]],
                  "failure_kinds": [None, None], "mean_nll": 15.5}
    assert risk_decomposition(all_failed, 30.0)["state"] == "fully_nonfinite_or_failed"
    assert risk_decomposition(finite_cap, 30.0)["state"] == "finite_catastrophic"
