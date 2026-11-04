"""Small deterministic contracts for a future full-population FKeras gate."""

import numpy as np
import pytest

from netsurf.w8a8_fkeras_preflight import (
    bounded_eigenpair_probe, project_full_population, tier_first_bit_order,
    tier_first_score_keys, symmetry_probe,
    validate_full_mapping,
)


def test_full_mapping_rejects_subset_and_shape_changes():
    entries = [{"module": "a", "shape": [2, 3]},
               {"module": "b", "shape": [1, 4]}]
    shapes = {"a": (2, 3), "b": (1, 4)}
    assert validate_full_mapping(entries, shapes, 80) == (("a", 0, 6), ("b", 6, 4))
    with pytest.raises(ValueError, match="omits"):
        validate_full_mapping(entries, {"a": (2, 3)}, 80)
    with pytest.raises(ValueError, match="shape"):
        validate_full_mapping(entries, {"a": (3, 2), "b": (1, 4)}, 80)
    with pytest.raises(ValueError, match="bit population"):
        validate_full_mapping(entries, shapes, 48)


def test_projection_accounts_for_all_codes_and_full_lanczos_basis():
    result = project_full_population(10_000, eigenpairs=8, lanczos_ncv=20)
    assert result.eligible_bits == 80_000
    assert result.float64_lanczos_basis_bytes == 1_600_000
    assert result.float64_eigenvectors_bytes == 640_000
    assert result.four_packed_masks_bytes == 40_000
    assert result.eigensolver_numeric_peak_bytes == sum((result.float64_lanczos_basis_bytes,
                                                         result.float64_eigenvectors_bytes,
                                                         result.float32_decoded_weights_bytes,
                                                         result.float32_hvp_vectors_bytes))
    assert result.numeric_peak_projection_bytes == max(result.eigensolver_numeric_peak_bytes,
                                                        result.score_mask_numeric_peak_bytes)


def test_eigenpair_preflight_reports_residual_and_work_bound():
    diagonal = np.array([30., 2., -1., .5], dtype=np.float32)
    result = bounded_eigenpair_probe(lambda vector: diagonal * vector, 4,
                                     seed=7, max_products=2, tolerance=1e-8)
    assert result["hvp_calls"] == 2
    assert result["normalized_residual"] > 0
    assert result["converged"] is False
    with pytest.raises(ValueError, match="incomplete"):
        bounded_eigenpair_probe(lambda vector: vector[:2], 4)
    converged = bounded_eigenpair_probe(lambda vector: diagonal * vector, 4,
                                        seed=7, max_products=8, tolerance=1e-3)
    assert converged["converged"] is True
    assert converged["normalized_residual"] <= 1e-3


def test_full_vector_symmetry_probe_detects_nonsymmetric_operator():
    diagonal = np.array([1., 3., 5.], dtype=np.float32)
    good = symmetry_probe(lambda vector: diagonal * vector, 3)
    assert good["relative_symmetry_error"] < 1e-6
    bad_matrix = np.array([[0., 10., 0.], [0., 0., 0.], [0., 0., 0.]], dtype=np.float32)
    bad = symmetry_probe(lambda vector: bad_matrix @ vector, 3)
    assert bad["relative_symmetry_error"] > 0.1


def test_bit_tier_precedes_any_weight_score():
    order = tier_first_bit_order(np.array([0., 100., 1.]))
    np.testing.assert_array_equal(order[:3], [15, 23, 7])
    assert all(int(x) % 8 == 7 for x in order[:3])
    assert all(int(x) % 8 == 6 for x in order[3:6])


def test_streaming_keys_preserve_tiers_and_exact_budget(tmp_path):
    from netsurf.w8a8_rank import write_top_masks

    scores = np.array([0., 100., 1.], dtype=np.float32)
    keys = tier_first_score_keys(scores)
    assert keys.shape == (24,) and keys.dtype == np.uint64
    assert keys[7] > keys[14] > keys[5]  # tier 7 outranks tier 6 regardless of score

    def chunks():
        yield 0, keys

    budgets = (1, 3, 4, 6)
    masks = write_top_masks(chunks, 24, budgets, tmp_path / "masks", "fkeras_fixture",
                            key_dtype=np.uint64)
    expected = tier_first_bit_order(scores)
    for budget in budgets:
        actual = [i for i in range(24) if masks[budget].contains(i)]
        assert sorted(actual) == sorted(map(int, expected[:budget]))
        assert len(actual) == budget
    with pytest.raises(ValueError, match="finite"):
        tier_first_score_keys(np.array([np.inf], dtype=np.float32))
