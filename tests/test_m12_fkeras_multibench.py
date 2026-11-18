"""Bounded published-form FKeras gates; never consume untouched final examples."""

from pathlib import Path
import hashlib

import numpy as np
import pytest

from netsurf.fkeras_baseline import fkeras_msb_tier_order, leading_hessian_eigenpairs
from netsurf.fkeras_estimator import ContinuousTFLiteHessian, fkeras_family
from netsurf.identities import BitIdentity
from netsurf.study_config import StudyConfigurationError, load_study
from netsurf.reporting import render_study_report
from netsurf.api import configuration_operation_plan


ROOT = Path(__file__).resolve().parents[1]
RESNET = ROOT / "config/examples/m12_resnet8_fkeras_smoke.toml"
TOYCAR = ROOT / "config/examples/m12_toycar_fkeras_smoke.toml"


@pytest.mark.parametrize("path,version", [
    (RESNET, "fkeras-paper-continuous-resnet8/v1"),
    (TOYCAR, "fkeras-paper-continuous-toycar/v1"),
])
def test_schema_36_accepts_only_published_form_for_new_models(path, version, tmp_path):
    config = load_study(path, check_inputs=False)
    assert config.schema_version == "3.6.0"
    assert next(row for row in config.estimators if row.name == "fkeras_paper").implementation_version == version
    assert config.benchmark.injectable_parameters == ("kernels",)
    text = path.read_text()
    modified = tmp_path / "wrong.toml"
    modified.write_text(text.replace("schema_version = \"3.6.0\"", "schema_version = \"3.5.0\""))
    with pytest.raises(StudyConfigurationError, match="FKeras requires"):
        load_study(modified, check_inputs=False)
    modified.write_text(text.replace("name = \"fkeras_paper\"", "name = \"fkeras_repository\""))
    with pytest.raises(StudyConfigurationError, match="FKeras requires"):
        load_study(modified, check_inputs=False)
    modified.write_text(text.replace(
        'priority = ["bit_significance_desc", "estimator_score_desc", "stable_identity"]',
        'priority = ["estimator_score_desc", "stable_identity"]',
    ))
    with pytest.raises(StudyConfigurationError, match="strict bit_significance"):
        load_study(modified, check_inputs=False)


def test_schema_35_kws_remains_accepted():
    path = ROOT / "config/examples/m97_kws_fkeras_protection_smoke.toml"
    config = load_study(path, check_inputs=False)
    assert config.schema_version == "3.5.0"
    assert {row.name for row in config.estimators if row.name.startswith("fkeras_")} == {
        "fkeras_paper", "fkeras_repository"
    }


def test_tier_first_mask_sets_are_exact_at_whole_tiers():
    bits = tuple(BitIdentity(f"kernel_{i}", 0, bit) for i in range(2) for bit in range(8))
    scores = {("kernel_0", 0): 1.0, ("kernel_1", 0): 2.0}
    order = fkeras_msb_tier_order(bits, scores)
    assert {bit.bit_index for bit in order[:2]} == {7}
    assert {bit.bit_index for bit in order[:4]} == {6, 7}
    assert order[0].parameter == "kernel_1"  # within-tier sensitivity matters
    assert all(bit.bit_index == 7 for bit in order[:2])  # no tier leakage


def _pinned():
    tf = pytest.importorskip("tensorflow")
    if tf.__version__ != "2.14.1":
        pytest.skip("requires pinned TensorFlow 2.14.1 BUILTIN_REF environment")
    if not Path("/projects/kona-assets/manuelbv/locks/mlperf-tiny/ad.lock.json").is_file():
        pytest.skip("pinned local MLPerf assets unavailable")


@pytest.mark.parametrize("target,count,seed", [
    ("resnet8_int8", 1, 12081),
    ("toycar_int8", 2, 12082),
])
def test_real_continuous_hvp_mapping_residual_and_restoration(target, count, seed):
    _pinned()
    from netsurf.benchmarks.capabilities import load_capability_adapter

    adapter = load_capability_adapter(target)
    x, y, _ = adapter.sample_partition("ranking", 1)
    operator = ContinuousTFLiteHessian(adapter, x[:count], y[:count])
    assert len(operator.mapping) == 10
    assert sum(row["flat_size"] for row in operator.mapping) == len(operator.theta)
    assert len(adapter.bit_identities()) == 8 * len(operator.theta)
    assert operator.companion.manifest()["authoritative_inference"] is False
    if target == "toycar_int8":
        assert operator.companion.loss_version == "continuous-window-reconstruction-mse/v1"
    original = operator.theta.copy()
    direction = np.random.default_rng(seed).normal(size=len(original))
    direction /= np.linalg.norm(direction)
    exact = operator(direction)
    np.testing.assert_array_equal(exact, operator(direction))
    epsilon = 1e-7
    difference = (
        operator.gradient_at(original + epsilon * direction)
        - operator.gradient_at(original - epsilon * direction)
    ) / (2 * epsilon)
    assert np.linalg.norm(difference - exact) / np.linalg.norm(exact) < 1e-5
    np.testing.assert_array_equal(
        np.concatenate([variable.numpy().ravel() for variable in operator.variables]),
        original,
    )
    _, diagnostics = leading_hessian_eigenpairs(
        operator, len(original), count=2, seed=seed,
        max_iterations=100, max_products=150, tolerance=1e-5,
    )
    assert diagnostics.hvp_calls <= 150
    assert max(diagnostics.normalized_residuals) < 1e-5
    assert diagnostics.orthogonality_error < 1e-5
    assert diagnostics.symmetry_error < 1e-5
    np.testing.assert_array_equal(adapter.values()[operator.names[0]], adapter.storage.values()[operator.names[0]])


def test_non_kws_repository_variant_rejected_before_hvp():
    adapter = type("Adapter", (), {"benchmark_id": "mlperf-tiny/ad"})()
    with pytest.raises(ValueError, match="retained only"):
        fkeras_family(adapter, None, None, (), seed=1, options={}, requested=("fkeras_repository",))


@pytest.mark.parametrize("adapter,loss_name", [
    ("mlperf_tiny_ic_int8_v1", "classification cross-entropy"),
    ("mlperf_tiny_ad_int8_v1", "window reconstruction MSE"),
])
def test_new_fkeras_report_identifies_correct_companion_and_only_published_form(adapter, loss_name):
    summary = {
        "clean_evaluation": {"accuracy": 1.0, "categorical_crossentropy": 0.0},
        "eligible_bits": 8, "fkeras_diagnostics": {"eigenpairs": 2},
        "scientific_payload_sha256": "test",
    }
    resolved = {
        "experiment": {"name": "bounded"},
        "benchmark": {"id": "bounded", "adapter": adapter, "quantization": "int8"},
        "scientific_protocol": "test",
        "samples": {"ranking_count": 1, "empirical_count": 1,
                    "evaluation_count": 1, "require_disjoint": True},
    }
    report = render_study_report(summary, resolved)
    assert loss_name in report
    assert "published-form" in report.lower()
    assert "repository sum-strategy" not in report
    assert "KWS companion" not in report


@pytest.mark.parametrize("filename,digest,fields,evaluations", [
    ("m12_resnet8_fkeras_matched_grid_v1.toml", "237f4697d69d9778cfd2f08510cc5ec4099ce1d1693b854c77906b83e901496a", 256, 7680),
    ("m12_resnet8_fkeras_paired_precision_v1.toml", "00ab9e7e7697142c3205a4476b64c3ce7efe1a3f17f9ee68c79fefa344a67e46", 640, 5120),
    ("m12_toycar_fkeras_matched_grid_v1.toml", "f8b59c0b3674f84ffd126fda82b6c03b3f441216b63e846aa95f591fc446d806", 160, 4800),
    ("m12_toycar_fkeras_paired_precision_v1.toml", "967f4048e939a04b98f0ff81a600e8a450b6ed5bb2ba65ac65b18c7c72614057", 768, 6144),
])
def test_frozen_matched_plan_identity_and_work_count(filename, digest, fields, evaluations):
    path = ROOT / "config/experiments" / filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    config = load_study(path, check_inputs=False)
    assert config.schema_version == "3.6.0"
    assert config.benchmark.injectable_parameters == ("kernels",)
    assert {row.name for row in config.estimators if row.name.startswith("fkeras_")} == {"fkeras_paper"}
    fkeras = next(row for row in config.policies if row.name == "fkeras_paper")
    assert fkeras.priority == ("bit_significance_desc", "estimator_score_desc", "stable_identity")
    assert config.samples.require_disjoint
    assert fields == len(config.protection.probabilities) * config.execution.trials * config.replications.count
    assert evaluations == fields * (2 + len(config.policies) * len(config.protection.coverages))
    assert configuration_operation_plan(config)["Fault evaluation"] == fields + evaluations
