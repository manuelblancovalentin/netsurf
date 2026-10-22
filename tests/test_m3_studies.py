from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from netsurf.api import regenerate_report
from netsurf.experiment_config import OutputConfiguration
from netsurf.result_bundle import ResultBundleError, read_result_bundle
from netsurf.study_bundle import create_study_bundle
from netsurf.study_config import StudyConfigurationError, load_study
from netsurf.study_engine import execute_study


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "config/experiments/numpy_single_bit.toml"


def test_schema3_configuration_and_numpy_study_are_deterministic():
    configuration = load_study(REFERENCE)
    first = execute_study(configuration)
    second = execute_study(configuration)
    assert configuration.schema_version == "3.0.0"
    assert len(first.empirical_measurements) == 102
    assert first.scientific_payload_sha256 == second.scientific_payload_sha256
    assert first.scientific_payload["checks"] == {
        "partitions_disjoint": True,
        "model_state_restored": True,
        "matched_schedules": "not_applicable",
        "quantized_state_exact": True,
    }
    assert all(value.evaluation_sample_sha256 == first.partitions["empirical_validation"].sample_sha256 for value in first.empirical_measurements)


def test_all_maintained_m3_templates_validate_without_framework_import():
    names = (
        "numpy_single_bit.toml", "mnist_single_bit.toml", "mnist_msb_policies.toml",
        "mnist_estimator_comparison.toml", "mnist_multi_coverage.toml",
    )
    for name in names:
        configuration = load_study(ROOT / "config/experiments" / name, check_inputs=False)
        assert configuration.schema_version == "3.0.0"
        assert configuration.samples.require_disjoint


def test_schema3_rejects_unsupported_version(tmp_path):
    text = REFERENCE.read_text(encoding="utf-8").replace('schema_version = "3.0.0"', 'schema_version = "3.1.0"', 1)
    path = tmp_path / "bad.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(StudyConfigurationError, match="schema_version"):
        load_study(path)


def test_study_bundle_is_finalized_verified_and_report_regenerates(tmp_path):
    configuration = load_study(REFERENCE)
    output = replace(configuration.output, root=tmp_path / "runs", plots=False)
    configuration = replace(configuration, output=output)
    outcome = execute_study(configuration)
    destination = create_study_bundle(outcome, repository_root=ROOT, run_id="fixed-study")
    bundle = read_result_bundle(destination)
    assert bundle.scientific_payload_sha256 == outcome.scientific_payload_sha256
    assert json.loads((destination / "artifacts/sample_partitions.json").read_text())["ranking"]["indices"] == [0, 4]
    assert (destination / "measurements/empirical_bits.csv").is_file()
    assert len(bundle.empirical_bits()) == 102
    assert bundle.empirical_bits()[0].identity == outcome.empirical_measurements[0].identity
    report = regenerate_report(bundle, tmp_path / "regenerated", plots=False)
    assert "Exact isolated-bit reference" in report.read_text(encoding="utf-8")
    with pytest.raises(ResultBundleError, match="already exists"):
        create_study_bundle(outcome, repository_root=ROOT, run_id="fixed-study")
