from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile

from netsurf.experiment_config import load_experiment, template_text
from netsurf.experiment_engine import ExperimentOutcome, TrialMeasurement
from netsurf.result_bundle import ResultBundleError, create_result_bundle, read_result_bundle


def _outcome(root: Path) -> ExperimentOutcome:
    checkpoint, dataset = root / "checkpoint.bin", root / "dataset.bin"
    checkpoint.write_bytes(b"checkpoint")
    dataset.write_bytes(b"dataset")
    text = template_text("smoke").replace(
        '/ml/experiments/netsurf/legacy/benchmarks/mnist_fnn_lite/q6_2_1/pruned_0.125_159569_qflatten_dense10u_qapplyalpha_quantized_relu_dense10u_qsoftmax/models/pruned_0.125_159569_qflatten_dense10u_qapplyalpha_quantized_relu_dense10u_qsoftmax.20250414_152629.keras',
        str(checkpoint),
    ).replace('/ml/datasets/mnist/mnist.npz', str(dataset)).replace('root = ""', f'root = "{root / "runs"}"')
    configuration_path = root / "experiment.toml"
    configuration_path.write_text(text, encoding="utf-8")
    configuration = load_experiment(configuration_path)
    estimate = {"count": 2, "mean": 0.75, "standard_deviation": 0.1, "standard_error": 0.07, "confidence": 0.95, "confidence_interval": [0.6, 0.9]}
    payload = {
        "baseline": {"checkpoint": {"accuracy": 0.9, "categorical_crossentropy": 0.2, "sample_count": 16}},
        "policies": {
            "msb_first": {"protected_bits": 1, "coverage": 0.1, "mask_sha256": "a" * 64,
                          "accuracy": estimate, "categorical_crossentropy": estimate,
                          "mean_applied_events": 1.0, "mean_suppressed_events": 1.0},
            "first_order_gradient_delta": {"protected_bits": 1, "coverage": 0.1, "mask_sha256": "b" * 64,
                          "accuracy": estimate, "categorical_crossentropy": estimate,
                          "mean_applied_events": 1.0, "mean_suppressed_events": 1.0},
        },
        "checks": {"state_restored": True, "matched_schedules": True},
    }
    trials = tuple(
        TrialMeasurement(trial, policy, "c" * 64, 0.75, 0.25, "c" * 64, 2, 1, 1, 1)
        for trial in range(2) for policy in payload["policies"]
    )
    rankings = {"msb_first": [["weight", 0, 5, 1.0]], "first_order_gradient_delta": [["weight", 0, 5, 0.5]]}
    masks = {"msb_first": [["weight", 0, 5]], "first_order_gradient_delta": [["weight", 0, 5]]}
    schedules = ({"trial": 0, "seed": 1, "sha256": "c" * 64, "events": []},)
    return ExperimentOutcome(configuration, payload, trials, rankings, masks, schedules)


def test_bundle_round_trip_and_repeated_payload_equivalence():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        outcome = _outcome(root)
        repo = Path(__file__).resolve().parents[1]
        first = create_result_bundle(outcome, repository_root=repo, run_id="one")
        second = create_result_bundle(outcome, repository_root=repo, run_id="two")
        first_result, second_result = read_result_bundle(first), read_result_bundle(second)
        assert first_result.scientific_payload_sha256 == second_result.scientific_payload_sha256
        assert first_result.trials() == outcome.trials
        assert (first / "report.md").is_file()
        assert first_result.manifest["versions"]["experiment_schema"] == "1.0.0"


def test_bundle_refuses_overwrite_and_detects_corruption():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        outcome = _outcome(root)
        repo = Path(__file__).resolve().parents[1]
        destination = create_result_bundle(outcome, repository_root=repo, run_id="fixed")
        try:
            create_result_bundle(outcome, repository_root=repo, run_id="fixed")
        except ResultBundleError:
            pass
        else:
            raise AssertionError("finalized bundle was overwritten")
        target = destination / "summary.json"
        target.chmod(0o644)
        target.write_text("{}\n", encoding="utf-8")
        try:
            read_result_bundle(destination)
        except ResultBundleError:
            pass
        else:
            raise AssertionError("corruption was not detected")


def test_bundle_requires_finalization_marker():
    with tempfile.TemporaryDirectory() as directory:
        try:
            read_result_bundle(directory)
        except ResultBundleError as error:
            assert "not a finalized" in str(error)
        else:
            raise AssertionError("partial directory was accepted")
