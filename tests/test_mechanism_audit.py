from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from netsurf.mechanism_audit import (
    MechanismBundleSpec,
    run_mechanism_audit,
    write_toycar_file_score_artifact,
)


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _finalize_fixture(root: Path):
    files = {}
    for path in root.rglob("*"):
        if path.is_file() and path.name not in {"checksums.json", "FINALIZED"}:
            files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "checksums.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    (root / "FINALIZED").write_text("fixture\n", encoding="utf-8")


def _fixtures(tmp_path: Path):
    estimator, protection = tmp_path / "estimator", tmp_path / "protection"
    empirical = []
    predictions = []
    for index, (mean_z, absolute, signed_damage) in enumerate((
        (0.2, 0.4, 0.3), (-0.1, 0.5, -0.2), (0.0, 0.0, 0.0), (0.3, 0.6, 0.5),
    )):
        parameter = f"tensor_{index // 2}/kernel"
        empirical.append({
            "parameter": parameter, "flat_index": index, "bit_index": index % 2,
            "original_encoded": index, "flipped_encoded": index ^ (1 << (index % 2)),
            "numerical_perturbation": 0.25 * (1 << (index % 2)),
            "signed_loss_difference": signed_damage,
            "positive_loss_damage": max(0.0, signed_damage),
            "absolute_loss_difference": abs(signed_damage), "accuracy_damage": signed_damage / 10,
            "sampling_probability": 0.25, "score_quantile": index % 2,
            "inclusion_rule": "stratified",
        })
        methods = (
            ("s2", "relative-word-bit-index/v1", float(index % 2), ""),
            ("s4", "signed-gradient-times-exact-delta/v2", abs(mean_z), mean_z),
            ("s5", "mean-absolute-per-example-gradient-times-exact-delta/v1", absolute, ""),
            ("s6", "mean-positive-per-example-gradient-times-exact-delta/v1", 0.5 * (absolute + mean_z), ""),
        )
        for name, version, score, signed in methods:
            predictions.append({
                "estimator": name, "implementation_version": version,
                "parameter": parameter, "flat_index": index, "bit_index": index % 2,
                "score": score, "signed_prediction": signed, "empirical_candidate": "True",
            })
    _write_csv(estimator / "measurements/empirical_bits.csv", empirical)
    _write_csv(estimator / "measurements/estimator_predictions.csv", predictions)
    _write_csv(estimator / "measurements/ranking_metrics.csv", [{"scope": "overall", "estimator": "s4"}])
    _write_csv(estimator / "measurements/estimator_runtime.csv", [{"estimator": "s4", "runtime_seconds": 1.0}])

    protection_rows = []
    for condition, policy in (
        ("msb_global@0.500000", "msb_global"),
        ("gradient_global@0.500000", "gradient_global"),
        ("per_example_absolute_global@0.500000", "per_example_absolute_global"),
        ("per_example_positive_global@0.500000", "per_example_positive_global"),
    ):
        protection_rows.append({"condition": condition, "policy": policy, "coverage": 0.5})
    _write_csv(protection / "measurements/protection_results.csv", protection_rows)
    _write_csv(protection / "measurements/paired_comparisons.csv", [{
        "logical_ber": 0.001, "coverage": 0.5, "metric": "accuracy",
        "left": "gradient_global", "right": "no_protection", "mean_difference": 0.1,
        "ci_low": 0.05, "ci_high": 0.15, "matched_pairs": 4,
        "p_value_holm": 0.5,
    }])
    _write_csv(protection / "measurements/mask_composition.csv", [{
        "policy": "gradient_global@0.500000", "dimension": "bit_position",
        "category": "7", "protected_count": 2,
    }])
    # Protection masks are ranking-ordered, not identity-sorted.  Opposite
    # unsorted orders exercise the bounded external sort used by the audit.
    identities = [["tensor_0/kernel", 1, 1], ["tensor_0/kernel", 0, 0]]
    masks = {
        "msb_global@0.500000": identities,
        "gradient_global@0.500000": list(reversed(identities)),
        "per_example_absolute_global@0.500000": [identities[0], ["tensor_1/kernel", 2, 0]],
        "per_example_positive_global@0.500000": [identities[1], ["tensor_1/kernel", 3, 1]],
    }
    (protection / "artifacts").mkdir(parents=True)
    (protection / "artifacts/protection_masks.json").write_text(
        json.dumps(masks, separators=(",", ":")), encoding="utf-8"
    )
    _finalize_fixture(estimator); _finalize_fixture(protection)
    return estimator, protection


def test_mechanism_audit_is_deterministic_memory_bounded_derived_analysis(tmp_path):
    estimator, protection = _fixtures(tmp_path)
    first = run_mechanism_audit(
        (MechanismBundleSpec("fixture", estimator, protection),),
        tmp_path / "first", verify=True, plots=False,
        bootstrap_seed=7, bootstrap_replicates=50,
    )
    second = run_mechanism_audit(
        (MechanismBundleSpec("fixture", estimator, protection),),
        tmp_path / "second", verify=True, plots=False,
        bootstrap_seed=7, bootstrap_replicates=50,
    )
    left = json.loads((first / "audit.json").read_text())
    right = json.loads((second / "audit.json").read_text())
    assert left == right
    coherence = left["benchmarks"]["fixture"]["coherence"]
    assert coherence["maximum_s5_s6_identity_residual"] == 0.0
    assert coherence["zero_s5_denominator_count"] == 1
    with (first / "mask_overlap.csv").open(newline="", encoding="utf-8") as stream:
        overlaps = list(csv.DictReader(stream))
    exact = next(row for row in overlaps if row["left"] == "S2" and row["right"] == "S4")
    assert float(exact["jaccard"]) == 1.0
    assert (first / "report.md").is_file()


class _FileScoreAdapter:
    def file_score_records(self, inputs, labels, *, schedule=None, protection=None):
        offset = 0.0 if schedule is None else 0.5
        rows = tuple({
            "file_id": index, "label": index, "score": 1.0 + index + offset,
            "score_float64_hex": float(1.0 + index + offset).hex(), "window_count": 2,
        } for index in range(2))
        report = None if schedule is None else SimpleNamespace(
            schedule_hash="schedule", proposed_events=2, applied_events=1,
            suppressed_events=1, unique_applied_bits=1,
        )
        return rows, report


def test_optional_toycar_file_score_artifact_records_ids_scores_and_accounting(tmp_path):
    destination = write_toycar_file_score_artifact(
        _FileScoreAdapter(), None, None, ("normal.wav", "anomaly.wav"),
        tmp_path / "scores", fault_cases={"faulted": (object(), object())},
    )
    with (destination / "file_scores.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert {row["sample_id"] for row in rows} == {"normal.wav", "anomaly.wav"}
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["event_accounting"]["faulted"]["proposed_events"] == 2
    assert (destination / "checksums.json").is_file()
