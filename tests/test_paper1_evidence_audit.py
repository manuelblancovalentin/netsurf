"""Synthetic checks for offline manuscript arithmetic; no model/data execution."""
import hashlib
import json

import numpy as np
import pytest

from workflows.paper1_evidence_audit import audit_static, estimate, sha, verify, write_csv, write_json


def test_estimate_uses_all_values_and_student_interval():
    result = estimate([0., 0., 1., 1.])
    assert result["count"] == 4
    assert result["mean"] == .5
    assert result["sd"] == pytest.approx(np.sqrt(1/3))
    assert result["ci_high"] - .5 == pytest.approx(.9186931155)


def test_payload_verification_detects_changed_evidence(tmp_path):
    payload = {"result": 1}
    digest = hashlib.sha256(b'{"result":1}').hexdigest()
    write_json(tmp_path / "scientific_payload.json", payload)
    write_json(tmp_path / "manifest.json", {"scientific_payload_sha256": digest})
    (tmp_path / "FINALIZED").write_text("complete\n")
    files = {p: sha(tmp_path / p) for p in ("manifest.json", "scientific_payload.json")}
    write_json(tmp_path / "checksums.json", {"files": files})
    source = dict(path=str(tmp_path), scientific_payload_sha256=digest,
                  verified_files=files, checksums_sha256=sha(tmp_path / "checksums.json"))
    assert verify("synthetic", source)[0]["scientific_payload_sha256"] == digest
    write_json(tmp_path / "scientific_payload.json", {"result": 2})
    with pytest.raises(AssertionError):
        verify("synthetic", source)


def test_static_recomputation_checks_replications_and_full_schedule_identity(tmp_path):
    (tmp_path / "measurements").mkdir()
    write_json(tmp_path / "scientific_payload.json", {"replications": [
        {"replication": {"name": f"replication_{i}"},
         "benchmark": {"eligible_bits": 8, "partitions": {}},
         "clean_evaluation": {"accuracy": 1, "categorical_crossentropy": 0}} for i in range(2)]})
    trials, summaries = [], []
    values = {"no_protection": [.1, .2, .7, .8],
              "policy@0.125000": [.0, .2, .9, .9], "full_protection": [1.] * 4}
    for condition, accuracy in values.items():
        for i, value in enumerate(accuracy):
            trials.append(dict(replication=f"replication_{i//2}", trial=i % 2,
                policy=f"ber=0.001/{condition}", schedule_sha256=f"field-{i}",
                accuracy=value, categorical_crossentropy=1-value,
                proposed_events=2, applied_events=0 if condition == "full_protection" else 2,
                suppressed_events=2 if condition == "full_protection" else 0))
        e = estimate(accuracy)
        summaries.append(dict(logical_ber=.001, condition=condition,
            policy=condition.split("@")[0], coverage={"full_protection": 1, "no_protection": 0}.get(condition, .125),
            protected_bits={"full_protection": 8, "no_protection": 0}.get(condition, 1),
            accuracy_mean=e["mean"], accuracy_standard_deviation=e["sd"],
            loss_mean=1-e["mean"], trial_count=4, fault_realization_count=4,
            ranking_mask_count=1, mean_proposed_events=2,
            mean_applied_events=0 if condition == "full_protection" else 2))
    e = estimate(np.array(values["policy@0.125000"]) - values["no_protection"])
    pairs = [dict(logical_ber=.001, coverage=.125, left="policy", right="no_protection",
                  metric="accuracy", mean_difference=e["mean"], standard_deviation=e["sd"],
                  ci_low=e["ci_low"], ci_high=e["ci_high"], matched_pairs=4)]
    for name, data in [("trials", trials), ("protection_results", summaries), ("paired_comparisons", pairs)]:
        write_csv(tmp_path / f"measurements/{name}.csv", data)
    manifest = dict(benchmark={}, scientific_payload_sha256="synthetic",
                    execution={"parallel_execution": {}, "protection": {"probabilities": [.001], "coverages": [.125]}})
    curves, _, inventory = audit_static("static_synthetic", tmp_path, manifest)
    assert inventory["distinct_fields"] == 4
    assert next(r for r in curves if r["policy"] == "no_protection")["mean"] == pytest.approx(.45)
    # Merely matching replication/trial while ignoring the field hash must fail.
    trials[4]["schedule_sha256"] = "unpaired"
    write_csv(tmp_path / "measurements/trials.csv", trials)
    with pytest.raises(AssertionError):
        audit_static("static_synthetic", tmp_path, manifest)


def test_bounded_counterexample_by_exact_enumeration():
    risk = [0, .2, .1, .3, 0, 0, .9, .5]
    for p in [.01, .2]:
        def expected(protected):
            return sum(p**int(s).bit_count() * (1-p)**(3-int(s).bit_count()) *
                       risk[s & ~protected] for s in range(8))
        assert expected(1) == pytest.approx(.1*p + .8*p*p)
        assert expected(2) == pytest.approx(.2*p - .2*p*p)
        assert (expected(1) < expected(2)) == (p < .1)
