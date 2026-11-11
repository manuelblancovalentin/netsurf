"""Read-only M11 production-analysis receipt; does not import an inference engine.

The finalized analysis is reverified. Linked selection/final payload identities
are retained, not represented as freshly rehashed source-bundle bytes.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from verify_conditional_history import read, sha, verify


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipts, summaries, comparisons, links = [], [], [], []
    for benchmark, name in (("KWS", "m11_kws_matched_lambda_v1"),
                            ("ResNet-8", "m11_resnet_matched_lambda_v1")):
        root = args.source_root / name / "analysis"
        receipts.append(verify(root))
        payload = read(root / "scientific_payload.json")
        assert len(payload["rows"]) == 48 and len(payload["groups"]) == 12
        for k in (6, 12, 32, 64):
            rows = [r for r in payload["rows"] if r["k"] == k]
            regrets = [r["selection_regret"] for r in rows if r["selection_regret"] is not None]
            errors = [r["outside_error"] for r in rows if r["outside_error"] is not None]
            summaries.append({"benchmark": benchmark, "candidate_bits": k,
                              "pool_ber_rows": len(rows), "pools_per_ber": 4,
                              "expected_applied_fault_grid": "10;100;1000",
                              "cohort_requests_per_selection": rows[0]["authoritative_calls"],
                              "selection_seconds_min": min(r["seconds"] for r in rows),
                              "selection_seconds_max": max(r["seconds"] for r in rows),
                              "exact_reference_rows": len(regrets),
                              "maximum_saved_selection_regret": max(regrets) if regrets else "not_measured",
                              "maximum_saved_outside_additive_error": max(errors) if errors else "not_measured",
                              "source_scientific_payload_sha256": receipts[-1]["scientific_payload_sha256"]})
        for group in payload["groups"]:
            for contrast in group["equal_pool_comparisons"]:
                comparisons.append({"benchmark": benchmark, "arm": group["arm"],
                                    "candidate_bits": group["k"],
                                    "target": contrast["target"], "baseline": contrast["baseline"],
                                    "mean_target_minus_baseline": contrast["mean_difference"],
                                    "simultaneous_interval_low": contrast["simultaneous_interval"][0],
                                    "simultaneous_interval_high": contrast["simultaneous_interval"][1],
                                    "family_size": contrast["family_size"],
                                    "independent_pools": contrast["independent_pools"],
                                    "method": contrast["method"]})
        links.extend({"benchmark": benchmark, "k": r["k"], "arm": r["arm"],
                      "pool": r["pool"], "selection_payload_sha256": r["selection_payload_sha256"],
                      "final_payload_sha256": r["final_payload_sha256"]} for r in payload["rows"])
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = []
    for filename, rows in (("m11_bounded_scaling_summary.csv", summaries),
                           ("m11_bounded_scaling_comparisons.csv", comparisons)):
        path = args.output / filename
        with path.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        outputs.append({"path": str(path), "sha256": sha(path), "rows": len(rows)})
    receipt = {"schema_version": "paper1-m11-history/1", "verified_analysis_bundles": receipts,
               "linked_source_payloads_not_rehashed": links, "outputs": outputs,
               "new_inference_count": 0,
               "limitations": [
                   "Fresh verification covers each finalized analysis bundle; linked source bundle bytes were not rehashed by this export.",
                   "The four pool identities recur across exposure arms and are not independent new pools.",
                   "K>12 has no exact restricted optimum or measured approximation error.",
                   "Identically zero saved natural-field differences do not establish equality of expected protection risk.",
                   "No equal-pool simultaneous interval excludes zero in these two saved analyses."]}
    assert all(r["simultaneous_interval_low"] <= 0 <= r["simultaneous_interval_high"] for r in comparisons)
    with (args.output / "m11_bounded_scaling_provenance.json").open("x") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"verified_analysis_bundles": 2, "summary_rows": len(summaries),
                      "comparison_rows": len(comparisons), "new_inference_count": 0}))


if __name__ == "__main__":
    main()
