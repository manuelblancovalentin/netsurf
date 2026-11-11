"""Offline-only extraction of the finalized historical conditional evidence.

Reads and hashes saved bytes. No NetSuRF/model/backend import is permitted here.
Outputs are exclusive-created; use a new destination for another extraction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import t


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def verify(root: Path, expected_payload: str | None = None) -> dict:
    manifest, inventory, payload = (read(root / name) for name in
                                    ("manifest.json", "checksums.json", "scientific_payload.json"))
    digest = canonical_sha(payload)
    assert digest == manifest["scientific_payload_sha256"], root
    if expected_payload is not None:
        assert digest == expected_payload, root
    assert (root / "FINALIZED").read_text().strip() == "scientific_payload_sha256=" + digest
    for name, expected in inventory["files"].items():
        assert sha(root / name) == expected, (root, name)
    return {"path": str(root), "scientific_payload_sha256": digest,
            "manifest_sha256": sha(root / "manifest.json"),
            "checksums_sha256": sha(root / "checksums.json"),
            "verified_files": len(inventory["files"]),
            "verified_bytes": sum((root / name).stat().st_size for name in inventory["files"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    verified_root = verify(args.source)
    source_evidence = read(args.source / "source_verification.json")["verified_before_and_after"]
    verified_sources = []
    for receipt in source_evidence:
        for phase in ("selection", "final"):
            root = Path(receipt[phase + "_bundle"])
            assert sha(root / "checksums.json") == receipt[phase + "_checksums_sha256"]
            verified_sources.append(verify(root, receipt[phase + "_payload_sha256"]))
    assert len(verified_sources) == len({r["path"] for r in verified_sources}) == 112
    payload = read(args.source / "scientific_payload.json")
    aggregate = read(args.source / "aggregate_summary.json")["comparisons"]
    with (args.source / "mechanism_summary.csv").open(newline="") as handle:
        mechanism = {row["campaign"]: row for row in csv.DictReader(handle)}
    rows = []
    for campaign in payload["campaigns"]:
        name = campaign["name"]
        pools = campaign["pools"]
        reported = next(row for row in aggregate if row["campaign"] == name and
                        row["comparison"] == "restricted_selected_vs_exact_isolated")["pool_summary"]
        means = np.asarray([next(row for row in p["final_comparisons"]["comparisons"]
                                 if row["comparison"] == "restricted_selected_vs_exact_isolated")
                            ["mean_effect"]["mean"] for p in pools])
        mean = float(means.mean())
        std = float(means.std(ddof=1))
        half = float(t.ppf(.975, len(means) - 1) * std / math.sqrt(len(means)))
        for reconstructed, saved in zip((mean, mean - half, mean + half),
                                         (reported["mean"], *reported["confidence_interval"])):
            assert math.isclose(reconstructed, saved, rel_tol=1e-12, abs_tol=1e-15), name
        counters = {}
        for label, key in (("clean", "clean_vs_exact_ordering"),
                           ("conditioned", "conditioned_vs_exact_ordering")):
            counters[label + "_strict_reversals"] = sum(p[key]["strict_reversals"] for p in pools)
            counters[label + "_strict_pairs"] = sum(p[key]["strict_comparable_pairs"] for p in pools)
        for key, value in counters.items():
            assert value == int(mechanism[name][key]), (name, key)
        benchmark = "ToyCar" if "toycar" in name else "ResNet-8" if "resnet" in name else "KWS"
        rows.append({"campaign_id": name, "benchmark": benchmark,
                     "logical_ber": campaign["logical_ber"], "candidate_bits": len(pools[0]["candidates"]),
                     "candidate_protection_budget": pools[0]["budget"], "pool_count": len(pools),
                     "risk_metric": campaign["risk_metric"],
                     "units": "AUC risk" if benchmark == "ToyCar" else "nats/clip" if benchmark == "KWS" else "nats/image",
                     **counters, "heldout_selected_minus_isolated_mean": mean,
                     "heldout_ci95_low": mean - half, "heldout_ci95_high": mean + half,
                     "heldout_pool_standard_deviation": std,
                     "uncertainty_unit": "candidate_pool_conditional_on_fixed_cohort",
                     "inference_status": "exploratory_descriptive_marginal_t_interval",
                     "source_scientific_payload_sha256": verified_root["scientific_payload_sha256"]})
    assert len(rows) == 6 and sum(row["pool_count"] for row in rows) == 56
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "conditional_historical_summary.csv"
    with csv_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    receipt = {"schema_version": "paper1-conditional-history/1", "source": verified_root,
               "source_bundles": verified_sources, "source_bundle_count": len(verified_sources),
               "source_file_count": sum(r["verified_files"] for r in verified_sources),
               "source_bytes_verified": sum(r["verified_bytes"] for r in verified_sources),
               "output": {"path": str(csv_path), "sha256": sha(csv_path), "rows": len(rows)},
               "new_inference_count": 0, "verification_seconds": time.monotonic() - started,
               "reconstruction": "Pool means and Student-t intervals reconstructed from per-pool saved final contrasts; strict reversal counts resummed from all per-pool diagnostics.",
               "limitations": ["KWS pool identities recur across BER arms and are not independent new samples.",
                               "Pair-reversal counts share masks and fields; no binomial independence is claimed.",
                               "Intervals are descriptive marginal pool-t intervals, not new multiplicity-adjusted tests.",
                               "Selection-side attribution is not a decomposition of held-out effects."]}
    with (args.output / "conditional_historical_provenance.json").open("x") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"rows": len(rows), "pool_ber_conditions": sum(r["pool_count"] for r in rows),
                      "verified_source_files": receipt["source_file_count"],
                      "source_payload": verified_root["scientific_payload_sha256"],
                      "seconds": receipt["verification_seconds"]}))


if __name__ == "__main__":
    main()
