"""Read-only file exposure inventory, including training/ranking distinctions."""
from .cohort_risk import SOURCE
from .manifests import digest_object


def role_inventory(value, path=()):
    rows = []
    if isinstance(value, dict):
        if "source_split" in value and "indices" in value:
            rows.append({"location": ".".join(path), "role": value.get("role", "unspecified"),
                         "source_split": value["source_split"], "indices": list(value["indices"]),
                         "sample_sha256": value.get("sample_sha256")})
        for key, child in value.items():
            rows.extend(role_inventory(child, path + (key,)))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            if isinstance(child, (dict, list)):
                rows.extend(role_inventory(child, path + (str(i),)))
    return rows


def inventory(roots=None):
    from .conditional_benchmarks import conditional_capability, audit_exposure
    audit = audit_exposure(conditional_capability("toycar_int8"), roots=roots)
    exposed = set(audit["exposed_test_indices"])
    audit["official_file_population"] = {"count": 248, "normal": 140, "anomalous": 108,
        "index_order": "normal 0..139, anomalous 140..247; verified by pinned adapter at cohort load"}
    audit["remaining_without_recorded_fault_exposure"] = {
        "normal": sorted(set(range(140)) - exposed), "anomalous": sorted(set(range(140, 248)) - exposed)}
    audit["prior_clean_reference"] = {"indices": list(range(248)), "file_ROC_AUC": .862037037037037,
        "source": "pinned Kona Assets full-conformance record", "not_new_confirmation_outcomes": True}
    audit["mandatory_load_gate"] = {"original_indices": [1, 141], "role": "engineering clean conformance"}
    audit["source_split"] = SOURCE
    audit["limitations"] += (" Distinguish clean-reference exposure of all 248 files from recorded fault-outcome exposure. "
        "Training/ranking normal development-train files are a different source population. "
        "No reservation across other campaigns or manual processes; re-audit immediately before execution.")
    return audit


def write_inventory(output):
    """Finalize an independently checksummed metadata-only exposure snapshot."""
    from pathlib import Path
    from .result_bundle import _write_json, _write_bytes
    from .manifests import sha256_file
    audit = inventory()
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=False)
    _write_json(root / "scientific_payload.json", audit)
    remaining = audit["remaining_without_recorded_fault_exposure"]
    lines = ["# ToyCar prior file exposure inventory", "",
        f"{len(audit['runs'])} finalized bundles verified. {len(audit['exposed_test_indices'])}/248 official files have recorded fault-outcome exposure.", "",
        f"Remaining without recorded fault exposure: {len(remaining['normal'])} normal, {len(remaining['anomalous'])} anomalous files.", "",
        "All 248 files previously underwent clean reference inference. This ledger does not make them an independently collected test dataset. "
        "Normal development-train ranking inputs are a separate population.", "",
        "| Finalized source | Payload SHA-256 | Exposed official files |", "|---|---|---:|"]
    for run in audit["runs"]:
        lines.append(f"| {run['bundle']} | {run['scientific_payload_sha256']} | {len(run['original_test_indices'])} |")
    lines += ["", audit["limitations"], "", f"Scientific payload SHA-256: `{digest_object(audit)}`", ""]
    _write_bytes(root / "report.md", "\n".join(lines).encode())
    _write_json(root / "checksums.json", {"schema_version": "1.0.0", "files": {
        p.name: sha256_file(p) for p in sorted(root.iterdir()) if p.is_file()}})
    _write_bytes(root / "FINALIZED", f"scientific_payload_sha256={digest_object(audit)}\n".encode())
    return {"output": str(root), "scientific_payload_sha256": digest_object(audit)}


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(write_inventory(args.output), indent=2))
