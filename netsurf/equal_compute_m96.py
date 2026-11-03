"""Read-only, retrospective equal-query search on frozen M9.6 response tables.

All model calls were made by M9.6. This diagnostic divides its selection
background fields into two halves; input clips are shared, and the second half
is not a new publication final set. It never mutates the source campaign.
"""
from __future__ import annotations

from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
import statistics
import subprocess
import zipfile

from .equal_compute_search import evaluate_held_out, search_allocation
from .manifests import canonical_json_bytes, digest_object, sha256_file

SOURCE = Path("/projects/netsurf/manuelbv/workspace/implementation-worktrees/m96/workspace/runs/m96_kws_cross_pool_k6_v1/campaign.json")
SOURCE_SHA256 = "46674ad4f90e8bc49fcee3ef826bbb57130005f239d192700cd09d8f106d3ff3"
SEEDS = (901, 902, 903)
QUERY_BUDGET = 10


def _validated_table(summary: dict) -> dict[tuple[int, int], tuple[float, ...]]:
    if summary.get("budget") != 2 or summary.get("candidate_count") != 6:
        raise ValueError("M9.6 frozen K=6/B=2 table required")
    rows = summary.get("eligible_masks")
    if not isinstance(rows, list) or len(rows) != 15:
        raise ValueError("all 15 exact mask risks required")
    table = {}
    for row in rows:
        mask = tuple(row.get("indices", ()))
        values = row.get("field_risks")
        if (len(mask) != 2 or tuple(sorted(mask)) != mask or
            not isinstance(values, list) or len(values) != 32 or
            any(not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in values)):
            raise ValueError("invalid mask or saved background field risks")
        if mask in table:
            raise ValueError("duplicate mask response")
        table[mask] = tuple(float(value) for value in values)
    if set(table) != set(itertools.combinations(range(6), 2)):
        raise ValueError("missing exact mask in saved response table")
    return table


def _source_pool(source: Path, pool: dict) -> tuple[dict, Path]:
    receipt_path = source.parent / "receipts" / f"{pool['pool_id']}_selection.json"
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("campaign_sha256") != SOURCE_SHA256 or
        receipt.get("pool_id") != pool["pool_id"] or
        receipt.get("phase") != "selection"):
        raise ValueError("wrong M9.6 selection receipt")
    bundle = Path(receipt["bundle"])
    payload_path = bundle / "scientific_payload.json"
    payload = json.loads(payload_path.read_text())
    checks = json.loads((bundle / "checksums.json").read_text())["files"]
    for name in ("scientific_payload.json", "artifacts/restricted_oracle_masks.json"):
        if sha256_file(bundle / name) != checks[name]:
            raise ValueError("M9.6 finalized response bundle changed")
    if digest_object(payload) != receipt["payload_sha256"]:
        raise ValueError("M9.6 selection payload differs from receipt")
    if payload["campaign_binding"]["campaign_sha256"] != SOURCE_SHA256:
        raise ValueError("M9.6 selection payload has wrong campaign binding")
    return payload, bundle


def retrospective_search(source: Path = SOURCE) -> dict:
    source = Path(source)
    if sha256_file(source) != SOURCE_SHA256:
        raise ValueError("frozen M9.6 campaign manifest changed")
    campaign = json.loads(source.read_text())
    pools = campaign["pools"]
    if len(pools) != 12 or any(len(pool["candidates"]) != 6 for pool in pools):
        raise ValueError("all 12 frozen K=6 pools required")
    if len({row["identity"] for pool in pools for row in pool["candidates"]}) != 72:
        raise ValueError("frozen pools unexpectedly overlap or changed")
    outcomes = []
    for pool in pools:
        payload, bundle = _source_pool(source, pool)
        summaries = payload["restricted_oracle"]["summaries"]
        if len(summaries) != 1:
            raise ValueError("one exact B=2 response summary required")
        summary = summaries[0]
        table = _validated_table(summary)
        initial = tuple(summary["original_mask_indices"])
        if initial != tuple(pool["original_mask_indices"]) or len(initial) != 2:
            raise ValueError("original S5 pool mask changed")
        table_sha = sha256_file(bundle / "artifacts/restricted_oracle_masks.json")
        select_id = f"{table_sha}:fields_00_15"
        final_id = f"{table_sha}:fields_16_31"

        def risk(mask, first: int, last: int):
            values = table[tuple(sorted(mask))]
            return statistics.mean(values[first:last])

        runs = [search_allocation(tuple(range(6)), initial, method=method,
            query_budget=QUERY_BUDGET, seed=seed,
            selection_identity=select_id, calls_per_query=0,
            clip_inferences_per_query=0, cost_type="saved_table_lookup",
            evaluate=lambda mask: risk(mask, 0, 16))
            for method in ("random_search", "greedy_swap") for seed in SEEDS]
        heldout = evaluate_held_out(runs, {"initial_s5": initial},
            final_identity=final_id, evaluate=lambda mask: risk(mask, 16, 32))
        outcomes.append({"pool_id": pool["pool_id"],
            "candidate_order_sha256": pool["candidate_order_sha256"],
            "selection_payload_sha256": digest_object(payload),
            "response_table_file_sha256": table_sha,
            "selection_background_records_sha256": payload["restricted_oracle"]["background_records_sha256"],
            "response_records_sha256": payload["restricted_oracle"]["response_records_sha256"],
            "selection_identity": select_id, "heldout_identity": final_id,
            "selection_initial_risk": risk(frozenset(initial), 0, 16),
            "heldout_risks": heldout,
            "searches": [asdict(run) for run in runs]})
    return {"schema_version": "m96-offline-equal-query-engineering/v1",
        "status": "RETROSPECTIVE_ENGINEERING_ONLY", "source_manifest": str(source),
        "source_manifest_sha256": SOURCE_SHA256, "query_budget": QUERY_BUDGET,
        "selection_seeds": list(SEEDS), "query_unit": "one exact B=2 mask-risk lookup from the sealed response table",
        "new_model_requests": 0, "new_clip_inferences": 0,
        "historical_table_generation_requests": 12 * 32 * 64,
        "selection_fields_per_pool": 16, "heldout_fields_per_pool": 16,
        "same_input_clips_across_halves": True,
        "limitation": "M9.6 selection responses were already finalized; heldout backgrounds are independent within the table but inputs are shared and the exercise was designed after the data were observed. No confirmatory or new final-effect claim.",
        "pools": outcomes}


def write_retrospective(output: Path) -> dict:
    output = Path(output)
    if output.exists():
        checks = json.loads((output / "checksums.json").read_text())
        if any(sha256_file(output / name) != value for name, value in checks.items()):
            raise ValueError("existing retrospective search evidence changed")
        with zipfile.ZipFile(output / "complete.zip") as archive:
            if archive.testzip() is not None:
                raise ValueError("existing retrospective search ZIP corrupt")
        return {"status": "VERIFIED_EXISTING", "output": str(output)}
    root = Path(__file__).resolve().parents[1]
    scientific_files = ("netsurf/equal_compute_m96.py", "netsurf/equal_compute_search.py")
    subprocess.run(["git", "ls-files", "--error-unmatch", *scientific_files],
                   cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *scientific_files],
                   cwd=root, check=True)
    data = retrospective_search()
    data["source_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    data["source_files_sha256"] = {name: sha256_file(root / name)
                                    for name in scientific_files}
    output.mkdir(parents=True, exist_ok=False)
    (output / "payload.json").write_bytes(canonical_json_bytes(data) + b"\n")
    (output / "FINALIZED").write_text(digest_object(data) + "\n")
    with zipfile.ZipFile(output / "complete.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(output / "payload.json", "payload.json")
        archive.write(SOURCE, "source/m96_campaign.json")
        archive.write(Path(__file__), "source/equal_compute_m96.py")
        archive.write(Path(__file__).with_name("equal_compute_search.py"), "source/equal_compute_search.py")
    checks = {name: sha256_file(output / name) for name in ("payload.json", "FINALIZED", "complete.zip")}
    (output / "checksums.json").write_bytes(canonical_json_bytes(checks) + b"\n")
    return {"status": "FINALIZED_ENGINEERING_ONLY", "output": str(output),
            "payload_sha256": digest_object(data), "complete_zip_sha256": checks["complete.zip"]}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(write_retrospective(arguments.output), sort_keys=True))
