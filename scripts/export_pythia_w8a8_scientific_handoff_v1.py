"""Offline compact handoff of finalized Pythia whole-qweight W8A8 evidence.

Reads the immutable v3 run and its verified reporting v3 export.  Never loads
the model, reruns a field, constructs a mask, or writes into those sources.
"""

from __future__ import annotations

from collections import defaultdict
import csv
from html import escape
import json
import math
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from netsurf.manifests import canonical_json_bytes, digest_object, sha256_file  # noqa: E402
from netsurf.w8a8_campaign import _read_bound  # noqa: E402
from netsurf.w8a8_faults import BoundaryMask  # noqa: E402
from report_lm_w8a8_v3 import _verify as verify_reporting  # noqa: E402

STEM = "pythia160m_w8a8_reliability_v3"
PLAN = ROOT / "workspace/plans" / STEM
RUN = ROOT / "workspace/runs" / (STEM + "_run1")
CORE = RUN.with_name(RUN.name + "_export")
REPORT = RUN.with_name(RUN.name + "_reporting_v3")
DEST = RUN.with_name(RUN.name + "_scientific_handoff_v1")
METHODS = ("random", "msb_first", "numerical_bit_jump", "global_first_order",
           "per_example_absolute", "per_example_positive")
COLORS = {"random": "#6b7280", "msb_first": "#2563eb", "numerical_bit_jump": "#d97706",
          "global_first_order": "#059669", "per_example_absolute": "#7c3aed",
          "per_example_positive": "#dc2626"}


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("empty scientific table")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def mask_summary(plan: dict, seal: dict) -> list[dict]:
    bits = plan["eligible_universe"]["bits"]
    budgets = sorted(int(x) for x in plan["interior_budgets_bits"].values())
    if tuple(plan["methods"]) != METHODS or len(budgets) != 4 or set(seal["masks"]) != set(METHODS)-{"msb_first"}:
        raise ValueError("frozen six-method mask family changed")
    rows = []
    for method in METHODS:
        for budget in budgets:
            if method == "msb_first":
                digest = seal["msb"][str(budget)]
                payload_hash = "logical_mask_no_packed_payload"
                payload_bytes = 0
            else:
                item = seal["masks"][method][str(budget)]
                digest, payload_hash, payload_bytes = item["digest"], item["sha256"], item["bytes"]
                if payload_bytes != bits // 8:
                    raise ValueError("packed mask byte count changed")
            rows.append({"policy_id": f"{method}_b{budget}", "method": method,
                         "protected_bits": budget, "coverage_fraction": budget/bits,
                         "mask_sha256": digest, "packed_payload_sha256_in_original_complete_archive": payload_hash,
                         "packed_payload_bytes_omitted": payload_bytes,
                         "eligible_population_bits": bits,
                         "eligible_population_sha256": plan["eligible_universe"]["sha256"]})
    for method, full in (("unprotected", False), ("full_protection", True)):
        rows.append({"policy_id": method, "method": method,
                     "protected_bits": bits if full else 0,
                     "coverage_fraction": 1.0 if full else 0.0,
                     "mask_sha256": BoundaryMask(bits, full).digest,
                     "packed_payload_sha256_in_original_complete_archive": "logical_boundary_mask",
                     "packed_payload_bytes_omitted": 0,
                     "eligible_population_bits": bits,
                     "eligible_population_sha256": plan["eligible_universe"]["sha256"]})
    if len(rows) != 26 or len({x["policy_id"] for x in rows}) != 26:
        raise ValueError("mask summary missing or duplicates a policy")
    return rows


def audit_observations(plan: dict, analysis: dict, raw: list[dict], masks: list[dict]) -> tuple[list[dict], dict]:
    bits = plan["eligible_universe"]["bits"]
    policies = {(x["method"], x["protected_bits"]): x for x in masks}
    expected_fields = {(spec["ber"], i, seed, spec["lambda"])
                       for spec in plan["fields"] for i, seed in enumerate(spec["seeds"])}
    if len(expected_fields) != 72 or len(raw) != 1872 or len(analysis["field_rows"]) != 72:
        raise ValueError("72 paired fields or 1872 policy observations missing")
    clean = analysis["clean"]["metrics"]
    groups = defaultdict(list)
    enriched = []
    for row in raw:
        ber, index, seed, lam = (float(row["ber"]), int(row["field_index"]),
                                 int(row["seed"]), int(row["expected_proposed_faults"]))
        key = (ber, index, seed, lam)
        if key not in expected_fields or row["plan_sha256"] != plan["plan_sha256"]:
            raise ValueError("field identity outside frozen plan")
        method, budget = row["method"], int(row["protected_bits"])
        mask = policies.get((method, budget))
        if mask is None or row["mask_sha256"] != mask["mask_sha256"]:
            raise ValueError("policy omitted, duplicated, or mask identity changed")
        proposed, applied, suppressed = (int(row[x]) for x in
            ("proposed_faults", "applied_faults", "suppressed_faults"))
        failed, scored = int(row["failed_tokens"]), int(row["scored_tokens"])
        loss = float(row["mean_nll"])
        if (proposed != applied + suppressed or scored != 4096 or not 0 <= failed <= scored or
            not math.isfinite(loss) or budget not in (0, bits) and budget not in
            (int(x) for x in plan["interior_budgets_bits"].values())):
            raise ValueError("fault, failure, or budget accounting changed")
        if method == "full_protection" and (applied != 0 or loss != clean["mean_nll"] or
            failed != 0 or float(row["perplexity"]) != clean["perplexity"]):
            raise ValueError("full-protection control differs from clean")
        if method == "unprotected" and (suppressed != 0 or applied != proposed):
            raise ValueError("unprotected event accounting changed")
        groups[key].append(row)
        enriched.append({**row, "policy_id": mask["policy_id"],
                         "failure_fraction": failed/scored,
                         "loss_degradation_nats": loss-clean["mean_nll"],
                         "eligible_population_bits": bits,
                         "eligible_population_sha256": plan["eligible_universe"]["sha256"]})
    if set(groups) != expected_fields:
        raise ValueError("frozen field missing or duplicated")
    for key, members in groups.items():
        if len(members) != 26 or {(x["method"], int(x["protected_bits"])) for x in members} != set(policies):
            raise ValueError("field has missing or duplicated ranking method")
        identities = {(x["field_sha256"], x["field_record"], int(x["proposed_faults"])) for x in members}
        if len(identities) != 1:
            raise ValueError("compared methods did not share proposed field")
    return enriched, {"independent_fields": len(groups), "observations": len(raw),
                      "policies_per_field": 26, "ranking_methods": list(METHODS),
                      "all_proposed_fields_paired": True, "all_event_identities_valid": True,
                      "all_full_protection_matches_clean": True,
                      "all_failure_fractions_recorded": True}


def budget_plot(conditions: list[dict], clean_nll: float) -> str:
    by = {(int(r["expected_proposed_faults"]), r["method"], int(r["protected_bits"])): r
          for r in conditions}
    lambdas = (4, 32, 256)
    budgets = sorted({int(r["protected_bits"]) for r in conditions if r["method"] in METHODS})
    if len(budgets) != 4 or len(conditions) != 78:
        raise ValueError("budget plot requires 78 complete conditions")
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1260" height="1230" viewBox="0 0 1260 1230">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#111827}.grid{stroke:#e5e7eb;stroke-width:1}.axis{stroke:#6b7280;stroke-width:1}</style>',
             '<text x="70" y="35" font-size="22" font-weight="bold">Pythia-160M W8A8: token NLL change versus protected budget</text>',
             '<text x="70" y="58" font-size="13">Categorical x positions; 24 independent paired fault fields per BER. Bars: 95% field bootstrap intervals.</text>']
    labels = [0, *budgets, 988545024]
    for panel, lam in enumerate(lambdas):
        top = 105 + panel*335; left = 105; width = 1060; height = 235
        members = [r for r in conditions if int(r["expected_proposed_faults"]) == lam]
        ymin = min(0.0, *(float(r["ci95_degradation_low"]) for r in members))
        ymax = max(0.0, *(float(r["ci95_degradation_high"]) for r in members))
        pad = max(.001, (ymax-ymin)*.12); ymin -= pad; ymax += pad
        y = lambda v: top + height - (v-ymin)/(ymax-ymin)*height
        x = lambda i: left + i*width/5
        parts.append(f'<text x="{left}" y="{top-16}" font-size="17" font-weight="bold">Expected proposed faults: {lam}</text>')
        for tick in range(5):
            value = ymin+(ymax-ymin)*tick/4; yy=y(value)
            parts.append(f'<line class="grid" x1="{left}" x2="{left+width}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
            parts.append(f'<text x="{left-12}" y="{yy+4:.1f}" font-size="11" text-anchor="end">{value:.4f}</text>')
        parts.append(f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{top+height}"/>')
        parts.append(f'<line class="axis" x1="{left}" x2="{left+width}" y1="{top+height}" y2="{top+height}"/>')
        for i, budget in enumerate(labels):
            parts.append(f'<text x="{x(i):.1f}" y="{top+height+20}" font-size="11" text-anchor="middle">{100*budget/988545024:g}%</text>')
        parts.append(f'<line x1="{left}" x2="{left+width}" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="#111827" stroke-dasharray="4,4"/>')
        for method in METHODS:
            series = [by[(lam, method, budget)] for budget in budgets]
            color = COLORS[method]
            coords = " ".join(f'{x(i+1):.1f},{y(float(r["mean_loss_degradation"])):.1f}' for i,r in enumerate(series))
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
            for i,r in enumerate(series):
                xx=x(i+1); low=y(float(r["ci95_degradation_low"])); high=y(float(r["ci95_degradation_high"]))
                parts.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{low:.1f}" y2="{high:.1f}" stroke="{color}"/>')
                parts.append(f'<circle cx="{xx:.1f}" cy="{y(float(r["mean_loss_degradation"])):.1f}" r="3" fill="{color}"/>')
        for method, budget, color in (("unprotected",0,"#111827"),("full_protection",988545024,"#0f766e")):
            r=by[(lam,method,budget)]; i=0 if not budget else 5
            parts.append(f'<circle cx="{x(i):.1f}" cy="{y(float(r["mean_loss_degradation"])):.1f}" r="5" fill="{color}"/>')
    legend = [*METHODS,"unprotected","full_protection"]
    for i,method in enumerate(legend):
        color=COLORS.get(method, "#111827" if method=="unprotected" else "#0f766e")
        xx=70+(i%4)*290; yy=1135+(i//4)*30
        parts.append(f'<line x1="{xx}" x2="{xx+22}" y1="{yy}" y2="{yy}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{xx+28}" y="{yy+4}" font-size="12">{escape(method.replace("_"," "))}</text>')
    parts.append(f'<text x="70" y="1210" font-size="12">Clean NLL = {clean_nll:.6f} nats; zero line = clean loss. 100% point is a full-protection control.</text>')
    parts.append('</svg>')
    return "\n".join(parts)+"\n"


def create() -> dict:
    if DEST.exists():
        raise ValueError("write-once compact handoff already exists")
    if not REPORT.is_dir():
        raise ValueError("verified Pythia reporting v3 export required first")
    plan = json.loads((PLAN/"plan.json").read_text())
    if sha256_file(PLAN/"plan.json") != (PLAN/"FROZEN").read_text().strip() or plan["target"] != "pythia160m_int8":
        raise ValueError("frozen whole-qweight Pythia plan changed")
    reporting = verify_reporting(REPORT, RUN)
    analysis = _read_bound(RUN/"analysis.json")
    payload = _read_bound(RUN/"scientific_payload.json")
    seal = _read_bound(RUN/"selection/seal.json")
    if (analysis["plan_sha256"] != plan["plan_sha256"] or payload["plan_sha256"] != plan["plan_sha256"] or
        payload["analysis_sha256"] != sha256_file(RUN/"analysis.json") or
        payload["selection_seal_sha256"] != sha256_file(RUN/"selection/seal.json") or
        len(payload["final_member_hashes"]) != 145):
        raise ValueError("finalized scientific payload does not bind analysis and fields")
    full_hash = sha256_file(CORE/"complete.zip")
    core_checks = json.loads((CORE/"checksums.json").read_text())
    if (full_hash != reporting["complete_zip_sha256"] or
        (CORE/"FINALIZED").read_text().strip() != digest_object({
            "plan_sha256": plan["plan_sha256"], "zip_sha256": full_hash, "members": core_checks})):
        raise ValueError("original complete archive provenance changed")
    masks = mask_summary(plan, seal)
    raw = _read_csv(REPORT/"field_observations.csv")
    rows, audit = audit_observations(plan, analysis, raw, masks)
    conditions = _read_csv(REPORT/"condition_level.csv")
    if len(conditions) != 78 or len({(r["ber"],r["method"],r["protected_bits"]) for r in conditions}) != 78:
        raise ValueError("condition table incomplete or duplicated")
    comparisons = analysis["comparisons"]
    if len(comparisons) != 60 or any("p_holm_60" not in x or "ci95" not in x for x in comparisons):
        raise ValueError("original paired multiplicity family incomplete")
    runtime = json.loads(RUN.with_name(RUN.name+"_runtime_receipt.json").read_text())
    if runtime["exit_code"] != 0 or len(runtime["stages"]) != 6:
        raise ValueError("foreground runtime receipt incomplete")
    stages = {x["phase"]: x for x in runtime["stages"]}
    costs = {"measured_total_seconds": runtime["seconds"], "measured_total_hours": runtime["seconds"]/3600,
             "peak_group_rss_bytes": runtime["peak_group_rss_bytes"],
             "selection_stage_seconds_including_surrogate_and_all_mask_construction": stages["selection"]["seconds"],
             "selection_stage_peak_rss_bytes": stages["selection"]["peak_group_rss_bytes"],
             "final_stage_seconds_including_fault_application_inference_and_restoration": stages["final"]["seconds"],
             "final_stage_peak_rss_bytes": stages["final"]["peak_group_rss_bytes"],
             "final_scored_token_requests": plan["counts"]["total_scored_final_unit_requests"],
             "stage_inclusive_scored_token_requests_per_second":
                plan["counts"]["total_scored_final_unit_requests"]/stages["final"]["seconds"],
             "mask_payloads_constructed": 20,
             "mask_payload_bytes_constructed": plan["counts"]["stored_interior_mask_bytes"],
             "msb_masks_logical_count": 4,
             "per_method_mask_construction_seconds": "not separately measured; selection stage also includes surrogate moments",
             "pinned_asset_clean_reference_forward_tokens_per_second_16_threads":
                json.loads((PLAN/"target_ready_record.json").read_text())["reference_results"]["throughput"]["forward_tokens_per_second"],
             "cost_interpretation": "Campaign final throughput includes mutation, integer inference and restoration on four threads; asset clean throughput used 16 threads and is not directly comparable."}
    source_paths = {name: ROOT/name for name in plan["source_files_sha256"]}
    if any(sha256_file(path) != plan["source_files_sha256"][name] for name,path in source_paths.items()):
        raise ValueError("hash-bound scientific source changed")
    DEST.mkdir(parents=True, exist_ok=False)
    _write_csv(DEST/"field_observations.csv", rows)
    shutil.copyfile(REPORT/"condition_level.csv", DEST/"condition_level.csv")
    _write_csv(DEST/"paired_comparisons_original.csv", comparisons)
    _write_csv(DEST/"mask_summary.csv", masks)
    shutil.copyfile(REPORT/"loss_degradation_vs_ber.svg", DEST/"loss_degradation_vs_ber.svg")
    (DEST/"loss_degradation_vs_budget.svg").write_text(budget_plot(conditions,analysis["clean"]["metrics"]["mean_nll"]))
    (DEST/"costs.json").write_bytes(canonical_json_bytes(costs)+b"\n")
    (DEST/"audit.json").write_bytes(canonical_json_bytes({**audit,
        "condition_count":78,"paired_comparisons":60,"min_holm_p":min(x["p_holm_60"] for x in comparisons),
        "holm_p_below_0_05":sum(x["p_holm_60"]<.05 for x in comparisons),
        "clean_nll":analysis["clean"]["metrics"]["mean_nll"],
        "clean_perplexity":analysis["clean"]["metrics"]["perplexity"],
        "failed_scored_tokens":sum(int(x["failed_tokens"]) for x in raw)})+b"\n")
    provenance = {"schema":"netsurf-pythia-w8a8-compact-handoff/v1",
        "source_run":str(RUN.resolve()),"frozen_plan":str(PLAN.resolve()),
        "original_complete_archive":str((CORE/"complete.zip").resolve()),
        "original_complete_archive_sha256":full_hash,
        "original_complete_archive_bytes":(CORE/"complete.zip").stat().st_size,
        "original_complete_archive_finalized_digest":(CORE/"FINALIZED").read_text().strip(),
        "original_reporting_v3_compact_sha256":reporting["compact_zip_sha256"],
        "plan_sha256":plan["plan_sha256"],"plan_file_sha256":sha256_file(PLAN/"plan.json"),
        "scientific_payload_sha256":payload["payload_sha256"],
        "analysis_file_sha256":sha256_file(RUN/"analysis.json"),
        "target_ready_record_sha256":plan["target_record_sha256"],
        "quantization_manifest_sha256":plan["quantization_manifest_sha256"],
        "eligible_population_bits":plan["eligible_universe"]["bits"],
        "eligible_population_sha256":plan["eligible_universe"]["sha256"],
        "omission":"Packed masks and full per-field JSON remain in the original complete archive. This compact export links to their original checksums but does not independently reverify omitted bytes."}
    (DEST/"provenance.json").write_bytes(canonical_json_bytes(provenance)+b"\n")
    (DEST/"README.md").write_text(
        "# Pythia-160M whole-qweight W8A8 compact scientific handoff\n\n"
        "This is a presentation and numerical audit of the finalized v3 campaign. "
        "It contains no model inference or newly selected masks. The original complete ZIP "
        "and its hash remain authoritative for omitted packed masks and field JSON.\n\n"
        "The unit of uncertainty is the independent whole-model fault field. "
        "All 26 policy evaluations within each field use the same proposed event set. "
        "The 60 Holm-adjusted comparisons in `paired_comparisons_original.csv` are copied "
        "from the original finalized analysis; condition-level bootstrap intervals are "
        "descriptive. Field failure fractions use 4096 scored targets per policy.\n\n"
        "The six rankings are random, MSB-first, numerical bit jump, global first-order, "
        "per-example absolute and per-example positive. FKeras and PrioriFI were unmeasured. "
        "Budget plots use categorical x spacing (0, 6.25, 12.5, 18.75, 25, 100 percent). "
        "Storage bits cover only the declared INT8 qweight universe, not every model weight.\n")
    inputs = {"plan/"+x:PLAN/x for x in ("FROZEN","plan.json","equivalence_audit.json",
            "target_ready_record.json","quantization_manifest.json")}
    inputs.update({"run/"+x:RUN/x for x in ("run_identity.json","scientific_payload.json",
                    "analysis.json","selection/seal.json")})
    inputs.update({"run/admission.json":RUN.with_name(RUN.name+"_admission.json"),
                   "run/runtime_receipt.json":RUN.with_name(RUN.name+"_runtime_receipt.json"),
                   "original_complete/checksums.json":CORE/"checksums.json",
                   "original_complete/FINALIZED":CORE/"FINALIZED",
                   "reporting_v3/receipt.json":REPORT/"receipt.json",
                   "reporting_v3/FINALIZED":REPORT/"FINALIZED"})
    inputs.update({"source/"+name:path for name,path in source_paths.items()})
    inputs["source/scripts/export_pythia_w8a8_scientific_handoff_v1.py"] = Path(__file__)
    for name in ("README.md","provenance.json","audit.json","costs.json","condition_level.csv",
                 "field_observations.csv","paired_comparisons_original.csv","mask_summary.csv",
                 "loss_degradation_vs_ber.svg","loss_degradation_vs_budget.svg"):
        inputs["handoff/"+name] = DEST/name
    hashes = {name:sha256_file(path) for name,path in inputs.items()}
    archive_path = DEST/"compact_scientific_evidence.zip"
    with zipfile.ZipFile(archive_path,"x",compression=zipfile.ZIP_DEFLATED,allowZip64=True) as archive:
        for name,path in sorted(inputs.items()):
            archive.write(path,name)
        archive.writestr("evidence_checksums.json",canonical_json_bytes(hashes)+b"\n")
    if archive_path.stat().st_size >= 20_000_000:
        raise ValueError("compact scientific export exceeded 20 MB; partial immutable output retained")
    import hashlib
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None or json.loads(archive.read("evidence_checksums.json")) != hashes:
            raise ValueError("compact ZIP integrity check failed")
        if any(hashlib.sha256(archive.read(name)).hexdigest()!=value for name,value in hashes.items()):
            raise ValueError("compact ZIP member SHA-256 check failed")
    zip_hash=sha256_file(archive_path)
    receipt={"schema":"netsurf-pythia-w8a8-compact-handoff-receipt/v1",
             "status":"VERIFIED_OFFLINE_EXPORT","zip_sha256":zip_hash,
             "zip_bytes":archive_path.stat().st_size,"source_scientific_payload_sha256":payload["payload_sha256"],
             "original_complete_zip_sha256":full_hash,"members":len(hashes)}
    (DEST/"receipt.json").write_bytes(canonical_json_bytes(receipt)+b"\n")
    (DEST/"FINALIZED").write_text(sha256_file(DEST/"receipt.json")+"\n")
    command=f"scp kona:{archive_path.resolve()} ~/Downloads/pythia160m_w8a8_compact_scientific_v1.zip"
    (DEST/"mac_scp_command.txt").write_text(command+"\n")
    return {**receipt,"archive":str(archive_path),"scp":command,"audit":audit,"costs":costs}


if __name__ == "__main__":
    result=create()
    print(json.dumps({k:result[k] for k in ("status","archive","zip_sha256","zip_bytes","source_scientific_payload_sha256","original_complete_zip_sha256","scp","audit","costs")},sort_keys=True))
