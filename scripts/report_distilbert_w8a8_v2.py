"""Offline, write-once DistilBERT W8A8 field CSVs, BER figure and ZIP export."""

from __future__ import annotations

import csv
from html import escape
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

from netsurf.manifests import canonical_json_bytes, sha256_file
from netsurf.w8a8_campaign import aggregate, export, validate, _read_bound


ROOT = Path(__file__).resolve().parent.parent
READY = Path("/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json")
COLORS = {"random": "#6b7280", "msb_first": "#2563eb",
          "numerical_bit_jump": "#d97706", "global_first_order": "#059669",
          "per_example_absolute": "#7c3aed", "per_example_positive": "#dc2626",
          "unprotected": "#111827", "full_protection": "#0f766e"}
METHODS = tuple(COLORS)
CONDITIONS = ("target", "plan_sha256", "ber", "expected_proposed_faults", "method",
              "protected_bits", "coverage", "independent_fields", "mean_cross_entropy",
              "mean_loss_degradation", "ci95_degradation_low", "ci95_degradation_high",
              "mean_accuracy", "failed_fraction", "mean_proposed_faults", "mean_applied_faults",
              "mean_suppressed_faults", "expected_applied_faults")
OBSERVATIONS = ("target", "plan_sha256", "ber", "expected_proposed_faults", "field_index",
                "seed", "field_sha256", "field_record", "method", "protected_bits", "coverage",
                "mask_sha256", "proposed_faults", "applied_faults", "suppressed_faults",
                "expected_applied_faults", "mean_cross_entropy", "accuracy", "failed_examples",
                "evaluated_examples", "logit_statistics_sha256")


def _method(policy: str) -> str:
    if policy in ("unprotected", "full_protection"):
        return policy
    name, marker, budget = policy.rpartition("_b")
    if not marker or not budget.isdecimal() or name not in COLORS:
        raise ValueError(f"unknown frozen W8A8 policy: {policy}")
    return name


def observations(plan: dict, run: Path) -> list[dict]:
    rows = []
    n = plan["eligible_universe"]["bits"]
    for arm, spec in enumerate(plan["fields"]):
        for field_index, seed in enumerate(spec["seeds"]):
            rel = Path(f"final/arm_{arm:02d}/field_{field_index:03d}.json")
            record = _read_bound(run / rel)
            if (record["plan_sha256"] != plan["plan_sha256"] or
                record["seed"] != seed or record["ber"] != spec["ber"]):
                raise ValueError("reporting field identity differs from frozen plan")
            for policy, item in sorted(record["policies"].items()):
                ev, risk = item["events"], item["risk"]
                if ev["field_sha256"] != record["field_sha256"] or \
                   ev["proposed"] != ev["applied"]+ev["suppressed"]:
                    raise ValueError("reporting fault accounting differs from field record")
                rows.append({"target": plan["target"], "plan_sha256": plan["plan_sha256"],
                             "ber": spec["ber"], "expected_proposed_faults": spec["lambda"],
                             "field_index": field_index, "seed": seed,
                             "field_sha256": record["field_sha256"], "field_record": str(rel),
                             "method": _method(policy), "protected_bits": item["budget_bits"],
                             "coverage": item["budget_bits"]/n, "mask_sha256": item["mask_sha256"],
                             "proposed_faults": ev["proposed"], "applied_faults": ev["applied"],
                             "suppressed_faults": ev["suppressed"],
                             "expected_applied_faults": item["expected_applied_faults"],
                             "mean_cross_entropy": risk["metrics"]["mean_nll"],
                             "accuracy": risk["metrics"].get("accuracy"),
                             "failed_examples": risk["failed_units"],
                             "evaluated_examples": risk["total_units"],
                             "logit_statistics_sha256": risk["logit_statistics_sha256"]})
    if len(rows) != plan["counts"]["final_model_requests"]:
        raise ValueError("field CSV would omit a prespecified evaluation")
    return rows


def conditions(plan: dict, rows: list[dict], clean_loss: float) -> list[dict]:
    groups: dict[tuple[float, str, int], list[dict]] = {}
    for row in rows:
        key = (row["ber"], row["method"], row["protected_bits"])
        groups.setdefault(key, []).append(row)
    output = []
    n = plan["eligible_universe"]["bits"]
    for index, ((ber, method, budget), group) in enumerate(sorted(groups.items())):
        if len(group) != plan["fields_per_ber"] or \
           len({x["field_index"] for x in group}) != len(group):
            raise ValueError("condition has duplicated or missing independent fields")
        values = np.asarray([x["mean_cross_entropy"] for x in group], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("field loss is nonfinite after frozen penalty")
        rng = np.random.default_rng(2204001 + index)
        boot = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)-clean_loss
        output.append({"target": plan["target"], "plan_sha256": plan["plan_sha256"],
                       "ber": ber, "expected_proposed_faults": group[0]["expected_proposed_faults"],
                       "method": method, "protected_bits": budget, "coverage": budget/n,
                       "independent_fields": len(group), "mean_cross_entropy": float(values.mean()),
                       "mean_loss_degradation": float(values.mean()-clean_loss),
                       "ci95_degradation_low": float(np.quantile(boot, .025)),
                       "ci95_degradation_high": float(np.quantile(boot, .975)),
                       "mean_accuracy": float(np.mean([x["accuracy"] for x in group])),
                       "failed_fraction": sum(x["failed_examples"] for x in group)/
                                          sum(x["evaluated_examples"] for x in group),
                       "mean_proposed_faults": float(np.mean([x["proposed_faults"] for x in group])),
                       "mean_applied_faults": float(np.mean([x["applied_faults"] for x in group])),
                       "mean_suppressed_faults": float(np.mean([x["suppressed_faults"] for x in group])),
                       "expected_applied_faults": group[0]["expected_applied_faults"]})
    if len(output) != 3*(6*4+2):
        raise ValueError("condition-level reliability family incomplete")
    return output


def _svg(rows: list[dict], plan: dict) -> str:
    width, height = 1250, 910
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#111827} .axis{stroke:#6b7280;stroke-width:1} .grid{stroke:#e5e7eb;stroke-width:1}</style>',
             '<text x="55" y="33" font-size="21" font-weight="bold">DistilBERT W8A8: loss degradation versus BER</text>',
             '<text x="55" y="54" font-size="13">Mean cross-entropy minus clean; 24 paired fault fields per BER; whiskers: field bootstrap 95% CI</text>']
    budgets = sorted(set(r["protected_bits"] for r in rows if r["method"] not in ("unprotected", "full_protection")))
    bers = sorted(set(r["ber"] for r in rows))
    if len(budgets) != 4 or len(bers) != 3:
        raise ValueError("SVG needs complete four-budget, three-BER grid")
    by_key = {(r["ber"], r["method"], r["protected_bits"]): r for r in rows}
    for panel, budget in enumerate(budgets):
        left = 70 + (panel % 2)*590
        top = 96 + (panel // 2)*360
        pw, ph = 500, 265
        members = [by_key[(b, method, budget)] for b in bers for method in METHODS[:6]]
        members += [by_key[(b, control, 0 if control == "unprotected" else plan["eligible_universe"]["bits"])]
                    for b in bers for control in ("unprotected", "full_protection")]
        ymin = min(0., *(r["ci95_degradation_low"] for r in members))
        ymax = max(0., *(r["ci95_degradation_high"] for r in members))
        pad = max(.005, .08*(ymax-ymin))
        ymin -= pad; ymax += pad
        y = lambda value: top+ph-(value-ymin)/(ymax-ymin)*ph
        x = lambda ber: left+(math.log10(ber)-math.log10(bers[0]))/\
            (math.log10(bers[-1])-math.log10(bers[0]))*pw
        parts.append(f'<text x="{left}" y="{top-13}" font-size="16" font-weight="bold">{100*budget/plan["eligible_universe"]["bits"]:.2f}% protected</text>')
        for tick in range(5):
            value = ymin+(ymax-ymin)*tick/4
            yy = y(value)
            parts.append(f'<line class="grid" x1="{left}" x2="{left+pw}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
            parts.append(f'<text x="{left-8}" y="{yy+4:.1f}" font-size="11" text-anchor="end">{value:.3g}</text>')
        parts.append(f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{top+ph}"/>')
        parts.append(f'<line class="axis" x1="{left}" x2="{left+pw}" y1="{top+ph}" y2="{top+ph}"/>')
        for b in bers:
            xx=x(b)
            parts.append(f'<text x="{xx:.1f}" y="{top+ph+19}" font-size="11" text-anchor="middle">{b:.2e}</text>')
        for method in METHODS:
            target_budget = budget if method not in ("unprotected", "full_protection") else \
                (0 if method == "unprotected" else plan["eligible_universe"]["bits"])
            series = [by_key[(b, method, target_budget)] for b in bers]
            color = COLORS[method]
            points = " ".join(f'{x(r["ber"]):.1f},{y(r["mean_loss_degradation"]):.1f}' for r in series)
            dash = ' stroke-dasharray="5,4"' if method in ("unprotected", "full_protection") else ""
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
            for r in series:
                xx=x(r["ber"])
                parts.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{y(r["ci95_degradation_low"]):.1f}" y2="{y(r["ci95_degradation_high"]):.1f}" stroke="{color}" stroke-width="1"/>')
                parts.append(f'<circle cx="{xx:.1f}" cy="{y(r["mean_loss_degradation"]):.1f}" r="3" fill="{color}"/>')
    for i, method in enumerate(METHODS):
        x0 = 50 + (i % 4)*300; y0 = 845 + (i//4)*24
        parts.append(f'<line x1="{x0}" x2="{x0+22}" y1="{y0}" y2="{y0}" stroke="{COLORS[method]}" stroke-width="3"/>')
        parts.append(f'<text x="{x0+30}" y="{y0+4}" font-size="12">{escape(method.replace("_", " "))}</text>')
    parts.append('</svg>')
    return "\n".join(parts)+"\n"


def _csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def report(plan_dir: Path, run: Path) -> dict:
    plan, _ = validate(READY, plan_dir)
    summary = aggregate(READY, plan_dir, run)
    full = export(READY, plan_dir, run)
    if summary["fields"] != 72 or full["status"] not in ("FINALIZED", "VERIFIED_FINALIZED_REUSED"):
        raise ValueError("scientific campaign incomplete before reporting")
    analysis = _read_bound(run / "analysis.json")
    if analysis["plan_sha256"] != plan["plan_sha256"] or len(analysis["field_rows"]) != 72:
        raise ValueError("saved field-level analysis differs from frozen plan")
    all_rows = observations(plan, run)
    clean = analysis["clean"]["metrics"]["mean_nll"]
    grouped = conditions(plan, all_rows, clean)
    destination = run.with_name(run.name + "_reporting_v2")
    if destination.exists():
        raise ValueError("write-once reporting output already exists")
    destination.mkdir(parents=True, exist_ok=False)
    _csv(destination / "condition_level.csv", CONDITIONS, grouped)
    _csv(destination / "field_observations.csv", OBSERVATIONS, all_rows)
    (destination / "loss_degradation_vs_ber.svg").write_text(_svg(grouped, plan))
    inputs = {"plan/plan.json": plan_dir / "plan.json",
              "plan/equivalence_audit.json": plan_dir / "equivalence_audit.json",
              "run/analysis.json": run / "analysis.json",
              "run/scientific_payload.json": run / "scientific_payload.json",
              "run/admission.json": run.with_name(run.name + "_admission.json"),
              "run/stages.json": run.with_name(run.name + "_stages.json"),
              "core_export/checksums.json": run.with_name(run.name + "_export") / "checksums.json",
              "core_export/FINALIZED": run.with_name(run.name + "_export") / "FINALIZED",
              "report/condition_level.csv": destination / "condition_level.csv",
              "report/field_observations.csv": destination / "field_observations.csv",
              "report/loss_degradation_vs_ber.svg": destination / "loss_degradation_vs_ber.svg"}
    for name, expected in plan["source_files_sha256"].items():
        path = ROOT / name
        if sha256_file(path) != expected:
            raise ValueError(f"bound reporting or campaign source changed: {name}")
        inputs["source/"+name] = path
    hashes = {name: sha256_file(path) for name, path in inputs.items()}
    with zipfile.ZipFile(destination / "compact.zip", "x", compression=zipfile.ZIP_DEFLATED,
                         allowZip64=True) as archive:
        for name, path in inputs.items():
            archive.write(path, name)
        archive.writestr("evidence_checksums.json", canonical_json_bytes(hashes)+b"\n")
    with zipfile.ZipFile(destination / "compact.zip") as archive:
        if archive.testzip() is not None:
            raise ValueError("compact ZIP CRC verification failed")
    full_inputs = {"run/"+str(path.relative_to(run)): path for path in sorted(run.rglob("*"))
                   if path.is_file()}
    full_inputs.update({"plan/"+str(path.relative_to(plan_dir)): path
                        for path in sorted(plan_dir.rglob("*")) if path.is_file()})
    full_inputs.update(inputs)
    full_hashes = {name: sha256_file(path) for name, path in full_inputs.items()}
    with zipfile.ZipFile(destination / "complete_v2.zip", "x", allowZip64=True) as archive:
        for name, path in full_inputs.items():
            archive.write(path, name, compress_type=zipfile.ZIP_STORED if path.suffix in (".bits", ".f32")
                          else zipfile.ZIP_DEFLATED)
        archive.writestr("evidence_checksums.json", canonical_json_bytes(full_hashes)+b"\n")
    with zipfile.ZipFile(destination / "complete_v2.zip") as archive:
        if archive.testzip() is not None:
            raise ValueError("complete v2 ZIP CRC verification failed")
    receipt = {"schema": "netsurf-w8a8-distilbert-reporting/v2",
               "plan_sha256": plan["plan_sha256"], "core_zip_sha256": full["zip_sha256"],
               "full_zip_sha256": sha256_file(destination / "complete_v2.zip"),
               "compact_zip_sha256": sha256_file(destination / "compact.zip"),
               "condition_count": len(grouped), "field_observation_count": len(all_rows),
               "compact_member_sha256": hashes, "full_member_sha256": full_hashes}
    (destination / "receipt.json").write_bytes(canonical_json_bytes(receipt)+b"\n")
    (destination / "FINALIZED").write_text(sha256_file(destination / "receipt.json")+"\n")
    commands = [
        f"scp kona:{(destination / 'complete_v2.zip').resolve()} ~/Downloads/{plan['target']}_w8a8_complete_v2.zip",
        f"scp kona:{(destination / 'compact.zip').resolve()} ~/Downloads/{plan['target']}_w8a8_compact_v2.zip",
        f"scp kona:{(destination / 'receipt.json').resolve()} ~/Downloads/{plan['target']}_w8a8_export_receipt_v2.json",
        f"scp kona:{run.with_name(run.name + '_runtime_receipt.json').resolve()} ~/Downloads/{plan['target']}_w8a8_runtime_receipt_v2.json",
    ]
    (destination / "mac_scp_commands.txt").write_text("\n".join(commands)+"\n")
    return {"status": "FINALIZED_OFFLINE_REPORT", "plan_sha256": plan["plan_sha256"],
            "conditions": len(grouped), "field_observations": len(all_rows),
            "full_zip_sha256": receipt["full_zip_sha256"], "compact_zip_sha256": receipt["compact_zip_sha256"],
            "reporting_receipt_sha256": sha256_file(destination / "receipt.json"),
            "reporting_dir": str(destination)}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: report_distilbert_w8a8_v2.py PLAN_DIR RUN_DIR")
    print(json.dumps(report(Path(sys.argv[1]), Path(sys.argv[2])), sort_keys=True))
