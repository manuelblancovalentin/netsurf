"""Build the current 552-cell INT8 reliability product from finalized sources only.

Inference, ranking, and fault scheduling are intentionally absent. The three
source studies remain separate; the historical 280-cell primary view is copied
unchanged and never reclassified as the current comparative cohort.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sqlite3
import tempfile
import zipfile
from pathlib import Path

from netsurf.manifests import sha256_file
from netsurf.reliability_database import METHODS, _verified_bundle
from .build_reliability_database_v3 import build_product

SCHEMA = "m12-current-comparative/2.0.0"
STUDIES = (
    ("resnet8", "ResNet-8 / CIFAR-10", "resnet8_partial_tier_matched_v1_followup", 208),
    ("kws", "DS-CNN / Speech Commands", "kws_partial_tier_fkeras_matched_v1_followup", 240),
    ("toycar", "ToyCar / anomaly detection", "toycar_partial_tier_matched_v1_followup", 104),
)
COLORS = {
    "random_global": "#7f7f7f", "msb_global": "#1f77b4",
    "numerical_jump_global": "#ff7f0e", "gradient_global": "#2ca02c",
    "per_example_absolute_global": "#d62728",
    "per_example_positive_global": "#9467bd", "fkeras_paper": "#8c564b",
}
BUDGETS = (0.0625, 0.125, 0.1875, 0.25)


def _export_query(db: sqlite3.Connection, target: Path, query: str) -> int:
    with target.open("x", newline="") as file:
        cursor = db.execute(query)
        writer = csv.writer(file)
        writer.writerow([column[0] for column in cursor.description])
        count = 0
        for row in cursor:
            writer.writerow(row)
            count += 1
    return count


def _current_views(source_db: Path, target_db: Path, selected: dict[str, dict]) -> None:
    with sqlite3.connect(f"file:{source_db}?mode=ro", uri=True) as source, sqlite3.connect(target_db) as target:
        source.backup(target)
    with sqlite3.connect(target_db) as db:
        if db.execute("SELECT count(*) FROM primary_conditions").fetchone()[0] != 280:
            raise ValueError("historical primary view is not the verified 280-row table")
        db.execute("INSERT INTO metadata VALUES (?,?)", ("current_comparative_schema", SCHEMA))
        db.execute("""CREATE TABLE current_study_registry (
            model_key TEXT PRIMARY KEY, model_name TEXT NOT NULL,
            study_id TEXT NOT NULL UNIQUE REFERENCES studies(study_id),
            source_payload_sha256 TEXT NOT NULL, source_checksums_sha256 TEXT NOT NULL,
            expected_conditions INTEGER NOT NULL, task_loss_name TEXT NOT NULL,
            quality_name TEXT NOT NULL, eligible_bits INTEGER NOT NULL,
            evaluation_cohort_json TEXT NOT NULL)""")
        for model_key, label, sid, expected in STUDIES:
            source = selected[sid]
            row = db.execute("""SELECT s.source_payload_sha256,s.source_checksums_sha256,
                m.task_loss_name,m.quality_name,m.eligible_bits,m.evaluation_cohort_json
                FROM studies s JOIN models m USING(model_id) WHERE s.study_id=?""", (sid,)).fetchone()
            if row is None or row[:2] != (source["scientific_payload_sha256"], source["checksums_sha256"]):
                raise ValueError(f"source identities differ for {sid}")
            count = db.execute("SELECT count(*) FROM aggregated_conditions WHERE study_id=?", (sid,)).fetchone()[0]
            if count != expected:
                raise ValueError(f"current condition count differs for {sid}: {count} != {expected}")
            db.execute("INSERT INTO current_study_registry VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (model_key, label, sid, row[0], row[1], expected, *row[2:]))
        db.execute("""CREATE VIEW current_comparative_conditions_v2 AS
            SELECT g.* FROM reliability_grid g JOIN current_study_registry r USING(study_id)
            JOIN studies s USING(study_id)
            WHERE s.source_payload_sha256=r.source_payload_sha256
              AND s.source_checksums_sha256=r.source_checksums_sha256""")
        db.execute("CREATE VIEW current_comparative_conditions AS SELECT * FROM current_comparative_conditions_v2")
        db.execute("""CREATE TABLE current_method_coverage (
            model_key TEXT NOT NULL REFERENCES current_study_registry(model_key),
            method_id TEXT NOT NULL REFERENCES methods(method_id),
            status TEXT NOT NULL, measured_conditions INTEGER NOT NULL,
            ber_values_json TEXT NOT NULL, budget_values_json TEXT NOT NULL,
            reason TEXT NOT NULL, PRIMARY KEY(model_key,method_id))""")
        for model_key, _, sid, _ in STUDIES:
            for method in METHODS:
                cells = db.execute("""SELECT ber,coverage FROM current_comparative_conditions
                    WHERE study_id=? AND method_id=? ORDER BY ber,coverage""", (sid, method)).fetchall()
                if cells:
                    status, reason = "FINALIZED_MEASURED", "Measured in this model's selected matched study"
                elif method == "fkeras_repository":
                    status, reason = "INCOMPATIBLE", "The public-source variant is archival; only published-form FKeras is eligible"
                elif method == "exponent_first":
                    status, reason = "INCOMPATIBLE", "INT8 stored kernels have no exponent bits"
                elif method == "priorifi_derived":
                    status, reason = "NOT_MEASURED", "No validated fixed-budget PrioriFI-derived mask exists"
                elif method == "fkeras_paper":
                    status, reason = "NOT_MEASURED", "Published-form FKeras is not measured on this matched study"
                elif method == "empirical_isolated":
                    status, reason = "NOT_MEASURED", "Sampled isolated-bit effects do not define a whole-kernel mask"
                else:
                    status, reason = "NOT_MEASURED", "No finalized matched condition in this study"
                db.execute("INSERT INTO current_method_coverage VALUES (?,?,?,?,?,?,?)",
                           (model_key, method, status, len(cells),
                            json.dumps(sorted({row[0] for row in cells})),
                            json.dumps(sorted({row[1] for row in cells})), reason))
        if db.execute("SELECT count(*) FROM current_comparative_conditions").fetchone()[0] != sum(item[3] for item in STUDIES):
            raise ValueError("current comparative view differs from the selected measured conditions")
        db.commit()


def _plots(database: Path, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir()
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        for key, label, study, _ in STUDIES:
            rows = [dict(row) for row in db.execute(
                "SELECT * FROM current_comparative_conditions WHERE study_id=?", (study,))]
            index = {(row["method_id"], row["coverage"], row["ber"]): row for row in rows}
            bers = sorted({row["ber"] for row in rows})
            methods = [method for method in COLORS if any(row["method_id"] == method for row in rows)]
            unit = ("Cross-entropy increase from clean (nats)" if key != "toycar"
                    else "Increase in 1 − complete-file ROC-AUC")
            count = min(row["field_count"] for row in rows)

            fig, axes = plt.subplots(1, 4, figsize=(17, 4.5), sharey=True, layout="constrained")
            for ax, budget in zip(axes, BUDGETS):
                ax.plot(bers, [index[("no_protection", 0.0, ber)]["loss_degradation"] for ber in bers],
                        "k--", linewidth=1.2, label="Unprotected")
                ax.axhline(0, color="black", linestyle=":", linewidth=1, label="Fully protected / clean")
                for method in methods:
                    if key == "kws" and method == "fkeras_paper" and budget in (0.125, 0.25):
                        continue  # exact same protected set and outcomes as MSB-first
                    points = [index.get((method, budget, ber)) for ber in bers]
                    points = [point for point in points if point is not None]
                    if not points:
                        continue
                    ax.errorbar([row["ber"] for row in points], [row["loss_degradation"] for row in points],
                                yerr=[[max(0, row["mean_loss"] - row["loss_ci_low"]) for row in points],
                                      [max(0, row["loss_ci_high"] - row["mean_loss"]) for row in points]],
                                color=COLORS[method], marker="o", markersize=3, capsize=2,
                                linewidth=1, label=METHODS[method][0])
                ax.set_xscale("log")
                if key != "toycar":
                    ax.set_yscale("symlog", linthresh=0.05)
                ax.set_title(f"{budget:.2%} protected")
                ax.set_xlabel("Logical bit-flip probability")
                ax.grid(alpha=0.2)
            axes[0].set_ylabel(unit)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
            fig.suptitle(f"{label}: task-loss degradation versus BER ({count} paired fields per BER)")
            fig.savefig(output / f"{key}_loss_vs_ber.png", dpi=180)
            plt.close(fig)

            displayed_bers = [bers[i] for i in sorted({0, len(bers)//3, 2*len(bers)//3, len(bers)-1})]
            fig, axes = plt.subplots(1, len(displayed_bers), figsize=(4.5 * len(displayed_bers), 4.5),
                                     sharey=True, layout="constrained", squeeze=False)
            for ax, ber in zip(axes[0], displayed_bers):
                baseline = index[("no_protection", 0.0, ber)]["loss_degradation"]
                ax.axhline(baseline, color="black", linestyle="--", linewidth=1.2, label="Unprotected")
                ax.axhline(0, color="black", linestyle=":", linewidth=1, label="Fully protected / clean")
                for method in methods:
                    points = [index.get((method, budget, ber)) for budget in BUDGETS
                              if not (key == "kws" and method == "fkeras_paper" and budget in (0.125, 0.25))]
                    points = [point for point in points if point is not None]
                    if not points:
                        continue
                    ax.errorbar([row["coverage"] for row in points],
                                [row["loss_degradation"] for row in points],
                                yerr=[[max(0, row["mean_loss"] - row["loss_ci_low"]) for row in points],
                                      [max(0, row["loss_ci_high"] - row["mean_loss"]) for row in points]],
                                color=COLORS[method], marker="o", markersize=3, capsize=2,
                                linewidth=1, linestyle="none" if key == "kws" and method == "fkeras_paper" else "-",
                                label=METHODS[method][0])
                ax.set_title(f"Logical BER {ber:g}")
                if key != "toycar":
                    ax.set_yscale("symlog", linthresh=0.05)
                ax.set_xticks(BUDGETS, [f"{budget:.2%}" for budget in BUDGETS], rotation=30)
                ax.set_xlabel("Protected eligible-bit fraction")
                ax.grid(alpha=0.2)
            axes[0][0].set_ylabel(unit)
            handles, labels = axes[0][0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
            fig.suptitle(f"{label}: task-loss degradation versus budget ({count} paired fields per BER)")
            fig.savefig(output / f"{key}_loss_vs_budget.png", dpi=180)
            plt.close(fig)


def _readme(database: Path, output: Path, sources: dict) -> None:
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
        matrix = {(model, method): (status, count) for model, method, status, count in db.execute(
            "SELECT model_key,method_id,status,measured_conditions FROM current_method_coverage")}
    lines = ["# Current matched whole-kernel INT8 reliability comparison", "",
             "This versioned view selects one separate, finalized matched study per model. It contains "
             "552 measured conditions: ResNet-8 208, keyword spotting 240, ToyCar 104. "
             "The historical 280-row `primary_conditions` view is unchanged. Different model task losses "
             "are not pooled or compared in common units.", "",
             "| Model | Selected source study | Conditions | Risk |", "|---|---|---:|---|",
             "| ResNet-8 / CIFAR-10 | `resnet8_partial_tier_matched_v1_followup` | 208 | Categorical cross-entropy |",
             "| DS-CNN / Speech Commands | `kws_partial_tier_fkeras_matched_v1_followup` | 240 | Categorical cross-entropy |",
             "| ToyCar / anomaly detection | `toycar_partial_tier_matched_v1_followup` | 104 | 1 − complete-file ROC-AUC |",
             "", "## Method coverage", "", "| Method | ResNet-8 | KWS | ToyCar |", "|---|---|---|---|"]
    for method, (label, *_rest) in METHODS.items():
        if method == "fkeras_repository":
            continue
        cells = [f"{matrix[(model, method)][0]} ({matrix[(model, method)][1]})" if matrix[(model, method)][1]
                 else matrix[(model, method)][0] for model in ("resnet8", "kws", "toycar")]
        lines.append(f"| {label} | {' | '.join(cells)} |")
    lines += ["", "FKeras means the **published mathematical form** only. ResNet-8 and ToyCar "
              "FKeras and PrioriFI-derived allocations remain unmeasured; PrioriFI is also unmeasured "
              "for KWS. No method is filled from a different cohort. The KWS 128-clip FKeras cohort "
              "remains separate. At KWS 12.5% and 25%, published-form FKeras and MSB-first select "
              "identical full bit tiers, so the duplicated FKeras curve points are omitted from plots. "
              "Source trial observations and exact mask identities remain in the "
              "checksum-verified source bundles.", "",
              "Plots use source-derived 95% intervals over the pooled scheduled fields. A method may "
              "reuse one ranking mask or vary masks across replications; inspect `ranking_mask_count` and "
              "the source replication-level tables before interpreting uncertainty. These are not "
              "independent training replicates. Paired policies share proposed fields, so marginal intervals do not "
              "constitute paired significance tests. Classification loss axes use a labeled symmetric-log "
              "scale above 0.05 nats so high-BER tail values remain visible; ToyCar risk uses a linear axis. "
              "Logical BER is not radiation-calibrated and "
              "ideal event suppression is not physical ECC.", "",
              "The source receipt, artifact hashes, complete queryable SQLite database and long-form "
              "CSV are included. Full source-bundle ZIPs are exported by each campaign recovery.", ""]
    (output / "CURRENT_README.md").write_text("\n".join(lines))


def _sources(base: Path, recovered: tuple[Path, Path], target: Path) -> dict[str, dict]:
    document = json.loads(base.read_text())
    known = {item["id"] for item in document["studies"]}
    selected = {item["id"]: item for item in document["studies"]}
    for path in recovered:
        result = json.loads((path / "recovery_audit_v1.json").read_text())
        receipt = json.loads((path / "source_receipt.json").read_text())
        new = [item for item in receipt["studies"] if item["id"] not in known]
        if result["status"] != "FINALIZED_OFFLINE" or len(new) != 1:
            raise ValueError("recovered source receipt is incomplete or ambiguous")
        item = new[0]
        if (item["scientific_payload_sha256"] != result["scientific_payload_sha256"] or
                item["checksums_sha256"] != result["source_checksums_sha256"]):
            raise ValueError("recovery receipt and source bundle identities differ")
        _verified_bundle(item)
        document["studies"].append(item)
        selected[item["id"]] = item
        known.add(item["id"])
    if {sid for _, _, sid, _ in STUDIES} - set(selected):
        raise ValueError("one of the three selected studies is absent")
    with target.open("x") as file:
        json.dump(document, file, indent=2, sort_keys=True)
        file.write("\n")
    return selected


def build(base_sources: Path, resnet: Path, toycar: Path, assets: Path,
          baseline: Path, output: Path) -> dict:
    base_sources, resnet, toycar, assets, baseline, output = (
        path.resolve() for path in (base_sources, resnet, toycar, assets, baseline, output)
    )
    if output.exists():
        raise FileExistsError(f"write-once current comparative product already exists: {output}")
    historical = baseline / "primary_conditions.csv"
    checks = json.loads((baseline / "checksums.json").read_text())
    if sha256_file(historical) != checks["files"]["primary_conditions.csv"]:
        raise ValueError("historical primary source is not checksum-verified")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".m12-current-stage-", dir=output.parent))
    candidate = stage / output.name
    candidate.mkdir()
    try:
        selected = _sources(base_sources, (resnet, toycar), candidate / "source_receipt.json")
        build_product(candidate / "source_receipt.json", assets, candidate / "source_database")
        if sha256_file(candidate / "source_database/primary_conditions.csv") != sha256_file(historical):
            raise ValueError("historical 280-row primary CSV changed")
        database = candidate / "current_comparative.sqlite"
        _current_views(candidate / "source_database/reliability_v3.sqlite", database, selected)
        with sqlite3.connect(database) as db:
            if _export_query(db, candidate / "current_comparative_conditions.csv",
                             "SELECT * FROM current_comparative_conditions ORDER BY study_id,ber,coverage,method_id") != 552:
                raise ValueError("current comparative CSV is incomplete")
            _export_query(db, candidate / "current_study_registry.csv", "SELECT * FROM current_study_registry ORDER BY model_key")
            _export_query(db, candidate / "current_method_coverage.csv", "SELECT * FROM current_method_coverage ORDER BY method_id,model_key")
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("current SQLite integrity check failed")
        shutil.copyfile(historical, candidate / "historical_primary_conditions.csv")
        _plots(database, candidate / "plots")
        _readme(database, candidate, selected)
        (candidate / "provenance.json").write_text(json.dumps({
            "schema_version": SCHEMA, "selected_sources": selected,
            "historical_primary_csv_sha256": sha256_file(historical),
            "asset_manifest_sha256": sha256_file(assets),
            "source_receipt_sha256": sha256_file(candidate / "source_receipt.json"),
            "model_inferences_executed": 0,
        }, indent=2, sort_keys=True) + "\n")
        files = {str(path.relative_to(candidate)): sha256_file(path)
                 for path in sorted(candidate.rglob("*")) if path.is_file()}
        (candidate / "checksums.json").write_text(json.dumps({"schema_version": SCHEMA, "files": files},
                                                        indent=2, sort_keys=True) + "\n")
        (candidate / "FINALIZED").write_text(f"database_sha256={files['current_comparative.sqlite']}\n")
        if output.exists():
            raise FileExistsError(output)
        os.rename(candidate, output)
        stage.rmdir()
    except Exception as error:
        raise RuntimeError(f"current-comparative staging retained for inspection: {stage}") from error
    return {"product": str(output), "current_conditions": 552,
            "historical_primary_conditions": 280,
            "database_sha256": sha256_file(output / "current_comparative.sqlite")}


def export(product: Path, destination: Path) -> dict:
    product, destination = product.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    checks = json.loads((product / "checksums.json").read_text())
    if checks["schema_version"] != SCHEMA or (product / "FINALIZED").read_text().strip() != f"database_sha256={checks['files']['current_comparative.sqlite']}":
        raise ValueError("current product is not finalized")
    for relative, digest in checks["files"].items():
        if sha256_file(product / relative) != digest:
            raise ValueError(f"current product file changed: {relative}")
    destination.mkdir(parents=True, exist_ok=False)
    records = {}
    for flavor in ("compact", "complete"):
        target = destination / f"m12_current_comparative_v2_{flavor}.zip"
        with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(product.rglob("*")):
                if path.is_file() and (flavor == "complete" or path.name not in {"reliability_v3.sqlite", "current_comparative.sqlite"}):
                    archive.write(path, arcname=str(Path(product.name) / path.relative_to(product)))
        with zipfile.ZipFile(target) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"archive failed integrity test: {target}")
        records[flavor] = {"path": str(target), "sha256": sha256_file(target), "bytes": target.stat().st_size}
    (destination / "export_receipt.json").write_text(json.dumps({"schema_version": "1.0.0",
        "database_sha256": checks["files"]["current_comparative.sqlite"], "archives": records},
        indent=2, sort_keys=True) + "\n")
    (destination / "mac_scp_commands.txt").write_text("\n".join(
        f"scp kona:{record['path']} ~/Downloads/" for record in records.values()) + "\n")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sources", type=Path, default=Path("workspace/m12_kws_seven_method_followup/source_receipt.json"))
    parser.add_argument("--resnet-recovery", type=Path, required=True)
    parser.add_argument("--toycar-recovery", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=Path("/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json"))
    parser.add_argument("--baseline", type=Path, default=Path("workspace/m12_kws_seven_method_followup/derived_database_recovered_v2"))
    parser.add_argument("--output", type=Path, default=Path("workspace/m12_current_comparative_v2"))
    args = parser.parse_args()
    result = build(args.base_sources, args.resnet_recovery, args.toycar_recovery,
                   args.assets, args.baseline, args.output)
    result["archives"] = export(args.output, args.output.with_name(args.output.name + "_export"))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
