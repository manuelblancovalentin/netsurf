"""Build the M12 read-only reliability product with verified W8A8 asset gates."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path

from netsurf.manifests import sha256_file
from netsurf.reliability_database import METHODS, PRIMARY_STUDIES, export_csv
from netsurf.reliability_database_v3 import SCHEMA_VERSION, build_database_v3
from .build_reliability_database import (
    COLORS, MODEL_LABELS, INTERIOR_BUDGETS, freeze_primary_plans,
    primary_dashboard, primary_plots, primary_readme, plots,
)


def measured_primary_plots(database: Path, directory: Path) -> None:
    """Display only measured interior budgets; never interpolate absent cells."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=False)
    with sqlite3.connect(database) as db:
        db.row_factory = sqlite3.Row
        names = dict(db.execute("SELECT method_id,display_name FROM methods"))
        for sid in MODEL_LABELS:
            rows = list(db.execute("SELECT * FROM primary_conditions WHERE study_id=?", (sid,)))
            if not rows:
                continue
            budgets = [b for b in INTERIOR_BUDGETS if any(abs(r["coverage"]-b)<1e-10 and r["method_id"] not in ("no_protection","full_protection") for r in rows)]
            fig, axes = plt.subplots(1, len(budgets), figsize=(5.2*len(budgets), 4.4), sharey=True, squeeze=False, constrained_layout=True)
            for ax, budget in zip(axes[0], budgets):
                for method in sorted({r["method_id"] for r in rows if abs(r["coverage"]-budget)<1e-10 and r["method_id"] not in ("no_protection","full_protection","fkeras_repository")}):
                    points = sorted((r for r in rows if r["method_id"]==method and abs(r["coverage"]-budget)<1e-10), key=lambda r:r["ber"])
                    ax.errorbar([r["ber"] for r in points], [r["loss_degradation"] for r in points],
                                yerr=[[r["mean_loss"]-r["loss_ci_low"] for r in points], [r["loss_ci_high"]-r["mean_loss"] for r in points]],
                                marker="o", markersize=3, capsize=2, linewidth=1, color=COLORS.get(method), label=names[method])
                for method, style, label in (("no_protection", "--", "Unprotected"), ("full_protection", ":", "Full protection")):
                    points = sorted((r for r in rows if r["method_id"]==method), key=lambda r:r["ber"])
                    ax.plot([r["ber"] for r in points], [r["loss_degradation"] for r in points],
                            linestyle=style, color="black", linewidth=1.2, label=label)
                ax.set_xscale("log")
                ax.set_xlabel("Logical bit-flip probability")
                ax.set_title(f"{budget:.2%} protected")
                ax.grid(alpha=.2)
            metric = "cross-entropy increase (nats)" if rows[0]["task_loss_name"]=="categorical_crossentropy" else "increase in 1 − file ROC-AUC"
            axes[0][0].set_ylabel(metric)
            handles={label: handle for ax in axes[0] for handle,label in zip(*ax.get_legend_handles_labels())}
            fig.legend(list(handles.values()), list(handles), loc="upper center", bbox_to_anchor=(.5,1.13), ncol=4, fontsize=8)
            fig.suptitle(f"{MODEL_LABELS[sid]} | measured cells only; 95% field-level CI", y=1.24, fontsize=10)
            fig.savefig(directory/f"{sid}_loss_vs_ber.png", dpi=180, bbox_inches="tight")
            plt.close(fig)


def expanded_method_dashboard(database: Path, output: Path) -> None:
    """One explicit row per method × primary or READY-asset target."""
    targets = tuple(PRIMARY_STUDIES) + (
        "distilbert_sst2_int8_full", "pythia160m_int8_full", "qwen3_0_6b_int8_full",
    )
    records=[]
    with sqlite3.connect(database) as db:
        for model in targets:
            primary=model in PRIMARY_STUDIES
            asset=("FINALIZED_SOURCE" if primary else "READY")
            adapter=("VALIDATED" if primary else "NOT_VALIDATED")
            for method in METHODS:
                if method=="fkeras_repository":
                    continue
                measured=(db.execute("SELECT count(*) FROM primary_conditions WHERE study_id=? AND method_id=?",(model,method)).fetchone()[0] if primary else 0)
                planned=(db.execute("SELECT status,reason FROM planned_model_methods WHERE target_id=? AND method_id=?",(model,method)).fetchone() if not primary else None)
                if measured:
                    status,reason="MEASURED",f"{measured} primary-cohort BER/budget cells"
                elif planned:
                    status,reason=planned
                elif method=="exponent_first":
                    status,reason="INCOMPATIBLE","INT8 has no exponent field"
                elif method=="priorifi_derived":
                    status,reason="BLOCKED","feedback scheduler implemented; full-universe protection conversion unvalidated"
                elif method=="fkeras_paper" and model!="kws_speechcommands_int8_full":
                    status,reason="BLOCKED","no validated architecture-specific continuous HVP"
                else:
                    status,reason="MISSING","no matched primary-cohort protection measurement"
                records.append((model,method,asset,adapter,status,measured,reason))
    with (output/"method_model_readiness.csv").open("x",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(("model_id","method_id","asset_status","netsurf_adapter_status","primary_method_status","measured_primary_cells","reason"))
        writer.writerows(records)
    lines=["# Method × model readiness", "",
           "READY means hash-verified source assets, not a validated NetSuRF fault adapter or measured protection condition.", "",
           "| Method | ResNet-8 | KWS | ToyCar | DistilBERT W8A8 | Pythia W8A8 | Qwen W8A8 |",
           "|---|---|---|---|---|---|---|"]
    for method in METHODS:
        if method=="fkeras_repository":
            continue
        cells=[]
        for model in targets:
            row=next(r for r in records if r[0]==model and r[1]==method)
            cells.append(row[4].lower().replace("_"," "))
        lines.append(f"| {METHODS[method][0]} | {' | '.join(cells)} |")
    lines.extend(("", "The 128-clip FKeras result is a separate cohort and therefore does not fill the KWS primary column. All W8A8 cells are unmeasured; the three source assets are READY but NetSuRF adapters are not validated.", ""))
    (output/"M12_METHOD_MATRIX.md").write_text("\n".join(lines))


def build_product(sources: Path, asset_manifest: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    db = output/"reliability_v3.sqlite"
    build_database_v3(sources, db, asset_manifest)
    for table, filename in (
        ("primary_conditions", "primary_conditions.csv"),
        ("primary_coverage", "primary_coverage.csv"),
        ("reliability_grid", "all_conditions.csv"),
        ("fkeras_cohort_conditions", "fkeras_cohort_conditions.csv"),
        ("engineering_conditions", "engineering_conditions.csv"),
        ("restricted_grid", "restricted_conditions_long.csv"),
        ("planned_model_methods", "planned_model_methods.csv"),
    ):
        export_csv(db, output/filename, table)
    with sqlite3.connect(db) as connection, (output/"asset_readiness.csv").open("x", newline="") as file:
        cursor = connection.execute("SELECT target_id,model_id,asset_status,adapter_status,experiment_status,checkpoint_sha256,quantization_manifest_sha256,eligible_bits,backend_contract,clean_metric_json FROM asset_readiness ORDER BY target_id")
        writer = csv.writer(file)
        writer.writerow([column[0] for column in cursor.description])
        writer.writerows(cursor)
    with sqlite3.connect(db) as connection, (output/"primary_condition_context.csv").open("x", newline="") as file:
        cursor = connection.execute("SELECT * FROM primary_condition_context ORDER BY study_id,ber,coverage,method_id")
        writer = csv.writer(file)
        writer.writerow([column[0] for column in cursor.description])
        writer.writerows(cursor)
    measured_primary_plots(db, output/"plots_measured_only")
    primary_plots(db, output/"plots_target_matrix")
    if any(item["kind"]=="whole_kernel_followup" for item in json.loads(sources.read_text())["studies"]):
        plots(db, output/"plots_all_studies")
    primary_dashboard(db, output)
    expanded_method_dashboard(db,output)
    primary_readme(db, output)
    freeze_primary_plans(sources, output)
    with (output/"PRIMARY_README.md").open("a") as file:
        file.write("\n## M12 asset gate\n\nThe three W8A8 transformer assets are hash-verified READY, but their NetSuRF full-universe adapters and protection experiments are NOT VALIDATED/NOT MEASURED. See `asset_readiness.csv`. Their INT8 Linear weights use dynamic INT8 activations and INT32 accumulation; non-Linear operations remain floating point. No transformer row enters `primary_conditions.csv`.\n\n`plots_measured_only/` contains only actually measured interior budgets; `plots_target_matrix/` also shows the 6.25% and 18.75% missing-budget panels explicitly.\n")
    files = {str(path.relative_to(output)):sha256_file(path) for path in sorted(output.rglob("*")) if path.is_file()}
    checks = {"schema_version":SCHEMA_VERSION,"sources_sha256":sha256_file(sources),"asset_manifest_sha256":sha256_file(asset_manifest),"files":files}
    (output/"checksums.json").write_text(json.dumps(checks, indent=2, sort_keys=True)+"\n")
    (output/"FINALIZED").write_text(f"reliability_database_sha256={files['reliability_v3.sqlite']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path("config/reliability_sources_v1.json"))
    parser.add_argument("--assets", type=Path, default=Path("/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json"))
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    build_product(args.sources,args.assets,args.output)
    print(args.output/"reliability_v3.sqlite")


if __name__ == "__main__":
    main()
