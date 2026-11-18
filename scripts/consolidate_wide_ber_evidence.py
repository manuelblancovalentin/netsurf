"""Read-only audit and write-once consolidation of saved W8A8 wide-BER evidence.

This module imports no model, ranking, or fault-generation code. It never writes to
either scientific source or the previous comparative database.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import zipfile

import numpy as np

from netsurf.manifests import digest_object, sha256_file


BER_GRID = (1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
BUDGETS = (0.0625, 0.125, 0.1875, 0.25)
METHODS = ("random", "msb_first", "numerical_bit_jump", "global_first_order",
           "per_example_absolute", "per_example_positive")
LABELS = ("Random", "MSB-first", "Numerical jump", "Global gradient",
          "Absolute gradient", "Positive gradient")


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _float(value: str | float | None) -> float | None:
    return None if value in (None, "") else float(value)


def _verify_manifest(root: Path, marker: str, *, full_source: bool) -> tuple[dict, str]:
    manifest = _load(root / "manifest.json")
    marker_data = _load(root / marker)
    digest = sha256_file(root / "manifest.json")
    if marker_data["manifest_sha256"] != digest:
        raise ValueError(f"{root}: manifest marker mismatch")
    if manifest["payload_sha256"] != digest_object({
        key: value for key, value in manifest.items() if key != "payload_sha256"
    }):
        raise ValueError(f"{root}: manifest payload digest mismatch")
    hashes = (manifest["scientific_file_sha256"] if full_source
              else manifest["output_sha256"])
    for relative, expected in hashes.items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f"{root}: scientific-file digest mismatch: {relative}")
    return manifest, digest


def audit_study(root: Path, *, full_source: bool) -> dict:
    marker = "FINALIZED" if full_source else "DERIVED_FINALIZED.json"
    manifest, manifest_sha = _verify_manifest(root, marker, full_source=full_source)
    expected_status = ("FINALIZED_MEASURED" if full_source else
                       "FINALIZED_DERIVED_VALIDATED_SUBSET_NOT_FROZEN_FULL_STUDY")
    if manifest["status"] != expected_status:
        raise ValueError(f"{root}: study status differs from expected status")
    prefix = root / "derived" if full_source else root
    trials = _csv(prefix / "trials.csv")
    conditions = _csv(prefix / "conditions.csv")
    pairs = _csv(prefix / "paired_comparisons.csv")
    analysis = _load(prefix / "analysis.json")
    plan = _load(root / "plan.json") if full_source else None
    count = 24 if full_source else 18
    expected_fields = len(BER_GRID) * count
    if (len(trials), len(conditions), len(pairs)) != (expected_fields * 26, 208, 160):
        raise ValueError(f"{root}: incomplete trials, conditions or contrasts")
    if full_source and (analysis["field_count"] != expected_fields or
                        analysis["trial_observations"] != len(trials) or
                        manifest["plan_sha256"] != plan["plan_sha256"]):
        raise ValueError(f"{root}: analysis/plan mismatch")
    if not full_source and (analysis["accepted_fields"] != expected_fields or
                            analysis["trial_observations"] != len(trials)):
        raise ValueError(f"{root}: recovered-subset analysis mismatch")
    by_field: dict[tuple[float, int], dict[str, dict]] = {}
    by_condition: dict[tuple[float, str], list[dict]] = {}
    field_identity: dict[tuple[float, int], tuple[str, str, str]] = {}
    for row in trials:
        ber, index = float(row["ber"]), int(row["field_index"])
        if ber not in BER_GRID:
            raise ValueError("unexpected BER in saved trial")
        key = (ber, index)
        identity = (row["field_seed"], row["field_sha256"], row["proposed"])
        if key in field_identity and field_identity[key] != identity:
            raise ValueError("policies do not share a proposed fault field")
        field_identity[key] = identity
        if int(row["proposed"]) != int(row["applied"]) + int(row["suppressed"]):
            raise ValueError("fault accounting mismatch")
        if int(row["failed_units"]) < 0 or not math.isfinite(float(row["mean_nll_nats"])):
            raise ValueError("invalid saved risk")
        policy = row["policy"]
        if policy in by_field.setdefault(key, {}):
            raise ValueError("duplicate policy/field observation")
        by_field[key][policy] = row
        by_condition.setdefault((ber, policy), []).append(row)
    if len(field_identity) != expected_fields or any(len(rows) != 26 for rows in by_field.values()):
        raise ValueError("missing paired policy or proposed field")
    for rows in by_field.values():
        clean = rows["full_protection"]
        if int(clean["applied"]) != 0 or int(rows["unprotected"]["suppressed"]) != 0:
            raise ValueError("control event accounting mismatch")
    condition_keys = set()
    for condition in conditions:
        ber, policy = float(condition["ber"]), condition["policy"]
        key = (ber, policy)
        if key in condition_keys:
            raise ValueError("duplicate condition")
        condition_keys.add(key)
        group = by_condition[key]
        values = np.asarray([float(r["mean_nll_nats"]) for r in group], dtype=np.float64)
        if (len(group) != count or int(condition["field_count"]) != count or
            not math.isclose(float(condition["mean_nll_nats"]), float(values.mean()), abs_tol=1e-11) or
            not math.isclose(float(condition["loss_std_nats"]), float(values.std(ddof=1)), abs_tol=1e-11) or
            int(condition["failed_unit_count"]) != sum(int(r["failed_units"]) for r in group)):
            raise ValueError("condition summary differs from saved field records")
        if policy == "full_protection" and any(float(r["mean_nll_nats"]) !=
                                               float(condition["clean_nll_nats"]) for r in group):
            raise ValueError("full protection differs from clean risk")
    for pair in pairs:
        ber, coverage = float(pair["ber"]), float(pair["coverage"])
        if coverage not in BUDGETS or int(pair["field_count"]) != count:
            raise ValueError("invalid paired contrast")
        budget = int(pair["budget_bits"])
        compared = f'{pair["compared_method"]}_b{budget}'
        reference = f'{pair["reference_method"]}_b{budget}'
        diffs = [float(rows[compared]["mean_nll_nats"]) -
                 float(rows[reference]["mean_nll_nats"])
                 for (b, _), rows in by_field.items() if b == ber]
        if (len(diffs) != count or
            not math.isclose(float(pair["mean_nll_difference_nats"]),
                             float(np.mean(diffs)), abs_tol=1e-11) or
            sum(int(pair[k]) for k in ("favorable_fields", "adverse_fields", "tied_fields")) != count):
            raise ValueError("paired contrast differs from matched saved fields")
    return {"root": root, "full_source": full_source, "manifest": manifest,
            "manifest_sha256": manifest_sha, "analysis": analysis, "plan": plan,
            "trials": trials, "conditions": conditions, "pairs": pairs,
            "field_count": expected_fields, "per_ber": count}


def _plot(study: dict, destination: Path, axis: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = "DistilBERT W8A8" if study["full_source"] else "Pythia W8A8"
    metric = "SST-2 cross-entropy" if study["full_source"] else "WikiText-2 token NLL"
    lookup = {(float(r["ber"]), r["policy"]): r for r in study["conditions"]}
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=axis == "ber", sharey=axis == "ber")
    panels = BUDGETS if axis == "ber" else (1e-4, 1e-3, 1e-2, 1e-1)
    eligible = int(study["plan"]["eligible_bits"] if study["full_source"] else
                   study["analysis"]["clean_risk"].get("eligible_bits", 0))
    if not eligible:
        eligible = int(next(r["budget_bits"] for r in study["conditions"]
                            if math.isclose(float(r["coverage"]), .125))) * 8
    for ax, panel in zip(axes.flat, panels):
        for method, label, color in zip(METHODS, LABELS, colors):
            if axis == "ber":
                x = np.asarray(BER_GRID)
                budget = int(eligible * panel)
                records = [lookup[(ber, f"{method}_b{budget}")] for ber in BER_GRID]
            else:
                x = np.asarray(BUDGETS) * 100
                records = [lookup[(panel, f"{method}_b{int(eligible * c)}")] for c in BUDGETS]
            mean = np.asarray([float(r["mean_nll_nats"]) for r in records])
            low = np.asarray([float(r["loss_ci95_low"]) for r in records])
            high = np.asarray([float(r["loss_ci95_high"]) for r in records])
            ax.plot(x, mean, marker="o", ms=3, lw=1.3, color=color, label=label)
            ax.fill_between(x, low, high, color=color, alpha=.10)
        if axis == "ber":
            baseline = [float(lookup[(ber, "unprotected")]["mean_nll_nats"]) for ber in BER_GRID]
            ax.plot(BER_GRID, baseline, "k--", lw=1.6, label="Unprotected")
            clean = float(lookup[(BER_GRID[0], "full_protection")]["mean_nll_nats"])
            ax.axhline(clean, color="0.35", ls=":", lw=1.4, label="Clean / fully protected")
            ax.set_xscale("log")
            ax.set_title(f"{panel * 100:g}% protected")
            ax.set_xlabel("Logical stored-bit BER")
        else:
            unprotected = float(lookup[(panel, "unprotected")]["mean_nll_nats"])
            clean = float(lookup[(panel, "full_protection")]["mean_nll_nats"])
            ax.axhline(unprotected, color="black", ls="--", lw=1.5, label="Unprotected")
            ax.axhline(clean, color="0.35", ls=":", lw=1.4, label="Clean / fully protected")
            ax.set_title(f"BER {panel:g}")
            ax.set_xticks(np.asarray(BUDGETS) * 100)
            ax.set_xlabel("Eligible bits protected (%)")
        ax.grid(alpha=.2)
        ax.set_ylabel(f"Mean {metric} (nats)")
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="center", bbox_to_anchor=(.5, .075), ncol=4, fontsize=9)
    qualifier = ("finalized, 24 paired fields per BER" if study["full_source"] else
                 "derived clean-conformant subset, 18 paired fields per BER")
    fig.suptitle(f"{target}: saved wide-BER observations ({qualifier})")
    note = ("Bands are 95% field-bootstrap intervals for each policy mean; matched-policy tests use paired differences."
            if study["full_source"] else
            "Original 24-field campaign failed shard-0 clean conformance; this subset is exploratory.")
    fig.text(.5, .018, note, ha="center", fontsize=9,
             color="darkred" if not study["full_source"] else "0.3")
    fig.tight_layout(rect=(0, .16, 1, .95))
    fig.savefig(destination / f"{study['name']}_loss_vs_{axis}.png", dpi=150)
    fig.savefig(destination / f"{study['name']}_loss_vs_{axis}.svg")
    plt.close(fig)


def consolidate(base_db: Path, distil_root: Path, pythia_root: Path, destination: Path) -> dict:
    if destination.exists() or destination.with_suffix(".zip").exists():
        raise ValueError("write-once derived destination or archive already exists")
    distil = audit_study(distil_root, full_source=True)
    pythia = audit_study(pythia_root, full_source=False)
    distil["name"] = "distilbert_sst2_wide_ber"
    pythia["name"] = "pythia160m_wide_ber_clean_shards"
    studies = (distil, pythia)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".wide-ber-consolidation-", dir=destination.parent))
    try:
        source = sqlite3.connect(f"file:{base_db.resolve()}?mode=ro", uri=True)
        db = sqlite3.connect(staging / "current_comparative_wide_v1.sqlite")
        source.backup(db)
        source.close()
        if db.execute("select count(*) from current_comparative_conditions").fetchone()[0] != 864:
            raise ValueError("base comparative database differs from the audited 864-condition version")
        db.executescript("""
            CREATE TABLE wide_ber_studies (
                study_id TEXT PRIMARY KEY, benchmark_id TEXT NOT NULL,
                evidence_status TEXT NOT NULL, source_root TEXT NOT NULL,
                source_manifest_sha256 TEXT NOT NULL, source_payload_sha256 TEXT NOT NULL,
                source_plan_sha256 TEXT NOT NULL, eligible_bits INTEGER NOT NULL,
                metric TEXT NOT NULL, field_count INTEGER NOT NULL, interpretation TEXT NOT NULL
            );
            CREATE TABLE wide_ber_conditions (
                study_id TEXT NOT NULL, benchmark_id TEXT NOT NULL,
                evidence_status TEXT NOT NULL, ber REAL NOT NULL, method_id TEXT NOT NULL,
                coverage REAL NOT NULL, protected_bits INTEGER NOT NULL,
                field_count INTEGER NOT NULL, mean_loss REAL NOT NULL, loss_sd REAL NOT NULL,
                loss_ci_low REAL NOT NULL, loss_ci_high REAL NOT NULL,
                clean_loss REAL NOT NULL, loss_degradation REAL NOT NULL,
                mean_accuracy REAL, failed_unit_count INTEGER NOT NULL, mask_sha256 TEXT NOT NULL,
                PRIMARY KEY (study_id, ber, method_id, coverage)
            );
            CREATE TABLE wide_ber_trials (
                study_id TEXT NOT NULL, ber REAL NOT NULL, field_index INTEGER NOT NULL,
                field_seed INTEGER NOT NULL, field_sha256 TEXT NOT NULL, method_id TEXT NOT NULL,
                coverage REAL NOT NULL, mask_sha256 TEXT NOT NULL,
                proposed INTEGER NOT NULL, applied INTEGER NOT NULL, suppressed INTEGER NOT NULL,
                mean_loss REAL NOT NULL, accuracy REAL, failed_units INTEGER NOT NULL,
                total_units INTEGER NOT NULL,
                PRIMARY KEY (study_id, ber, field_index, method_id)
            );
            CREATE TABLE wide_ber_paired_comparisons (
                study_id TEXT NOT NULL, ber REAL NOT NULL, coverage REAL NOT NULL,
                compared_method TEXT NOT NULL, reference_method TEXT NOT NULL,
                field_count INTEGER NOT NULL, mean_difference REAL NOT NULL,
                ci_low REAL NOT NULL, ci_high REAL NOT NULL, p_signflip REAL NOT NULL,
                p_holm_160 REAL NOT NULL, favorable_fields INTEGER NOT NULL,
                adverse_fields INTEGER NOT NULL, tied_fields INTEGER NOT NULL,
                PRIMARY KEY (study_id, ber, coverage, compared_method)
            );
            CREATE VIEW current_wide_ber_conditions AS
                SELECT * FROM wide_ber_conditions;
            CREATE VIEW finalized_wide_ber_conditions AS
                SELECT * FROM wide_ber_conditions WHERE evidence_status='FINALIZED_MEASURED';
            CREATE VIEW exploratory_wide_ber_conditions AS
                SELECT * FROM wide_ber_conditions WHERE evidence_status<>'FINALIZED_MEASURED';
        """)
        flat_conditions = []
        for study in studies:
            status = study["manifest"]["status"]
            plan_sha = (study["plan"]["plan_sha256"] if study["full_source"] else
                        study["analysis"]["source_plan_sha256"])
            eligible = int(next(r["budget_bits"] for r in study["conditions"]
                                if math.isclose(float(r["coverage"]), .125))) * 8
            interpretation = ("full frozen wide-BER campaign" if study["full_source"] else
                              "post-failure clean-conformant subset; not full prespecified campaign")
            db.execute("INSERT INTO wide_ber_studies VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                study["name"], study["conditions"][0]["target"], status,
                str(study["root"]), study["manifest_sha256"],
                study["manifest"]["payload_sha256"], plan_sha, eligible,
                "mean cross-entropy in nats" if study["full_source"] else "mean token NLL in nats",
                study["field_count"], interpretation))
            for r in study["conditions"]:
                row = (study["name"], r["target"], status, float(r["ber"]),
                       r["policy"], float(r["coverage"]), int(r["budget_bits"]),
                       int(r["field_count"]), float(r["mean_nll_nats"]),
                       float(r["loss_std_nats"]), float(r["loss_ci95_low"]),
                       float(r["loss_ci95_high"]), float(r["clean_nll_nats"]),
                       float(r["mean_loss_degradation_nats"]), _float(r.get("mean_accuracy")),
                       int(r["failed_unit_count"]), r["mask_sha256"])
                db.execute("INSERT INTO wide_ber_conditions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
                flat_conditions.append(row)
            for r in study["trials"]:
                db.execute("INSERT INTO wide_ber_trials VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    study["name"], float(r["ber"]), int(r["field_index"]),
                    int(r["field_seed"]), r["field_sha256"], r["policy"],
                    float(r["coverage"]), r["mask_sha256"], int(r["proposed"]),
                    int(r["applied"]), int(r["suppressed"]), float(r["mean_nll_nats"]),
                    _float(r.get("accuracy")), int(r["failed_units"]), int(r["total_units"])))
            for r in study["pairs"]:
                db.execute("INSERT INTO wide_ber_paired_comparisons VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    study["name"], float(r["ber"]), float(r["coverage"]),
                    r["compared_method"], r["reference_method"], int(r["field_count"]),
                    float(r["mean_nll_difference_nats"]), float(r["ci95_low"]),
                    float(r["ci95_high"]), float(r["p_signflip"]), float(r["p_holm_160"]),
                    int(r["favorable_fields"]), int(r["adverse_fields"]), int(r["tied_fields"])))
            _plot(study, staging, "ber")
            _plot(study, staging, "budget")
        db.commit()
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
        counts = {table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                  for table in ("current_comparative_conditions", "wide_ber_conditions",
                                "wide_ber_trials", "wide_ber_paired_comparisons")}
        db.close()
        if integrity != "ok" or counts != {
            "current_comparative_conditions": 864, "wide_ber_conditions": 416,
            "wide_ber_trials": 8736, "wide_ber_paired_comparisons": 320,
        }:
            raise ValueError(f"database integrity/count mismatch: {integrity} {counts}")
        header = ["study_id", "benchmark_id", "evidence_status", "ber", "method_id",
                  "coverage", "protected_bits", "field_count", "mean_loss", "loss_sd",
                  "loss_ci_low", "loss_ci_high", "clean_loss", "loss_degradation",
                  "mean_accuracy", "failed_unit_count", "mask_sha256"]
        with (staging / "wide_ber_conditions.csv").open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(flat_conditions)
        readme = (
            "# W8A8 wide-BER offline consolidation\n\n"
            "The copied comparative v3 database and its 864-row historical current view remain "
            "unchanged. `current_wide_ber_conditions` contains 416 additional rows in two "
            "distinct studies. `finalized_wide_ber_conditions` contains only the 208-row "
            "DistilBERT full study; `exploratory_wide_ber_conditions` contains only the "
            "208-row Pythia clean-conformant derived subset. Do not pool the latter with "
            "the original failed 24-field campaign or earlier low-BER studies.\n\n"
            "Each study has eight BERs, four interior budgets, six ranking methods and "
            "unprotected/full controls: 26 arms per field. DistilBERT has 24 paired fields "
            "per BER (192 fields, 4,992 records); Pythia has 18 per BER (144 fields, "
            "3,744 records). One ranking mask per method and budget was held fixed. "
            "The plotted bands are marginal bootstrap intervals over fault fields, not "
            "paired-policy intervals; paired contrasts and Holm-adjusted p-values are "
            "stored separately. Loss units differ: SST-2 classification cross-entropy vs "
            "WikiText-2 token NLL. Both are in nats but must not be pooled. The high-loss "
            "regime uses the frozen 30-nat penalty/cap; inspect failure counts separately.\n"
        )
        (staging / "README.md").write_text(readme)
        manifest = {"schema": "netsurf-wide-ber-consolidation/v1",
                    "source_database_sha256": sha256_file(base_db),
                    "source_studies": {s["name"]: {
                        "source_manifest_sha256": s["manifest_sha256"],
                        "source_payload_sha256": s["manifest"]["payload_sha256"],
                        "status": s["manifest"]["status"]} for s in studies},
                    "counts": counts,
                    "output_sha256": {p.name: sha256_file(p) for p in sorted(staging.iterdir())
                                      if p.is_file() and p.name != "manifest.json"}}
        manifest["payload_sha256"] = digest_object(manifest)
        (staging / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        (staging / "DERIVED_FINALIZED.json").write_text(json.dumps({
            "manifest_sha256": sha256_file(staging / "manifest.json")}, sort_keys=True) + "\n")
        os.rename(staging, destination)
        zip_path = destination.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6) as archive:
            for path in sorted(destination.iterdir()):
                archive.write(path, arcname=f"{destination.name}/{path.name}")
        with zipfile.ZipFile(zip_path) as archive:
            bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP CRC failed: {bad}")
        return {"destination": str(destination), "zip": str(zip_path),
                "zip_sha256": sha256_file(zip_path), "counts": counts,
                "source_manifests": manifest["source_studies"]}
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-db", type=Path, required=True)
    parser.add_argument("--distil-root", type=Path, required=True)
    parser.add_argument("--pythia-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(consolidate(args.base_db, args.distil_root, args.pythia_root,
                                 args.destination), indent=2))


if __name__ == "__main__":
    main()
