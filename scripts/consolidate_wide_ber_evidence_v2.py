"""Offline, write-once Qwen and KWS-pilot addition to comparative v1.

The source database's 864-row main comparative view is copied unchanged.
W8A8 wide-BER studies and the exploratory KWS conditional pilot remain separate
tables because their tasks, candidate universes, and statistical units differ.
No model, mask, field generator, or inference adapter is imported here.
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
from scripts.consolidate_wide_ber_evidence import (
    BER_GRID, BUDGETS, LABELS, METHODS, _float, _load, _verify_manifest,
    audit_study,
)


CONTRASTS = (
    "restricted_selected_vs_original_s5",
    "restricted_selected_vs_exact_isolated",
    "exact_isolated_vs_original_s5",
)


def _verify_kws(analysis_root: Path, campaign_path: Path) -> tuple[dict, dict]:
    payload_path = analysis_root / "scientific_payload.json"
    payload = _load(payload_path)
    marker = (analysis_root / "FINALIZED").read_text().strip()
    if marker != f"scientific_payload_sha256={digest_object(payload)}":
        raise ValueError("KWS derived scientific payload/finalization mismatch")
    checksums = _load(analysis_root / "checksums.json")["files"]
    if set(checksums) != {"report.md", "scientific_payload.json"}:
        raise ValueError("KWS derived checksum inventory differs")
    for name, expected in checksums.items():
        if sha256_file(analysis_root / name) != expected:
            raise ValueError(f"KWS derived checksum differs: {name}")
    campaign = _load(campaign_path)
    if (sha256_file(campaign_path) != payload["joint_campaign_sha256"] or
        payload["ber_grid"] != [.02, .05, .075, .1] or
        len(payload["evidence"]) != 4 or
        len(payload["decision_changes"]) != 12 or
        len(payload["exploratory_within_ber_comparisons"]) != 12):
        raise ValueError("KWS campaign identity or four-BER pool analysis differs")
    if [arm["logical_ber"] for arm in campaign["arms"]] != payload["ber_grid"]:
        raise ValueError("KWS frozen BER arm order differs")
    return campaign, payload


def _read_large_csv(path: Path):
    prior = csv.field_size_limit()
    try:
        csv.field_size_limit(max(prior, 8 * 1024 * 1024))
        with path.open(newline="") as stream:
            yield from csv.DictReader(stream)
    finally:
        csv.field_size_limit(prior)


def _insert_qwen(db: sqlite3.Connection, study: dict) -> None:
    sid = "qwen3_0_6b_wide_ber"
    if study["manifest"]["status"] != "FINALIZED_MEASURED" or study["field_count"] != 192:
        raise ValueError("Qwen is not a complete finalized full-source study")
    db.execute("INSERT INTO wide_ber_studies VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
        sid, "qwen3_0_6b_int8", study["manifest"]["status"], str(study["root"]),
        study["manifest_sha256"], study["manifest"]["payload_sha256"],
        study["plan"]["plan_sha256"], int(study["plan"]["eligible_bits"]),
        "teacher-forced target-token NLL in nats", study["field_count"],
        "full frozen 24-field-per-BER wide-BER campaign; one fixed mask per method/budget"))
    for r in study["conditions"]:
        db.execute("INSERT INTO wide_ber_conditions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            sid, r["target"], "FINALIZED_MEASURED", float(r["ber"]),
            r["policy"], float(r["coverage"]), int(r["budget_bits"]),
            int(r["field_count"]), float(r["mean_nll_nats"]),
            float(r["loss_std_nats"]), float(r["loss_ci95_low"]),
            float(r["loss_ci95_high"]), float(r["clean_nll_nats"]),
            float(r["mean_loss_degradation_nats"]), _float(r["mean_accuracy"]),
            int(r["failed_unit_count"]), r["mask_sha256"]))
    for r in study["trials"]:
        db.execute("INSERT INTO wide_ber_trials VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            sid, float(r["ber"]), int(r["field_index"]), int(r["field_seed"]),
            r["field_sha256"], r["policy"], float(r["coverage"]),
            r["mask_sha256"], int(r["proposed"]), int(r["applied"]),
            int(r["suppressed"]), float(r["mean_nll_nats"]),
            _float(r["accuracy"]), int(r["failed_units"]), int(r["total_units"])))
    for r in study["pairs"]:
        db.execute("INSERT INTO wide_ber_paired_comparisons VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            sid, float(r["ber"]), float(r["coverage"]),
            r["compared_method"], r["reference_method"], int(r["field_count"]),
            float(r["mean_nll_difference_nats"]), float(r["ci95_low"]),
            float(r["ci95_high"]), float(r["p_signflip"]), float(r["p_holm_160"]),
            int(r["favorable_fields"]), int(r["adverse_fields"]), int(r["tied_fields"])))


def _kws_schema(db: sqlite3.Connection) -> None:
    db.executescript("""
        CREATE TABLE kws_high_ber_pilot_registry(
          study_id TEXT PRIMARY KEY, campaign_path TEXT NOT NULL,
          campaign_sha256 TEXT NOT NULL, derived_path TEXT NOT NULL,
          derived_payload_sha256 TEXT NOT NULL, status TEXT NOT NULL,
          metric TEXT NOT NULL, uncertainty_unit TEXT NOT NULL);
        CREATE TABLE kws_high_ber_pool_contrasts(
          ber REAL NOT NULL, pool_id TEXT NOT NULL, comparison TEXT NOT NULL,
          field_count INTEGER NOT NULL, mean_difference REAL NOT NULL,
          field_ci_low REAL NOT NULL, field_ci_high REAL NOT NULL,
          selection_payload_sha256 TEXT NOT NULL, final_payload_sha256 TEXT NOT NULL,
          PRIMARY KEY(ber,pool_id,comparison));
        CREATE TABLE kws_high_ber_field_contrasts(
          ber REAL NOT NULL, pool_id TEXT NOT NULL, field_index INTEGER NOT NULL,
          comparison TEXT NOT NULL, field_seed INTEGER NOT NULL,
          master_pairing_key TEXT NOT NULL, mean_difference REAL NOT NULL,
          source_csv_path TEXT NOT NULL, source_csv_sha256 TEXT NOT NULL,
          PRIMARY KEY(ber,pool_id,field_index,comparison));
        CREATE TABLE kws_high_ber_summary(
          ber REAL NOT NULL, comparison TEXT NOT NULL, pool_count INTEGER NOT NULL,
          mean_difference REAL NOT NULL, pool_ci_low REAL NOT NULL,
          pool_ci_high REAL NOT NULL, nominal_p REAL NOT NULL,
          favorable_pools INTEGER NOT NULL, adverse_pools INTEGER NOT NULL,
          zero_pools INTEGER NOT NULL, inferential_status TEXT NOT NULL,
          PRIMARY KEY(ber,comparison));
        CREATE TABLE kws_high_ber_mask_decisions(
          ber REAL NOT NULL, pool_id TEXT NOT NULL,
          original_local_mask TEXT NOT NULL, isolated_local_mask TEXT NOT NULL,
          selected_local_mask TEXT NOT NULL, selection_payload_sha256 TEXT NOT NULL,
          PRIMARY KEY(ber,pool_id));
    """)


def _insert_kws(db: sqlite3.Connection, campaign_path: Path,
                analysis_root: Path, campaign: dict, payload: dict) -> None:
    db.execute("INSERT INTO kws_high_ber_pilot_registry VALUES (?,?,?,?,?,?,?,?)", (
        "kws_higher_ber_pilot_v1", str(campaign_path), sha256_file(campaign_path),
        str(analysis_root), digest_object(payload), "FINALIZED_EXPLORATORY_DERIVED",
        "paired categorical cross-entropy difference, nats/clip",
        "32 independent fault fields per pool; 12 sampled candidate pools for across-pool interval"))
    for arm, evidence in zip(campaign["arms"], payload["evidence"], strict=True):
        ber = float(arm["logical_ber"])
        if evidence["logical_ber"] != ber or len(evidence["pools"]) != 12:
            raise ValueError("KWS frozen arm/pool evidence differs")
        receipt_root = Path(arm["manifest"]).parent / "receipts"
        for pool in evidence["pools"]:
            pid = pool["pool_id"]
            receipt = _load(receipt_root / f"{pid}_final.json")
            if (receipt["payload_sha256"] != pool["final_sha256"] or
                receipt["campaign_sha256"] != arm["manifest_sha256"] or
                receipt["pool_id"] != pid or receipt["phase"] != "final"):
                raise ValueError("KWS final receipt identity differs")
            bundle = Path(receipt["bundle"])
            source = bundle / "measurements/paired_swap_fields.csv"
            source_sha = _load(bundle / "checksums.json")["files"]["measurements/paired_swap_fields.csv"]
            if sha256_file(source) != source_sha:
                raise ValueError("KWS paired field CSV differs from finalized source checksum")
            by_comparison = {c["comparison"]: c for c in pool["field_analysis"]["comparisons"]}
            if set(by_comparison) != set(CONTRASTS) or pool["field_analysis"]["independent_field_count"] != 32:
                raise ValueError("KWS pool comparison family or field count differs")
            for name, c in by_comparison.items():
                estimate = c["mean_effect"]
                db.execute("INSERT INTO kws_high_ber_pool_contrasts VALUES (?,?,?,?,?,?,?,?,?)", (
                    ber, pid, name, 32, float(estimate["mean"]),
                    float(estimate["confidence_interval"][0]),
                    float(estimate["confidence_interval"][1]),
                    pool["selection_sha256"], pool["final_sha256"]))
            selection = pool["selection"]
            db.execute("INSERT INTO kws_high_ber_mask_decisions VALUES (?,?,?,?,?,?)", (
                ber, pid, json.dumps(selection["original_mask_indices"]),
                json.dumps(selection["isolated_mask_indices"]),
                json.dumps(selection["selected_mask_indices"]), pool["selection_sha256"]))
            values: dict[str, list[float]] = {name: [] for name in CONTRASTS}
            keys: dict[int, tuple[str, str]] = {}
            for row in _read_large_csv(source):
                name = row["comparison"]
                field = int(row["field"])
                if name not in values or float(row["logical_ber"]) != ber or not 0 <= field < 32:
                    raise ValueError("KWS field row violates frozen BER/contrast/field plan")
                identity = (row["field_seed"], row["master_pairing_key"])
                if field in keys and keys[field] != identity:
                    raise ValueError("KWS policy contrasts do not share the same proposed field")
                keys[field] = identity
                effect = float(row["target_minus_baseline"])
                values[name].append(effect)
                db.execute("INSERT INTO kws_high_ber_field_contrasts VALUES (?,?,?,?,?,?,?,?,?)", (
                    ber, pid, field, name, int(row["field_seed"]),
                    row["master_pairing_key"], effect, str(source), source_sha))
            if len(keys) != 32 or any(len(v) != 32 for v in values.values()):
                raise ValueError("KWS per-pool paired-field family incomplete")
            for name, group in values.items():
                if not math.isclose(float(np.mean(group)),
                                    by_comparison[name]["mean_effect"]["mean"], abs_tol=1e-11):
                    raise ValueError("KWS saved pool mean differs from imported fields")
    for row in payload["exploratory_within_ber_comparisons"]:
        est = row["estimate"]
        db.execute("INSERT INTO kws_high_ber_summary VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
            float(row["logical_ber"]), row["comparison"], int(est["count"]),
            float(est["mean"]), float(est["confidence_interval"][0]),
            float(est["confidence_interval"][1]), float(row["p_value_two_sided"]),
            int(row["favorable_pools"]), int(row["adverse_pools"]),
            int(row["zero_pools"]), "EXPLORATORY_UNADJUSTED"))
    for row in db.execute("SELECT ber,comparison,mean_difference FROM kws_high_ber_summary"):
        means = [r[0] for r in db.execute("""SELECT mean_difference FROM kws_high_ber_pool_contrasts
            WHERE ber=? AND comparison=? ORDER BY pool_id""", row[:2])]
        if len(means) != 12 or not math.isclose(np.mean(means), row[2], abs_tol=1e-11):
            raise ValueError("KWS across-pool mean differs from imported pool effects")


def _plots(db: sqlite3.Connection, destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sid = "qwen3_0_6b_wide_ber"
    rows = db.execute("""SELECT ber,method_id,coverage,mean_loss,loss_ci_low,loss_ci_high
        FROM wide_ber_conditions WHERE study_id=?""", (sid,)).fetchall()
    lookup = {(r[0], r[1]): r for r in rows}
    colors = plt.get_cmap("tab10").colors
    for axis in ("ber", "budget"):
        fig, panels = plt.subplots(2, 2, figsize=(13, 9), sharex=axis == "ber")
        grid = BUDGETS if axis == "ber" else (1e-4, 1e-3, 1e-2, 1e-1)
        bits = 4_767_875_072
        for ax, panel in zip(panels.flat, grid):
            for method, label, color in zip(METHODS, LABELS, colors[:len(METHODS)], strict=True):
                if axis == "ber":
                    x = BER_GRID
                    records = [lookup[(b, f"{method}_b{int(bits*panel)}")] for b in x]
                else:
                    x = [b*100 for b in BUDGETS]
                    records = [lookup[(panel, f"{method}_b{int(bits*b)}")] for b in BUDGETS]
                y = [r[3] for r in records]
                ax.plot(x, y, marker="o", ms=3, lw=1.2, color=color, label=label)
                ax.fill_between(x, [r[4] for r in records], [r[5] for r in records],
                                color=color, alpha=.1)
            if axis == "ber":
                ax.plot(BER_GRID, [lookup[(b, "unprotected")][3] for b in BER_GRID],
                        "k--", lw=1.5, label="Unprotected")
                ax.set_xscale("log")
                ax.set_title(f"{panel*100:g}% protected")
                ax.set_xlabel("Logical stored-bit BER")
            else:
                ax.axhline(lookup[(panel, "unprotected")][3], color="black", ls="--",
                           lw=1.5, label="Unprotected")
                ax.set_title(f"BER {panel:g}")
                ax.set_xticks([b*100 for b in BUDGETS])
                ax.set_xlabel("Eligible bits protected (%)")
            ax.axhline(lookup[(BER_GRID[0], "full_protection")][3], color="0.4", ls=":",
                       lw=1.3, label="Clean / fully protected")
            ax.set_ylabel("Mean token NLL (nats)")
            ax.grid(alpha=.2)
        handles, labels = panels.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center", bbox_to_anchor=(.5, .075), ncol=4, fontsize=9)
        fig.suptitle("Qwen3-0.6B W8A8: finalized wide-BER fault study")
        fig.text(.5, .018, "24 paired proposed fields per BER; bands are marginal 95% field-bootstrap intervals. "
                 "Paired comparisons and Holm-adjusted p-values are separate.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .16, 1, .95))
        fig.savefig(destination / f"qwen3_0_6b_loss_vs_{axis}.png", dpi=150)
        fig.savefig(destination / f"qwen3_0_6b_loss_vs_{axis}.svg")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, comparison, title in zip(axes, CONTRASTS,
                                      ("Selected − original gradient mask",
                                       "Selected − exact isolated mask",
                                       "Exact isolated − original gradient mask"), strict=True):
        summary = db.execute("""SELECT ber,mean_difference,pool_ci_low,pool_ci_high
            FROM kws_high_ber_summary WHERE comparison=? ORDER BY ber""", (comparison,)).fetchall()
        for i, (ber, mean, low, high) in enumerate(summary):
            effects = [r[0] for r in db.execute("""SELECT mean_difference
                FROM kws_high_ber_pool_contrasts WHERE ber=? AND comparison=? ORDER BY pool_id""",
                (ber, comparison))]
            offsets = np.linspace(-.004, .004, len(effects))
            ax.scatter(np.full(len(effects), ber)+offsets, effects, color="0.45",
                       alpha=.55, s=16, zorder=2)
            ax.errorbar([ber], [mean], yerr=[[mean-low], [high-mean]], fmt="o",
                        color="darkblue", capsize=4, zorder=3)
        ax.plot([r[0] for r in summary], [r[1] for r in summary], color="darkblue", lw=1.2)
        ax.axhline(0, color="black", ls="--", lw=1)
        ax.set_title(title)
        ax.set_xticks([.02, .05, .075, .1], ["2%", "5%", "7.5%", "10%"])
        ax.set_xlabel("Logical stored-bit BER")
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Paired cross-entropy difference (nats/clip)")
    fig.suptitle("KWS: local protection exchanges under higher BER")
    fig.text(.5, .01, "Gray dots: 12 sampled pools; blue: pool mean and nominal 95% pool-t interval. "
             "Negative favors the first named mask. Exploratory, previously exposed inputs.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .08, 1, .93))
    fig.savefig(destination / "kws_high_ber_pool_effects.png", dpi=150)
    fig.savefig(destination / "kws_high_ber_pool_effects.svg")
    plt.close(fig)


def consolidate(base: Path, qwen_root: Path, kws_analysis: Path,
                kws_campaign: Path, destination: Path) -> dict:
    if destination.exists() or destination.with_suffix(".zip").exists():
        raise ValueError("write-once derived destination or ZIP already exists")
    _verify_manifest(base, "DERIVED_FINALIZED.json", full_source=False)
    qwen = audit_study(qwen_root, full_source=True)
    campaign, kws = _verify_kws(kws_analysis, kws_campaign)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".wide-ber-v2-", dir=destination.parent))
    try:
        source_db = sqlite3.connect(f"file:{(base/'current_comparative_wide_v1.sqlite').resolve()}?mode=ro", uri=True)
        db = sqlite3.connect(staging / "current_comparative_wide_v2.sqlite")
        source_db.backup(db)
        source_db.close()
        if (db.execute("SELECT count(*) FROM current_comparative_conditions").fetchone()[0] != 864 or
            db.execute("SELECT count(*) FROM wide_ber_conditions").fetchone()[0] != 416):
            raise ValueError("v1 comparative source row counts differ")
        _insert_qwen(db, qwen)
        _kws_schema(db)
        _insert_kws(db, kws_campaign, kws_analysis, campaign, kws)
        db.commit()
        counts = {table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                  for table in ("current_comparative_conditions", "wide_ber_conditions",
                                "wide_ber_trials", "wide_ber_paired_comparisons",
                                "kws_high_ber_pool_contrasts", "kws_high_ber_field_contrasts",
                                "kws_high_ber_summary", "kws_high_ber_mask_decisions")}
        expected = {"current_comparative_conditions": 864,
                    "wide_ber_conditions": 624, "wide_ber_trials": 13_728,
                    "wide_ber_paired_comparisons": 480,
                    "kws_high_ber_pool_contrasts": 144,
                    "kws_high_ber_field_contrasts": 4_608,
                    "kws_high_ber_summary": 12, "kws_high_ber_mask_decisions": 48}
        if counts != expected or db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError(f"combined database count/integrity check failed: {counts}")
        _plots(db, staging)
        for table, name in (("wide_ber_conditions", "wide_ber_conditions.csv"),
                            ("kws_high_ber_summary", "kws_high_ber_summary.csv"),
                            ("kws_high_ber_pool_contrasts", "kws_high_ber_pool_contrasts.csv")):
            rows = db.execute(f"SELECT * FROM {table}")
            with (staging / name).open("x", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow([col[0] for col in rows.description])
                writer.writerows(rows)
        db.close()
        (staging / "README.md").write_text(
            "# Current reliability evidence plus high-BER extension\n\n"
            "The original 864-row `current_comparative_conditions` view is unchanged. "
            "The 624-row `wide_ber_conditions` table contains separately identified "
            "DistilBERT (finalized), Pythia (derived 18-field clean-conformant subset, "
            "not the failed 24-field study), and Qwen (finalized), each with 208 "
            "conditions. Do not pool cross-entropy and token NLL across tasks. "
            "The 160 within-study Holm-adjusted comparisons per model use paired "
            "fault fields and one fixed mask per method/budget.\n\n"
            "The KWS higher-BER study is a separate exploratory 12-pool local-mask "
            "exchange pilot, not a whole-network ranking campaign. It includes "
            "4,608 paired field contrasts, 144 pool means, and 12 BER-level "
            "pool-t summaries. Its nominal intervals and p-values are descriptive, "
            "not a pre-registered confirmation. The original KWS campaign directory "
            "retains its postprocessing-failed runner status; the separately finalized "
            "derived analysis verifies its 96 finalized member bundles.\n")
        manifest = {"schema": "netsurf-wide-ber-consolidation/v2",
                    "base_manifest_sha256": sha256_file(base / "manifest.json"),
                    "base_database_sha256": sha256_file(base / "current_comparative_wide_v1.sqlite"),
                    "qwen_manifest_sha256": qwen["manifest_sha256"],
                    "qwen_payload_sha256": qwen["manifest"]["payload_sha256"],
                    "kws_campaign_sha256": sha256_file(kws_campaign),
                    "kws_derived_payload_sha256": digest_object(kws),
                    "counts": counts,
                    "output_sha256": {p.name: sha256_file(p) for p in sorted(staging.iterdir())
                                      if p.is_file() and p.name != "manifest.json"}}
        manifest["payload_sha256"] = digest_object(manifest)
        (staging / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        (staging / "DERIVED_FINALIZED.json").write_text(json.dumps(
            {"manifest_sha256": sha256_file(staging / "manifest.json")}, sort_keys=True) + "\n")
        os.rename(staging, destination)
        zip_path = destination.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(destination.iterdir()):
                archive.write(path, arcname=f"{destination.name}/{path.name}")
        with zipfile.ZipFile(zip_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("derived ZIP CRC verification failed")
        return {"destination": str(destination), "zip": str(zip_path),
                "zip_sha256": sha256_file(zip_path), "counts": counts}
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "qwen_root", "kws_analysis", "kws_campaign", "destination"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(consolidate(args.base, args.qwen_root, args.kws_analysis,
                                 args.kws_campaign, args.destination), indent=2))


if __name__ == "__main__":
    main()
