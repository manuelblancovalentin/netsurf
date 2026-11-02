"""Offline scalar-cohort cross-pool analysis; never fabricate additive inputs."""
from pathlib import Path
import numpy as np

from .conditional_config import PAIRED_SWAP_COMPARISONS
from .cohort_risk import CONTRACT, cohort_risk, score_digest
from .manifests import digest_object, sha256_file
from .statistics import summarize
from .result_bundle import _write_json, _write_bytes


def scalar_pool_statistics(effects):
    from scipy.stats import t
    x = np.asarray(effects, dtype=np.float64)
    if x.ndim != 3 or x.shape[0] < 2 or x.shape[1] < 2 or x.shape[2] != 3 or not np.isfinite(x).all():
        raise ValueError("need P>=2 pools × F>=2 whole fields × 3 finite contrasts")
    means = x.mean(axis=1)
    rows = []
    for i, name in enumerate(PAIRED_SWAP_COMPARISONS):
        e = summarize(means[:, i]).to_dict()
        se = e["standard_error"]
        rows.append({"comparison": name, "pool_mean_effect": e, "pool_effects": means[:, i].tolist(),
            "p_value_two_sided": float(2 * t.sf(abs(e["mean"] / se), len(means) - 1)) if se else (1. if e["mean"] == 0 else None),
            "field_intervals_by_pool": [summarize(x[j, :, i]).to_dict() for j in range(len(x))]})
    previous = 0.
    for rank, i in enumerate(sorted((1, 2), key=lambda i: rows[i]["p_value_two_sided"] if rows[i]["p_value_two_sided"] is not None else 1.)):
        p = rows[i]["p_value_two_sided"]
        previous = max(previous, min(1., (2 - rank) * (1. if p is None else p)))
        rows[i]["secondary_holm_p_value"] = None if p is None else previous
        rows[i]["secondary_family_mean_ci"] = summarize(means[:, i], confidence=.975).to_dict()
    return {"schema_version": "1.0.0", "pool_count": len(x), "fields_per_pool": x.shape[1],
        "comparisons": rows, "scalar_cohort_contract": CONTRACT,
        "assumptions": "Equal-pool Student-t inference conditional on shared complete files, frozen checkpoint and selection protocol; finite-field noise included. Independent sampled pools/seeds required. Small-pool and tiny-cohort estimates are engineering evidence, not efficacy. No input-population interval; no pairs/windows treated as independent observations."}


def verify_score_evidence(evidence):
    from .cohort_risk import ordering_diagnostics
    if evidence.get("schema_version") != "1.0.0" or evidence.get("risk_contract") != CONTRACT:
        raise ValueError("unknown scalar-cohort contract")
    for key, cohort in evidence["cohorts"].items():
        if digest_object({k: v for k, v in cohort.items() if k != "cohort_sha256"}) != key:
            raise ValueError("cohort identity checksum mismatch")
        ids = [v["sample_id"] for v in cohort["files"]]
        if len(set(ids)) != len(ids) or any(v["window_count"] != 196 or len(v["ordered_window_ids"]) != 196 for v in cohort["files"]):
            raise ValueError("incomplete or duplicate file/window identities")
    for row in evidence["evaluations"]:
        cohort = evidence["cohorts"][row["cohort_sha256"]]
        scores = row["scores"]
        if (score_digest(scores) != row["scores_sha256"]
                or [float(v).hex() for v in scores] != row["scores_float64_hex"]
                or cohort_risk([v["label"] for v in cohort["files"]], scores) != row["risk"]):
            raise ValueError("saved complete-file scores do not reproduce scalar risk")
        if "file_roc_auc" in row and row["risk"] != 1 - row["file_roc_auc"]:
            raise ValueError("file AUC and scalar risk disagree")
        if "events" in row:
            e = row["events"]
            if e["proposed_events"] != e["applied_events"] + e["suppressed_events"]:
                raise ValueError("fault event accounting differs")


def aggregate_scalar_campaign(path, output, *, plots=True):
    from .cross_pool import load_campaign, verified_receipt, _receipt
    from .cross_pool_analysis import read_field_records
    from .replication_statistics import summarize_replication
    path = Path(path).resolve()
    record, digest = load_campaign(path), sha256_file(path)
    if record["schema_version"] != "1.2.0":
        raise ValueError("scalar cohort analysis requires campaign 1.2")
    evidence, tensors, common_ids, seeds = [], [], None, set()
    for pool in record["pools"]:
        selection_receipt, selection = verified_receipt(_receipt(path.parent, pool["pool_id"], "selection"), digest, pool["pool_id"], "selection")
        receipt, payload = verified_receipt(_receipt(path.parent, pool["pool_id"], "final"), digest, pool["pool_id"], "final")
        verify_score_evidence(payload["scalar_cohort_evidence"])
        ids = payload["sample_identities"]["held_out_evaluation"]
        if common_ids is not None and ids != common_ids:
            raise ValueError("cross-pool complete-file cohorts differ")
        common_ids = ids
        fields = read_field_records(Path(receipt["bundle"]) / "measurements/paired_swap_fields.csv")
        for row in fields:
            for key in ("field", "field_seed"):
                row[key] = int(row[key])
            for key in ("target_minus_baseline", "logical_ber"):
                row[key] = float(row[key])
        count = record["design"]["final_fields"]
        field_analysis = summarize_replication(fields, count)
        current = {r["field_seed"] for r in fields}
        if current & seeds:
            raise ValueError("cross-pool fault seeds overlap")
        seeds |= current
        mapping = {(r["field"], r["comparison"]): r["target_minus_baseline"] for r in fields}
        tensors.append([[mapping[(f, name)] for name in PAIRED_SWAP_COMPARISONS] for f in range(count)])
        evidence.append({"pool_id": pool["pool_id"], "selection_payload_sha256": selection_receipt["payload_sha256"],
            "final_payload_sha256": receipt["payload_sha256"], "field_analysis": field_analysis,
            "selection": selection["restricted_oracle"]["summaries"],
            "final_cohort_evidence_sha256": payload["scalar_cohort_evidence_sha256"]})
    stats = scalar_pool_statistics(tensors)
    payload = {"schema_version": "1.0.0", "campaign_sha256": digest, "sample_identities": common_ids,
               "pool_evidence": evidence, "statistics": stats}
    destination = Path(output).resolve(); destination.mkdir(parents=True, exist_ok=False)
    _write_json(destination / "scientific_payload.json", payload)
    lines = ["# Complete-file ToyCar conditional protection", "", f"Purpose: {record['purpose']}. Risk: **1 − file ROC-AUC**.", "",
        "One whole persistent fault field acts on every window/file. Negative paired risk differences favor the target. "
        "All pools retained; no post-hoc final optimum. Primary: selected−S5. Two secondary tests use Holm.", "",
        "| Contrast | Pools | Mean risk difference | Pool-t 95% CI | p | Secondary Holm p |", "|---|---:|---:|---|---|---|"]
    for row in stats["comparisons"]:
        e = row["pool_mean_effect"]
        lines.append(f"| {row['comparison']} | {e['count']} | {e['mean']:.9g} | {e['confidence_interval']} | {row['p_value_two_sided']} | {row.get('secondary_holm_p_value', 'primary')} |")
    lines += ["", stats["assumptions"], "", "Individual pool/field intervals and selection stability are in scientific_payload.json. File-score/order diagnostics remain in verified source bundles.", ""]
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(9, 9), constrained_layout=True)
        for i, ax in enumerate(axes):
            row = stats["comparisons"][i]
            vals = np.array(row["pool_effects"])
            ci = np.array([e["confidence_interval"] for e in row["field_intervals_by_pool"]]).T
            ax.errorbar(range(len(vals)), vals, yerr=np.abs(ci - vals), fmt="o", capsize=3)
            ax.axhline(0, color="gray"); ax.set_title(row["comparison"])
            ax.set_ylabel("Δ expected (1 − file ROC-AUC)"); ax.set_xlabel("Prespecified candidate pool")
        fig.savefig(destination / "pool_effects.png", dpi=140); plt.close(fig)
        lines += ["![Pool effects](pool_effects.png)", "", "Intervals describe whole-field uncertainty within each pool, not independent file pairs.", ""]
    lines += [f"Scientific payload SHA-256: `{digest_object(payload)}`", ""]
    _write_bytes(destination / "report.md", "\n".join(lines).encode())
    _write_json(destination / "checksums.json", {"schema_version": "1.0.0", "files": {p.name: sha256_file(p) for p in sorted(destination.iterdir()) if p.is_file()}})
    _write_bytes(destination / "FINALIZED", f"scientific_payload_sha256={digest_object(payload)}\n".encode())
    return {"output": str(destination), "scientific_payload_sha256": digest_object(payload)}
