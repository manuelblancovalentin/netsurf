"""Offline equal-pool inference; no model loading or outcome-dependent selection."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .conditional_config import PAIRED_SWAP_COMPARISONS
from .cross_pool import load_campaign, verified_receipt, _receipt, _read
from .manifests import digest_object, sha256_file
from .replication_statistics import summarize_replication
from .result_bundle import _write_json, _write_bytes
from .statistics import summarize


def cross_pool_statistics(effects, *, seed, replicates, field_effects=None):
    """P×F×I×C crossed data. Pool is primary, never flattened field/input cells.

    Bootstrap diagnostics are separate: nested/multiway percentile ranges are
    sensitivity checks, not a claim of exact finite-sample confidence coverage.
    """
    from scipy.stats import t
    x = np.asarray(effects, dtype=np.float64)
    if x.ndim != 4 or x.shape[0] < 2 or x.shape[1] < 2 or x.shape[2] < 1 or x.shape[3] != 3 or not np.isfinite(x).all():
        raise ValueError("need >=2 pools, >=2 fields, common inputs and three finite paired contrasts")
    p, f, n, c = x.shape
    field_means = x.mean(axis=2) if field_effects is None else np.asarray(field_effects, dtype=np.float64)
    if field_means.shape != (p, f, c) or not np.isfinite(field_means).all() or not np.allclose(field_means, x.mean(axis=2), rtol=0, atol=1e-12):
        raise ValueError("authoritative scalar fields disagree with diagnostic contributions")
    means = field_means.mean(axis=1)
    result = []
    for i, name in enumerate(PAIRED_SWAP_COMPARISONS):
        est = summarize(means[:, i]).to_dict()
        se = est["standard_error"]
        result.append({"comparison": name, "pool_mean_effect": est,
            "p_value_two_sided": float(2 * t.sf(abs(est["mean"] / se), p - 1)) if se else (1. if est["mean"] == 0 else None),
            "pool_effects": means[:, i].tolist(), "median": float(np.median(means[:, i])),
            "quantiles": {str(q): float(np.quantile(means[:, i], q)) for q in (0, .05, .25, .5, .75, .95, 1)},
            "estimated_improving_pools": int(np.count_nonzero(means[:, i] < 0)),
            "estimated_harmful_pools": int(np.count_nonzero(means[:, i] > 0)),
            "estimated_zero_pools": int(np.count_nonzero(means[:, i] == 0)),
            "fraction_negative_estimated_effect": float(np.mean(means[:, i] < 0)),
            "sign_interpretation": "Signs of noisy pool effect estimates, not a proven probability of true improvement"})
        fraction = float(np.mean(means[:, i] < 0)); z = 1.959963984540054
        center = (fraction + z*z/(2*p)) / (1 + z*z/p)
        half = z * np.sqrt(fraction*(1-fraction)/p + z*z/(4*p*p)) / (1 + z*z/p)
        result[-1]["negative_estimate_fraction_wilson95"] = [center-half, center+half]
    order = sorted((1, 2), key=lambda i: result[i]["p_value_two_sided"] if result[i]["p_value_two_sided"] is not None else 1.)
    previous = 0.
    for j, i in enumerate(order):
        raw = result[i]["p_value_two_sided"]
        previous = max(previous, min(1., (2-j) * (raw if raw is not None else 1.)))
        result[i]["secondary_holm_p_value"] = previous if raw is not None else None
        result[i]["secondary_family_mean_ci"] = summarize(means[:, i], confidence=.975).to_dict()
    rng = np.random.Generator(np.random.PCG64(seed))
    boot = {name: [] for name in ("pools", "fields_conditional", "inputs_shared_conditional", "crossed_diagnostic")}
    input_means = x.mean(axis=(0, 1))
    for _ in range(replicates):
        picked_pools = rng.integers(p, size=p)
        input_weights = rng.multinomial(n, np.full(n, 1/n)) / n
        field_only = []; crossed = []
        for j in range(p):
            weights = rng.multinomial(f, np.full(f, 1/f)) / f
            field_only.append(weights @ field_means[j])
            # Shared input weights preserve covariance across every pool and field.
            crossed.append(np.einsum("f,fic,i->c", weights, x[picked_pools[j]], input_weights))
        boot["pools"].append(means[picked_pools].mean(axis=0))
        boot["fields_conditional"].append(np.mean(field_only, axis=0))
        boot["inputs_shared_conditional"].append(input_weights @ input_means)
        boot["crossed_diagnostic"].append(np.mean(crossed, axis=0))
    intervals = {name: {"percentile_95_range": np.quantile(values, [.025, .975], axis=0).T.tolist(),
                        "standard_deviation": np.std(values, axis=0, ddof=1).tolist()}
                 for name, values in boot.items()}
    return {"schema_version": "1.0.0", "principal_unit": "candidate pool",
        "pool_count": p, "fields_per_pool": f, "shared_input_count": n,
        "comparisons": result, "bootstrap": {"seed": seed, "replicates": replicates, "diagnostics": intervals},
        "assumptions": "Primary pool-t interval conditions on shared inputs and selection data, includes finite-field noise and assumes independent probability-sampled pools/seeds. Shared-input resampling assumes exchangeable clips despite fixed contiguous TFDS ordering. Crossed/nested resampling can overcount finite-field noise and is an approximate sensitivity diagnostic, not an additional confirmatory test. No inference over trained checkpoints or physical exposure."}


def aggregate_campaign(path, output, *, plots=True):
    path = Path(path).resolve(); record = load_campaign(path); digest = sha256_file(path)
    root = path.parent
    evidence, arrays, scalar_arrays, common_ids, seeds = [], [], [], None, set()
    for pool in record["pools"]:
        pid = pool["pool_id"]
        selection_receipt, selection = verified_receipt(_receipt(root, pid, "selection"), digest, pid, "selection")
        receipt, payload = verified_receipt(_receipt(root, pid, "final"), digest, pid, "final")
        bundle = Path(receipt["bundle"])
        with (bundle / "measurements/paired_swap_fields.csv").open() as handle:
            fields = list(csv.DictReader(handle))
        for row in fields:
            for key in ("field", "field_seed", "path_length", "authoritative_evaluation_requests"):
                row[key] = int(row[key])
            for key in ("logical_ber", "target_minus_baseline"):
                row[key] = float(row[key])
        count = record["design"]["final_fields"]
        analysis = summarize_replication(fields, count)
        current_seeds = {r["field_seed"] for r in fields}
        if current_seeds & seeds:
            raise ValueError("cross-pool field seeds overlap")
        seeds.update(current_seeds)
        diag = payload["paired_input_diagnostics"]
        ids = diag["sample_identities"]
        if common_ids is not None and ids != common_ids:
            raise ValueError("cross-pool shared input identities differ")
        common_ids = ids
        tensor = np.zeros((count, len(ids), 3), dtype=np.float64)
        seen = set()
        field_map = {(r["field"], r["comparison"]): r for r in fields}
        for row in diag["records"]:
            key = (row["field"], row["comparison"])
            if key in seen or key not in field_map:
                raise ValueError("duplicated/unprespecified input diagnostic record")
            seen.add(key); scalar = field_map[key]
            if any(row[k] != scalar[k] for k in ("field_seed", "logical_ber", "master_pairing_key")):
                raise ValueError("input diagnostics are not paired with scalar fields")
            vector = np.asarray(row["contributions"], dtype=np.float64)
            if vector.shape != (len(ids),) or not np.isfinite(vector).all() or not np.isclose(vector.mean(), scalar["target_minus_baseline"], rtol=0, atol=1e-12):
                raise ValueError("input diagnostic reduction mismatch")
            tensor[row["field"], :, PAIRED_SWAP_COMPARISONS.index(row["comparison"])] = vector
        if len(seen) != count * 3:
            raise ValueError("incomplete input diagnostics")
        arrays.append(tensor)
        scalar_arrays.append(np.array([[field_map[(f, name)]["target_minus_baseline"]
                                       for name in PAIRED_SWAP_COMPARISONS] for f in range(count)]))
        selection_rows = _read(Path(selection_receipt["bundle"]) / "artifacts/restricted_oracle_masks.json")
        evidence.append({"pool_id": pid, "candidates": pool["candidates"],
            "selection_payload_sha256": selection_receipt["payload_sha256"], "final_payload_sha256": receipt["payload_sha256"],
            "selection": selection_rows, "field_analysis": analysis,
            "comparisons": payload["paired_swap_evaluation"]["summaries"],
            "selection_runtime_seconds": _read(Path(selection_receipt["bundle"]) / "manifest.json")["execution"]["runtime_seconds"],
            "final_runtime_seconds": _read(bundle / "manifest.json")["execution"]["runtime_seconds"]})
    statistics = cross_pool_statistics(np.stack(arrays), seed=record["statistics"]["bootstrap_seed"], replicates=record["statistics"]["bootstrap_replicates"], field_effects=np.stack(scalar_arrays))
    # Execution durations are incidental metadata, never scientific hash content.
    costs = [{k: r[k] for k in ("pool_id", "selection_runtime_seconds", "final_runtime_seconds")} for r in evidence]
    for row in evidence:
        row.pop("selection_runtime_seconds"); row.pop("final_runtime_seconds")
    payload = {"schema_version": "1.0.0", "campaign_sha256": digest, "pool_evidence": evidence,
               "sample_identities": common_ids, "statistics": statistics}
    destination = Path(output).resolve(); destination.mkdir(parents=True, exist_ok=False)
    _write_json(destination / "scientific_payload.json", payload)
    _write_json(destination / "costs.json", {"schema_version": "1.0.0", "pools": costs, "operation_plan": record["operation_counts"]})
    lines = ["# M9.6 cross-pool protection comparison", "", f"Campaign SHA-256: `{digest}`", "",
             f"Purpose: **{record['purpose']}**. " + ("Bounded software validation on previously exposed inputs; not confirmation evidence." if record['purpose'] == "engineering" else "Prespecified campaign; interpretation remains conditional on recorded exposure and sampling assumptions."), "",
             "All prespecified pools retained. Negative CE differences favor the target. Ideal event suppression is not physical mitigation.", "",
             "| Contrast | Pools | Mean CE difference | Pool-t 95% CI | Negative / zero / positive estimated pool effects |", "|---|---:|---:|---|---|"]
    for row in statistics["comparisons"]:
        e = row["pool_mean_effect"]
        lines.append(f"| {row['comparison']} | {e['count']} | {e['mean']:.9g} | {e['confidence_interval']} | {row['estimated_improving_pools']} / {row['estimated_zero_pools']} / {row['estimated_harmful_pools']} |")
    lines += ["", "## Pool-level selection and held-out effects", "",
              "Masks contain indices in each pool's frozen candidate order. Switches concern selection fields, not final outcomes.", "",
              "| Pool | Original / isolated / selected | Selection winner switches | Selected↔S5 exchanges | Selected−S5 CE | Field 95% CI | Unique final states (shared by contrasts) |",
              "|---|---|---:|---:|---:|---|---:|"]
    for row in evidence:
        selection = row["selection"][0]
        comparison = row["comparisons"][0]
        effect = comparison["target_minus_baseline"]
        lines.append(f"| {row['pool_id']} | {selection['original_mask_indices']} / {selection['isolated_mask_indices']} / {selection['selected_mask_indices']} | "
                     f"{selection['model_selection_stability']['cumulative_switch_count']} | {len(comparison['path'])} | {effect['mean']:.9g} | "
                     f"{effect['confidence_interval']} | {comparison['actual_unique_authoritative_inferences_shared_across_comparisons']} |")
    lines += ["", "## Prespecified cross-pool tests", "",
              "| Contrast | Two-sided pool-t p | Secondary Holm p | Secondary 97.5% CI |",
              "|---|---:|---:|---|"]
    for row in statistics["comparisons"]:
        lines.append(f"| {row['comparison']} | {row['p_value_two_sided']} | {row.get('secondary_holm_p_value', 'primary; not adjusted')} | "
                     f"{row.get('secondary_family_mean_ci', {}).get('confidence_interval', 'primary 95% above')} |")
    lines += ["", "A null p-value means the empirical standard error is zero for a nonzero effect; no artificial significance is assigned."]
    lines += ["", statistics["assumptions"], "", "Per-pool masks, isolated damages, runner-up margins, all mask risks, selection switches, field intervals and actual inference counts remain in scientific_payload.json. Timings are in costs.json. Primary is selected−S5; two secondary p-values use Holm. Finite-sample signs do not prove true improvement probabilities.", "",
              f"Scientific payload SHA-256: `{digest_object(payload)}`"]
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(3, 1, figsize=(9, 10), constrained_layout=True)
        for i, ax in enumerate(axes):
            vals = statistics["comparisons"][i]["pool_effects"]
            intervals = [r["field_analysis"]["comparisons"][i]["mean_effect"]["confidence_interval"] for r in evidence]
            ax.errorbar(range(len(vals)), vals, yerr=np.abs(np.array(intervals).T - np.asarray(vals)), fmt="o", capsize=3)
            ax.axhline(0, color="gray", linewidth=.8); ax.set_title(PAIRED_SWAP_COMPARISONS[i]); ax.set_ylabel("Δ expected CE")
            ax.set_xticks(range(len(vals)), [r["pool_id"] for r in evidence], rotation=45)
        figure.savefig(destination / "pool_effects.png", dpi=150); plt.close(figure)
        lines += ["", "![Pool effects](pool_effects.png)", "", "Each interval is over independent whole fault fields within its fixed pool and shared input set, not the cross-pool interval."]
    _write_bytes(destination / "report.md", ("\n".join(lines) + "\n").encode())
    _write_json(destination / "checksums.json", {"schema_version": "1.0.0", "files": {p.name: sha256_file(p) for p in sorted(destination.iterdir()) if p.is_file()}})
    _write_bytes(destination / "FINALIZED", (f"scientific_payload_sha256={digest_object(payload)}\n").encode())
    return {"output": str(destination), "scientific_payload_sha256": digest_object(payload)}
