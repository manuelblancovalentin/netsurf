"""Offline joint BER analysis; pools, not clips or fields, are generalization units."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .conditional_config import PAIRED_SWAP_COMPARISONS
from .cross_pool import _read, _receipt, verified_receipt
from .cross_pool_analysis import cross_pool_statistics
from .manifests import digest_object, sha256_file
from .multiber_campaign import PILOT_PROTOCOL, VERSION, load_joint
from .replication_statistics import summarize_replication
from .result_bundle import _write_bytes, _write_json
from .statistics import summarize


def _holm(rows):
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p_value_two_sided"] if rows[i]["p_value_two_sided"] is not None else 1.)
    prior = 0.
    for rank, index in enumerate(order):
        raw = rows[index]["p_value_two_sided"]
        prior = max(prior, min(1., (len(rows) - rank) * (1. if raw is None else raw)))
        rows[index]["holm_p_value"] = None if raw is None else prior


def _test(values, confidence=.95):
    from scipy.stats import t
    values = np.asarray(values, dtype=np.float64)
    estimate = summarize(values, confidence=confidence).to_dict()
    mean, se = estimate["mean"], estimate["standard_error"]
    p = float(2 * t.sf(abs(mean / se), len(values) - 1)) if se else (1. if mean == 0 else None)
    return {"estimate": estimate, "p_value_two_sided": p,
            "favorable_pools": int(np.count_nonzero(values < 0)),
            "adverse_pools": int(np.count_nonzero(values > 0)),
            "zero_pools": int(np.count_nonzero(values == 0)),
            "pool_effects": values.tolist()}


def _member(arm, pool):
    root = Path(arm["manifest"]).parent
    pid = pool["pool_id"]
    s, _ = verified_receipt(_receipt(root, pid, "selection"), arm["manifest_sha256"], pid, "selection")
    f, payload = verified_receipt(_receipt(root, pid, "final"), arm["manifest_sha256"], pid, "final")
    selection = _read(Path(s["bundle"]) / "artifacts/restricted_oracle_masks.json")
    if len(selection) != 1 or not selection[0]["global_protection_cardinality_preserved"]:
        raise ValueError("selection mask budget differs")
    chosen = selection[0]
    with (Path(f["bundle"]) / "measurements/paired_swap_fields.csv").open() as handle:
        fields = list(csv.DictReader(handle))
    with (Path(f["bundle"]) / "measurements/paired_swap_steps.csv").open() as handle:
        steps = list(csv.DictReader(handle))
    for step in steps:
        for side in ("removed", "added"):
            proposed = int(step[f"{side}_proposed_events"])
            applied = int(step[f"{side}_applied_events"])
            suppressed = int(step[f"{side}_suppressed_events"])
            if proposed != applied + suppressed:
                raise ValueError("paired-step event accounting differs")
    event_counts = {side: {kind: float(np.mean([int(s[f"{side}_{kind}_events"]) for s in steps]))
                           for kind in ("proposed", "applied", "suppressed")}
                    for side in ("removed", "added")} if steps else {}
    count = arm["operation_counts"]["final_fields_per_pool"]
    for row in fields:
        for key in ("field", "field_seed", "path_length", "authoritative_evaluation_requests"):
            row[key] = int(row[key])
        for key in ("logical_ber", "target_minus_baseline"):
            row[key] = float(row[key])
        if row["logical_ber"] != arm["logical_ber"]:
            raise ValueError("BER field differs from frozen arm")
    field_analysis = summarize_replication(fields, count)
    diag = payload["paired_input_diagnostics"]
    ids = diag["sample_identities"]
    matrix = np.full((count, len(ids), 3), np.nan)
    field_map = {(r["field"], r["comparison"]): r for r in fields}
    if len(field_map) != 3 * count:
        raise ValueError("incomplete field family")
    for row in diag["records"]:
        key = (row["field"], row["comparison"])
        if key not in field_map or row["field_seed"] != field_map[key]["field_seed"] or row["master_pairing_key"] != field_map[key]["master_pairing_key"]:
            raise ValueError("paired input record differs from field")
        vector = np.asarray(row["contributions"], dtype=np.float64)
        if vector.shape != (len(ids),) or not np.isclose(vector.mean(), field_map[key]["target_minus_baseline"], atol=1e-12, rtol=0):
            raise ValueError("paired input reduction differs")
        slot = PAIRED_SWAP_COMPARISONS.index(row["comparison"])
        if np.isfinite(matrix[row["field"], :, slot]).any():
            raise ValueError("duplicate paired input record")
        matrix[row["field"], :, slot] = vector
    if not np.isfinite(matrix).all():
        raise ValueError("missing paired input records")
    # Additive isolated prediction concerns mask contrasts only; the unknown
    # outside-background intercept cancels. Signed isolated damages are retained.
    damages = np.asarray([r["damage"] for r in chosen["isolated_damages"]])
    original, selected = map(tuple, (chosen["original_mask_indices"], chosen["selected_mask_indices"]))
    risks = {tuple(r["indices"]): r["mean_risk"] for r in chosen["eligible_masks"]}
    p = arm["logical_ber"]
    predicted = p * (sum(damages[i] for i in original) - sum(damages[i] for i in selected))
    observed = risks[selected] - risks[original]
    return {"ids": ids, "matrix": matrix, "fields": fields, "field_analysis": field_analysis,
            "selection": chosen, "selection_sha256": s["payload_sha256"], "final_sha256": f["payload_sha256"],
            "paired_step_event_counts_per_request": event_counts,
            "additive_prediction_error": observed - predicted,
            "additive_prediction": predicted, "selection_exact_contrast": observed,
            "selection_regret_among_15": risks[selected] - min(risks.values()),
            "bounded_heldout_regret_vs_s5_and_isolated": max(0.,
                matrix[:, :, 0].mean(), matrix[:, :, 1].mean())}


def aggregate_joint(path, output):
    parent = load_joint(path)
    pilot = parent["protocol"] == PILOT_PROTOCOL
    arrays, evidence, all_ids, all_seeds = [], [], None, set()
    for arm in parent["arms"]:
        child = _read(arm["manifest"])
        arm_arrays, rows = [], []
        for pool in child["pools"]:
            member = _member(arm, pool)
            if all_ids is not None and member["ids"] != all_ids:
                raise ValueError("cross-BER or cross-pool input identity differs")
            all_ids = member["ids"]
            seeds = {r["field_seed"] for r in member["fields"]}
            if seeds & all_seeds:
                raise ValueError("cross-BER or cross-pool final field seed collision")
            all_seeds.update(seeds)
            arm_arrays.append(member.pop("matrix"))
            member.pop("ids")
            member.pop("fields")
            rows.append({"pool_id": pool["pool_id"], **member})
        arrays.append(np.stack(arm_arrays))
        evidence.append({"arm": arm["id"], "logical_ber": arm["logical_ber"], "pools": rows})
    cube = np.stack(arrays)  # B x P x F x I x three policy contrasts
    if cube.shape[0] != len(parent["ber_grid"]) or cube.shape[1] != len(parent["source_m96"]["candidate_pools"]):
        raise ValueError("incomplete joint BER-by-pool cube")
    within = []
    for b, arm in enumerate(parent["arms"]):
        if cube.shape[1] >= 2:
            stats = cross_pool_statistics(cube[b], seed=parent["statistics"]["bootstrap_seed"] + b,
                                          replicates=parent["statistics"]["bootstrap_replicates"])
        else:
            stats = {"comparisons": [], "engineering_only": True}
        within.append({"logical_ber": arm["logical_ber"], "statistics": stats})
    pool_means = cube.mean(axis=(2, 3))  # B x P x C: shared pool ordering
    primary = []
    if not pilot and len(parent["ber_grid"]) == 3 and cube.shape[1] >= 2:
        for b in (0, 1):
            row = {"comparison": "selected_vs_exact_isolated_change",
                   "lower_ber": parent["ber_grid"][b], "anchor_ber": .01,
                   "definition": "lower-BER selected-minus-isolated CE minus same-window BER-0.01 selected-minus-isolated CE",
                   **_test(pool_means[b, :, 1] - pool_means[2, :, 1], confidence=.975)}
            primary.append(row)
        _holm(primary)
    secondary = []
    if cube.shape[1] >= 2:
        for b, probability in enumerate(parent["ber_grid"]):
            for c, comparison in enumerate(PAIRED_SWAP_COMPARISONS):
                secondary.append({"logical_ber": probability, "comparison": comparison,
                                  **_test(pool_means[b, :, c], confidence=.95 if pilot else 1-.05/9)})
        if not pilot:
            _holm(secondary)
        else:
            for row in secondary:
                row["exploratory_only"] = True
    decisions = []
    for pool_index, source_pool in enumerate(parent["source_m96"]["candidate_pools"]):
        masks = [row["pools"][pool_index]["selection"]["selected_mask_indices"] for row in evidence]
        decisions.append({"pool_id": source_pool["pool_id"], "selected_by_ber": masks,
                          "distinct_selected_masks": len({tuple(m) for m in masks}),
                          "changes_between_adjacent_ber": [a != b for a, b in zip(masks, masks[1:])]})
    transfer = []
    if not pilot and len(evidence) == 3:
        for b in (0, 1):
            for pool_index, source_pool in enumerate(parent["source_m96"]["candidate_pools"]):
                low = evidence[b]["pools"][pool_index]["selection"]
                anchor = evidence[2]["pools"][pool_index]["selection"]
                risks = {tuple(r["indices"]): r["mean_risk"] for r in low["eligible_masks"]}
                transferred = tuple(anchor["selected_mask_indices"])
                transfer.append({"logical_ber": parent["ber_grid"][b], "pool_id": source_pool["pool_id"],
                                 "anchor_mask": transferred, "lower_ber_selected": low["selected_mask_indices"],
                                 "selection_data_transfer_regret": risks[transferred] - risks[tuple(low["selected_mask_indices"])]})
    payload = {"schema_version": VERSION, "joint_campaign_sha256": sha256_file(path),
               "sample_identities": all_ids, "ber_grid": parent["ber_grid"], "evidence": evidence,
               "within_ber": within, "primary_change_family": primary,
               **({"exploratory_within_ber_comparisons": secondary} if pilot else {"secondary_nine_comparison_family": secondary}),
               "decision_changes": decisions,
               "optional_selection_data_transfer_diagnostic": transfer,
               "limitations": ("Previously exposed inputs and reduced field counts make this an exploratory engineering pilot. Selection and final fault fields are independent within this pilot, but no confirmatory effectiveness claim follows. Twelve sampled pools, shared clips, heavy tails and selection uncertainty remain. No whole-network optimum or physical SEU rate is inferred." if pilot else "Transfer diagnostic uses model-selection response tables; held-out transfer requires separately frozen fourth-policy final evaluations. Pool-t inference has 12 sampled pools, shared clips and independent fault fields; heavy tails and selection uncertainty remain. No whole-network optimum or physical SEU rate is inferred.")}
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    _write_json(destination / "scientific_payload.json", payload)
    lines = ["# KWS higher-BER exploratory pilot" if pilot else "# M9.7 joint KWS BER campaign", "", f"Frozen campaign SHA-256: `{sha256_file(path)}`", "",
             "Negative cross-entropy contrast favors the first named policy. Pools are the cross-pool units; fields are stochastic replicates and clips are shared across all arms.", "",
             ("All inputs were previously exposed. This pilot has no confirmatory hypothesis or untouched evaluation cohort." if pilot else "BER 0.01 is a same-window anchor with a historically known M9.6 outcome, not an untouched discovery hypothesis."), ""]
    if pilot:
        lines += ["## Exploratory study", "", "No BER endpoint or policy contrast is treated as confirmatory. Descriptive intervals are not multiplicity-adjusted."]
    else:
        lines += ["## Prespecified primary BER-change family", "",
                  "| Lower BER | Mean change | Pool CI | Raw p | Holm p | Favorable/adverse pools |", "|---:|---:|---|---:|---:|---:|"]
    for row in primary:
        e = row["estimate"]
        lines.append(f"| {row['lower_ber']} | {e['mean']:.9g} | {e['confidence_interval']} | {row['p_value_two_sided']} | {row['holm_p_value']} | {row['favorable_pools']}/{row['adverse_pools']} |")
    lines += ["", "## Descriptive within-BER comparisons (exploratory)" if pilot else "## Within-BER policy comparisons (secondary nine-test Holm family)", "",
              "| BER | Contrast | Mean CE | Pool CI | Favorable/adverse pools |" if pilot else "| BER | Contrast | Mean CE | Pool CI | Holm p | Favorable/adverse pools |",
              "|---:|---|---:|---|---:|" if pilot else "|---:|---|---:|---|---:|---:|"]
    for row in secondary:
        e = row["estimate"]
        if pilot:
            lines.append(f"| {row['logical_ber']} | {row['comparison']} | {e['mean']:.9g} | {e['confidence_interval']} | {row['favorable_pools']}/{row['adverse_pools']} |")
        else:
            lines.append(f"| {row['logical_ber']} | {row['comparison']} | {e['mean']:.9g} | {e['confidence_interval']} | {row['holm_p_value']} | {row['favorable_pools']}/{row['adverse_pools']} |")
    lines += ["", "All 15 local masks retain the 22,016-bit global budget. Selection stability, signed isolated damage, additive prediction error, bounded regret, field-tail sensitivity, and verified event/schedule records are in scientific_payload.json.",
              "", ("No cross-BER transfer or confirmatory significance test is specified for this pilot." if pilot else "The optional BER-0.01 mask transfer is calculated only on lower-BER model-selection tables. It is descriptive and excluded from confirmatory tests. Held-out transfer remains unresolved."),
              "", payload["limitations"], "", f"Scientific payload SHA-256: `{digest_object(payload)}`"]
    _write_bytes(destination / "report.md", ("\n".join(lines) + "\n").encode())
    _write_json(destination / "checksums.json", {"schema_version": VERSION,
        "files": {p.name: sha256_file(p) for p in destination.iterdir() if p.is_file()}})
    _write_bytes(destination / "FINALIZED", (f"scientific_payload_sha256={digest_object(payload)}\n").encode())
    return {"output": str(destination), "scientific_payload_sha256": digest_object(payload)}
