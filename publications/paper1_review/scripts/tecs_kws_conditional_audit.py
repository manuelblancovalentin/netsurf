#!/usr/bin/env python3
"""Offline audit of the 32-pool KWS conditional-vs-isolated study (read-only source; no inference).

Reports: (i) background risk level at selection, (ii) attainable within-pool effect
(best minus worst of the 15 fixed-budget masks on the selection response tables),
(iii) selection noise (best-second margin vs field SD), (iv) finite-sample
non-transitivity of the three separately path-estimated held-out contrasts,
(v) candidate-bit composition.  Writes one JSON to a new file (exclusive create).
"""
import json, sys, hashlib
from pathlib import Path
import numpy as np
SRC = Path("/projects/netsurf/manuelbv/workspace/implementation-worktrees/kws-transition-validation/workspace/runs/kws_transition_independent_pools_p05_derived_recovery_v1/scientific_payload.json")
out = Path(sys.argv[1])
d = json.loads(SRC.read_text())
pools = d["pool_evidence"]
rec = []
for p in pools:
    sel = p["selection"][0] if isinstance(p["selection"], list) else p["selection"]
    masks = sel["eligible_masks"]
    means = np.array([m["mean_risk"] for m in masks])
    fr = np.array([m["field_risks"] for m in masks])            # masks x selection fields
    best = int(np.argmin(means))
    # paired (common outside field) SD of each mask's difference from the best mask
    diffs = fr - fr[best]
    paired_se = diffs.std(axis=1, ddof=1) / np.sqrt(fr.shape[1])
    order = np.argsort(means)
    c = {x["comparison"]: x for x in p["comparisons"]}
    s5 = c["restricted_selected_vs_original_s5"]["target_minus_baseline"]["mean"]
    si = c["restricted_selected_vs_exact_isolated"]["target_minus_baseline"]["mean"]
    i5 = c["exact_isolated_vs_original_s5"]["target_minus_baseline"]["mean"]
    se = {k: v["target_minus_baseline"]["standard_error"] for k, v in c.items()}
    idx = lambda key: tuple(sel[key])
    def risk_of(ind):
        return float(next(m["mean_risk"] for m in masks if tuple(m["indices"]) == tuple(ind)))
    rec.append(dict(
        pool=p["pool_id"], selection_fields=int(fr.shape[1]), masks=len(masks),
        selection_mean_risk_level=float(means.mean()),
        selection_field_sd_of_risk=float(fr.std(axis=1, ddof=1).mean()),
        attainable_range_selection=float(means.max() - means.min()),
        best_second_margin=float(means[order[1]] - means[order[0]]),
        paired_se_of_second_vs_best=float(paired_se[order[1]]),
        selection_gain_selected_vs_isolated=risk_of(idx("isolated_mask_indices")) - risk_of(idx("selected_mask_indices")),
        selection_gain_selected_vs_original=risk_of(idx("original_mask_indices")) - risk_of(idx("selected_mask_indices")),
        heldout_selected_minus_s5=s5, heldout_selected_minus_isolated=si, heldout_isolated_minus_s5=i5,
        transitivity_residual=s5 - (si + i5),
        heldout_se=se,
        masks_differ_selected_vs_isolated=idx("selected_mask_indices") != idx("isolated_mask_indices"),
        candidate_bits=[x["bit_index"] for x in p["candidates"]],
        candidate_protected=[x["source_protected"] for x in p["candidates"]],
        isolated_damages=[x["damage"] for x in sel["isolated_damages"]],
        winner_switches=sel["model_selection_stability"]["cumulative_switch_count"],
    ))
A = lambda k: np.array([r[k] for r in rec], dtype=float)
bits = np.concatenate([r["candidate_bits"] for r in rec])
summary = dict(
    source=str(SRC), source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(), pools=len(rec),
    logical_ber=0.05,
    global_protected_bits=22016,
    selection_mean_risk_level_nats_per_clip=dict(mean=float(A("selection_mean_risk_level").mean()), min=float(A("selection_mean_risk_level").min()), max=float(A("selection_mean_risk_level").max())),
    selection_field_sd_of_risk_mean=float(A("selection_field_sd_of_risk").mean()),
    attainable_range_selection=dict(median=float(np.median(A("attainable_range_selection"))), mean=float(A("attainable_range_selection").mean()), max=float(A("attainable_range_selection").max())),
    best_second_margin=dict(median=float(np.median(A("best_second_margin"))), max=float(A("best_second_margin").max())),
    pools_where_margin_below_one_paired_se=int(np.sum(A("best_second_margin") < A("paired_se_of_second_vs_best"))),
    pools_where_margin_below_two_paired_se=int(np.sum(A("best_second_margin") < 2 * A("paired_se_of_second_vs_best"))),
    in_sample_gain_selected_vs_isolated=dict(mean=float(A("selection_gain_selected_vs_isolated").mean()), median=float(np.median(A("selection_gain_selected_vs_isolated")))),
    heldout_means=dict(selected_minus_s5=float(A("heldout_selected_minus_s5").mean()), selected_minus_isolated=float(A("heldout_selected_minus_isolated").mean()), isolated_minus_s5=float(A("heldout_isolated_minus_s5").mean())),
    heldout_abs_effect_selected_minus_isolated=dict(median=float(np.median(np.abs(A("heldout_selected_minus_isolated")))), max=float(np.abs(A("heldout_selected_minus_isolated")).max())),
    transitivity=dict(mean_residual=float(A("transitivity_residual").mean()), max_abs_residual=float(np.abs(A("transitivity_residual")).max()),
                      pools_exactly_transitive=int(np.sum(np.abs(A("transitivity_residual")) < 1e-12)),
                      note="each contrast is a separately path-conditioned (Rao-Blackwellised) paired-swap estimate on the same master fields; estimands are transitive, finite estimates are not"),
    masks_differ_selected_vs_isolated=int(sum(r["masks_differ_selected_vs_isolated"] for r in rec)),
    candidate_bit_position_histogram={int(b): int(np.sum(bits == b)) for b in range(8)},
    abs_isolated_damage=dict(median=float(np.median(np.abs(np.concatenate([r["isolated_damages"] for r in rec])))), max=float(np.max(np.abs(np.concatenate([r["isolated_damages"] for r in rec]))))),
    minimum_detectable_effect_80pct_power=float((2.04 + 0.85) * d["statistics"]["comparisons"][1]["pool_mean_effect"]["standard_error"]),
    ratio_primary_ci_halfwidth_to_background_risk=float(0.0016111 / A("selection_mean_risk_level").mean()),
)
with open(out, "x") as f:
    json.dump(dict(summary=summary, pools=rec), f, indent=1)
print(json.dumps(summary, indent=1))
