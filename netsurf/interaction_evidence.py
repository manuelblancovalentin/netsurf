"""Strict, one-pool-at-a-time readers and immutable offline M910 output."""
from __future__ import annotations

import ast
import importlib.metadata
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from .manifests import digest_object, sha256_file, git_provenance
from .result_bundle import _write_json, _write_bytes, _write_csv as immutable_csv
from .conditional_config import PAIRED_SWAP_COMPARISONS
from .cross_pool import load_campaign, verified_receipt, _receipt
from .cross_pool_analysis import read_field_records
from .replication_statistics import summarize_replication
from .restricted_subcube import subcube_states
from .interaction_attribution import (
    VERSION, TIE_ATOL, contrast, decompose_field, mean_interval,
    mobius_coefficients, ordering_comparison, selection_diagnostics,
)


def read_json(path):
    return json.loads(Path(path).read_text())


def _csv_records(path):
    """Reconstruct native response rows, preserving optional old-schema fields."""
    ints = {"field", "field_seed", "replicate", "local_field", "proposed_events",
            "applied_events", "suppressed_events", "unique_applied_bits"}
    records = []
    for raw in read_field_records(path):
        row = {}
        for k, v in raw.items():
            if k == "schema_version" or v == "":
                continue
            if k in ints:
                v = int(v)
            elif k in {"risk", "logical_ber"}:
                v = float(v)
            elif k in {"candidate_state", "candidate_bits"}:
                # Historical CSV writer used Python list repr, not JSON.
                v = ast.literal_eval(v)
                if not isinstance(v, list):
                    raise ValueError("response candidate list required")
            elif k == "cache_hit":
                if v not in {"True", "False"}:
                    raise ValueError("invalid response cache flag")
                v = v == "True"
            row[k] = v
        records.append(row)
    return records


def read_selection(bundle, payload, pool, campaign):
    """Bind responses to ordered bits, independent fields, cohort, law and context."""
    rows = _csv_records(bundle / "measurements/restricted_oracle_responses.csv")
    backgrounds = read_json(bundle / "artifacts/restricted_oracle_backgrounds.json")
    derived = read_json(bundle / "artifacts/restricted_oracle_derived.json")
    for key, values in (("response", rows), ("background", backgrounds), ("derived", derived)):
        if digest_object(values) != payload["restricted_oracle"][key+"_records_sha256"]:
            raise ValueError(f"{key} scientific-record hash differs")
    ids = [r["identity"] for r in read_json(bundle / "artifacts/candidates.json")]
    if ids != [c["identity"] for c in pool["candidates"]] or digest_object(ids) != pool["candidate_order_sha256"]:
        raise ValueError("candidate population/order differs from frozen campaign")
    k = len(ids); p = campaign["design"]["logical_ber"]
    context = payload["protection_context"]["identity_sha256"]
    protected_outside = set(payload["protection_context"].get("outside_mask", []))
    if "outside_mask" in payload["protection_context"]:
        pc = payload["protection_context"]
        if (digest_object(pc["outside_mask"]) != pc["outside_mask_sha256"]
                or len(protected_outside) != len(pc["outside_mask"])
                or len(protected_outside)+len(pool["original_mask_indices"]) != campaign["protected_population"]):
            raise ValueError("outside protection identity/cardinality mismatch")
    partition = payload["partitions"]["model_selection"]["sample_sha256"]
    if len(payload["restricted_oracle"]["summaries"]) != 1:
        raise ValueError("one BER per child campaign required")
    summary = payload["restricted_oracle"]["summaries"][0]
    if summary["logical_ber"] != p or summary["original_mask_indices"] != pool["original_mask_indices"]:
        raise ValueError("BER or source mask differs")
    if not summary["global_protection_cardinality_preserved"] or summary["global_protection_cardinality"] != campaign["protected_population"]:
        raise ValueError("fixed global budget is not preserved")
    clean, tables, metadata = {}, {}, {}
    for bg in backgrounds:
        if bg["phase"] != "model_selection" or bg["field"] in metadata:
            raise ValueError("unexpected/duplicate outside field")
        events = bg["outside_events"]
        if len(set(events)) != len(events) or set(events) & (set(ids) | protected_outside) or digest_object(events) != bg["outside_sha256"] or len(events) != bg["outside_event_count"]:
            raise ValueError("outside event identities mismatch")
        law = bg["probability_law"]
        if law["name"] != "independent_bernoulli" or law["logical_ber"] != p or law["candidate_probability"] != p or digest_object(law) != bg["probability_law_sha256"]:
            raise ValueError("unsupported or corrupted fault law")
        if bg["sample_sha256"] != partition or bg["protection_context_sha256"] != context:
            raise ValueError("cohort or protection context mismatch")
        pairing = {"phase": bg["phase"], "logical_ber": p, "field": bg["field"],
                   "replicate": bg["replicate"], "seed": bg["seed"],
                   "outside_sha256": bg["outside_sha256"], "sample_sha256": partition,
                   "protection_context_sha256": context}
        if digest_object(pairing) != bg["pairing_key"]:
            raise ValueError("outside field pairing key mismatch")
        metadata[bg["field"]] = bg; tables[bg["field"]] = {}
    for row in rows:
        state = tuple(row["candidate_state"])
        if tuple(sorted(set(state))) != state or not set(state) <= set(range(k)):
            raise ValueError("invalid candidate state")
        if row["candidate_bits"] != [ids[i] for i in state] or row["logical_ber"] != p:
            raise ValueError("encoded identity or BER mismatch")
        if row["proposed_events"] != row["applied_events"]+row["suppressed_events"] or not np.isfinite(row["risk"]):
            raise ValueError("event accounting or finite risk failed")
        if row["phase"] == "model_selection_isolated":
            if len(state) > 1 or row["field"] != -1 or row["outside_background_sha256"] != digest_object([]):
                raise ValueError("invalid same-cohort clean isolated response")
            table = clean
        elif row["phase"] == "model_selection":
            bg = metadata[row["field"]]; table = tables[row["field"]]
            for key, value in (("field_seed", bg["seed"]), ("replicate", bg["replicate"]),
                               ("local_field", bg["local_field"]), ("outside_background_sha256", bg["outside_sha256"]),
                               ("pairing_key", bg["pairing_key"]), ("probability_law_sha256", bg["probability_law_sha256"]),
                               ("protection_context_sha256", context)):
                if row[key] != value:
                    raise ValueError(f"response/background mismatch: {key}")
        else:
            raise ValueError("final responses cannot enter model-selection attribution")
        if state in table:
            raise ValueError("duplicate candidate state")
        table[state] = row["risk"]
    expected = campaign["design"]["selection_fields_per_seed"]*campaign["design"]["selection_replicates"]
    if set(tables) != set(range(expected)) or len({r["seed"] for r in metadata.values()}) != expected:
        raise ValueError("missing or duplicated independent selection fields")
    if set(clean) != {(), *((i,) for i in range(k))}:
        raise ValueError("incomplete clean isolated table")
    if any(set(t) != set(subcube_states(k)) for t in tables.values()):
        raise ValueError("incomplete exact subcube")
    derived_map = {r["field"]: r for r in derived}
    if len(derived_map) != len(derived) or set(derived_map) != set(tables):
        raise ValueError("missing derived table identities")
    for f, table in tables.items():
        record = derived_map[f]
        if record["pairing_key"] != metadata[f]["pairing_key"] or record["response_table_sha256"] != digest_object([
            {"state": list(s), "risk": table[s]} for s in subcube_states(k)]):
            raise ValueError("derived response-table identity differs")
    if len(summary["isolated_damages"]) != k:
        raise ValueError("one same-cohort isolated damage required per candidate")
    for i, d in enumerate(summary["isolated_damages"]):
        if d["candidate"] != ids[i] or d["damage"] != clean[(i,)]-clean[()]:
            raise ValueError("saved isolated damage differs from same-cohort exact response")
    return rows, clean, tables, metadata, summary


def _field_rows(bundle, payload, campaign, selection):
    rows = read_field_records(bundle / "measurements/paired_swap_fields.csv")
    for r in rows:
        for key in ("field", "field_seed", "path_length", "authoritative_evaluation_requests"):
            r[key] = int(r[key])
        for key in ("logical_ber", "target_minus_baseline"):
            r[key] = float(r[key])
        if r["logical_ber"] != campaign["design"]["logical_ber"]:
            raise ValueError("final BER mismatch")
    summary = summarize_replication(rows, campaign["design"]["final_fields"])
    if payload["frozen_selection"] != selection["frozen_selection"]:
        raise ValueError("final masks differ from frozen selection")
    return rows, summary


def _score_map(evidence):
    from .cohort_analysis import verify_score_evidence
    verify_score_evidence(evidence)
    if len(evidence["cohorts"]) != 1:
        raise ValueError("one fixed complete-file cohort per separated phase required")
    cohort = next(iter(evidence["cohorts"].values()))
    clean = [r for r in evidence["evaluations"] if r["fault"] is None]
    if len(clean) != 1:
        raise ValueError("need one explicit clean score reference")
    mapping = {}
    for r in evidence["evaluations"]:
        if r["fault"]:
            key = r["fault"]["schedule_sha256"]
            if key in mapping and mapping[key]["scores_sha256"] != r["scores_sha256"]:
                raise ValueError("same persistent schedule has different saved file scores")
            mapping[key] = r
    return cohort, clean[0], mapping


def file_bootstrap(labels, evaluations, steps, fields, *, seed, replicates):
    """Resample WHOLE files within class, recompute AUC and complete paired paths.

    Identical file draws are shared across all states/fields/contrasts (and pools
    when called with the same seed/cohort). No windows/pairs are resampled.
    This conditions on observed fields and frozen masks, not selection stability.
    """
    if fields < 1 or replicates < 2:
        raise ValueError("positive whole-field count and >=2 resamples required")
    labels = np.asarray(labels); normals = np.flatnonzero(labels == 0); anomalies = np.flatnonzero(labels == 1)
    if not len(normals) or not len(anomalies):
        raise ValueError("both complete-file classes required")
    keys = sorted({r[k] for r in steps for k in ("removed_schedule_sha256", "added_schedule_sha256")})
    index = {s: i for i,s in enumerate(keys)}
    scores = np.array([evaluations[s]["scores"] for s in keys])
    # Full cohort AUC over dependent comparison pairs; this is not an additive
    # per-file loss and pairs never become inferential replication units.
    comparisons = PAIRED_SWAP_COMPARISONS
    weights = np.zeros((3, len(keys)))
    for row in steps:
        c = comparisons.index(row["comparison"]); p = float(row["logical_ber"])
        weights[c, index[row["removed_schedule_sha256"]]] += p/fields
        weights[c, index[row["added_schedule_sha256"]]] -= p/fields
    rng = np.random.Generator(np.random.PCG64(seed)); values = []
    for _ in range(replicates):
        n = rng.choice(normals, size=len(normals)); a = rng.choice(anomalies, size=len(anomalies))
        if len(keys):
            order = scores[:, a, None] - scores[:, None, n]
            risk = 1 - ((order > 0) + .5*(order == 0)).mean(axis=(1,2))
            values.append(weights @ risk)
        else:
            values.append(np.zeros(3))
    values = np.asarray(values)
    return {"seed": seed, "replicates": replicates,
            "contrast_draws": values.tolist(),
            "percentile95_sensitivity": np.quantile(values, [.025,.975], axis=0).T.tolist(),
            "interpretation": "Exploratory stratified complete-file bootstrap conditional on fixed fields and masks. Shared draws preserve pairing. Assumes exchangeable files within class; contiguous device/source ordering may violate it. Not formal confirmation or independent score-pair inference."}


def toycar_audit(selection, final, responses, final_bundle, seed, replicates, fields):
    from .cohort_risk import ordering_diagnostics
    sc, clean_s, mapping_s = _score_map(selection["scalar_cohort_evidence"])
    fc, clean_f, mapping_f = _score_map(final["scalar_cohort_evidence"])
    def cohort_summary(c, clean):
        labels = [f["label"] for f in c["files"]]
        n, a = labels.count(0), labels.count(1)
        return {"files": n+a, "normal": n, "anomalous": a, "clean_auc": 1-clean["risk"],
                "indices": [f["sample_id"] for f in c["files"]], "cohort_sha256": c["cohort_sha256"],
                "strict_order_auc_increment": 1/(n*a), "tie_half_increment": .5/(n*a),
                "resolution_caveat": "Per-cohort AUC lattice, not a minimum detectable expected-mask contrast; Bernoulli integration and field averaging yield smaller increments."}
    labels = [f["label"] for f in sc["files"]]
    tables = [r for r in responses if r["phase"] == "model_selection"]
    base = {r["field"]: r for r in tables if not r["candidate_state"]}
    diagnostics = []
    for r in tables:
        e = mapping_s[r["schedule_sha256"]]; b = mapping_s[base[r["field"]]["schedule_sha256"]]
        if e["risk"] != r["risk"]:
            raise ValueError("response table does not match saved complete-file AUC")
        order = ordering_diagnostics(labels, e["scores"], b["scores"])
        diagnostics.append({"field": r["field"], "candidate_state": r["candidate_state"],
                            "auc": 1-r["risk"], "risk_minus_field_empty": r["risk"]-base[r["field"]]["risk"],
                            "strict_pair_reversals_vs_field_empty": order["strict_reversals"],
                            "pair_order_changes_vs_field_empty": order["ordering_changes_including_ties"],
                            "risk_minus_clean": r["risk"]-clean_s["risk"]})
    steps = read_field_records(final_bundle / "measurements/paired_swap_steps.csv")
    for r in steps:
        for key, schedule in (("removed_risk", "removed_schedule_sha256"), ("added_risk", "added_schedule_sha256")):
            if float(r[key]) != mapping_f[r[schedule]]["risk"]:
                raise ValueError("paired-swap AUC differs from saved complete-file score evidence")
    nonempty = [r for r in diagnostics if r["candidate_state"]]
    actual_final = list(mapping_f.values())
    return {"selection": cohort_summary(sc, clean_s), "final": cohort_summary(fc, clean_f),
            "selection_candidate_states": diagnostics,
            "changed_nonempty_candidate_state_count": sum(r["risk_minus_field_empty"] != 0 for r in nonempty),
            "nonempty_candidate_states": len(nonempty),
            "background_auc": [1-r["risk"] for r in base.values()],
            "final_evaluated_state_auc": [1-r["risk"] for r in actual_final],
            "final_state_pair_reversals_vs_clean": [ordering_diagnostics(
                [f["label"] for f in fc["files"]], r["scores"], clean_f["scores"])["strict_reversals"] for r in actual_final],
            "file_bootstrap": file_bootstrap([f["label"] for f in fc["files"]], mapping_f, steps,
                         fields=fields, seed=seed, replicates=replicates),
            "interpretation": "Forced 2^K state distributions are not natural Bernoulli trial distributions. Final evaluated states are forced swap queries, not unconditional deployment AUC. Pairs are dependent; file-bootstrap is separate from field uncertainty."}


def analyze_pool(campaign_path, campaign, pool, out, *, seed, replicates):
    digest = sha256_file(campaign_path); pid = pool["pool_id"]
    receipt, selection = verified_receipt(_receipt(campaign_path.parent, pid, "selection"), digest, pid, "selection")
    bundle = Path(receipt["bundle"])
    contract = selection["frozen_selection"]["contract"]
    if contract["checkpoint_sha256"] != campaign["checkpoint_sha256"] or contract["asset_lock_sha256"] != campaign["asset_lock_sha256"]:
        raise ValueError("source model/lock contract differs from frozen campaign")
    responses, clean, tables, metadata, summary = read_selection(bundle, selection, pool, campaign)
    k, p, budget = len(pool["candidates"]), campaign["design"]["logical_ber"], summary["budget"]
    all_fields = [decompose_field(tables[f], clean, k=k, budget=budget, probability=p) for f in sorted(tables)]
    masks = [tuple(r["mask"]) for r in all_fields[0]]
    exact = np.array([[r["centered_exact"] for r in rows] for rows in all_fields])
    absolute = np.array([[r["exact_risk"] for r in rows] for rows in all_fields])
    saved = {tuple(r["indices"]): r["field_risks"] for r in summary["eligible_masks"]}
    if set(saved) != set(masks) or not np.allclose(absolute, np.array([saved[m] for m in masks]).T, rtol=0, atol=1e-12):
        raise ValueError("reintegrated risk differs from saved every-mask table")
    chosen = {"selected": tuple(summary["selected_mask_indices"]), "isolated": tuple(summary["isolated_mask_indices"]),
              "original": tuple(summary["original_mask_indices"])}
    mask_rows, pair_rows, order_rows = [], [], []
    comparisons = {PAIRED_SWAP_COMPARISONS[0]: ("selected","original"), PAIRED_SWAP_COMPARISONS[1]: ("selected","isolated"),
                   PAIRED_SWAP_COMPARISONS[2]: ("isolated","original")}
    comparison_rows = {name: [] for name in comparisons}
    for f, field in enumerate(all_fields):
        common = {"field": f, "field_seed": metadata[f]["seed"], "replicate": metadata[f]["replicate"],
                  "local_field":metadata[f]["local_field"], "pairing_key": metadata[f]["pairing_key"], "logical_ber": p}
        for r in field:
            mask_rows.append({**common, **r, "mask": json.dumps(r["mask"]), "within_by_order": json.dumps(r["within_by_order"])})
        for i,j in itertools.combinations(range(len(masks)), 2):
            c = contrast(field[j],field[i])
            pair_rows.append({**common, **c, "target": json.dumps(c["target"]), "baseline": json.dumps(c["baseline"]), "within_by_order": json.dumps(c["within_by_order"])})
        for name,(target,baseline) in comparisons.items():
            comparison_rows[name].append(contrast(field[masks.index(chosen[target])],field[masks.index(chosen[baseline])]))
        for state, value in mobius_coefficients(tables[f], k).items():
            order_rows.append({**common, "state": json.dumps(state), "order": len(state), "coefficient": value})
    _write_csv(out / "mask_decomposition.csv", mask_rows)
    _write_csv(out / "all_pair_decomposition.csv", pair_rows)
    _write_csv(out / "mobius_coefficients.csv", order_rows)
    clean_pred = np.array([r["clean_term"] for r in all_fields[0]])
    bg_pred = np.mean([[r["clean_term"]+r["outside_correction"] for r in field] for field in all_fields], axis=0)
    means = exact.mean(axis=0)
    stability = selection_diagnostics(exact, masks, chosen["selected"], seed=seed, replicates=replicates)
    stability["seed_batch_winners"] = [
        {"replicate": replicate, "fields": [f for f in metadata if metadata[f]["replicate"] == replicate],
         "winner":list(masks[int(np.argmin(exact[[f for f in metadata if metadata[f]["replicate"] == replicate]].mean(axis=0)))])}
        for replicate in sorted({r["replicate"] for r in metadata.values()})]
    names = ("clean_term", "outside_correction", "within_correction", "exact_contrast")
    comp = {name: {key: mean_interval([r[key] for r in rows]) for key in names}
            for name, rows in comparison_rows.items()}
    final_receipt, final = verified_receipt(_receipt(campaign_path.parent, pid, "final"), digest, pid, "final")
    final_bundle = Path(final_receipt["bundle"])
    if final["workflow"]["selection_payload_sha256"] != receipt["payload_sha256"] or final["workflow"]["selection_artifact_sha256"] != sha256_file(bundle/"artifacts/frozen_selection.json"):
        raise ValueError("final source selection binding differs")
    field_rows, final_stats = _field_rows(final_bundle, final, campaign, selection)
    field_scaling = []
    from scipy.stats import t
    for row in final_stats["comparisons"]:
        est = row["mean_effect"]; sd = est["standard_deviation"]
        for n in (32,64,128,256,512):
            field_scaling.append({"comparison": row["comparison"], "fields": n,
                "projected_t95_halfwidth": float(t.ppf(.975,n-1)*sd/np.sqrt(n)),
                "risk_metric": selection["risk"]["name"], "assumption": "same variance and fixed masks/cohort; not power; zero observed variance may miss rare events"})
    toycar = toycar_audit(selection, final, responses, final_bundle, seed=910991, replicates=replicates,
                         fields=campaign["design"]["final_fields"]) if "scalar_cohort_evidence" in selection else None
    input_diagnostic = None
    if "paired_input_diagnostics" in final:
        diag = final["paired_input_diagnostics"]
        rows = diag["records"]; lookup = {(r["field"],r["comparison"]):r for r in field_rows}
        per_comparison = {}
        for name in PAIRED_SWAP_COMPARISONS:
            vectors = []
            for r in rows:
                if r["comparison"] != name: continue
                scalar = lookup[(r["field"],name)]
                if abs(np.mean(r["contributions"])-scalar["target_minus_baseline"]) > 1e-12 or r["master_pairing_key"] != scalar["master_pairing_key"]:
                    raise ValueError("input diagnostics disagree with whole-field endpoint")
                vectors.append(r["contributions"])
            average = np.mean(vectors, axis=0)
            rng = np.random.Generator(np.random.PCG64(910991))
            draws = [float(average[rng.integers(len(average), size=len(average))].mean()) for _ in range(replicates)]
            per_comparison[name] = {"input_count": len(average), "percentile95_sensitivity": np.quantile(draws,[.025,.975]).tolist()}
        input_diagnostic = {"seed":910991,"replicates":replicates,"comparisons":per_comparison,
            "interpretation":"Shared-input resampling conditional on observed fields and fixed masks, not new held-out data. Assumes exchangeable clips despite source ordering; not a formal population CI."}
    source = {"selection_bundle": str(bundle), "final_bundle": str(final_bundle), "selection_payload_sha256": receipt["payload_sha256"],
              "final_payload_sha256": final_receipt["payload_sha256"],
              "selection_checksums_sha256": sha256_file(bundle/"checksums.json"), "final_checksums_sha256":sha256_file(final_bundle/"checksums.json"),
              "response_csv_sha256": sha256_file(bundle/"measurements/restricted_oracle_responses.csv"),
              "response_records_sha256":selection["restricted_oracle"]["response_records_sha256"],
              "checkpoint_sha256":campaign["checkpoint_sha256"], "context_sha256":selection["protection_context"]["identity_sha256"],
              "selection_partition":selection["partitions"]["model_selection"], "final_partition":final["partitions"]["held_out_evaluation"]}
    # Noncancelling magnitude diagnostics: absolute field effects and absolute
    # field-mean effects are distinct; never divide by a near-zero net contrast.
    by_pair = {}
    for r in pair_rows:
        by_pair.setdefault((r["target"],r["baseline"]),[]).append(r)
    magnitudes = {key:{"mean_absolute_field_contrast":float(np.mean([abs(r[key]) for r in pair_rows])),
                       "mean_absolute_field_mean_contrast":float(np.mean([abs(np.mean([r[key] for r in rows])) for rows in by_pair.values()]))}
                  for key in names}
    result = {"pool_id": pid, "logical_ber": p, "risk_metric":selection["risk"]["name"], "source":source,
              "candidates":pool["candidates"], "masks":{n:list(m) for n,m in chosen.items()},
              "selection_fields":len(tables), "final_fields":campaign["design"]["final_fields"], "budget":budget,
              "clean_selection_risk":clean[()], "background_risk":mean_interval([t[()] for t in tables.values()]),
              "mask_means":[{"mask":list(m),"centered_exact":float(means[i]),"absolute_exact":float(absolute[:,i].mean()),
                             "clean_additive_term":float(clean_pred[i]),"conditioned_additive_term":float(bg_pred[i])} for i,m in enumerate(masks)],
              "clean_vs_exact_ordering":ordering_comparison(clean_pred,means), "conditioned_vs_exact_ordering":ordering_comparison(bg_pred,means),
              "isolated_selection_regret":float(means[masks.index(chosen["isolated"]) ]-means.min()),
              "regret_scope":"empirical model-selection regret within fixed pool; selected on these fields, not unbiased held-out regret or whole-network optimum",
              "selection_attribution":comp,"selection_stability":stability,"final_comparisons":final_stats,
              "all_mask_pair_component_magnitudes":magnitudes,
              "final_field_sensitivity":field_scaling,"input_variation_diagnostic":input_diagnostic,"toycar":toycar,
              "max_decomposition_error":max(r["decomposition_error"] for field in all_fields for r in field)}
    _write_json(out / "summary.json", result)
    _write_json(out / "final_field_sensitivity.json",field_scaling)
    window_counts = ([f["window_count"] for c in selection["scalar_cohort_evidence"]["cohorts"].values()
                      for f in c["files"]] if toycar else [])
    costs = {"pool_id":pid,"selection_runtime_seconds":read_json(bundle/"manifest.json")["execution"]["runtime_seconds"],
             "final_runtime_seconds":read_json(final_bundle/"manifest.json")["execution"]["runtime_seconds"],
             "selection_state_requests":len(responses),"selection_input_count":len(source["selection_partition"]["indices"]),
             "selection_window_counts":window_counts,
             "windows_per_file":window_counts[0] if window_counts and len(set(window_counts)) == 1 else None,
             "candidate_states_per_field":2**k}
    return result, costs


def _write_csv(path, rows):
    immutable_csv(path,list(rows[0]),rows)


def analyze_campaigns(paths, output, *, seed=910001, replicates=500, plots=True):
    paths = [Path(p).resolve() for p in paths]; output=Path(output).resolve()
    if len(set(paths)) != len(paths) or not paths:
        raise ValueError("need distinct frozen campaigns")
    output.mkdir(parents=True,exist_ok=False)
    _write_bytes(output/"creation_note.txt", b"Read-only offline derivation. Only the FINALIZED marker establishes completion.\n")
    campaigns=[];costs=[];sources=[]
    for index,path in enumerate(paths):
        campaign=load_campaign(path); digest=sha256_file(path)
        if "joint_campaign" in campaign:
            from .multiber_campaign import verify_member
            verify_member(path,campaign,require_sealed=True)
        registry=path.parent/"final-plan/selections.json"
        if (registry.parent/"SEALED").read_text().strip() != sha256_file(registry):
            raise ValueError("source campaign selections were not sealed")
        bindings=read_json(registry)
        if bindings["campaign_sha256"] != digest or set(bindings["selections"]) != {p["pool_id"] for p in campaign["pools"]}:
            raise ValueError("incomplete sealed selection registry")
        print(f"Verifying and attributing {campaign['name']} ({len(campaign['pools'])} pools)",file=sys.stderr,flush=True)
        pools=[]
        for number,pool in enumerate(campaign["pools"]):
            out=output/"campaigns"/campaign["name"]/pool["pool_id"]
            out.mkdir(parents=True,exist_ok=False)
            result,cost=analyze_pool(path,campaign,pool,out,seed=seed+index*100003+number,replicates=replicates)
            pools.append(result); costs.append({"campaign":campaign["name"],**cost})
            sources.append(result["source"])
            print(f"  {pool['pool_id']}: verified selection + final, all masks/fields retained",file=sys.stderr,flush=True)
        campaigns.append({"name":campaign["name"],"path":str(path),"campaign_sha256":digest,
                          "purpose":campaign["purpose"],"risk_metric":pools[0]["risk_metric"],
                          "logical_ber":campaign["design"]["logical_ber"],"pools":pools,
                          "operation_plan":campaign["operation_counts"],"statistical_plan":campaign["statistical_plan"]})
    payload={"schema_version":VERSION,"protocol":"offline-exact-interaction-attribution/v1",
             "bootstrap_seed":seed,"bootstrap_replicates":replicates,"near_tie_atol":TIE_ATOL,
             "campaigns":campaigns,"limitations":["All attribution and reordering analyses exploratory, conditional on recorded inputs/masks/BER.",
             "Model-selection attribution is not a decomposition of unseen final effects; final response subcubes were not evaluated.",
             "Cross-architecture differences confound checkpoint, metric, dataset and BER; not causal architecture interventions.",
             "No new inference or optimization. Whole fields, pools and files remain distinct uncertainty levels."]}
    from .interaction_reporting import write_report
    write_report(payload,output,plots=plots)
    _write_json(output/"scientific_payload.json",payload)
    _write_json(output/"costs.json",{"schema_version":VERSION,"recorded_historical_execution_costs":costs})
    versions = {}
    for name in ("numpy", "scipy", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    _write_json(output/"manifest.json",{"schema_version":VERSION,"source":git_provenance(Path(__file__).resolve().parents[1]),
                "python_version":sys.version,"dependencies":versions,
                "scientific_payload_sha256":digest_object(payload),"new_model_inference_count":0})
    # Re-read every checksum inventory and small canonical payload at completion.
    # Full source artifact hashes were checked by verified_receipt on entry.
    for s in sources:
        for phase in ("selection","final"):
            bundle=Path(s[phase+"_bundle"])
            if sha256_file(bundle/"checksums.json") != s[phase+"_checksums_sha256"] or digest_object(read_json(bundle/"scientific_payload.json")) != s[phase+"_payload_sha256"]:
                raise ValueError("historical source changed during offline analysis")
    _write_json(output/"source_verification.json",{"schema_version":VERSION,"verified_before_and_after":sources})
    # Only FINALIZED confers completion; failures retain their creation note.
    _write_json(output/"checksums.json",{"schema_version":VERSION,"files":{str(p.relative_to(output)):sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()}})
    digest=digest_object(payload)
    _write_bytes(output/"FINALIZED",f"scientific_payload_sha256={digest}\n".encode())
    return {"output":str(output),"scientific_payload_sha256":digest,"campaigns":len(campaigns),"pools":sum(len(c["pools"]) for c in campaigns),"new_inferences":0}
