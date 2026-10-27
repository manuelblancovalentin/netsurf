"""Immutable schema-4 result bundles for conditional protection search."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from time import perf_counter
import uuid
import resource

import numpy as np

from . import __version__
from .analysis_m3 import spearman
from .experiment_config import MANIFEST_SCHEMA_VERSION, SUMMARY_SCHEMA_VERSION
from .manifests import git_provenance, runtime_provenance, sha256_file
from .result_bundle import ResultBundleError, _inventory, _write_bytes, _write_csv, _write_json
from .search_engine import SearchOutcome
from .search_reporting import paired_search_comparisons, plot_search_results, render_search_report, summarize_conditional_comparisons, summarize_search_trials


MEASUREMENT_SCHEMA_VERSION = "4.0.0"


def _union_fields(rows):
    """Keep first-row order while admitting optional fields from later rows."""
    fields = list(rows[0])
    fields.extend(sorted(set().union(*(row.keys() for row in rows)) - set(fields)))
    return fields


def _members(outcome):
    return outcome.replications or (outcome,)


def _rows(outcome):
    members = _members(outcome)
    trials = tuple(row for member in members for row in member.final_trials)
    convergence = tuple(row for member in members for row in member.convergence)
    validation = tuple({"replication": member.replication_name, **row} for member in members for row in member.incremental_validation)
    polar = []
    topology = []
    for member in members:
        for value in member.polar_measurements:
            row = value.to_dict(); left, right = value.left, value.right
            row.update({"replication": member.replication_name, "left_parameter": left.parameter, "left_flat_index": left.flat_index, "left_bit_index": left.bit_index, "right_parameter": right.parameter, "right_flat_index": right.flat_index, "right_bit_index": right.bit_index})
            row.pop("left", None); row.pop("right", None); polar.append(row)
        for value in member.topology_measurements:
            topology.append({"replication": member.replication_name, **{key: item for key, item in value.items() if key not in {"left_identity", "right_identity"}}})
    steps, proposal_evaluations = [], []
    for member in members:
        for result in member.search_results:
            for value in result.steps:
                row = value.to_dict(); removed, added = value.removed, value.added
                row.update({"replication": member.replication_name, "removed_parameter": removed.parameter, "removed_flat_index": removed.flat_index, "removed_bit_index": removed.bit_index, "added_parameter": added.parameter, "added_flat_index": added.flat_index, "added_bit_index": added.bit_index})
                if outcome.configuration.schema_version == "4.3.0": row["wall_seconds_semantics"]="exclusive_iteration_elapsed_including_exact_verification"
                row.pop("removed", None); row.pop("added", None); steps.append(row)
                for candidate in value.evaluated_candidates:
                    candidate_row = candidate.to_dict(); candidate_row.update({"replication": member.replication_name, "strategy": result.strategy, "iteration": value.iteration, "background_schedule_sha256": value.background_schedule_sha256})
                    proposal_evaluations.append(candidate_row)
    return trials, convergence, validation, tuple(polar), tuple(topology), tuple(steps), tuple(proposal_evaluations)


def _summary(outcome, trials, validation, conditional_contexts=(), proposal_quality=()):
    members = _members(outcome)
    exact = next(({"risk": member.exact_optimum_risk, "mask": [bit.key() for bit in sorted(member.exact_optimum_mask)]} for member in members if member.exact_optimum_risk is not None), None)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION, "search_schema_version": outcome.configuration.schema_version,
        "scientific_protocol": outcome.configuration.scientific_protocol, "experiment_name": outcome.configuration.name,
        "benchmark_id": outcome.configuration.benchmark.id, "eligible_bits": len(members[0].estimator_outputs[0].scores),
        "protection_budget_bits": list(outcome.configuration.initial_mask.budgets_bits or (outcome.configuration.initial_mask.budget_bits,)),
        "logical_ber": outcome.configuration.fault.probability, "replication_count": len(members),
        "trial_count": len(trials), "strategy_results": summarize_search_trials(trials, outcome.configuration.final_evaluation.confidence),
        "paired_comparisons": paired_search_comparisons(trials, outcome.configuration.final_evaluation.confidence),
        "conditional_results": summarize_conditional_comparisons(conditional_contexts, outcome.configuration.final_evaluation.confidence),
        "proposal_quality": list(proposal_quality),
        "incremental_validation": list(validation), "exact_optimum": exact,
        "scientific_payload_sha256": outcome.scientific_payload_sha256,
    }


def create_search_bundle(outcome: SearchOutcome, *, repository_root: Path, run_id: str | None = None) -> Path:
    bundle_started = perf_counter()
    config = outcome.configuration; members = _members(outcome); now = datetime.now(timezone.utc)
    identifier = run_id or f"{now.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:8]}"
    experiment_root = config.output.root / config.name
    destination, staging = experiment_root / identifier, experiment_root / f".{identifier}.incomplete"
    experiment_root.mkdir(parents=True, exist_ok=True)
    if destination.exists() or staging.exists():
        raise ResultBundleError(f"run directory already exists: {destination}")
    staging.mkdir()
    _write_bytes(staging / "config.toml", config.source_text.encode())
    resolved = config.resolved_document(); _write_json(staging / "resolved_config.json", resolved)
    _write_json(staging / "scientific_payload.json", outcome.scientific_payload)
    trials, convergence, validation, polar, topology, steps, proposal_evaluations = _rows(outcome)
    conditional_contexts=[]; conditional_summaries=[]; conditional_paths=[]
    for member in members:
        for comparison in member.conditional_comparisons:
            common={"replication":member.replication_name,"logical_ber":member.logical_ber,**({"protection_budget_bits":member.protection_budget_bits} if config.schema_version in {"4.2.0","4.3.0"} else {}),"baseline":comparison.baseline_name,"target":comparison.target_name,"input_sample_count":comparison.input_sample_count,"swap_count":len(comparison.swaps),"path_rule":comparison.path_rule}
            for contribution in comparison.contributions:
                conditional_contexts.append({**common,**contribution.to_dict()})
            conditional_summaries.append({**common,"background_context_count":comparison.context_count,"conditional_loss_mean":comparison.conditional_loss_difference["mean"],"conditional_loss_standard_deviation":comparison.conditional_loss_difference["standard_deviation"],"conditional_loss_standard_error":comparison.conditional_loss_difference["standard_error"],"conditional_loss_ci_low":comparison.conditional_loss_difference["confidence_interval"][0],"conditional_loss_ci_high":comparison.conditional_loss_difference["confidence_interval"][1],"conditional_zero_one_mean":comparison.conditional_zero_one_risk_difference["mean"],"direct_loss_mean":comparison.direct_loss_difference["mean"],"direct_loss_standard_error":comparison.direct_loss_difference["standard_error"],"direct_loss_ci_low":comparison.direct_loss_difference["confidence_interval"][0],"direct_loss_ci_high":comparison.direct_loss_difference["confidence_interval"][1],"direct_zero_one_mean":comparison.direct_zero_one_risk_difference["mean"],"exact_evaluations":comparison.exact_evaluations,"work_proxy_macs":comparison.work_proxy_macs,"wall_seconds":comparison.wall_seconds})
            conditional_paths.extend({**common,"step":swap.index,"removed":swap.removed.key(),"added":swap.added.key()} for swap in comparison.swaps)
    proposal_diagnostics=[]
    if config.schema_version in {"4.1.0", "4.2.0", "4.3.0"}:
        for member in members:
            polar_lookup={}
            for value in member.polar_measurements:
                polar_lookup.setdefault(tuple(sorted((value.left,value.right))),[]).append(value.finite_polar_interaction)
            final_paths={value.target_name:{(step.removed,step.added) for step in value.swaps} for value in member.conditional_comparisons}
            for result in member.search_results:
                guidance_names=tuple(result.proposal_provenance.get("guidance",()))
                rank_maps={name:{bit:index for index,bit in enumerate(member.orderings[name].identities,1)} for name in guidance_names}
                for step in result.steps:
                    for candidate in step.evaluated_candidates:
                        polar_values=polar_lookup.get(tuple(sorted((candidate.removed,candidate.added))),())
                        proposal_diagnostics.append({"replication":member.replication_name,"logical_ber":member.logical_ber,"protection_budget_bits":member.protection_budget_bits,"strategy":result.strategy,"iteration":step.iteration,"candidate_index":candidate.candidate_index,"removed":candidate.removed.key(),"added":candidate.added.key(),"removed_layer":candidate.removed.parameter.split('/',1)[0],"added_layer":candidate.added.parameter.split('/',1)[0],"proposal_probability_remove":candidate.proposal_probability_remove,"proposal_probability_add":candidate.proposal_probability_add,"proposal_score":candidate.proposal_probability_remove*candidate.proposal_probability_add,"guidance_ranks":";".join(f"{name}:{rank_maps[name][candidate.added]}" for name in guidance_names),"polar_interaction":None if not polar_values else sum(polar_values)/len(polar_values),"exact_search_risk_difference":candidate.risk_difference,"exact_search_benefit":-candidate.risk_difference,"selected":candidate.removed==step.removed and candidate.added==step.added,"accepted":bool(step.accepted and candidate.removed==step.removed and candidate.added==step.added),"improved_current_mask":candidate.risk_difference<0,"in_final_comparison_path":(candidate.removed,candidate.added) in final_paths.get(result.strategy,set()),"gate_changes":candidate.gate_changes,"exact_evaluations":candidate.exact_evaluations,"work_proxy_macs":candidate.work_proxy_macs,"iteration_wall_seconds":step.wall_seconds})
    proposal_quality=[]
    for strategy in sorted({row["strategy"] for row in proposal_diagnostics}):
        values=[row for row in proposal_diagnostics if row["strategy"]==strategy]
        scores=[float(row["proposal_score"]) for row in values]; benefits=[float(row["exact_search_benefit"]) for row in values]
        k=max(1,len(values)//4); top_score=set(np.argsort(scores)[-k:]); top_benefit=set(np.argsort(benefits)[-k:])
        polar_values=[row for row in values if row["polar_interaction"] is not None]
        proposal_correlation=spearman(scores,benefits); polar_correlation=None if len(polar_values)<2 else spearman([float(row["polar_interaction"]) for row in polar_values],[float(row["exact_search_benefit"]) for row in polar_values])
        proposal_quality.append({"strategy":strategy,"proposal_count":len(values),"proposal_score_benefit_spearman":None if not np.isfinite(proposal_correlation) else proposal_correlation,"top_quartile_recall":len(top_score&top_benefit)/k,"accepted_count":sum(bool(row["accepted"]) for row in values),"improving_count":sum(bool(row["improved_current_mask"]) for row in values),"polar_measured_count":len(polar_values),"polar_benefit_spearman":None if polar_correlation is None or not np.isfinite(polar_correlation) else polar_correlation})
    influence_rows=[{"replication":member.replication_name,"logical_ber":member.logical_ber,**({"protection_budget_bits":member.protection_budget_bits} if config.schema_version in {"4.2.0","4.3.0"} else {}),"strategy":result.strategy,**row} for member in members for result in member.search_results for row in result.proposal_influence]
    screening_rows=[{"replication":member.replication_name,"logical_ber":member.logical_ber,"protection_budget_bits":member.protection_budget_bits,"strategy":result.strategy,**row} for member in members for result in member.search_results for row in result.screening_rows]
    screening_context_rows=[{"replication":member.replication_name,"logical_ber":member.logical_ber,"protection_budget_bits":member.protection_budget_bits,"strategy":result.strategy,**row} for member in members for result in member.search_results for row in result.screening_context_rows]
    factorized_rows=[{"replication":member.replication_name,"logical_ber":member.logical_ber,"protection_budget_bits":member.protection_budget_bits,"strategy":result.strategy,**row} for member in members for result in member.search_results for row in result.factorized_rows]
    engine_timing_rows=[{"replication":member.replication_name,**row} for member in members for row in member.timing_records]
    summary = _summary(outcome, trials, validation, conditional_contexts, proposal_quality)
    summary["proposal_influence"] = influence_rows
    summary["screening_diagnostics"] = {
        "candidate_rows": len(screening_rows),
        "paired_context_rows": len(screening_context_rows),
        "factorized_decisions": len(factorized_rows),
        "screening_response_count": sum(int(row.get("response_count", 0)) for row in screening_rows),
        "screening_work_proxy_macs": sum(int(row.get("work_proxy_macs", 0)) for row in screening_rows),
        "screening_exclusive_wall_seconds": sum(float(row.get("exclusive_wall_seconds", 0.0)) for row in screening_rows),
        "maximum_aggregate_reconstruction_error": max((float(row["aggregate_reconstruction_error"]) for row in factorized_rows if row.get("aggregate_reconstruction_error") not in (None, "")), default=None),
        "aggregate_population_coverages": sorted({float(row["aggregate_population_coverage"]) for row in factorized_rows if row.get("aggregate_population_coverage") not in (None, "")}),
    }
    aggregate_rows=[row for row in factorized_rows if row.get("aggregate_population_coverage") not in (None,"")]
    direct_rows=[row for row in factorized_rows if row.get("aggregate_population_coverage") in (None,"")]
    factor_setup=sum(float(row.get("factor_build_seconds",0.0))+float(row.get("aggregate_build_seconds",0.0)) for row in aggregate_rows)
    factor_iteration=sum(float(row.get("aggregate_query_seconds",0.0))+float(row.get("aggregate_update_seconds",0.0))+float(row.get("exact_verification_seconds",0.0)) for row in aggregate_rows)
    exact_verification=sum(float(row.get("exact_verification_seconds",0.0)) for row in factorized_rows)
    direct_screening=sum(float(row.get("conditional_screening_seconds",0.0)) for row in direct_rows)
    per_direct=direct_screening/max(1,len(direct_rows)); per_factor=factor_iteration/max(1,len(aggregate_rows))
    summary["computational_costs"]={
        "factor_and_aggregate_setup_seconds":factor_setup,
        "conditional_screening_seconds":direct_screening,
        "factor_query_update_and_verification_seconds":factor_iteration,
        "exact_verification_seconds":exact_verification,
        "candidate_responses_per_second":None if not screening_rows or summary["screening_diagnostics"]["screening_exclusive_wall_seconds"]<=0 else summary["screening_diagnostics"]["screening_response_count"]/summary["screening_diagnostics"]["screening_exclusive_wall_seconds"],
        "verified_swaps_per_second":None if exact_verification<=0 else sum(int(row.get("verified_swaps",0)) for row in factorized_rows)/exact_verification,
        "component_break_even_iterations":None if not direct_rows or not aggregate_rows or per_direct<=per_factor else factor_setup/(per_direct-per_factor),
        "break_even_semantics":"hypothetical component substitution estimate; current hybrid also performs exact screening, so this is not end-to-end speedup",
    }
    summary["timing_audit"] = engine_timing_rows
    _write_json(staging / "summary.json", summary)
    _write_json(staging / "artifacts/sample_partitions.json", {member.replication_name: {name: value.to_dict() for name, value in member.partitions.items()} for member in members})
    _write_json(staging / "artifacts/rankings.json", {member.replication_name: {name: {"digest": value.digest(), "criteria": value.criteria, "estimator": value.estimator, "ordered_bits": [bit.key() for bit in value.identities]} for name, value in member.orderings.items()} for member in members})
    _write_json(staging / "artifacts/search_masks.json", {member.replication_name: {result.strategy: {"initial": [bit.key() for bit in sorted(result.initial_mask)], "final": [bit.key() for bit in sorted(result.final_mask)], "best": [bit.key() for bit in sorted(result.best_mask)], "proposal": result.proposal_provenance} for result in member.search_results} for member in members})
    _write_json(staging / "artifacts/candidate_pools.json", {member.replication_name: [bit.key() for bit in member.candidate_pool] for member in members})
    _write_json(staging / "artifacts/final_fault_schedules.json", {member.replication_name: member.scientific_payload.get("final_fault_states", []) for member in members})
    _write_json(staging / "artifacts/search_fault_schedules.json", {member.replication_name: {result.strategy: [{"iteration": step.iteration, "schedule_sha256": step.background_schedule_sha256, "full_background_states": step.background_fault_states} for step in result.steps] for result in member.search_results} for member in members})
    if conditional_contexts:
        _write_json(staging/"artifacts/conditional_final_fault_schedules.json",{member.replication_name:member.scientific_payload.get("conditional_final_fault_states",[]) for member in members})
        context_fields=["schema_version","replication","logical_ber"]+(["protection_budget_bits"] if config.schema_version in {"4.2.0","4.3.0"} else [])+["baseline","target","input_sample_count","swap_count","path_rule","context","schedule_sha256","base_fault_count","loss_difference","zero_one_risk_difference","direct_loss_difference","direct_zero_one_risk_difference","background_evaluations","incremental_evaluations","full_evaluations","work_proxy_macs","gate_changes","cache_hits","cache_misses","fallback_count","wall_seconds"]
        _write_csv(staging/"measurements/conditional_context_contributions.csv",context_fields,({"schema_version":config.schema_version,**row} for row in conditional_contexts))
        summary_fields=["schema_version",*conditional_summaries[0].keys()]
        _write_csv(staging/"measurements/conditional_mask_comparisons.csv",summary_fields,({"schema_version":config.schema_version,**row} for row in conditional_summaries))
        if conditional_paths:
            path_fields=["schema_version",*conditional_paths[0].keys()]
            _write_csv(staging/"measurements/conditional_swap_paths.csv",path_fields,({"schema_version":config.schema_version,**row} for row in conditional_paths))
    if proposal_diagnostics:
        diagnostic_fields=["schema_version",*proposal_diagnostics[0].keys()]
        _write_csv(staging/"measurements/proposal_diagnostics.csv",diagnostic_fields,({"schema_version":config.schema_version,**row} for row in proposal_diagnostics))
    if influence_rows:
        influence_fields=["schema_version",*_union_fields(influence_rows)]
        _write_csv(staging/"measurements/proposal_influence.csv",influence_fields,({"schema_version":config.schema_version,**row} for row in influence_rows))
    if screening_rows:
        _write_csv(staging/"measurements/candidate_screening.csv",["schema_version",*screening_rows[0].keys()],({"schema_version":config.schema_version,**row} for row in screening_rows))
    if screening_context_rows:
        _write_csv(staging/"measurements/candidate_screening_contexts.csv",["schema_version",*screening_context_rows[0].keys()],({"schema_version":config.schema_version,**row} for row in screening_context_rows))
    if factorized_rows:
        factorized_fields = _union_fields(factorized_rows)
        _write_csv(staging/"measurements/factorized_guidance.csv",["schema_version",*factorized_fields],({"schema_version":config.schema_version,**row} for row in factorized_rows))
    trial_fields=["schema_version","replication"]+(["logical_ber"] if config.schema_version in {"4.1.0", "4.2.0", "4.3.0"} else [])+(["protection_budget_bits"] if config.schema_version in {"4.2.0","4.3.0"} else [])+["strategy","trial","schedule_sha256","fault_count","active_fault_count","accuracy","categorical_crossentropy"]
    _write_csv(staging/"measurements/final_trials.csv", trial_fields, ({"schema_version":config.schema_version,**row} for row in trials))
    if steps:
        fields=["schema_version","replication","strategy","iteration","removed_parameter","removed_flat_index","removed_bit_index","added_parameter","added_flat_index","added_bit_index","proposal_probability_remove","proposal_probability_add","estimated_risk_difference","estimated_risk_ci_low","estimated_risk_ci_high","accepted","acceptance_probability","current_estimated_risk","best_estimated_risk","exact_evaluations","work_proxy_macs","gate_changes","background_evaluations","incremental_evaluations","full_evaluations","affected_layers","reused_layers","cache_hits","cache_misses","fallback_count","background_schedule_sha256","wall_seconds"]+(["wall_seconds_semantics"] if config.schema_version=="4.3.0" else [])
        _write_csv(staging/"measurements/search_steps.csv", fields, ({"schema_version":config.schema_version,**row} for row in steps))
    if proposal_evaluations:
        fields=["schema_version","replication","strategy","iteration","candidate_index","removed","added","proposal_probability_remove","proposal_probability_add","risk_difference","risk_ci_low","risk_ci_high","exact_evaluations","work_proxy_macs","gate_changes","background_schedule_sha256"]
        _write_csv(staging/"measurements/proposal_evaluations.csv", fields, ({"schema_version":config.schema_version,**row} for row in proposal_evaluations))
    convergence_fields=["schema_version","replication"]+(["logical_ber"] if config.schema_version in {"4.1.0", "4.2.0", "4.3.0"} else [])+(["protection_budget_bits"] if config.schema_version in {"4.2.0","4.3.0"} else [])+["strategy","iteration","exact_evaluations","work_proxy_macs","wall_seconds"]+(["wall_seconds_semantics"] if config.schema_version=="4.3.0" else [])+["heldout_loss_mean","heldout_loss_ci_low","heldout_loss_ci_high"]
    _write_csv(staging/"measurements/convergence.csv", convergence_fields, ({"schema_version":config.schema_version,**row,**({"wall_seconds_semantics":"cumulative_search_elapsed_to_mask_state_do_not_sum"} if config.schema_version=="4.3.0" else {})} for row in convergence))
    validation_fields=["schema_version","replication","case","earliest_layer","changed_bits","changed_parameters","relu_gate_changes","max_logit_absolute_error","loss_absolute_error","prediction_mismatches","full_seconds_mean","full_seconds_std","incremental_seconds_mean","incremental_seconds_std","full_work_proxy_macs","incremental_work_proxy_macs","fallback"]
    _write_csv(staging/"measurements/incremental_validation.csv", validation_fields, ({"schema_version":config.schema_version,**row} for row in validation))
    _write_csv(staging/"measurements/estimator_runtime.csv", ["schema_version","replication","estimator","implementation_version","runtime_seconds"], ({"schema_version":config.schema_version,"replication":member.replication_name,"estimator":value.name,"implementation_version":value.implementation_version,"runtime_seconds":value.runtime_seconds} for member in members for value in member.estimator_outputs))
    if polar:
        fields=["schema_version",*polar[0].keys()]; _write_csv(staging/"measurements/polar_interactions.csv", fields, ({"schema_version":config.schema_version,**row} for row in polar))
    if topology:
        fields=["schema_version",*topology[0].keys()]; _write_csv(staging/"measurements/topology_interactions.csv", fields, ({"schema_version":config.schema_version,**row} for row in topology))
    exact_rows=[]
    for member in members:
        for row in member.exact_enumeration: exact_rows.append({"replication":member.replication_name,"mask":";".join(row["mask"]),"risk":row["risk"]})
    if exact_rows: _write_csv(staging/"measurements/exact_mask_enumeration.csv", ["schema_version","replication","mask","risk"], ({"schema_version":config.schema_version,**row} for row in exact_rows))
    exact_lookup={(row["replication"],row["mask"]):float(row["risk"]) for row in exact_rows}
    result_rows=[]
    for member in members:
        optimum=member.exact_optimum_risk
        for result in member.search_results:
            mask=";".join(bit.key() for bit in sorted(result.best_mask)); restricted=exact_lookup.get((member.replication_name,mask))
            best_iteration=0; best_time=0.0; seen=float(result.initial_risk["mean"]); elapsed=0.0
            for step in result.steps:
                elapsed += step.wall_seconds
                if step.best_estimated_risk < seen:
                    seen=step.best_estimated_risk; best_iteration=step.iteration; best_time=elapsed
            result_rows.append({"replication":member.replication_name,**({"logical_ber":member.logical_ber} if config.schema_version in {"4.1.0", "4.2.0", "4.3.0"} else {}),**({"protection_budget_bits":member.protection_budget_bits} if config.schema_version in {"4.2.0","4.3.0"} else {}),"strategy":result.strategy,"protected_bits":len(result.best_mask),"initial_risk_mean":result.initial_risk["mean"],"best_search_risk_estimate":result.best_estimated_risk,"heldout_loss_mean":result.final_risk["mean"],"heldout_loss_ci_low":result.final_risk["confidence_interval"][0],"heldout_loss_ci_high":result.final_risk["confidence_interval"][1],"accepted":result.accepted,"rejected":result.rejected,"proposals_generated":result.proposals_generated,"proposals_evaluated":result.proposals_evaluated,"exact_evaluations":result.exact_evaluations,"work_proxy_macs":result.work_proxy_macs,"wall_seconds":result.wall_seconds,**({"wall_seconds_semantics":"inclusive_method_elapsed_do_not_add_to_step_times"} if config.schema_version=="4.3.0" else {}),"iteration_to_best":best_iteration,"seconds_to_best":best_time,"restricted_exact_risk":restricted,"restricted_exact_gap":None if restricted is None or optimum is None else restricted-optimum})
    result_fields=["schema_version","replication"]+(["logical_ber"] if config.schema_version in {"4.1.0", "4.2.0", "4.3.0"} else [])+(["protection_budget_bits"] if config.schema_version in {"4.2.0","4.3.0"} else [])+["strategy","protected_bits","initial_risk_mean","best_search_risk_estimate","heldout_loss_mean","heldout_loss_ci_low","heldout_loss_ci_high","accepted","rejected","proposals_generated","proposals_evaluated","exact_evaluations","work_proxy_macs","wall_seconds"]+(["wall_seconds_semantics"] if config.schema_version=="4.3.0" else [])+["iteration_to_best","seconds_to_best","restricted_exact_risk","restricted_exact_gap"]
    _write_csv(staging/"measurements/search_results.csv",result_fields,({"schema_version":config.schema_version,**row} for row in result_rows))
    overlap_rows=[]
    for member in members:
        for left,right in combinations(member.search_results,2):
            intersection=len(left.best_mask & right.best_mask); union=len(left.best_mask | right.best_mask)
            overlap_rows.append({"replication":member.replication_name,"left":left.strategy,"right":right.strategy,"intersection_bits":intersection,"union_bits":union,"jaccard":intersection/union if union else 1.0,"fraction_of_budget":intersection/len(left.best_mask) if left.best_mask else 1.0})
    if overlap_rows:
        _write_csv(staging/"measurements/mask_overlap.csv",["schema_version","replication","left","right","intersection_bits","union_bits","jaccard","fraction_of_budget"],({"schema_version":config.schema_version,**row} for row in overlap_rows))
    composition=[]
    for member in members:
        for result in member.search_results:
            for dimension, function in (("layer",lambda b:b.parameter.split("/",1)[0]),("parameter_class",lambda b:b.parameter.split("/",1)[1]),("bit_position",lambda b:str(b.bit_index))):
                for category,count in sorted(Counter(function(bit) for bit in result.best_mask).items()): composition.append({"replication":member.replication_name,"strategy":result.strategy,"dimension":dimension,"category":category,"protected_count":count})
    _write_csv(staging/"measurements/mask_composition.csv", ["schema_version","replication","strategy","dimension","category","protected_count"], ({"schema_version":config.schema_version,**row} for row in composition))
    figures=()
    if config.output.plots:
        figures=tuple(path.relative_to(staging).as_posix() for path in plot_search_results(summary, staging/"plots", trials=trials, convergence=convergence, validation=validation, polar=polar, topology=topology, search_steps=steps, composition=composition, search_results=result_rows, proposal_diagnostics=proposal_diagnostics, screening=screening_rows, factorized=factorized_rows, timing=engine_timing_rows))
    if config.output.report: _write_bytes(staging/"report.md", render_search_report(summary,resolved,figure_paths=figures).encode())
    timing_rows=list(engine_timing_rows)
    timing_rows.append({"replication":"bundle","phase":"bundle_reporting","category":"reporting","exclusive_wall_seconds":perf_counter()-bundle_started,"operation_count":1,"peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0})
    timing_fields=["schema_version","replication","phase","category","exclusive_wall_seconds","operation_count","peak_rss_mb","exclusive_phase_sum_seconds","uninstrumented_seconds"]
    _write_csv(staging/"measurements/timing_phases.csv",timing_fields,({"schema_version":config.schema_version,**row} for row in timing_rows))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION, "run_id": identifier,
        "status": "complete", "source": git_provenance(Path(repository_root).resolve()),
        "runtime": runtime_provenance(),
        "versions": {
            "netsurf": __version__, "experiment_schema": config.schema_version,
            "manifest_schema": MANIFEST_SCHEMA_VERSION, "summary_schema": SUMMARY_SCHEMA_VERSION,
            "measurement_schema": config.schema_version, "scientific_protocol": config.scientific_protocol,
            "incremental_evaluator": "dense-relu-dense-exact-incremental/v1",
            "topology_evaluator": "dense-relu-dense-topology-exact/v1" if config.schema_version in {"4.2.0","4.3.0"} else None,
            "conditional_estimator": "telescoping-three-state-conditional-risk/v1" if config.schema_version in {"4.1.0", "4.2.0", "4.3.0"} else "three-state matched conditional swap/v1",
        },
        "benchmark": {**resolved["benchmark"], "checkpoint_sha256": sha256_file(config.benchmark.checkpoint), "dataset_sha256": sha256_file(config.benchmark.dataset)},
        "experiment": {**resolved["experiment"], "requested_configuration": "config.toml", "resolved_configuration": "resolved_config.json"},
        "execution": {
            **{key: resolved[key] for key in ("samples", "estimators", "guidance", "initial_mask", "search", "polar", "fault", "final_evaluation", "replications")},
            **({"methods": resolved["methods"]} if "methods" in resolved else {}),
            **({"topology": resolved["topology"]} if "topology" in resolved else {}),
            **({"screening": resolved["screening"]} if "screening" in resolved else {}),
            "started_utc": now.isoformat(), "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
        "outputs": _inventory(staging), "scientific_payload_sha256": outcome.scientific_payload_sha256,
    }
    _write_json(staging/"manifest.json",manifest); _write_json(staging/"checksums.json",{"schema_version":"1.0.0","files":{entry["path"]:entry["sha256"] for entry in _inventory(staging)}}); _write_bytes(staging/"FINALIZED",f"scientific_payload_sha256={outcome.scientific_payload_sha256}\n".encode("ascii")); staging.rename(destination); return destination
