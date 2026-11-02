"""Frozen M11 measurement plans using existing exact adapters and fault laws.

This is a diagnostic baseline, not an adaptive protection search. Final commands
load frozen selection evidence; no final result can alter a mask or candidate.
"""
from __future__ import annotations

import itertools
import hashlib
import json
import math
from pathlib import Path
import resource
import re
from time import monotonic
import tomllib

import numpy as np

from .conditional_scaling import (measure_field, choose, compare_oracle, exposure,
    finite_sample_bound, paired_summary, crossed_bootstrap)
from .manifests import digest_object, sha256_file, git_provenance
from .result_bundle import _write_json, _write_bytes, _inventory

VERSION = "1.0.0"
PROTOCOL = "outside-conditioned-forced-clean/v1"


def _read(path):
    return json.loads(Path(path).read_text())


def _keys(value, keys, label):
    if set(value) != set(keys.split()):
        raise ValueError(f"{label} requires exactly {keys}; got {sorted(value)}")


def load_request(path):
    raw = tomllib.loads(Path(path).read_text())
    _keys(raw, "schema_version name purpose source_campaign source_campaign_sha256 design samples statistics", "M11")
    if raw["schema_version"] != VERSION or raw["purpose"] not in {"engineering", "exploratory", "confirmation"}:
        raise ValueError("unsupported M11 version/purpose")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",raw["name"]) or not Path(raw["source_campaign"]).is_absolute() or not re.fullmatch("[0-9a-f]{64}",raw["source_campaign_sha256"]):
        raise ValueError("safe name, absolute verified campaign path and SHA256 required")
    d = raw["design"]
    _keys(d, "candidate_counts pool_seeds sampling bit_positions axis values selection_seeds final_seeds exhaustive_max_k", "design")
    if not d["candidate_counts"] or any(type(k) is not int or k not in {6,8,12,32,64} for k in d["candidate_counts"]):
        raise ValueError("supported K: 6,8,12,32,64")
    if len(set(d["candidate_counts"]))!=len(d["candidate_counts"]) or len(set(d["values"]))!=len(d["values"]):
        raise ValueError("duplicate candidate sizes or exposure arms")
    if d["sampling"] not in {"primary", "structural_bit_tiers"}:
        raise ValueError("primary or separately labeled structural_bit_tiers sampling required")
    if not d["bit_positions"] or len(set(d["bit_positions"])) != len(d["bit_positions"]) or any(type(i) is not int or not 0 <= i < 8 for i in d["bit_positions"]):
        raise ValueError("unique INT8 bit positions required")
    if d["sampling"] == "primary" and sorted(d["bit_positions"]) != list(range(8)):
        raise ValueError("primary sample must support all bit positions")
    if d["axis"] not in {"ber", "expected_applied_flips"} or not d["values"] or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in d["values"]):
        raise ValueError("positive finite two-axis exposure design required")
    if d["axis"]=="ber" and any(v>1 for v in d["values"]):raise ValueError("BER exceeds one")
    if type(d["exhaustive_max_k"]) is not int or not 0 <= d["exhaustive_max_k"] <= 12:
        raise ValueError("exhaustive_max_k must be 0..12")
    for name in ("pool_seeds", "selection_seeds", "final_seeds"):
        v = d[name]
        if not v or len(set(v))!=len(v) or any(type(s) is not int or s<0 for s in v):
            raise ValueError(f"{name} requires distinct nonnegative integers")
    if set(d["selection_seeds"]) & set(d["final_seeds"]):
        raise ValueError("selection and final field seeds overlap")
    s = raw["samples"]
    _keys(s, "selection_offset selection_count final_offset final_count", "samples")
    if any(type(v) is not int or v < (1 if key.endswith("count") else 0) for key,v in s.items()):
        raise ValueError("positive counts and nonnegative offsets required")
    st = raw["statistics"]
    _keys(st, "confidence bootstrap_seed bootstrap_replicates", "statistics")
    if not 0 < st["confidence"] < 1 or type(st["bootstrap_seed"]) is not int or st["bootstrap_seed"] < 0 or type(st["bootstrap_replicates"]) is not int or st["bootstrap_replicates"]<2:
        raise ValueError("invalid uncertainty plan")
    return raw


def sample_candidates(universe, protected, k, seed, positions=tuple(range(8))):
    """SRSWOR in source-status strata; all primary eligible bits have positive pi.

    Approximately 1/3 protected. No measured outcome or susceptibility enters.
    A small draw cannot guarantee every tensor/tier; achieved coverage is recorded.
    """
    budget = (k+1)//3
    rng = np.random.default_rng(seed)
    before = rng.bit_generator.state
    rows = []
    for status, n in ((True,budget), (False,k-budget)):
        population = [b for b in universe if (b in protected)==status and b.bit_index in positions]
        if len(population)<n:
            raise ValueError("candidate stratum too small")
        for j in rng.choice(len(population),n,replace=False):
            bit=population[j]
            rows.append({"identity":bit.key(), "parameter":bit.parameter, "flat_index":bit.flat_index,
                "bit_index":bit.bit_index, "originally_protected":status,
                "stratum_population":len(population), "inclusion_probability":n/len(population),
                "sampling_weight":len(population)/n})
    rows.sort(key=lambda r:(r["parameter"],r["flat_index"],r["bit_index"]))
    return {"candidates":rows, "budget":budget, "seed":seed, "rng_before":before,
            "rng_after":rng.bit_generator.state, "sampling":"SRSWOR within original S5 protection status",
            "candidate_sha256":digest_object([r["identity"] for r in rows]),
            "original_mask": [i for i,r in enumerate(rows) if r["originally_protected"]]}


def field_seeds(design,pool,arm,phase):
    # Independent per-pool/per-arm streams; exact derived seeds are frozen.
    return [int(np.random.SeedSequence([seed,pool["seed"],pool["k"],arm,
        0 if phase=="selection" else 1]).generate_state(1,dtype=np.uint64)[0])
        for seed in design[f"{phase}_seeds"]]


def _source(path, expected):
    from .cross_pool import load_campaign
    from .conditional_config import load_conditional
    from .conditional_benchmarks import verified_storage
    from .protection_context import read_mask_identities, verify_protection_context
    path=Path(path).resolve()
    campaign=load_campaign(path,expected)
    config_path=path.parent/"configs/pool_000_selection.toml"
    config=load_conditional(config_path)
    cap,lock,storage=verified_storage(config.benchmark)
    universe=storage.bit_identities()
    ctx=config.protection_context
    original=frozenset(read_mask_identities(ctx.source_bundle/ctx.artifact,ctx.mask_key))
    probe=(next(b for b in universe if b in original),next(b for b in universe if b not in original))
    context=verify_protection_context(ctx,universe,probe,expected_checkpoint_sha256=storage.sha256)
    payload=ctx.source_bundle/"scientific_payload.json"
    if sha256_file(payload)!=_read(ctx.source_bundle/"checksums.json")["files"]["scientific_payload.json"] or digest_object(_read(payload))!=ctx.scientific_payload_sha256:
        raise ValueError("source S5 payload is corrupted")
    if (len(universe),len(original))!=(campaign["eligible_population"],campaign["protected_population"]):
        raise ValueError("source population mismatch")
    return campaign,config,cap,storage,universe,context


def _partitions(config,cap,samples):
    if cap.target=="toycar_int8":
        # Explicit source cohorts only; never truncate files or create window samples.
        a=list(config.cohorts["conditional_model_selection"])
        b=list(config.cohorts["conditional_final_evaluation"])
        if samples!={"selection_offset":0,"selection_count":len(a),"final_offset":0,"final_count":len(b)}:
            raise ValueError("ToyCar counts must match COMPLETE source cohorts exactly")
    else:
        a=list(cap.indices("conditional_model_selection",samples["selection_offset"],samples["selection_count"]))
        b=list(cap.indices("conditional_final_evaluation",samples["final_offset"],samples["final_count"]))
    if set(a)&set(b):
        raise ValueError("selection and final inputs overlap")
    return {"selection":a,"final":b}


def exposure_gate(partitions, audit, purpose):
    known=set(audit["exposed_test_indices"])
    for role,indices in partitions.items():
        if purpose=="engineering" and not set(indices)<=known:
            raise ValueError(f"engineering {role} would consume previously unexposed identities")
        if purpose=="confirmation" and set(indices)&known:
            raise ValueError(f"confirmation {role} overlaps prior fault-outcome exposure")


def _seal_directory(root, payload, *, frozen=False, metadata=None, report=None, artifacts=None):
    root=Path(root).resolve()
    root.mkdir(parents=True,exist_ok=False)
    for name,value in (artifacts or {}).items():
        if Path(name).is_absolute() or ".." in Path(name).parts:raise ValueError("unsafe artifact name")
        (root/name).parent.mkdir(parents=True,exist_ok=True)
        _write_json(root/name,value)
    file="plan.json" if frozen else "scientific_payload.json"
    _write_json(root/file,payload)
    digest=digest_object(payload)
    if not frozen:
        _write_json(root/"manifest.json",{"schema_version":VERSION,"protocol":PROTOCOL,
            "benchmark":{"checkpoint_sha256":payload["checkpoint_sha256"]},
            "scientific_payload_sha256":digest,"source":git_provenance(Path(__file__).resolve().parents[1]),
            "execution":metadata or {}, "status":"complete"})
    if report is not None:
        _write_bytes(root/"report.md",report.encode())
    _write_json(root/"checksums.json",{"schema_version":VERSION,"files":{r["path"]:r["sha256"] for r in _inventory(root)}})
    _write_bytes(root/("FROZEN" if frozen else "FINALIZED"),
        (sha256_file(root/file) if frozen else "scientific_payload_sha256="+digest).encode())
    return {"output":str(root),"scientific_payload_sha256":digest,"file_sha256":sha256_file(root/file)}


def verify_directory(root, *, frozen=False):
    root=Path(root).resolve()
    name="plan.json" if frozen else "scientific_payload.json"
    for rel,sha in _read(root/"checksums.json")["files"].items():
        if Path(rel).is_absolute() or ".." in Path(rel).parts or sha256_file(root/rel)!=sha:
            raise ValueError("M11 inventory checksum mismatch")
    obj=_read(root/name)
    expected=sha256_file(root/name) if frozen else "scientific_payload_sha256="+digest_object(obj)
    if (root/("FROZEN" if frozen else "FINALIZED")).read_text().strip()!=expected or obj["schema_version"]!=VERSION or obj["protocol"]!=PROTOCOL:
        raise ValueError("unsupported or unfinalized M11 evidence")
    return obj


def freeze(request_path, output):
    from .conditional_benchmarks import audit_exposure
    raw=load_request(request_path)
    t0=monotonic()
    campaign,config,cap,storage,universe,context=_source(raw["source_campaign"],raw["source_campaign_sha256"])
    partitions=_partitions(config,cap,raw["samples"])
    audit=audit_exposure(cap)
    exposure_gate(partitions,audit,raw["purpose"])
    d=raw["design"]
    values=storage.values(); descriptors={s.parameter:s for s in storage.descriptors}
    pools=[]
    for k,seed in itertools.product(d["candidate_counts"],d["pool_seeds"]):
        pool=sample_candidates(universe,context.original,k,seed,d["bit_positions"])
        pool.update({"id":f"k{k}_seed{seed}","k":k,"design":d["sampling"]})
        for row in pool["candidates"]:
            desc=descriptors[row["parameter"]]; q=desc.quantization
            ch=np.unravel_index(row["flat_index"],desc.shape)[q.quantized_dimension]
            code=int(values[row["parameter"]].flat[row["flat_index"]])
            row.update({"tensor_index":desc.tensor_index,"layer":desc.original_name,
                "encoded_unsigned":code&255,"stored_signed":code,
                "scale":q.scales[ch if len(q.scales)>1 else 0],
                "zero_point":q.zero_points[ch if len(q.zero_points)>1 else 0],
                "storage":"INT8 two's complement kernel, bit 0 LSB"})
        pools.append(pool)
    arms=[exposure(len(universe),len(context.original),**{
        "probability" if d["axis"]=="ber" else "expected":v}) for v in d["values"]]
    operations=[]
    all_seeds=set()
    for pool in pools:
        k=pool["k"]; nstate=2**k if k<=d["exhaustive_max_k"] else k+1
        m=k-pool["budget"]
        for ai,arm in enumerate(arms):
            streams={phase:field_seeds(d,pool,ai,phase) for phase in ("selection","final")}
            seeds=streams["selection"]+streams["final"]
            if all_seeds.intersection(seeds) or len(set(seeds))!=len(seeds):raise ValueError("derived field seed collision")
            all_seeds.update(seeds)
            p=arm["logical_ber"]
            operations.append({"pool":pool["id"],"arm":ai,**arm,"unprotected_candidates":m,
                "field_seeds":streams,
                "mp":m*p,"probability_two_or_more":max(0.,1-(1-p)**m-m*p*(1-p)**(m-1)),
                "selection_evaluations":1+k+1+len(d["selection_seeds"])*nstate,
                "final_evaluations":1+len(d["final_seeds"])*(4 if k<=d["exhaustive_max_k"] else 3),
                "input_evaluations_per_call":len(partitions["selection"])*(196 if cap.target=="toycar_int8" else 1)})
    plan={"schema_version":VERSION,"protocol":PROTOCOL,"request":raw,"capability":cap.record(),
        "checkpoint_sha256":storage.sha256,"source_campaign":campaign["name"],
        "template_sha256":sha256_file(Path(raw["source_campaign"]).parent/"configs/pool_000_selection.toml"),
        "population":len(universe),"global_budget":len(context.original),
        "universe_sha256":digest_object([b.key() for b in universe]),
        "source_context":context.record,"partitions":partitions,"source_split":cap.source_split,
        "exposure_audit":audit,"pools":pools,"arms":arms,"operation_counts":operations,
        "hypotheses":["outside-additive improves over clean-isolated finite-field risk ordering",
            "higher-order approximation error grows with candidate exposure, not necessarily architecture"],
        "primary_comparison":"outside_additive minus original_S5; held-out paired whole fields",
        "sampling_limits":"status-stratified SRS supports every primary eligible bit; small draws need not cover every tensor/tier; no outcome filtering; mask quality is a pool-level estimand, not a weighted bit mean",
        "selection_separation":"final samples and fields never accessed during selection; all pool selections must seal before final",
        "cost_metadata":{"freeze_seconds":monotonic()-t0}}
    # Runtime is not part of the reproducibility-critical frozen scientific plan.
    plan.pop("cost_metadata")
    return _seal_directory(output,plan,frozen=True)


def _identity(row):
    from .identities import BitIdentity
    return BitIdentity(row["parameter"],row["flat_index"],row["bit_index"])


def _run_context(plan_path,pool_id,arm):
    plan=verify_directory(Path(plan_path).parent,frozen=True)
    raw=plan["request"]
    _,config,cap,storage,universe,context=_source(raw["source_campaign"],raw["source_campaign_sha256"])
    if sha256_file(Path(raw["source_campaign"]).parent/"configs/pool_000_selection.toml")!=plan["template_sha256"] or digest_object([b.key() for b in universe])!=plan["universe_sha256"]:
        raise ValueError("source template/population changed after freeze")
    pool=next((v for v in plan["pools"] if v["id"]==pool_id),None)
    if pool is None or not 0<=arm<len(plan["arms"]):
        raise ValueError("unknown frozen pool/arm")
    candidates=tuple(_identity(r) for r in pool["candidates"])
    from .protection_context import verify_protection_context
    context=verify_protection_context(config.protection_context,universe,candidates,expected_checkpoint_sha256=storage.sha256)
    if list(context.original_inside_indices)!=pool["original_mask"]:
        raise ValueError("source S5 mask changed")
    return plan,config,cap,universe,context,pool,candidates


def execute(plan_path,pool_id,arm,phase,output,*,selection_registry=None):
    from .conditional_benchmarks import create_conditional_adapter,audit_exposure
    from .conditional_damage import _state_sha256,_schedule,_risk_value,_sample_identity_records
    from .protection import ProtectionMask
    from .faults import FaultSchedule
    from .conditional_damage import _model
    if phase not in {"selection","final"} or Path(output).exists():
        raise ValueError("selection/final phase and a NEW output directory required")
    t0=monotonic()
    plan,config,cap,universe,context,pool,candidates=_run_context(plan_path,pool_id,arm)
    plan_sha=sha256_file(plan_path); p=plan["arms"][arm]["logical_ber"]
    raw=plan["request"]; design=raw["design"]; k=pool["k"]; r=pool["budget"]
    operation=next(v for v in plan["operation_counts"] if v["pool"]==pool_id and v["arm"]==arm)
    if "field_seeds" not in operation:
        raise ValueError("development prototype plan lacks frozen independent streams; freeze a NEW plan")
    seeds=operation["field_seeds"][phase]
    source=None
    if phase=="final":
        if selection_registry is None:
            raise ValueError("final requires an explicitly sealed ALL-selection registry")
        registry=verify_directory(Path(selection_registry).parent,frozen=True)
        if registry["plan_sha256"]!=plan_sha:
            raise ValueError("selection registry belongs to another plan")
        item=next(v for v in registry["selections"] if v["pool"]==pool_id and v["arm"]==arm)
        source=verify_directory(item["bundle"])
        if digest_object(source)!=item["payload_sha256"] or source["plan_sha256"]!=plan_sha:
            raise ValueError("selection evidence mismatch")
    audit=audit_exposure(cap,allowed_campaign=plan_sha)
    # The same frozen plan may share cohorts across pools, not across unbound plans.
    exposure_gate({phase:plan["partitions"][phase]},audit,raw["purpose"])
    benchmark=create_conditional_adapter(config); benchmark.load()
    samples=raw["samples"]
    role="conditional_model_selection" if phase=="selection" else "conditional_final_evaluation"
    x,y,partition=benchmark.sample_partition(role,samples[f"{phase}_count"],offset=samples[f"{phase}_offset"])
    if list(partition.indices)!=plan["partitions"][phase]:
        raise ValueError("loaded input identities differ from frozen plan")
    before=_state_sha256(benchmark.values())
    model_before=hashlib.sha256(benchmark._model_content).hexdigest()
    outside=context.original-set(candidates)
    protected=ProtectionMask("M11_fixed_outside",frozenset(outside),len(universe),len(outside)/len(universe),{})
    costs={"initialization_seconds":monotonic()-t0,"background_seconds":0.,"evaluation_including_apply_restore_seconds":0.,
        "restoration_verification_seconds":0.,"risk_extraction_seconds":0.,"selection_seconds":0.,"authoritative_calls":0,
        "exact_outputs_seconds":0.,"model_apply_seconds":0.,"model_restore_seconds":0.}
    # Instance-local timing wrappers preserve the existing implementation and arguments.
    # Both validated backends perform set(faulted), evaluate, set(clean) in finally.
    original_set=benchmark._set_model_content; original_outputs=benchmark._outputs
    setters=[0]
    def timed_set(content):
        start=monotonic()
        try:return original_set(content)
        finally:
            key="model_apply_seconds" if setters[0]%2==0 else "model_restore_seconds"
            costs[key]+=monotonic()-start; setters[0]+=1
    def timed_outputs(inputs):
        start=monotonic()
        try:return original_outputs(inputs)
        finally:costs["exact_outputs_seconds"]+=monotonic()-start
    benchmark._set_model_content=timed_set; benchmark._outputs=timed_outputs
    # Complete-file adapter needs one explicit clean score reference; classification
    # uses the same call for provenance, without treating it as an independent field.
    start=monotonic(); clean_evaluation=benchmark.evaluate(x,y)
    costs["evaluation_including_apply_restore_seconds"]+=monotonic()-start; costs["authoritative_calls"]+=1
    records=[]
    def evaluate(events,mask,seed):
        schedule=_schedule(p,seed,events,len(universe),len(x))
        start=monotonic()
        evaluation,report=benchmark.evaluate_fault_schedule(x,y,schedule,mask)
        costs["evaluation_including_apply_restore_seconds"]+=monotonic()-start
        start=monotonic(); restored=(_state_sha256(benchmark.values())==before and
            hashlib.sha256(benchmark._model_content).hexdigest()==model_before)
        costs["restoration_verification_seconds"]+=monotonic()-start
        if not restored or report.proposed_events!=report.applied_events+report.suppressed_events:
            raise RuntimeError("restoration/event accounting failure")
        start=monotonic(); risk=_risk_value(evaluation,config.risk_metric)
        costs["risk_extraction_seconds"]+=monotonic()-start; costs["authoritative_calls"]+=1
        record={"schedule_sha256":schedule.digest(),"proposed":report.proposed_events,"applied":report.applied_events,
            "suppressed":report.suppressed_events,"risk":risk,"restored":True}
        records.append(record)
        return risk
    output_payload={"schema_version":VERSION,"protocol":PROTOCOL,"phase":phase,"plan_sha256":plan_sha,
        "checkpoint_sha256":plan["checkpoint_sha256"],"pool":pool_id,"arm":arm,
        "probability_law":{"name":"independent_bernoulli","logical_ber":p,"lifetime":"persistent_whole_cohort"},
        "protection_context":context.record,"candidate_manifest":pool,
        "campaign_binding":{"campaign_sha256":plan_sha},
        "partitions":{phase:partition.to_dict()},"source_split":cap.source_split,
        "clean_risk":_risk_value(clean_evaluation,config.risk_metric),"risk_metric":config.risk_metric,
        "records":records,"restoration_sha256":before,"restored_model_bytes_sha256":model_before,
        "input_identities":list(benchmark.sample_identity_records(x,y,partition)) if hasattr(benchmark,"sample_identity_records") else
            _sample_identity_records(x,y,partition)}
    if phase=="selection":
        clean=measure_field(lambda state:evaluate([candidates[i] for i in state],protected,0),k)
        tables=[]; backgrounds=[]
        candidate_set=set(candidates)
        available=[b for b in universe if b not in outside and b not in candidate_set]
        for fi,seed in enumerate(seeds):
            start=monotonic(); rng=np.random.default_rng(seed)
            events=tuple(b for b,flip in zip(available,rng.random(len(available))<p) if flip)
            costs["background_seconds"]+=monotonic()-start
            bg={"field":fi,"seed":seed,"outside_events":[b.key() for b in events],
                "outside_sha256":digest_object([b.key() for b in events])}
            bg["pairing_key"]=digest_object({"pool":pool_id,"arm":arm,"field":fi,"seed":seed,
                "context":digest_object(context.record),"sample":partition.sample_sha256,"events":bg["outside_sha256"]})
            table=measure_field(lambda state:evaluate((*events,*(candidates[i] for i in state)),protected,seed),
                k,exhaustive=k<=design["exhaustive_max_k"])
            tables.append(table)
            bg["responses"]=[{"state":list(s),"risk":v} for s,v in table.items()]
            backgrounds.append(bg)
        start=monotonic()
        deltas=np.array([[t[(i,)]-t[()] for i in range(k)] for t in tables])
        masks={"original_S5":pool["original_mask"],"clean_isolated":list(choose([clean[(i,)]-clean[()] for i in range(k)],r)),
            "outside_additive":list(choose(deltas.mean(axis=0),r))}
        output_payload.update({"clean_responses":[{"state":list(s),"risk":v} for s,v in clean.items()],
            "outside_fields":backgrounds,"signed_deltas":deltas.tolist(),
            "mean_signed_deltas":deltas.mean(axis=0).tolist(),
            "delta_standard_errors":(deltas.std(axis=0,ddof=1)/math.sqrt(len(deltas))).tolist() if len(deltas)>1 else None,
            "cumulative_masks":[list(choose(deltas[:n].mean(axis=0),r)) for n in range(1,len(deltas)+1)],
            "bounds":finite_sample_bound(k,r,p,1. if cap.target=="toycar_int8" else -math.log(1e-12),len(tables),1-raw["statistics"]["confidence"])})
        if k<=design["exhaustive_max_k"]:
            comparison=compare_oracle(tables,clean,k,r,p,pool["original_mask"])
            output_payload["oracle_comparison"]=comparison
            masks["restricted_exact"]=comparison["exact_empirical_optimum"]
        for mask in masks.values():
            if len(set(mask))!=r or len(outside)+len(mask)!=plan["global_budget"]:
                raise RuntimeError("global fixed budget failed")
        output_payload["frozen_masks"]=masks
        costs["selection_seconds"]=monotonic()-start
    else:
        masks=source["frozen_masks"]; names=list(masks); fields=[]; risks=[]; per_input=[]
        capture=getattr(benchmark,"capture_input_losses",None)
        from contextlib import nullcontext
        for fi,seed in enumerate(seeds):
            start=monotonic(); schedule=FaultSchedule.generate(_model(p),universe,seed=seed,inference_count=len(x))
            # The existing schedule defines the full-network independent Bernoulli field.
            events=tuple(event.bit for event in schedule.events)
            costs["background_seconds"]+=monotonic()-start
            fields.append({"field":fi,"seed":seed,"events":[b.key() for b in events],"schedule_sha256":schedule.digest()})
            row=[]; inputs=[]
            for name,indices in masks.items():
                mask=ProtectionMask(name,frozenset(outside|{candidates[i] for i in indices}),len(universe),plan["global_budget"]/len(universe),{})
                with capture() if capture else nullcontext(None) as loss_rows:
                    row.append(evaluate(events,mask,seed))
                if loss_rows is not None:
                    inputs.append(np.asarray(loss_rows[-1]).tolist())
            risks.append(row)
            if inputs: per_input.append(inputs)
        output_payload.update({"frozen_masks":masks,"selection_payload_sha256":digest_object(source),
            "mask_order":names,"final_fields":fields,"field_risks":risks,
            "paired_comparisons":paired_summary(risks,names,raw["statistics"]["confidence"]),
            "per_input_losses":per_input})
        if per_input:
            draws=crossed_bootstrap(per_input,seed=raw["statistics"]["bootstrap_seed"],replicates=raw["statistics"]["bootstrap_replicates"])
            output_payload["crossed_bootstrap"]= {"contrasts_vs":names[0],"draws":draws.tolist(),
                "assumptions":"exchangeable independent inputs and iid whole fields, fixed selected masks; not selection uncertainty"}
    artifacts={}
    if hasattr(benchmark,"cohort_evidence"):
        evidence=benchmark.cohort_evidence(); output_payload["scalar_cohort_evidence"]=evidence
        if phase=="final":
            rows=[e for e in evidence["evaluations"] if e["fault"] is not None]
            cohort=next(iter(evidence["cohorts"].values()))
            scores=np.array([e["scores"] for e in rows]).reshape(len(design["final_seeds"]),len(masks),-1)
            labels=[f["label"] for f in cohort["files"]]
            draws=crossed_bootstrap(scores,labels=labels,seed=raw["statistics"]["bootstrap_seed"],replicates=raw["statistics"]["bootstrap_replicates"])
            output_payload["crossed_bootstrap"]={"contrasts_vs":names[0],"draws":draws.tolist(),
                "assumptions":"whole files within class and whole fault fields resampled jointly; fixed masks, no pair/window independence"}
        # Keep the exposure-critical payload bounded; exact scores remain immutable,
        # inventory-hashed and scientifically bound as a separately readable artifact.
        artifacts["artifacts/cohort_scores.json"]=evidence
        output_payload["scalar_cohort_evidence"]={"artifact":"artifacts/cohort_scores.json",
            "scientific_sha256":digest_object(evidence),"cohorts":evidence["cohorts"],
            "evaluated_states":len(evidence["evaluations"])}
    costs.update({"total_seconds":monotonic()-t0,"peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "inherited_conformance_reference_inputs":cap.reference_inputs,
        "reference_representation":benchmark.representation_manifest() if hasattr(benchmark,"representation_manifest") else None,
        "exposure_audit_sha256":digest_object(audit),
        "source_sampling_cost":"reported separately by plan freezing; not an inference operation",
        "aggregation_and_schedule_overhead_seconds":max(0.,costs["evaluation_including_apply_restore_seconds"]-
            costs["exact_outputs_seconds"]-costs["model_apply_seconds"]-costs["model_restore_seconds"]),
        "timer_limitation":"model_apply/restore time interpreter reinitialization; aggregation residual also includes schedule application/decoding; conformance initialization separate"})
    benchmark._set_model_content=original_set; benchmark._outputs=original_outputs
    return _seal_directory(output,output_payload,metadata=costs,report=report(output_payload,costs),artifacts=artifacts)


def seal(plan_path, selection_root, output):
    plan=verify_directory(Path(plan_path).parent,frozen=True); sha=sha256_file(plan_path)
    needed={(p["id"],i) for p in plan["pools"] for i in range(len(plan["arms"]))}
    found={}
    for path in sorted(Path(selection_root).rglob("FINALIZED")):
        payload=verify_directory(path.parent)
        if payload.get("plan_sha256")!=sha or payload.get("phase")!="selection":continue
        key=(payload["pool"],payload["arm"])
        if key in found:raise ValueError("duplicate selections: cannot choose using outcomes")
        found[key]={"pool":key[0],"arm":key[1],"bundle":str(path.parent.resolve()),"payload_sha256":digest_object(payload)}
    if set(found)!=needed:raise ValueError("all prespecified pools/arms must finalize before sealing")
    return _seal_directory(output,{"schema_version":VERSION,"protocol":PROTOCOL,"plan_sha256":sha,
        "selections":[found[key] for key in sorted(found)]},frozen=True)


def report(payload,costs=None):
    text=["# M11 conditional scaling", "",f"Phase: {payload['phase']}; pool: {payload['pool']}; risk: {payload['risk_metric']}.",
        "", "Exact TFLite scalar task risks. Logical BER, ideal suppression; not physical radiation reliability.",
        "Outside-conditioned means all other candidate bits are forced clean; not full conditional vulnerability.",
        "No optimum is known for K>12. Finite-field selection regret is NOT held-out regret.", "",
        f"Payload: `{digest_object(payload)}`", "", "Frozen masks:", "", "```json",json.dumps(payload["frozen_masks"],indent=2),"```"]
    if "oracle_comparison" in payload:
        text += ["", "## Exact versus additive (same selection fields)","", "```json",json.dumps(payload["oracle_comparison"]["diagnostics"],indent=2),"```"]
    if "paired_comparisons" in payload:
        text += ["", "## Independent paired final fields", "", "```json",json.dumps(payload["paired_comparisons"],indent=2),"```",
            "Intervals condition on the frozen model, mask and cohort. Input bootstrap is separately recorded. "
            "Small engineering cohorts/field counts do not establish efficacy or generalization."]
    if costs:text += ["","## Measured execution cost","","```json",json.dumps(costs,indent=2),"```"]
    return "\n".join(text)+"\n"
