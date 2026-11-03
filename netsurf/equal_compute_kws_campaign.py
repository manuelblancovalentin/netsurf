"""Prospective 72-bit KWS equal-query search, with write-once field receipts.

The candidate panel is the exact union of all twelve frozen M9.6 pools. Model
selection and final fields are independent; no inference occurs at freeze.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import zipfile

import numpy as np

from .campaign_exposure import audit_exposure
from .conditional_config import load_conditional
from .conditional_damage import _master_backgrounds, _sample_identity_records, _schedule
from .equal_compute_search import search_allocation
from .faults import FaultEvent, FaultModel, FaultSchedule
from .identities import BitIdentity
from .manifests import canonical_json_bytes, digest_object, sha256_file
from .protection import ProtectionMask
from .protection_context import verify_protection_context

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path("/projects/netsurf/manuelbv/workspace/implementation-worktrees/m96/workspace/runs/m96_kws_cross_pool_k6_v1/campaign.json")
SOURCE_SHA="46674ad4f90e8bc49fcee3ef826bbb57130005f239d192700cd09d8f106d3ff3"
KWS_SHA="aeea436800704fce17b17292e4412630ad856e9d777c044c64ef748a880bd0ae"
BER=.01
SELECTION_FIRST,SELECTION_CLIPS=2216,64
FINAL_FIRST,FINAL_CLIPS=4474,416
SELECTION_FIELDS,FINAL_FIELDS=64,128
QUERIES=64
RANDOM_SEEDS=(970101,970102,970103)
GREEDY_SEEDS=(970201,970202,970203)
ANALYSIS_SEED=970901
SOURCE_FILES=("netsurf/equal_compute_kws_campaign.py","netsurf/equal_compute_search.py",
    "netsurf/campaign_exposure.py","netsurf/conditional_damage.py",
    "netsurf/conditional_config.py","netsurf/adapters/registry.py",
    "netsurf/adapters/mlperf_tiny_multibench.py","netsurf/faults.py",
    "netsurf/protection_context.py","netsurf/protection.py",
    "scripts/run_kws_equal_compute_once.sh")


def _write(path:Path,document:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as stream:
        stream.write(canonical_json_bytes(document).decode()+"\n")
        stream.flush();os.fsync(stream.fileno())


def _bound(document:dict)->dict:
    return {**document,"payload_sha256":digest_object(document)}


def _read(path:Path,plan_hash:str)->dict:
    item=json.loads(path.read_text())
    if (item.get("plan_sha256")!=plan_hash or
        item.get("payload_sha256")!=digest_object({k:v for k,v in item.items() if k!="payload_sha256"})):
        raise ValueError(f"corrupt or wrong-plan campaign artifact: {path}")
    return item


def _source()->tuple[dict,tuple[BitIdentity,...]]:
    if sha256_file(SOURCE)!=SOURCE_SHA:raise ValueError("frozen M9.6 candidate manifest changed")
    doc=json.loads(SOURCE.read_text())
    pools=doc["pools"]
    if len(pools)!=12 or [p["pool_id"] for p in pools]!=[f"pool_{i:03d}" for i in range(12)]:
        raise ValueError("all twelve ordered M9.6 pools required")
    records=[row for pool in pools for row in pool["candidates"]]
    bits=tuple(sorted(BitIdentity(r["parameter"],r["flat_index"],r["bit_index"])
                      for r in records))
    if len(bits)!=72 or len(set(bits))!=72 or sum(r["source_protected"] for r in records)!=24:
        raise ValueError("source candidate union/protection count changed")
    if any(digest_object([r["identity"] for r in pool["candidates"]])!=pool["candidate_order_sha256"]
           for pool in pools):
        raise ValueError("source pool order hash changed")
    return doc,bits


def _source_config():
    return load_conditional(SOURCE.parent/"configs/pool_000_selection.toml",check_inputs=False)


def _storage(config):
    from .tflite_storage import TFLiteModelStorage
    return TFLiteModelStorage.from_path(config.benchmark.checkpoint,
        expected_sha256=KWS_SHA,eligible_roles=("kernel",))


def _context(config,universe,candidates):
    result=verify_protection_context(config.protection_context,universe,candidates,
                                     expected_checkpoint_sha256=KWS_SHA)
    if len(result.original)!=22016 or len(result.outside)!=21992 or len(result.original_inside_indices)!=24:
        raise ValueError("verified S5 global/panel budget changed")
    return result


def _window(adapter,role,first,count):
    base=2024 if role=="conditional_model_selection" else 3048
    inputs,labels,partition=adapter.sample_partition(role,count,offset=first-base)
    if list(partition.indices)!=list(range(first,first+count)) or \
       partition.source_split!="speech_commands_tfds_test_interleaved_blocks16":
        raise ValueError("KWS source-index window changed")
    records=_sample_identity_records(inputs,labels,partition)
    return (inputs,labels),{"first":first,"count":count,"role":role,
            "source_split":partition.source_split,"sample_sha256":partition.sample_sha256,
            "identity_records_sha256":digest_object(records),"identity_records":list(records)}


def _check_exposure()->dict:
    audit=audit_exposure()
    known=set(audit["exposed_test_indices"])
    for first,count in ((SELECTION_FIRST,SELECTION_CLIPS),(FINAL_FIRST,FINAL_CLIPS)):
        overlap=known.intersection(range(first,first+count))
        if overlap:raise ValueError(f"prospective KWS input exposure detected: {sorted(overlap)[:12]}")
    return {"ledger_sha256":audit["prior_ledger_sha256"],
        "inventory_sha256":digest_object(audit),"finalized_bundles":len(audit["runs"]),
        "known_fault_outcome_overlap":[],"limitations":audit["limitations"],
        "clean_reference_inference_previously_performed":True}


def _seeds()->dict:
    selection=[970301+j*1_000_003 for j in range(SELECTION_FIELDS)]
    final=[970501+j*1_000_003 for j in range(FINAL_FIELDS)]
    isolated=[970401+j for j in range(72)]
    other=list(RANDOM_SEEDS)+list(GREEDY_SEEDS)+[970400,ANALYSIS_SEED]
    all_seeds=selection+final+isolated+other
    if len(all_seeds)!=len(set(all_seeds)):
        raise ValueError("selection/final/isolated/search seed collision")
    return {"selection_fields":selection,"final_fields":final,
        "isolated_bits":isolated,"search_random":list(RANDOM_SEEDS),
        "search_greedy":list(GREEDY_SEEDS),"analysis":ANALYSIS_SEED}


def freeze(output:Path)->dict:
    if Path(output).exists():raise FileExistsError("immutable KWS plan already exists")
    source,candidates=_source()
    config=_source_config();storage=_storage(config);universe=storage.bit_identities()
    context=_context(config,universe,candidates)
    if len(universe)!=176128 or context.record["original_mask_sha256"]!=source["source_mask_sha256"]:
        raise ValueError("M9.6 eligible universe or S5 mask changed")
    exposure=_check_exposure()
    from .adapters.registry import create_adapter
    adapter=create_adapter(config,None);adapter.load()
    _,selection=_window(adapter,"conditional_model_selection",SELECTION_FIRST,SELECTION_CLIPS)
    _,final=_window(adapter,"conditional_final_evaluation",FINAL_FIRST,FINAL_CLIPS)
    if set(r["original_index"] for r in selection["identity_records"]) & \
       set(r["original_index"] for r in final["identity_records"]):
        raise ValueError("selection/final clip overlap")
    if subprocess.run(["git","diff","--quiet","HEAD","--",*SOURCE_FILES],cwd=ROOT).returncode:
        raise ValueError("scientific source must be committed before freezing")
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    plan={"schema_version":"kws-equal-compute-prospective/v1","status":"FROZEN",
        "source_commit":commit,"source_files_sha256":{n:sha256_file(ROOT/n) for n in SOURCE_FILES},
        "m96_manifest_sha256":SOURCE_SHA,"checkpoint_sha256":KWS_SHA,
        "asset_lock_sha256":sha256_file(config.benchmark.asset_lock),
        "benchmark_config_sha256":sha256_file(config.source_path),
        "eligible_population":176128,"eligible_population_sha256":digest_object([b.key() for b in universe]),
        "s5_mask_sha256":context.record["original_mask_sha256"],
        "s5_context_artifact_sha256":config.protection_context.artifact_sha256,
        "global_budget":22016,"panel_budget":24,
        "candidate_bits":[b.key() for b in candidates],
        "candidate_union_sha256":digest_object([b.key() for b in candidates]),
        "source_pools":[{"pool_id":p["pool_id"],"candidate_order_sha256":p["candidate_order_sha256"],
                         "candidates_sha256":digest_object(p["candidates"])} for p in source["pools"]],
        "initial_mask":[b.key() for b in candidates if b in context.original],
        "selection_window":selection,"final_window":final,"exposure_audit":exposure,
        "logical_ber":BER,"fault_law":"independent Bernoulli over all 176128 eligible kernel bits; persistent per field",
        "seeds":_seeds(),"selection_fields":SELECTION_FIELDS,"final_fields":FINAL_FIELDS,
        "queries_per_search_seed":QUERIES,"search_methods":["random_search","greedy_swap"],
        "primary":"greedy minus random, each averaged over three fixed search seeds; two-sided paired field test",
        "secondary":["random minus S5","greedy minus S5","random minus isolated","greedy minus isolated"],
        "secondary_adjustment":"Holm across four two-sided paired field tests",
        "bootstrap_replicates":10000,"signflip_replicates":10000,
        "statistical_unit":"final fault field, conditional on one fixed 72-bit panel and final input cohort",
        "costs":{"search_requests":6*QUERIES*SELECTION_FIELDS,
            "isolated_requests":73,"final_requests":FINAL_FIELDS*8,
            "total_requests":6*QUERIES*SELECTION_FIELDS+73+FINAL_FIELDS*8,
            "selection_clip_inferences":6*QUERIES*SELECTION_FIELDS*SELECTION_CLIPS+73*SELECTION_CLIPS,
            "final_clip_inferences":FINAL_FIELDS*8*FINAL_CLIPS,
            "total_clip_inferences":6*QUERIES*SELECTION_FIELDS*SELECTION_CLIPS+73*SELECTION_CLIPS+FINAL_FIELDS*8*FINAL_CLIPS}}
    if plan["costs"]["total_requests"]!=25673 or plan["costs"]["total_clip_inferences"]!=2003520:
        raise ValueError("prespecified exact cost changed")
    plan["plan_sha256"]=digest_object(plan)
    output=Path(output);output.mkdir(parents=True)
    _write(output/"plan.json",plan)
    (output/"FROZEN").write_text(sha256_file(output/"plan.json")+"\n")
    return {"plan":str(output/"plan.json"),"plan_sha256":plan["plan_sha256"],
            "costs":plan["costs"],"exposure":exposure}


def _static(plan_path:Path)->dict:
    path=Path(plan_path);plan=json.loads(path.read_text())
    if (plan.get("schema_version")!="kws-equal-compute-prospective/v1" or
        plan.get("plan_sha256")!=digest_object({k:v for k,v in plan.items() if k!="plan_sha256"}) or
        (path.parent/"FROZEN").read_text().strip()!=sha256_file(path) or
        plan["candidate_union_sha256"]!=digest_object(plan["candidate_bits"]) or
        len(plan["candidate_bits"])!=72 or len(plan["initial_mask"])!=24 or
        plan["logical_ber"]!=BER or plan["queries_per_search_seed"]!=QUERIES or
        plan["selection_fields"]!=SELECTION_FIELDS or plan["final_fields"]!=FINAL_FIELDS or
        plan["costs"]["total_requests"]!=25673 or
        any(sha256_file(ROOT/name)!=value for name,value in plan["source_files_sha256"].items())):
        raise ValueError("frozen KWS plan/source changed")
    if sha256_file(SOURCE)!=plan["m96_manifest_sha256"] or _seeds()!=plan["seeds"]:
        raise ValueError("M9.6 source or seed family differs from freeze")
    _check_exposure()
    return plan


def _runtime(plan:dict,*, final:bool):
    config=_source_config();storage=_storage(config);universe=storage.bit_identities()
    candidates=tuple(BitIdentity(*(_parse(key))) for key in plan["candidate_bits"])
    context=_context(config,universe,candidates)
    if (sha256_file(config.benchmark.asset_lock)!=plan["asset_lock_sha256"] or
        sha256_file(config.source_path)!=plan["benchmark_config_sha256"] or
        digest_object([b.key() for b in universe])!=plan["eligible_population_sha256"] or
        context.record["original_mask_sha256"]!=plan["s5_mask_sha256"]):
        raise ValueError("KWS asset, universe or S5 identity changed")
    from .adapters.registry import create_adapter
    adapter=create_adapter(config,None);adapter.load()
    role,first,count,expected=(("conditional_final_evaluation",FINAL_FIRST,FINAL_CLIPS,plan["final_window"])
                               if final else ("conditional_model_selection",SELECTION_FIRST,SELECTION_CLIPS,plan["selection_window"]))
    data,actual=_window(adapter,role,first,count)
    if actual!=expected:raise ValueError("KWS labels or encoded input identities changed")
    return adapter,universe,candidates,context,data


def _parse(key:str)->tuple[str,int,int]:
    return key.split("[")[0],int(key.split("[")[1].split("]")[0]),int(key.rsplit("#b",1)[1])


def _mask(name:str,local,context,universe_size=176128)->ProtectionMask:
    bits=frozenset(context.outside)|frozenset(local)
    if len(bits)!=22016 or len(local)!=24:
        raise ValueError("every KWS policy must preserve 22016/176128 global budget")
    return ProtectionMask(name,bits,universe_size,22016,{})


def _fields(universe,seeds,n):
    result=[]
    for seed in seeds:
        master=_master_backgrounds(universe,BER,1,seed)[0]
        if master[0]!=seed:raise ValueError("field seed changed")
        result.append(_schedule(BER,seed,master[1],len(universe),n))
    return result


def _evaluate(adapter,data,schedule,mask):
    evaluation,report=adapter.evaluate_fault_schedule(*data,schedule,mask)
    value=float(evaluation.categorical_crossentropy)
    if not math.isfinite(value):raise ValueError("nonfinite KWS CE without prespecified penalty")
    return {"mean_ce":value,"accuracy":float(evaluation.accuracy),
            "report":asdict(report),"schedule_sha256":schedule.digest()}


def _isolated(plan,root,adapter,data,candidates,context):
    base=ProtectionMask("isolated_calibration",context.outside,176128,len(context.outside),{})
    clean_path=root/"selection/isolated_clean.json"
    empty=FaultSchedule(FaultModel("weight","logical_trainable_parameter_bits","persistent_trial",
        "forced_engineering",0.),970400,(),176128,SELECTION_CLIPS)
    if clean_path.exists():clean=_read(clean_path,plan["plan_sha256"])
    else:
        clean=_bound({"plan_sha256":plan["plan_sha256"],"result":_evaluate(adapter,data,empty,base)})
        _write(clean_path,clean)
    scores=[];hashes=[]
    for ordinal,bit in enumerate(candidates):
        seed=plan["seeds"]["isolated_bits"][ordinal]
        schedule=FaultSchedule(FaultModel("weight","logical_trainable_parameter_bits","persistent_trial",
            "forced_engineering",1.),seed,(FaultEvent(None,bit),),176128,SELECTION_CLIPS)
        path=root/"selection/isolated"/f"bit_{ordinal:03d}.json"
        intent=path.with_suffix(".intent.json")
        identity={"plan_sha256":plan["plan_sha256"],"ordinal":ordinal,"bit":bit.key(),
                  "seed":seed,"schedule_sha256":schedule.digest()}
        if path.exists():
            item=_read(path,plan["plan_sha256"])
            if _read(intent,plan["plan_sha256"])!=_bound(identity) or any(item[k]!=v for k,v in identity.items()):
                raise ValueError("completed isolated request differs from frozen identity")
        else:
            if intent.exists():raise ValueError(f"partial isolated request retained: {intent}")
            _write(intent,_bound(identity))
            item=_bound({**identity,"result":_evaluate(adapter,data,schedule,base)})
            _write(path,item)
        scores.append(item["result"]["mean_ce"]-clean["result"]["mean_ce"])
        hashes.append(sha256_file(path))
    chosen=sorted(candidates,key=lambda bit:(-scores[candidates.index(bit)],bit))[:24]
    record=_bound({"plan_sha256":plan["plan_sha256"],"scores":scores,
        "selected_mask":[b.key() for b in sorted(chosen)],"member_file_sha256":hashes,
        "clean_sha256":sha256_file(clean_path)})
    summary=root/"selection/isolated.json"
    if summary.exists():
        if _read(summary,plan["plan_sha256"])!=record:raise ValueError("isolated selection changed")
    else:_write(summary,record)
    return tuple(chosen)


def _query(plan,root,adapter,data,fields,context,method,seed,ordinal,local):
    name=f"{method}_{seed}"
    path=root/"selection/search"/name/f"query_{ordinal:03d}.json"
    intent=path.with_suffix(".intent.json")
    mask=_mask(name,local,context)
    identity={"plan_sha256":plan["plan_sha256"],"method":method,"seed":seed,
        "query":ordinal,"local_mask":[b.key() for b in sorted(local)],
        "global_mask_sha256":mask.digest(),
        "selection_field_schedule_sha256":[s.digest() for s in fields]}
    if path.exists():
        saved=_read(path,plan["plan_sha256"])
        if _read(intent,plan["plan_sha256"])!=_bound(identity) or any(saved[k]!=v for k,v in identity.items()):
            raise ValueError("completed equal-query receipt changed")
    else:
        if intent.exists():raise ValueError(f"partial equal-query request retained: {intent}")
        _write(intent,_bound(identity))
        outcomes=[_evaluate(adapter,data,s,mask) for s in fields]
        saved=_bound({**identity,"field_results":outcomes,
            "mean_ce":statistics.mean(item["mean_ce"] for item in outcomes),
            "model_requests":len(fields),"clip_inferences":len(fields)*SELECTION_CLIPS})
        _write(path,saved)
    if len(saved["field_results"])!=SELECTION_FIELDS or saved["model_requests"]!=SELECTION_FIELDS:
        raise ValueError("equal-query receipt has incomplete fields")
    if (saved["clip_inferences"]!=SELECTION_FIELDS*SELECTION_CLIPS or
        not math.isclose(saved["mean_ce"],statistics.mean(x["mean_ce"] for x in saved["field_results"]),abs_tol=1e-12)):
        raise ValueError("equal-query risk or inference cost changed")
    for field_result,schedule in zip(saved["field_results"],fields):
        expected_applied=sum(event.bit not in mask.protected for event in schedule.events)
        report=field_result["report"]
        if (field_result["schedule_sha256"]!=schedule.digest() or
            report["proposed_events"]!=len(schedule.events) or
            report["applied_events"]!=expected_applied or
            report["suppressed_events"]!=len(schedule.events)-expected_applied):
            raise ValueError("equal-query field accounting changed")
    return saved["mean_ce"]


def run_selection(plan_path:Path,run_root:Path)->dict:
    plan=_static(plan_path)
    adapter,universe,candidates,context,data=_runtime(plan,final=False)
    root=Path(run_root);root.mkdir(parents=True,exist_ok=True)
    fields=_fields(universe,plan["seeds"]["selection_fields"],SELECTION_CLIPS)
    schedule_doc=_bound({"plan_sha256":plan["plan_sha256"],"fields":[
        {"seed":s.seed,"schedule_sha256":s.digest(),"events":[e.key() for e in s.events]} for s in fields]})
    schedule_path=root/"selection/schedules.json"
    if schedule_path.exists():
        if _read(schedule_path,plan["plan_sha256"])!=schedule_doc:raise ValueError("selection fields changed")
    else:_write(schedule_path,schedule_doc)
    isolated=_isolated(plan,root,adapter,data,candidates,context)
    initial=tuple(c for c in candidates if c.key() in plan["initial_mask"])
    names=[]
    for method,seeds in (("random_search",RANDOM_SEEDS),("greedy_swap",GREEDY_SEEDS)):
        for seed in seeds:
            counter=[0]
            def evaluate(local):
                counter[0]+=1
                return _query(plan,root,adapter,data,fields,context,method,seed,counter[0],local)
            result=search_allocation(candidates,initial,method=method,query_budget=QUERIES,
                seed=seed,selection_identity=digest_object({"plan":plan["plan_sha256"],
                    "window":plan["selection_window"]["identity_records_sha256"],
                    "schedules":schedule_doc["payload_sha256"]}),
                calls_per_query=SELECTION_FIELDS,
                clip_inferences_per_query=SELECTION_FIELDS*SELECTION_CLIPS,evaluate=evaluate)
            trajectory=asdict(result);trajectory.pop("elapsed_seconds")
            trajectory["initial_mask"]=[b.key() for b in result.initial_mask]
            trajectory["selected_mask"]=[b.key() for b in result.selected_mask]
            trajectory["observations"]=[{**asdict(observation),
                "candidate":[b.key() for b in observation.candidate]}
                for observation in result.observations]
            document=_bound({"plan_sha256":plan["plan_sha256"],"result":trajectory})
            path=root/"selection"/f"{method}_{seed}.json"
            if path.exists():
                if _read(path,plan["plan_sha256"])!=document:raise ValueError("search trajectory changed")
            else:_write(path,document)
            names.append(path.name)
            print(f"{method} seed {seed}: {QUERIES} paired queries sealed",flush=True)
    return {"status":"SELECTION_COMPLETE","isolated_mask_sha256":digest_object([b.key() for b in sorted(isolated)]),
            "search_trajectories":names}


def seal(plan_path:Path,run_root:Path)->dict:
    plan=_static(plan_path);root=Path(run_root)
    names=["selection/schedules.json","selection/isolated.json"]+[
        f"selection/{method}_{seed}.json" for method,seeds in
        (("random_search",RANDOM_SEEDS),("greedy_swap",GREEDY_SEEDS)) for seed in seeds]
    for name in names:_read(root/name,plan["plan_sha256"])
    isolated=_read(root/"selection/isolated.json",plan["plan_sha256"])
    if len(isolated["member_file_sha256"])!=72 or any(
        sha256_file(root/"selection/isolated"/f"bit_{ordinal:03d}.json")!=expected
        for ordinal,expected in enumerate(isolated["member_file_sha256"])):
        raise ValueError("isolated response files differ from sealed selection")
    schedule=_read(root/"selection/schedules.json",plan["plan_sha256"])
    schedule_hashes=[item["schedule_sha256"] for item in schedule["fields"]]
    if len(schedule_hashes)!=SELECTION_FIELDS or len(set(schedule_hashes))!=SELECTION_FIELDS:
        raise ValueError("incomplete selection fault-field family")
    query_hashes={}
    for method,seeds in (("random_search",RANDOM_SEEDS),("greedy_swap",GREEDY_SEEDS)):
        for seed in seeds:
            trajectory=_read(root/"selection"/f"{method}_{seed}.json",plan["plan_sha256"])["result"]
            if trajectory["query_budget"]!=QUERIES or len(trajectory["observations"])!=QUERIES:
                raise ValueError("equal-query trajectory incomplete")
            for ordinal,observation in enumerate(trajectory["observations"],1):
                receipt=_read(root/"selection/search"/f"{method}_{seed}"/
                    f"query_{ordinal:03d}.json",plan["plan_sha256"])
                query_name=f"selection/search/{method}_{seed}/query_{ordinal:03d}.json"
                query_hashes[query_name]=sha256_file(root/query_name)
                if (receipt["local_mask"]!=observation["candidate"] or
                    receipt["selection_field_schedule_sha256"]!=schedule_hashes or
                    receipt["mean_ce"]!=observation["measured_risk"] or
                    len(receipt["field_results"])!=SELECTION_FIELDS):
                    raise ValueError("equal-query trace differs from sealed receipts")
    document=_bound({"plan_sha256":plan["plan_sha256"],
        "artifact_sha256":{name:sha256_file(root/name) for name in names},
        "query_receipt_sha256":query_hashes,
        "all_six_equal_query_trajectories_complete":True})
    path=root/"selection/seal.json"
    if path.exists():
        if _read(path,plan["plan_sha256"])!=document:raise ValueError("selection seal changed")
    else:_write(path,document)
    return {"selection_seal_sha256":sha256_file(path),"artifacts":len(names)}


def _require_seal(plan,root):
    saved=_read(root/"selection/seal.json",plan["plan_sha256"])
    if (not saved["all_six_equal_query_trajectories_complete"] or
        len(saved["query_receipt_sha256"])!=6*QUERIES or
        any(sha256_file(root/name)!=sha for name,sha in
            {**saved["artifact_sha256"],**saved["query_receipt_sha256"]}.items())):
        raise ValueError("all-selection-before-final seal or artifact changed")
    return saved


def _selected_masks(plan,root,candidates):
    mapping={b.key():b for b in candidates}
    isolated=_read(root/"selection/isolated.json",plan["plan_sha256"])
    masks={"s5":tuple(mapping[key] for key in plan["initial_mask"]),
           "exact_isolated":tuple(mapping[key] for key in isolated["selected_mask"])}
    for method,seeds in (("random_search",RANDOM_SEEDS),("greedy_swap",GREEDY_SEEDS)):
        for seed in seeds:
            item=_read(root/"selection"/f"{method}_{seed}.json",plan["plan_sha256"])
            if item["result"]["query_budget"]!=QUERIES or item["result"]["model_requests"]!=QUERIES*SELECTION_FIELDS:
                raise ValueError("search budget changed")
            masks[f"{method}_{seed}"]=tuple(mapping[key] for key in item["result"]["selected_mask"])
    if len(masks)!=8 or any(len(mask)!=24 for mask in masks.values()):
        raise ValueError("final mask family incomplete")
    return masks


def run_final(plan_path:Path,run_root:Path)->dict:
    plan=_static(plan_path);root=Path(run_root);_require_seal(plan,root)
    adapter,universe,candidates,context,data=_runtime(plan,final=True)
    masks={name:_mask(name,local,context) for name,local in _selected_masks(plan,root,candidates).items()}
    fields=_fields(universe,plan["seeds"]["final_fields"],FINAL_CLIPS)
    for index,schedule in enumerate(fields):
        path=root/"final"/f"field_{index:03d}.json"
        intent=path.with_suffix(".intent.json")
        identity={"plan_sha256":plan["plan_sha256"],"field":index,"seed":schedule.seed,
            "schedule_sha256":schedule.digest(),"events":[e.key() for e in schedule.events],
            "policy_names":sorted(masks)}
        if path.exists():
            item=_read(path,plan["plan_sha256"])
            if _read(intent,plan["plan_sha256"])!=_bound(identity) or \
               any(item.get(k)!=v for k,v in identity.items() if k!="policy_names") or \
               set(item["policies"])!=set(masks):
                raise ValueError("completed final KWS field differs from frozen identity")
            _verify_final_result(item,schedule,masks)
            continue
        if intent.exists():raise ValueError(f"partial final field retained: {intent}")
        _write(intent,_bound(identity))
        results={name:_evaluate(adapter,data,schedule,mask) for name,mask in masks.items()}
        document=_bound({**{k:v for k,v in identity.items() if k!="policy_names"},
                         "policies":results})
        _verify_final_result(document,schedule,masks)
        _write(path,document)
        print(f"KWS final field {index+1}/{FINAL_FIELDS} sealed",flush=True)
    return {"status":"FINAL_MEMBERS_COMPLETE","fields":FINAL_FIELDS}


def _verify_final_result(item,schedule,masks):
    if (item["seed"]!=schedule.seed or item["schedule_sha256"]!=schedule.digest() or
        item["events"]!=[event.key() for event in schedule.events] or
        set(item["policies"])!=set(masks)):
        raise ValueError("final KWS schedule or policy family changed")
    for name,mask in masks.items():
        result=item["policies"][name];report=result["report"]
        applied=sum(event.bit not in mask.protected for event in schedule.events)
        if (result["schedule_sha256"]!=schedule.digest() or
            report["proposed_events"]!=len(schedule.events) or
            report["applied_events"]!=applied or
            report["suppressed_events"]!=len(schedule.events)-applied or
            not math.isfinite(result["mean_ce"]) or
            not 0<=result["accuracy"]<=1):
            raise ValueError("final KWS mask/event/risk accounting changed")


def aggregate(plan_path:Path,run_root:Path)->dict:
    plan=_static(plan_path);root=Path(run_root);_require_seal(plan,root)
    config=_source_config();universe=_storage(config).bit_identities()
    candidates=tuple(BitIdentity(*_parse(key)) for key in plan["candidate_bits"])
    context=_context(config,universe,candidates)
    masks={name:_mask(name,local,context) for name,local in _selected_masks(plan,root,candidates).items()}
    schedules=_fields(universe,plan["seeds"]["final_fields"],FINAL_CLIPS)
    fields=[_read(root/"final"/f"field_{i:03d}.json",plan["plan_sha256"])
            for i in range(FINAL_FIELDS)]
    for index,(field,schedule) in enumerate(zip(fields,schedules)):
        intent=_read(root/"final"/f"field_{index:03d}.intent.json",plan["plan_sha256"])
        if any(field[key]!=intent[key] for key in ("field","seed","schedule_sha256","events")):
            raise ValueError("final KWS intent/outcome mismatch")
        _verify_final_result(field,schedule,masks)
    names=set(fields[0]["policies"])
    if len(fields)!=128 or len(names)!=8 or any(set(x["policies"])!=names for x in fields):
        raise ValueError("incomplete final KWS policy matrix")
    rows={name:{"mean_ce":statistics.mean(f["policies"][name]["mean_ce"] for f in fields),
                "accuracy":statistics.mean(f["policies"][name]["accuracy"] for f in fields)} for name in sorted(names)}
    def per_field(method):
        return np.array([statistics.mean(f["policies"][f"{method}_{seed}"]["mean_ce"]
            for seed in (RANDOM_SEEDS if method=="random_search" else GREEDY_SEEDS)) for f in fields])
    random=per_field("random_search");greedy=per_field("greedy_swap")
    s5=np.array([f["policies"]["s5"]["mean_ce"] for f in fields])
    isolated=np.array([f["policies"]["exact_isolated"]["mean_ce"] for f in fields])
    contrasts=[("greedy_minus_random",greedy-random),
        ("random_minus_s5",random-s5),("greedy_minus_s5",greedy-s5),
        ("random_minus_isolated",random-isolated),("greedy_minus_isolated",greedy-isolated)]
    comparisons=[]
    for index,(name,diffs) in enumerate(contrasts):
        rng=np.random.default_rng(ANALYSIS_SEED+index)
        draws=diffs[rng.integers(0,FINAL_FIELDS,size=(10000,FINAL_FIELDS))].mean(axis=1)
        signs=rng.choice([-1.,1.],size=(10000,FINAL_FIELDS))
        p=(1+int(np.count_nonzero(np.abs((signs*diffs).mean(axis=1))>=abs(diffs.mean()))))/10001
        trimmed=np.sort(diffs)[12:-12]
        comparisons.append({"contrast":name,"mean":float(diffs.mean()),
            "ci95":[float(v) for v in np.quantile(draws,[.025,.975])],
            "p_unadjusted":p,"favorable_fields":int(np.count_nonzero(diffs<0)),
            "adverse_fields":int(np.count_nonzero(diffs>0)),
            "tied_fields":int(np.count_nonzero(diffs==0)),
            "median":float(np.median(diffs)),"trimmed_mean":float(trimmed.mean()),
            "max_abs":float(np.max(np.abs(diffs))),
            "omit_largest_abs_mean":float(np.delete(diffs,np.argmax(np.abs(diffs))).mean())})
    secondary=comparisons[1:];previous=0.
    for rank,item in enumerate(sorted(secondary,key=lambda x:x["p_unadjusted"])):
        previous=max(previous,min(1.,(len(secondary)-rank)*item["p_unadjusted"]))
        item["p_holm_secondary4"]=previous
    report=_bound({"schema_version":"kws-equal-compute-analysis/v1",
        "plan_sha256":plan["plan_sha256"],"risk_rows":rows,"comparisons":comparisons,
        "limitation":"one fixed union panel; fields are stochastic units, not 12 independent pools or 416 independent clips"})
    path=root/"analysis.json"
    if path.exists():
        if _read(path,plan["plan_sha256"])!=report:raise ValueError("offline analysis changed")
    else:_write(path,report)
    return report


def export(plan_path:Path,run_root:Path,host="kona"):
    plan=_static(plan_path);report=aggregate(plan_path,run_root)
    root=Path(run_root);destination=root.with_name(root.name+"_export")
    if destination.exists():
        checks=json.loads((destination/"checksums.json").read_text())
        if any(sha256_file(destination/name)!=sha for name,sha in checks.items()):
            raise ValueError("existing KWS export checksum changed")
        with zipfile.ZipFile(destination/"complete.zip") as archive:
            if archive.testzip() is not None:raise ValueError("existing ZIP corrupt")
        return destination
    stage=destination.with_name("."+destination.name+".incomplete");stage.mkdir()
    _write(stage/"summary.json",{"plan_sha256":plan["plan_sha256"],
        "risk_rows":report["risk_rows"],"comparisons":report["comparisons"]})
    with zipfile.ZipFile(stage/"complete.zip","x",compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(plan_path,"plan/plan.json")
        archive.write(Path(plan_path).parent/"FROZEN","plan/FROZEN")
        archive.write(SOURCE,"source/m96_campaign.json")
        for path in sorted(root.rglob("*")):
            if path.is_file():archive.write(path,"run/"+str(path.relative_to(root)))
        archive.write(stage/"summary.json","analysis/summary.json")
        for name in SOURCE_FILES:archive.write(ROOT/name,"source/"+name)
    (stage/"mac_scp_command.txt").write_text(f"scp {host}:{(destination/'complete.zip').resolve()} ~/Downloads/kws_equal_compute_prospective_v1_complete.zip\n")
    _write(stage/"checksums.json",{n:sha256_file(stage/n) for n in
        ("complete.zip","summary.json","mac_scp_command.txt")})
    stage.rename(destination)
    return destination


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("freeze","validate","selection","seal","final","aggregate","export"))
    parser.add_argument("plan",type=Path)
    parser.add_argument("run_root",type=Path)
    args=parser.parse_args()
    if args.phase=="freeze":out=freeze(args.plan.parent)
    elif args.phase=="validate":
        p=_static(args.plan);out={"plan_sha256":p["plan_sha256"],"costs":p["costs"]}
    elif args.phase=="selection":out=run_selection(args.plan,args.run_root)
    elif args.phase=="seal":out=seal(args.plan,args.run_root)
    elif args.phase=="final":out=run_final(args.plan,args.run_root)
    elif args.phase=="aggregate":out={"analysis_payload_sha256":aggregate(args.plan,args.run_root)["payload_sha256"]}
    else:out={"export":str(export(args.plan,args.run_root))}
    print(json.dumps(out))


if __name__=="__main__":main()
