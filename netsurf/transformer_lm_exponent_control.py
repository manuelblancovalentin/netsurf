"""Prospective paired exponent-first control on fresh LM target tokens.

Original 96-bit panels and selection masks are copied by hash from finalized
pilots. The new data-free mask is frozen before any new final fault field.
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

from .adapters.transformers_pinned import PinnedTransformerAdapter
from .identities import BitIdentity
from .manifests import canonical_json_bytes, digest_object, sha256_file
from .protection import ProtectionMask
from .torch_storage import TorchStoredLinearWeights, sparse_bernoulli_schedule
from .transformer_assets import load_ready_asset
from .transformer_lm_representation_audit import structural_mask

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "pythia160m_fp32stored_fp32compute": "pythia160m_fp32_targeted96_v1",
    "pythia160m_bf16stored_fp32compute": "pythia160m_bf16fp32_targeted96_v1",
    "qwen3_0_6b_bf16": "qwen3_0_6b_bf16_targeted96_v1",
}
SOURCE_FILES = ("netsurf/transformer_lm_exponent_control.py",
                "netsurf/transformer_lm_representation_audit.py",
                "netsurf/adapters/transformers_pinned.py", "netsurf/torch_storage.py",
                "netsurf/transformer_assets.py", "netsurf/transformer_risk.py",
                "netsurf/storage_bit_codec.py", "netsurf/faults.py",
                "netsurf/protection.py", "scripts/run_lm_exponent_control_once.sh")
BERS = (1 / 96, 4 / 96)
BUDGETS = (12, 24)
FIELD_COUNT = 32
SEED_BASE = 530001
ANALYSIS_SEED = 590001
OLD_ROOT = ROOT / "workspace/runs"


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json_bytes(document).decode() + "\n")
        stream.flush(); os.fsync(stream.fileno())


def _bound(document: dict) -> dict:
    return {**document, "payload_sha256": digest_object(document)}


def _read(path: Path, plan_hash: str) -> dict:
    item = json.loads(path.read_text())
    if (item.get("plan_sha256") != plan_hash or
        item.get("payload_sha256") != digest_object({k:v for k,v in item.items() if k != "payload_sha256"})):
        raise ValueError(f"wrong-plan or corrupt control artifact: {path}")
    return item


def _bit(key: str) -> BitIdentity:
    return BitIdentity(key.split("[")[0], int(key.split("[")[1].split("]")[0]),
                       int(key.rsplit("#b", 1)[1]))


def _token_sha(ids) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i4").tobytes()).hexdigest()


def _pilot(path: Path, target: str) -> tuple[dict, dict]:
    from .transformer_lm_representation_audit import audit_archive
    audit = audit_archive(path)
    if audit["target"] != target:
        raise ValueError("wrong frozen pilot target")
    with zipfile.ZipFile(path) as archive:
        original = json.loads(archive.read("plan.json"))
        snapshot = archive.read("assets/NETSURF_ASSET_READINESS.json")
        embedded_seal_sha = hashlib.sha256(archive.read("run/selection/seal.json")).hexdigest()
    live_seal_sha = sha256_file(OLD_ROOT / (TARGETS[target] + "_run1") / "selection/seal.json")
    if embedded_seal_sha != live_seal_sha:
        raise ValueError("live pilot selection seal differs from immutable archive")
    return original, {"archive_sha256": audit["archive_sha256"],
                      "selection_masks": {k:v for k,v in audit["selected_masks"].items()
                                          if k.startswith(("random_", "significance_", "exact_isolated_", "exponent_first_"))},
                      "source_snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
                      "source_selection_seal_sha256": live_seal_sha}


def freeze(target: str, pilot_zip: Path, snapshot: Path, registry: Path, output: Path) -> dict:
    if target not in TARGETS or Path(output).exists():
        raise ValueError("unsupported target or existing immutable control plan")
    original, derived = _pilot(Path(pilot_zip), target)
    if (sha256_file(snapshot) != original["asset_manifest_sha256"] or
        sha256_file(registry) != original["registry_sha256"] or
        derived["source_snapshot_sha256"] != original["asset_manifest_sha256"]):
        raise ValueError("frozen READY/runtime snapshot differs from pilot")
    asset = load_ready_asset(snapshot, target, registry)
    if asset.record_sha256 != original["asset_record_sha256"]:
        raise ValueError("pinned model/data/runtime identity differs from pilot")
    adapter = PinnedTransformerAdapter(asset); adapter.load()
    if adapter.storage.bit_count != original["full_linear_weight_eligible_bits"]:
        raise ValueError("eligible tensor universe differs from pilot")
    ids = adapter.token_sequence(split="test")
    token_identity = dict(adapter.last_token_sequence_identity)
    if token_identity != original["test_token_identity"]:
        raise ValueError("pinned full test-token identity differs from pilot")
    fresh = ids[257:514]
    if len(fresh) != 257:
        raise ValueError("fresh held-out WikiText token window missing")
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *SOURCE_FILES], cwd=ROOT).returncode:
        raise ValueError("uncommitted scientific source")
    source_hashes = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    masks = derived["selection_masks"]
    keys = original["candidate_bits"]
    if len(keys) != 96 or any(len(masks[f"{m}_b{b}"]) != b for b in BUDGETS
                           for m in ("random", "significance", "exponent_first", "exact_isolated")):
        raise ValueError("pilot bit population or control masks changed")
    plan = {"schema_version": "lm-exponent-control-plan/v1", "status": "FROZEN",
            "target": target, "source_commit": current, "source_files_sha256": source_hashes,
            "pilot_archive_sha256": derived["archive_sha256"],
            "pilot_plan_sha256": original["plan_sha256"],
            "pilot_selection_seal_sha256": derived["source_selection_seal_sha256"],
            "asset_snapshot_sha256": sha256_file(snapshot), "registry_sha256": sha256_file(registry),
            "asset_record_sha256": asset.record_sha256,
            "stored_precision": asset.precision, "computation_precision": asset.computation_precision,
            "candidate_bits": keys, "candidate_sha256": digest_object(keys),
            "masks": masks, "masks_sha256": digest_object(masks),
            "old_selection_window": original["selection_token_slice"],
            "old_final_window": original["final_token_slice"],
            "new_final_window": {"split": "test", "start": 257, "stop": 514,
                                 "scored_token_ids": "258..513", "token_ids_sha256": _token_sha(fresh)},
            "test_token_identity": token_identity, "ber": list(BERS), "budgets": list(BUDGETS),
            "fields_per_ber": FIELD_COUNT, "field_seed_base": SEED_BASE,
            "failure_penalty_nats": 30.0,
            "primary_family": "exponent_first minus exact_isolated at BER 4/96, budgets 12 and 24; two-sided Holm across two",
            "secondary_family": "significance, exponent_first and random minus exact_isolated at two BERs and two budgets; Holm across twelve",
            "analysis_seed": ANALYSIS_SEED, "bootstrap_replicates": 10000,
            "statistical_unit": "paired independent fault field; fixed shared token cohort",
            "status_limit": "prospective follow-up prompted by old final outcomes; not untouched discovery",
            "costs": {"final_model_requests": 1 + len(BERS)*FIELD_COUNT*10,
                      "scored_target_requests": (1 + len(BERS)*FIELD_COUNT*10)*256,
                      "new_selection_model_requests": 0}}
    plan["plan_sha256"] = digest_object(plan)
    output = Path(output); output.mkdir(parents=True)
    _write(output / "plan.json", plan)
    _write(output / "mask_seal.json", _bound({"plan_sha256": plan["plan_sha256"],
            "masks_sha256": plan["masks_sha256"],
            "source_selection_seal_sha256": plan["pilot_selection_seal_sha256"]}))
    (output / "FROZEN").write_text(sha256_file(output / "plan.json") + "\n")
    return plan


def _static_plan(plan_dir: Path) -> dict:
    root = Path(plan_dir); path = root / "plan.json"; plan = json.loads(path.read_text())
    if (plan.get("schema_version") != "lm-exponent-control-plan/v1" or
        plan.get("plan_sha256") != digest_object({k:v for k,v in plan.items() if k != "plan_sha256"}) or
        (root / "FROZEN").read_text().strip() != sha256_file(path) or
        _read(root / "mask_seal.json", plan["plan_sha256"])["masks_sha256"] != digest_object(plan["masks"]) or
        plan["ber"] != list(BERS) or plan["budgets"] != list(BUDGETS) or
        plan["fields_per_ber"] != FIELD_COUNT or plan["field_seed_base"] != SEED_BASE or
        plan["new_final_window"]["start"] != 257 or plan["new_final_window"]["stop"] != 514 or
        plan["costs"]["final_model_requests"] != 641 or
        any(sha256_file(ROOT / name) != value for name,value in plan["source_files_sha256"].items())):
        raise ValueError("control plan or mask seal changed")
    return plan


def validate(plan_dir: Path, pilot_zip: Path, snapshot: Path, registry: Path):
    plan = _static_plan(plan_dir)
    if (sha256_file(pilot_zip) != plan["pilot_archive_sha256"] or
        sha256_file(snapshot) != plan["asset_snapshot_sha256"] or
        sha256_file(registry) != plan["registry_sha256"]):
        raise ValueError("pilot, asset, runtime or scientific source differs from freeze")
    original, derived = _pilot(Path(pilot_zip), plan["target"])
    if (original["plan_sha256"] != plan["pilot_plan_sha256"] or
        derived["source_selection_seal_sha256"] != plan["pilot_selection_seal_sha256"] or
        derived["selection_masks"] != plan["masks"]):
        raise ValueError("old sealed masks differ from new control plan")
    asset = load_ready_asset(snapshot, plan["target"], registry)
    if asset.record_sha256 != plan["asset_record_sha256"]:
        raise ValueError("target asset record changed")
    adapter = PinnedTransformerAdapter(asset);adapter.load()
    if adapter.storage.bit_count != original["full_linear_weight_eligible_bits"]:
        raise ValueError("eligible stored population changed")
    ids = adapter.token_sequence(split="test")
    if (adapter.last_token_sequence_identity != plan["test_token_identity"] or
        _token_sha(ids[257:514]) != plan["new_final_window"]["token_ids_sha256"]):
        raise ValueError("held-out token identities changed")
    bits = tuple(_bit(key) for key in plan["candidate_bits"])
    source_model = adapter.storage_model or adapter.model
    adapter.storage = TorchStoredLinearWeights(source_model, asset.precision, eligible_bits=bits)
    return plan, adapter, ids[257:514], bits


def _policies(plan, bits):
    mapping = {bit.key():bit for bit in bits}
    masks = {name: ProtectionMask(name, frozenset(mapping[key] for key in names),96,len(names),{})
             for name,names in plan["masks"].items()}
    masks.update({"unprotected":ProtectionMask("unprotected",frozenset(),96,0,{}),
                  "full":ProtectionMask("full",frozenset(bits),96,96,{})})
    if len(masks)!=10 or any(not set(m.protected)<=set(bits) for m in masks.values()):
        raise ValueError("invalid sealed policy family")
    return masks


def _verify_field(record: dict, intent: dict, plan: dict, masks: dict, bits: tuple,
                  clean_risk: dict, arm: int, field: int) -> None:
    class Panel:
        bit_count = 96
        def bit_at(self, ordinal): return bits[ordinal]
    seed = SEED_BASE + arm * 100000 + field
    schedule = sparse_bernoulli_schedule(Panel(), ber=BERS[arm-1], seed=seed,
                                         inference_count=256, max_events=96)
    identity = {"plan_sha256": plan["plan_sha256"], "arm": arm, "field": field,
                "ber": BERS[arm-1], "seed": seed, "schedule_sha256": schedule.digest(),
                "events": [event.key() for event in schedule.events]}
    if (any(record.get(key) != value or intent.get(key) != value for key,value in identity.items()) or
        set(record.get("policies", {})) != set(masks) or
        set(intent.get("policy_names", ())) != set(masks)):
        raise ValueError("final control field identity or schedule changed")
    for name, mask in masks.items():
        member = record["policies"][name]
        applied = sum(event.bit not in mask.protected for event in schedule.events)
        report = member["report"]
        if (member["mask_sha256"] != mask.digest() or
            member["expected_applied_faults"] != (96-mask.budget)*BERS[arm-1] or
            report["schedule_hash"] != schedule.digest() or
            report["proposed_events"] != len(schedule.events) or
            report["applied_events"] != applied or
            report["suppressed_events"] != len(schedule.events)-applied or
            member["risk"]["total_units"] != 256):
            raise ValueError("final control policy accounting changed")
        if name == "full" and member["risk"] != clean_risk:
            raise ValueError("full control protection differs from clean")


def run_final(plan_dir: Path, pilot_zip: Path, snapshot: Path, registry: Path,
              run_root: Path) -> dict:
    plan, adapter, ids, bits = validate(plan_dir,pilot_zip,snapshot,registry)
    root=Path(run_root);root.mkdir(parents=True,exist_ok=True)
    policies=_policies(plan,bits)
    clean_path=root/"clean.json"
    if clean_path.exists():clean=_read(clean_path,plan["plan_sha256"])
    else:
        clean=_bound({"plan_sha256":plan["plan_sha256"],
            "risk":asdict(adapter.evaluate_language_model(ids,context=1024,stride=512))})
        _write(clean_path,clean)
    completed=0
    for arm,ber in enumerate(BERS,1):
        for field in range(FIELD_COUNT):
            seed=SEED_BASE+arm*100000+field
            schedule=sparse_bernoulli_schedule(adapter.storage,ber=ber,seed=seed,
                inference_count=256,max_events=96)
            identity={"plan_sha256":plan["plan_sha256"],"arm":arm,"field":field,
                "ber":ber,"seed":seed,"schedule_sha256":schedule.digest(),
                "events":[event.key() for event in schedule.events],
                "policy_names":sorted(policies)}
            path=root/f"ber_{arm:02d}"/f"field_{field:03d}.json"
            intent=path.with_suffix(".intent.json")
            if path.exists():
                saved=_read(path,plan["plan_sha256"])
                _verify_field(saved,_read(intent,plan["plan_sha256"]),plan,policies,bits,
                              clean["risk"],arm,field)
                completed+=1;continue
            if intent.exists():
                raise ValueError(f"partial final field retained; same-identity recovery audit required: {intent}")
            _write(intent,_bound(identity))
            results={}
            for name,mask in policies.items():
                with adapter.injected(schedule,mask) as report:
                    risk=asdict(adapter.evaluate_language_model(ids,context=1024,stride=512))
                if name=="full" and risk!=clean["risk"]:
                    raise ValueError("full protection differs from clean reference")
                results[name]={"mask_sha256":mask.digest(),"report":asdict(report),"risk":risk,
                               "expected_applied_faults":(96-mask.budget)*ber}
            outcome=_bound({**{k:v for k,v in identity.items() if k!="policy_names"},
                            "policies":results})
            _verify_field(outcome,_bound(identity),plan,policies,bits,clean["risk"],arm,field)
            _write(path,outcome)
            completed+=1
            print(f"{plan['target']} BER {arm}/2 field {field+1}/32 sealed",flush=True)
    return {"completed":completed,"expected":len(BERS)*FIELD_COUNT}


def aggregate(plan_dir: Path, run_root: Path) -> dict:
    plan=_static_plan(plan_dir)
    root=Path(run_root)
    bits=tuple(_bit(key) for key in plan["candidate_bits"])
    policies=_policies(plan,bits)
    clean=_read(root/"clean.json",plan["plan_sha256"])
    fields={}
    for arm in (1,2):
        fields[arm]=[]
        for field in range(FIELD_COUNT):
            prefix=root/f"ber_{arm:02d}"/f"field_{field:03d}"
            record=_read(prefix.with_suffix(".json"),plan["plan_sha256"])
            intent=_read(prefix.with_suffix(".intent.json"),plan["plan_sha256"])
            _verify_field(record,intent,plan,policies,bits,clean["risk"],arm,field)
            fields[arm].append(record)
    rows=[];comparisons=[]
    for arm,items in fields.items():
        for name in sorted(items[0]["policies"]):
            risks=[item["policies"][name]["risk"] for item in items]
            rows.append({"arm":arm,"ber":BERS[arm-1],"policy":name,
                "mean_nll":statistics.mean(r["mean_nll"] for r in risks),
                "failed_token_fraction":statistics.mean(r["failed_units"]/256 for r in risks)})
        for budget in BUDGETS:
            for baseline in ("random","significance","exponent_first"):
                diffs=np.array([item["policies"][f"{baseline}_b{budget}"]["risk"]["mean_nll"]-
                    item["policies"][f"exact_isolated_b{budget}"]["risk"]["mean_nll"] for item in items])
                rng=np.random.default_rng(ANALYSIS_SEED+len(comparisons))
                boot=diffs[rng.integers(0,FIELD_COUNT,size=(10000,FIELD_COUNT))].mean(axis=1)
                signs=rng.choice([-1.,1.],size=(10000,FIELD_COUNT))
                p=(1+int(np.count_nonzero(np.abs((signs*diffs).mean(axis=1))>=abs(diffs.mean()))))/10001
                comparisons.append({"arm":arm,"ber":BERS[arm-1],"budget":budget,
                    "contrast":f"{baseline}_minus_exact_isolated","mean":float(diffs.mean()),
                    "ci95":[float(v) for v in np.quantile(boot,[.025,.975])],
                    "p_unadjusted":p,"favorable_fields":int(np.count_nonzero(diffs>0)),
                    "adverse_fields":int(np.count_nonzero(diffs<0)),
                    "tied_fields":int(np.count_nonzero(diffs==0))})
    def holm(rows,key):
        previous=0.
        for rank,item in enumerate(sorted(rows,key=lambda x:x["p_unadjusted"])):
            previous=max(previous,min(1.,(len(rows)-rank)*item["p_unadjusted"]))
            item[key]=previous
    holm(comparisons,"p_holm_secondary12")
    primary=[c for c in comparisons if c["arm"]==2 and c["contrast"]=="exponent_first_minus_exact_isolated"]
    if len(primary)!=2:raise ValueError("primary control family incomplete")
    holm(primary,"p_holm_primary2")
    report=_bound({"plan_sha256":plan["plan_sha256"],"schema_version":"lm-exponent-control-analysis/v1",
        "scope":plan["status_limit"],"clean":clean["risk"],"rows":rows,"comparisons":comparisons})
    path=root/"analysis.json"
    if path.exists():
        if _read(path,plan["plan_sha256"])!=report:raise ValueError("analysis differs from finalized fields")
    else:_write(path,report)
    return report


def export(plan_dir: Path, pilot_zip: Path, snapshot: Path, registry: Path,
           run_root: Path, host: str="kona") -> Path:
    plan=_static_plan(plan_dir)
    if sha256_file(pilot_zip)!=plan["pilot_archive_sha256"] or sha256_file(snapshot)!=plan["asset_snapshot_sha256"]:
        raise ValueError("pilot or asset identity changed")
    report=aggregate(plan_dir,run_root)
    root=Path(run_root);destination=root.with_name(root.name+"_export")
    if destination.exists():
        checks=json.loads((destination/"checksums.json").read_text())
        if any(sha256_file(destination/name)!=sha for name,sha in checks.items()):
            raise ValueError("existing finalized export changed")
        with zipfile.ZipFile(destination/"complete.zip") as archive:
            if archive.testzip() is not None:raise ValueError("existing ZIP corrupt")
        return destination
    stage=destination.with_name("."+destination.name+".incomplete")
    stage.mkdir()
    _write(stage/"summary.json",{"plan_sha256":plan["plan_sha256"],"rows":report["rows"],
                                "comparisons":report["comparisons"]})
    with zipfile.ZipFile(stage/"complete.zip","x",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(Path(plan_dir).rglob("*")):
            if path.is_file():archive.write(path,"plan/"+str(path.relative_to(plan_dir)))
        archive.write(snapshot,"assets/NETSURF_ASSET_READINESS.json")
        archive.write(registry,"assets/transformer_runtime_registry.json")
        for path in sorted(root.rglob("*")):
            if path.is_file():archive.write(path,"run/"+str(path.relative_to(root)))
        archive.write(stage/"summary.json","analysis/summary.json")
        for name in SOURCE_FILES:archive.write(ROOT/name,"source/"+name)
    (stage/"mac_scp_command.txt").write_text(f"scp {host}:{(destination/'complete.zip').resolve()} ~/Downloads/{plan['target']}_exponent_control_complete.zip\n")
    checks={name:sha256_file(stage/name) for name in ("complete.zip","summary.json","mac_scp_command.txt")}
    _write(stage/"checksums.json",checks)
    stage.rename(destination)
    return destination


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("freeze","validate","final","aggregate","export"))
    parser.add_argument("target",choices=tuple(TARGETS))
    parser.add_argument("pilot_zip",type=Path)
    parser.add_argument("snapshot",type=Path)
    parser.add_argument("registry",type=Path)
    parser.add_argument("plan_dir",type=Path)
    parser.add_argument("run_root",type=Path)
    args=parser.parse_args()
    if args.phase=="freeze":result=freeze(args.target,args.pilot_zip,args.snapshot,args.registry,args.plan_dir)
    elif args.phase=="validate":result={"plan_sha256":validate(args.plan_dir,args.pilot_zip,args.snapshot,args.registry)[0]["plan_sha256"]}
    elif args.phase=="final":result=run_final(args.plan_dir,args.pilot_zip,args.snapshot,args.registry,args.run_root)
    elif args.phase=="aggregate":result={"analysis_sha256":digest_object(aggregate(args.plan_dir,args.run_root))}
    else:result={"export":str(export(args.plan_dir,args.pilot_zip,args.snapshot,args.registry,args.run_root))}
    print(json.dumps(result))


if __name__=="__main__":main()
