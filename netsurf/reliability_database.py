"""Read-only, versioned reliability database derived from finalized result bundles.

No model execution, fault sampling, mask construction or source mutation occurs here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

from .manifests import digest_object, sha256_file

SCHEMA_VERSION = "1.0.0"
METHODS = {
    "no_protection": ("Unprotected", "No protected eligible bits", "control"),
    "random_global": ("Random", "Seeded uniform bit ordering", "static"),
    "msb_global": ("MSB-first", "Descending word-relative bit significance", "static"),
    "numerical_jump_global": ("Numerical bit jump", "Descending absolute decoded XOR jump", "static"),
    "gradient_global": ("Global first-order", "Absolute mean gradient times exact XOR jump", "static"),
    "per_example_absolute_global": ("Per-example absolute gradient", "Mean absolute per-example gradient times exact XOR jump", "static"),
    "per_example_positive_global": ("Per-example positive gradient", "Mean positive per-example gradient times exact XOR jump", "static"),
    "fkeras_paper": ("FKeras (published)", "Published Hessian projection; significance tier before parameter sensitivity", "static"),
    "fkeras_repository": ("FKeras (source variant; archival)", "Pinned public-repository variant; retained only to account for source trial rows", "source_only"),
    "priorifi_derived": ("PrioriFI-derived allocation", "Not specified until faithful feedback prioritization and conversion are validated", "not_implemented"),
    "empirical_isolated": ("Exact isolated-bit", "Descending exact clean-background bit damage; full-universe score required", "static"),
    "exponent_first": ("Exponent-first", "Representation-aware exponent/significand tier ordering; contract pending", "not_implemented"),
    "full_protection": ("Full protection", "All eligible bits protected", "control"),
}
ALIASES = {
    "msb_first": "msb_global", "first_order_gradient_delta": "gradient_global",
    "per_sample_first_order_magnitude": "per_example_absolute_global",
    "per_sample_positive_damage": "per_example_positive_global",
}
SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE models (
 model_id TEXT PRIMARY KEY, benchmark_id TEXT NOT NULL, architecture TEXT NOT NULL,
 checkpoint_sha256 TEXT NOT NULL, precision TEXT NOT NULL, dataset_identity TEXT NOT NULL,
 backend TEXT NOT NULL, eligible_bits INTEGER NOT NULL CHECK(eligible_bits>0),
 universe_kind TEXT NOT NULL, clean_task_loss REAL NOT NULL, clean_quality REAL NOT NULL,
 task_loss_name TEXT NOT NULL, quality_name TEXT NOT NULL,
 calibration_cohort_json TEXT NOT NULL, evaluation_cohort_json TEXT NOT NULL);
CREATE TABLE studies (
 study_id TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES models(model_id),
 source_path TEXT NOT NULL, source_payload_sha256 TEXT NOT NULL,
 source_checksums_sha256 TEXT NOT NULL, source_schema_version TEXT NOT NULL,
 study_kind TEXT NOT NULL, status TEXT NOT NULL);
CREATE TABLE methods (
 method_id TEXT PRIMARY KEY, display_name TEXT NOT NULL, mathematical_definition TEXT NOT NULL,
 ordering_policy TEXT NOT NULL, calibration_requirement TEXT NOT NULL, adaptivity TEXT NOT NULL,
 implementation_status TEXT NOT NULL);
CREATE TABLE method_implementations (
 study_id TEXT NOT NULL REFERENCES studies(study_id), method_id TEXT NOT NULL REFERENCES methods(method_id),
 implementation_version TEXT NOT NULL, source_commit TEXT NOT NULL,
 PRIMARY KEY(study_id,method_id));
CREATE TABLE source_artifacts (
 study_id TEXT NOT NULL REFERENCES studies(study_id), relative_path TEXT NOT NULL,
 sha256 TEXT NOT NULL, bytes INTEGER NOT NULL, PRIMARY KEY(study_id,relative_path));
CREATE TABLE protection_masks (
 study_id TEXT NOT NULL REFERENCES studies(study_id), replication TEXT NOT NULL,
 method_id TEXT NOT NULL REFERENCES methods(method_id), coverage REAL NOT NULL,
 selected_bit_count INTEGER NOT NULL, mask_sha256 TEXT NOT NULL,
 bit_identities_artifact TEXT NOT NULL, construction_seed TEXT,
 ranking_cost_seconds REAL, PRIMARY KEY(study_id,replication,method_id,coverage));
CREATE TABLE trial_observations (
 study_id TEXT NOT NULL REFERENCES studies(study_id), method_id TEXT NOT NULL REFERENCES methods(method_id),
 coverage REAL NOT NULL, ber REAL NOT NULL, replication TEXT NOT NULL, trial INTEGER NOT NULL,
 schedule_sha256 TEXT NOT NULL, proposed_events INTEGER NOT NULL, applied_events INTEGER NOT NULL,
 suppressed_events INTEGER NOT NULL, task_loss REAL NOT NULL, task_quality REAL NOT NULL,
 auxiliary_loss REAL, failure INTEGER, timestamp_utc TEXT,
 source_relative_path TEXT NOT NULL,
 PRIMARY KEY(study_id,method_id,coverage,ber,replication,trial));
CREATE TABLE aggregated_conditions (
 study_id TEXT NOT NULL REFERENCES studies(study_id), method_id TEXT NOT NULL REFERENCES methods(method_id),
 coverage REAL NOT NULL, ber REAL NOT NULL, protected_bits INTEGER NOT NULL,
 field_count INTEGER NOT NULL, replication_count INTEGER NOT NULL,
 ranking_mask_count INTEGER NOT NULL, mean_loss REAL NOT NULL, loss_sd REAL NOT NULL,
 loss_se REAL NOT NULL, loss_ci_low REAL NOT NULL, loss_ci_high REAL NOT NULL,
 loss_degradation REAL NOT NULL, mean_quality REAL NOT NULL, quality_sd REAL NOT NULL,
 mean_auxiliary_loss REAL, failure_fraction REAL,
 paired_unprotected_delta_mean REAL, paired_unprotected_ci_low REAL, paired_unprotected_ci_high REAL,
 status TEXT NOT NULL,
 PRIMARY KEY(study_id,method_id,coverage,ber));
CREATE TABLE coverage_matrix (
 study_id TEXT NOT NULL REFERENCES studies(study_id), method_id TEXT NOT NULL REFERENCES methods(method_id),
 coverage REAL NOT NULL, ber REAL NOT NULL, status TEXT NOT NULL, reason TEXT NOT NULL,
 PRIMARY KEY(study_id,method_id,coverage,ber));
CREATE TABLE restricted_studies (
 study_id TEXT PRIMARY KEY, target TEXT NOT NULL, stored_precision TEXT NOT NULL,
 computation_precision TEXT NOT NULL, eligible_bits INTEGER NOT NULL,
 universe_description TEXT NOT NULL, eligible_population_sha256 TEXT NOT NULL,
 asset_record_sha256 TEXT NOT NULL, plan_sha256 TEXT NOT NULL,
 analysis_payload_sha256 TEXT NOT NULL, source_path TEXT NOT NULL,
 clean_task_loss REAL NOT NULL, task_loss_name TEXT NOT NULL,
 final_cohort_json TEXT NOT NULL, status TEXT NOT NULL);
CREATE TABLE restricted_trials (
 study_id TEXT NOT NULL REFERENCES restricted_studies(study_id), method TEXT NOT NULL REFERENCES methods(method_id),
 protected_bits INTEGER NOT NULL, coverage REAL NOT NULL, ber REAL NOT NULL,
 field INTEGER NOT NULL, schedule_sha256 TEXT NOT NULL, task_loss REAL NOT NULL,
 quality REAL, failed_fraction REAL NOT NULL, finite_loss REAL,
 proposed_events INTEGER NOT NULL, applied_events INTEGER NOT NULL,
 suppressed_events INTEGER NOT NULL, source_relative_path TEXT NOT NULL,
 field_payload_sha256 TEXT NOT NULL,
 PRIMARY KEY(study_id,method,protected_bits,ber,field));
CREATE TABLE restricted_conditions (
 study_id TEXT NOT NULL REFERENCES restricted_studies(study_id), method TEXT NOT NULL REFERENCES methods(method_id),
 protected_bits INTEGER NOT NULL, coverage REAL NOT NULL, ber REAL NOT NULL,
 field_count INTEGER NOT NULL, mean_loss REAL NOT NULL, loss_sd REAL NOT NULL,
 loss_ci_low REAL NOT NULL, loss_ci_high REAL NOT NULL,
 loss_degradation REAL NOT NULL, mean_quality REAL,
 failed_fraction REAL NOT NULL, finite_loss_mean REAL,
 status TEXT NOT NULL, PRIMARY KEY(study_id,method,protected_bits,ber));
CREATE TABLE restricted_coverage_matrix (
 study_id TEXT NOT NULL REFERENCES restricted_studies(study_id), method_id TEXT NOT NULL REFERENCES methods(method_id),
 coverage REAL NOT NULL, ber REAL NOT NULL, status TEXT NOT NULL, reason TEXT NOT NULL,
 PRIMARY KEY(study_id,method_id,coverage,ber));
CREATE TABLE planned_model_methods (
 target_id TEXT NOT NULL, method_id TEXT NOT NULL REFERENCES methods(method_id),
 target_representation TEXT NOT NULL, status TEXT NOT NULL, reason TEXT NOT NULL,
 PRIMARY KEY(target_id,method_id));
CREATE TABLE excluded_studies (
 source_id TEXT PRIMARY KEY, source_path TEXT NOT NULL,
 scientific_payload_sha256 TEXT NOT NULL, status TEXT NOT NULL,
 reason TEXT NOT NULL);
CREATE VIEW reliability_grid AS
 SELECT c.study_id,s.study_kind,m.benchmark_id,m.checkpoint_sha256,m.precision,
        m.backend,m.eligible_bits,m.task_loss_name,m.quality_name,
        c.method_id,method.display_name AS method_name,c.coverage,c.protected_bits,c.ber,
        c.field_count,c.replication_count,c.ranking_mask_count,c.mean_loss,c.loss_sd,
        c.loss_ci_low,c.loss_ci_high,c.loss_degradation,c.mean_quality,c.quality_sd,
        c.mean_auxiliary_loss,c.failure_fraction,c.paired_unprotected_delta_mean,
        c.paired_unprotected_ci_low,c.paired_unprotected_ci_high,c.status
 FROM aggregated_conditions c JOIN studies s ON c.study_id=s.study_id
 JOIN models m ON s.model_id=m.model_id JOIN methods method ON c.method_id=method.method_id;
CREATE VIEW restricted_grid AS
 SELECT c.study_id,s.target,s.stored_precision,s.computation_precision,
        s.eligible_bits,s.universe_description,s.task_loss_name,
        c.method AS method_id,method.display_name AS method_name,c.coverage,
        c.protected_bits,c.ber,c.field_count,c.mean_loss,c.loss_sd,
        c.loss_ci_low,c.loss_ci_high,c.loss_degradation,c.mean_quality,
        c.failed_fraction,c.finite_loss_mean,c.status
 FROM restricted_conditions c JOIN restricted_studies s ON c.study_id=s.study_id
 JOIN methods method ON c.method=method.method_id;
CREATE INDEX trial_fields ON trial_observations(study_id,ber,replication,trial,schedule_sha256);
"""


def _mean_ci(values: list[float]) -> tuple[float, float, float, float, float]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    se = sd / math.sqrt(len(values))
    if len(values) < 2 or se == 0:
        return mean, sd, se, mean, mean
    from scipy.stats import t
    half = float(t.ppf(0.975, len(values) - 1)) * se
    return mean, sd, se, mean - half, mean + half


def _verified_bundle(source: dict) -> tuple[Path, dict, dict, dict]:
    root = Path(source["path"])
    checks_path = root / "checksums.json"
    assert sha256_file(checks_path) == source["checksums_sha256"], f"checksum inventory changed: {root}"
    checks = json.loads(checks_path.read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    payload = json.loads((root / "scientific_payload.json").read_text())
    expected = source["scientific_payload_sha256"]
    assert (root / "FINALIZED").read_text().strip() == f"scientific_payload_sha256={expected}", root
    assert manifest["status"] == "complete" and manifest["scientific_payload_sha256"] == expected
    assert digest_object(payload) == expected, f"scientific payload changed: {root}"
    for name, digest in checks["files"].items():
        path = root / name
        assert path.is_file() and sha256_file(path) == digest, f"artifact changed: {path}"
    return root, manifest, json.loads((root / "summary.json").read_text()), checks


def _study_model(source: dict, manifest: dict, summary: dict) -> tuple:
    b = manifest["benchmark"]
    clean = summary["clean_evaluation"]
    toycar = "roc_auc" in clean
    if toycar:
        loss, quality, loss_name, quality_name = 1.0 - clean["roc_auc"], clean["roc_auc"], "one_minus_file_roc_auc", "file_roc_auc"
    else:
        loss, quality, loss_name, quality_name = clean["categorical_crossentropy"], clean["accuracy"], "categorical_crossentropy", "accuracy"
    partitions = b.get("sample_partitions", {})
    # The complete role maps are retained, including every replication's hashes and indices.
    calibration = {k: {r: v for r, v in roles.items() if r != "evaluation"} for k, roles in partitions.items()}
    evaluation = {k: roles.get("evaluation") for k, roles in partitions.items()}
    model_id = source["id"]
    architecture = b["id"]
    dataset = b.get("dataset_identity", b.get("dataset_sha256", b.get("dataset")))
    return (model_id, b["id"], architecture, b["checkpoint_sha256"], "INT8", json.dumps(dataset,sort_keys=True),
            b["inference_backend"], summary["eligible_bits"], source["kind"], loss, quality,
            loss_name, quality_name, json.dumps(calibration,sort_keys=True), json.dumps(evaluation,sort_keys=True))


def _parse_policy(value: str, default_ber: float | None = None) -> tuple[float, str, float]:
    if "/" in value:
        ber_text, condition = value.split("/", 1)
        ber = float(ber_text.removeprefix("ber="))
    else:
        if default_ber is None:
            raise ValueError("legacy policy omitted BER but study has multiple BER conditions")
        ber,condition=default_ber,value
    if "@" in condition:
        method, coverage = condition.rsplit("@", 1)
        return ber, ALIASES.get(method, method), float(coverage)
    method = ALIASES.get(condition, condition)
    return ber, method, 1.0 if method == "full_protection" else 0.0


def _insert_study(db: sqlite3.Connection, source: dict) -> None:
    root, manifest, summary, checks = _verified_bundle(source)
    sid = source["id"]
    model = _study_model(source, manifest, summary)
    db.execute("INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", model)
    db.execute("INSERT INTO studies VALUES (?,?,?,?,?,?,?,?)", (sid,sid,str(root),source["scientific_payload_sha256"],source["checksums_sha256"],manifest["schema_version"],source["kind"],"FINALIZED_MEASURED"))
    estimator_versions=manifest.get("versions",{}).get("estimators",{})
    for raw_name,version in estimator_versions.items():
        normalized={"random_reference":"random_global","s2_msb_first":"msb_global","s3_numerical_jump":"numerical_jump_global","s4_gradient_delta":"gradient_global","s5_per_example_absolute":"per_example_absolute_global","s6_per_example_positive":"per_example_positive_global"}.get(raw_name,ALIASES.get(raw_name,raw_name))
        if normalized in METHODS:
            db.execute("INSERT OR REPLACE INTO method_implementations VALUES (?,?,?,?)",(sid,normalized,version,manifest.get("source",{}).get("git_commit","unknown")))
    for name, digest in checks["files"].items():
        db.execute("INSERT INTO source_artifacts VALUES (?,?,?,?)",(sid,name,digest,(root/name).stat().st_size))
    aggregates = {}
    rep_masks = {}
    for r in csv.DictReader((root/"measurements/replication_protection_results.csv").open()):
        rep_masks[(r["replication"],float(r["logical_ber"]),ALIASES.get(r["policy"],r["policy"]),float(r["coverage"]))]=r
    runtimes={}
    for r in csv.DictReader((root/"measurements/estimator_runtime.csv").open()):
        runtimes[(r.get("replication") or "replication_0",r["estimator"])]=float(r["runtime_seconds"])
    runtime_alias={"random_global":("random_reference",),"msb_global":("s2_msb_first","msb_first"),"numerical_jump_global":("s3_numerical_jump",),"gradient_global":("s4_gradient_delta","first_order_gradient_delta"),"per_example_absolute_global":("s5_per_example_absolute","per_sample_first_order_magnitude"),"per_example_positive_global":("s6_per_example_positive","per_sample_positive_damage"),"fkeras_paper":("fkeras_paper",)}
    for r in csv.DictReader((root/"measurements/protection_results.csv").open()):
        key=(float(r["logical_ber"]), ALIASES.get(r["policy"],r["policy"]),float(r["coverage"]))
        assert key not in aggregates, (sid,key)
        aggregates[key]=r
    groups=defaultdict(list)
    field_keys=defaultdict(dict)
    distinct_bers={key[0] for key in aggregates}
    default_ber=next(iter(distinct_bers)) if len(distinct_bers)==1 else None
    for r in csv.DictReader((root/"measurements/trials.csv").open()):
        ber,method,coverage=_parse_policy(r["policy"],default_ber)
        assert method in METHODS, (sid,method)
        rep=r.get("replication") or "replication_0"
        trial=int(r["trial"])
        schedule=r["schedule_sha256"]
        field_key=(ber,rep,trial)
        prior=field_keys[field_key].setdefault("schedule",schedule)
        assert prior==schedule, f"unpaired schedule in {sid} {field_key}"
        proposed,applied,suppressed=[int(r[k]) for k in ("proposed_events","applied_events","suppressed_events")]
        assert proposed==applied+suppressed, (sid,field_key)
        if "categorical_crossentropy" in r:
            loss=float(r["categorical_crossentropy"]); quality=float(r["accuracy"]); aux=None
        else:
            assert r["primary_metric_name"]=="roc_auc" and r["loss_metric_name"]=="reconstruction_mse"
            quality=float(r["primary_metric_value"]); loss=1.0-quality; aux=float(r["loss_metric_value"])
        assert math.isfinite(loss) and math.isfinite(quality)
        db.execute("INSERT INTO trial_observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (sid,method,coverage,ber,rep,trial,schedule,proposed,applied,suppressed,loss,quality,aux,None,None,"measurements/trials.csv"))
        groups[(ber,method,coverage)].append((rep,trial,schedule,loss,quality,aux))
    clean=model[9]
    by_ber={}
    for (ber,method,coverage),records in groups.items():
        by_ber.setdefault(ber,{})[(method,coverage)]={(r,t,s):loss for r,t,s,loss,*_ in records}
    for (ber,method,coverage),records in groups.items():
        key=(ber,method,coverage)
        assert key in aggregates, f"missing saved aggregate {sid} {key}"
        saved=aggregates[key]
        losses=[v[3] for v in records]; qualities=[v[4] for v in records]
        mean,sd,se,low,high=_mean_ci(losses)
        assert len(records)==int(saved["trial_count"]), (sid,key)
        expected=float(saved.get("loss_mean",0)) if "loss_mean" in saved else 1.0-float(saved["primary_metric_mean"])
        if "loss_mean" in saved or model[11] != "one_minus_file_roc_auc":
            assert math.isclose(mean,expected,abs_tol=1e-10), (sid,key,mean,expected)
        else:
            assert math.isclose(mean,expected,abs_tol=1e-10), (sid,key,mean,expected)
        control=by_ber[ber].get(("no_protection",0.0),{})
        matched=[v[3]-control[(v[0],v[1],v[2])] for v in records if (v[0],v[1],v[2]) in control]
        if method=="no_protection": matched=[0.0]*len(records)
        assert len(matched)==len(records), f"missing paired control {sid} {key}"
        dmean,_,_,dlow,dhigh=_mean_ci(matched)
        aux=[v[5] for v in records if v[5] is not None]
        db.execute("INSERT INTO aggregated_conditions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (sid,method,coverage,ber,int(saved["protected_bits"]),len(records),len({v[0] for v in records}),int(saved["ranking_mask_count"]),mean,sd,se,low,high,mean-clean,statistics.mean(qualities),statistics.stdev(qualities) if len(qualities)>1 else 0.0,statistics.mean(aux) if aux else None,None,dmean,dlow,dhigh,"FINALIZED_MEASURED"))
        # Exact bit identities remain in the checksum-verified source artifact. They are
        # deliberately not duplicated into a multi-GB SQLite blob.
        reps={v[0] for v in records}
        for rep in reps:
            rr=rep_masks[(rep,ber,method,coverage)]
            assert int(rr["protected_bits"])==int(saved["protected_bits"])
            db.execute("INSERT OR IGNORE INTO protection_masks VALUES (?,?,?,?,?,?,?,?,?)",
                       (sid,rep,method,coverage,int(rr["protected_bits"]),rr["mask_sha256"],str(root/"artifacts/protection_masks.json"),json.dumps({"replication":rep,"estimators":manifest["execution"].get("estimators",[]),"randomization":manifest["execution"].get("randomization",{})},sort_keys=True),next((runtimes[(rep,name)] for name in runtime_alias.get(method,(method,)) if (rep,name) in runtimes),None)))
    assert set(groups)==set(aggregates), f"aggregate/trial mismatch {sid}"
    desired_coverages=(0.0,0.0625,0.125,0.1875,0.25,1.0)
    for ber in sorted(by_ber):
        for coverage in desired_coverages:
            for method in METHODS:
                if method == "fkeras_repository":
                    continue
                measured=(ber,method,coverage) in groups
                if measured:
                    status,reason="FINALIZED_MEASURED","checksum-verified source trial records"
                elif method=="no_protection" and coverage!=0 or method=="full_protection" and coverage!=1:
                    status,reason="INCOMPATIBLE","control is defined only at its boundary budget"
                elif method not in ("no_protection","full_protection") and coverage in (0,1):
                    status,reason="INCOMPATIBLE","all rankers collapse to the measured no/full-protection control at this boundary"
                elif method=="priorifi_derived":
                    status,reason="NOT_IMPLEMENTED","faithful feedback ranking and allocation conversion unvalidated"
                elif method=="exponent_first":
                    status,reason="INCOMPATIBLE","INT8 has no exponent storage field"
                elif method=="fkeras_paper" and source["id"]!="kws_speechcommands_int8_fkeras128":
                    status,reason="NOT_MEASURED","different-cohort FKeras result cannot be borrowed"
                elif method=="empirical_isolated":
                    status,reason="NOT_MEASURED","sampled isolated-bit studies do not define full-universe masks"
                else:
                    status,reason="NOT_MEASURED","no verified finalized cell at this study/cohort/budget"
                db.execute("INSERT INTO coverage_matrix VALUES (?,?,?,?,?,?)",(sid,method,coverage,ber,status,reason))


def _insert_restricted(db: sqlite3.Connection, source: dict) -> None:
    """Retain targeted experiments in separate tables; never join their BER with full universes."""
    base=Path("/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration")
    root=base/"workspace/runs"/source["run"]
    plan=json.loads((base/"config/experiments"/source["plan"]).read_text())
    analysis_path=root/"analysis.json"
    assert sha256_file(analysis_path)==source["analysis_sha256"]
    report=json.loads(analysis_path.read_text())
    assert plan["plan_sha256"]==digest_object({k:v for k,v in plan.items() if k!="plan_sha256"})
    assert report["plan_sha256"]==plan["plan_sha256"]
    assert report["payload_sha256"]==source["payload_sha256"]==digest_object({k:v for k,v in report.items() if k!="payload_sha256"})
    sid=source["id"]
    is_lm="pythia" in sid or "qwen" in sid
    clean=report["clean"]
    clean_risk=clean["mean_nll"]
    cohort=plan.get("final_token_slice",plan.get("final_source_indices",plan.get("final",{})))
    stored=plan.get("stored_precision","FP32")
    compute=plan.get("computation_precision","FP32")
    db.execute("INSERT INTO restricted_studies VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
               (sid,plan["target"],stored,compute,plan["eligible_bits"],plan["scope"],plan["eligible_population_sha256"],plan["asset_record_sha256"],plan["plan_sha256"],source["payload_sha256"],str(root),clean_risk,"penalized_token_nll" if is_lm else "capped_cross_entropy",json.dumps(cohort,sort_keys=True),"FINALIZED_RESTRICTED"))
    groups=defaultdict(list)
    fields=sorted(p for p in root.glob("final/ber_*/field_*.json") if not p.name.endswith(".intent.json"))
    assert fields
    seen=set()
    for path in fields:
        item=json.loads(path.read_text())
        assert item["plan_sha256"]==plan["plan_sha256"]
        assert item["payload_sha256"]==digest_object({k:v for k,v in item.items() if k!="payload_sha256"})
        ber=float(item["ber"]); field=int(item["field"])
        assert (ber,field) not in seen
        seen.add((ber,field))
        for policy,data in item["policies"].items():
            method=("empirical_isolated" if policy.startswith("exact_isolated") else "msb_global" if policy.startswith("significance") else "random_global" if policy.startswith("random") else "full_protection" if policy=="full" else "no_protection" if policy=="unprotected" else policy)
            budget=0 if policy=="unprotected" else plan["eligible_bits"] if policy=="full" else int(policy.rsplit("b",1)[-1])
            risk=data["risk"]
            loss=float(risk["mean_nll"])
            quality=risk.get("accuracy")
            failure=float(risk.get("failed_units",0))/float(risk["total_units"])
            assert len(risk["failure_kinds"])==risk["total_units"]
            assert sum(kind is not None for kind in risk["failure_kinds"])==risk["failed_units"]
            penalty=float(plan["failure_penalty_nats"])
            # Exact algebra on the saved capped mean, not a new model evaluation.
            # Finite tokens may themselves have been capped at the same penalty.
            finite=(loss-penalty*failure)/(1.0-failure) if failure<1.0 else None
            events=data["report"]
            proposed=int(events["proposed_events"]); applied=int(events["applied_events"]); suppressed=int(events["suppressed_events"])
            assert proposed==applied+suppressed
            db.execute("INSERT INTO restricted_trials VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (sid,method,budget,budget/plan["eligible_bits"],ber,field,item["schedule_sha256"],loss,quality,failure,finite,proposed,applied,suppressed,str(path.relative_to(root)),item["payload_sha256"]))
            groups[(method,budget,ber)].append((loss,quality,failure,finite))
    # Compare the independent analysis receipt against field-level means.
    indexed={(r["method"],int(r["budget"]),float(r["ber"])):r for r in report["rows"]}
    for (method,budget,ber),values in groups.items():
        base_method="exact_isolated" if method=="empirical_isolated" else "significance" if method=="msb_global" else "random" if method=="random_global" else "full" if method=="full_protection" else method
        if method=="no_protection": base_method="unprotected"
        original_method=f"{base_method}_b{budget}" if is_lm and method not in ("no_protection","full_protection") else base_method
        row=indexed.get((original_method,budget,ber))
        assert row is not None, (sid,method,budget,ber)
        losses=[v[0] for v in values]
        mean,sd,_,lo,hi=_mean_ci(losses)
        assert math.isclose(mean,float(row["mean_nll"]),abs_tol=1e-9), (sid,method,budget,ber,mean,row["mean_nll"])
        quality=[v[1] for v in values if v[1] is not None]
        finite=[v[3] for v in values if v[3] is not None]
        db.execute("INSERT INTO restricted_conditions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (sid,method,budget,budget/plan["eligible_bits"],ber,len(values),mean,sd,lo,hi,mean-clean_risk,statistics.mean(quality) if quality else None,statistics.mean(v[2] for v in values),statistics.mean(finite) if finite else None,"FINALIZED_RESTRICTED"))
    for ber in sorted({k[2] for k in groups}):
        for coverage in (0.0,0.0625,0.125,0.1875,0.25,1.0):
            budget=int(round(coverage*plan["eligible_bits"]))
            for method in METHODS:
                if method=="fkeras_repository": continue
                if method=="no_protection" and coverage!=0 or method=="full_protection" and coverage!=1:
                    status,reason="INCOMPATIBLE","control is defined only at its boundary budget"
                elif method not in ("no_protection","full_protection") and coverage in (0,1):
                    status,reason="INCOMPATIBLE","ranker collapses to the measured no/full-protection control"
                elif (method,budget,ber) in groups:
                    status,reason="FINALIZED_RESTRICTED","verified targeted-universe final field records"
                elif method in ("priorifi_derived","exponent_first"):
                    status,reason="NOT_IMPLEMENTED","no validated ranking-to-mask implementation for this targeted protocol"
                elif method in ("gradient_global","per_example_absolute_global","per_example_positive_global","fkeras_paper"):
                    status,reason="NOT_IMPLEMENTED","no validated matching surrogate/eigenpair mask in this targeted protocol"
                else:
                    status,reason="NOT_MEASURED","no verified final field cell for this restricted universe"
                db.execute("INSERT INTO restricted_coverage_matrix VALUES (?,?,?,?,?,?)",(sid,method,coverage,ber,status,reason))
    missing=db.execute("SELECT count(*) FROM restricted_conditions c LEFT JOIN restricted_coverage_matrix g ON c.study_id=g.study_id AND c.method=g.method_id AND c.coverage=g.coverage AND c.ber=g.ber WHERE c.study_id=? AND g.study_id IS NULL",(sid,)).fetchone()[0]
    assert missing==0, f"restricted condition missing from coverage matrix: {sid}"


def build_database(sources: Path, destination: Path) -> None:
    """Create a new database exclusively; refuse overwrite and verify every source byte."""
    spec=json.loads(Path(sources).read_text())
    if spec["schema_version"]!=SCHEMA_VERSION:
        raise ValueError("unsupported reliability source schema")
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation protects an earlier derived product from accidental replacement.
    with destination.open("x"):
        pass
    db=sqlite3.connect(destination)
    try:
        db.executescript(SCHEMA)
        db.execute("INSERT INTO metadata VALUES (?,?)",("schema_version",SCHEMA_VERSION))
        db.execute("INSERT INTO metadata VALUES (?,?)",("sources_sha256",sha256_file(Path(sources))))
        db.execute("INSERT INTO metadata VALUES (?,?)",("builder_module_sha256",sha256_file(Path(__file__))))
        for method,(name,definition,kind) in METHODS.items():
            ordering="published significance-tier then parameter sensitivity" if method=="fkeras_paper" else "global score descending" if kind=="static" else "control" if kind=="control" else "unvalidated"
            calibration="held-out model inputs" if method in ("gradient_global","per_example_absolute_global","per_example_positive_global","fkeras_paper","empirical_isolated","priorifi_derived") else "seeded ordering only" if method=="random_global" else "none"
            db.execute("INSERT INTO methods VALUES (?,?,?,?,?,?,?)",(method,name,definition,ordering,calibration,kind,"NOT_IMPLEMENTED" if kind=="not_implemented" else "SOURCE_ONLY" if kind=="source_only" else "IMPLEMENTED_OR_MEASURED"))
        for source in spec["studies"]:
            _insert_study(db,source)
        for source in spec.get("restricted_studies",[]):
            _insert_restricted(db,source)
        for source in spec.get("excluded_studies",[]):
            _verified_bundle(source)
            db.execute("INSERT INTO excluded_studies VALUES (?,?,?,?,?)",(source["id"],source["path"],source["scientific_payload_sha256"],"INCOMPATIBLE",source["reason"]))
        for target,representation,status,reason in (
            ("distilbert_sst2_int8_full","INT8","ASSET_BLOCKED","no hash-bound validated full INT8 source in this receipt"),
            ("pythia160m_int8_full","INT8","ASSET_BLOCKED","no hash-bound validated full INT8 source in this receipt"),
            ("qwen3_0_6b_int8_full","INT8","ASSET_BLOCKED","no hash-bound validated full INT8 source in this receipt"),
            ("qwen3_0_6b_int4_full","INT4","ASSET_BLOCKED","no validated packed-INT4 source and bit layout in this receipt"),
            ("distilbert_sst2_fp32_full","FP32","NOT_MEASURED","only a 1,024-bit one-row classifier subset is finalized"),
            ("pythia160m_bf16fp32_full","BF16 storage / FP32 compute","NOT_MEASURED","only a selected 96-bit universe is finalized"),
            ("qwen3_0_6b_bf16_full","BF16","NOT_MEASURED","only a selected 96-bit universe is finalized"),
        ):
            for method in METHODS:
                if method=="fkeras_repository": continue
                cell_status=("INCOMPATIBLE" if method=="exponent_first" and representation=="INT8" else
                             "NOT_IMPLEMENTED" if method in ("priorifi_derived","exponent_first") else status)
                cell_reason=("INT8 has no exponent storage field" if cell_status=="INCOMPATIBLE" else
                             "validated ranking-to-mask implementation absent" if cell_status=="NOT_IMPLEMENTED" else reason)
                db.execute("INSERT INTO planned_model_methods VALUES (?,?,?,?,?)",(target,method,representation,cell_status,cell_reason))
        db.commit()
    except Exception:
        db.close()
        destination.unlink(missing_ok=True)
        raise
    db.close()


def export_csv(database: Path, destination: Path, table: str) -> None:
    if table not in ("aggregated_conditions","coverage_matrix","trial_observations","restricted_conditions","restricted_trials","restricted_coverage_matrix","planned_model_methods","reliability_grid","restricted_grid"):
        raise ValueError(table)
    with sqlite3.connect(database) as db, Path(destination).open("x",newline="") as out:
        order=("target_id,method_id" if table=="planned_model_methods" else
               "study_id,ber,coverage,method" if table in ("restricted_conditions","restricted_trials") else
               "study_id,ber,coverage,method_id")
        cur=db.execute(f"SELECT * FROM {table} ORDER BY {order}")
        writer=csv.writer(out)
        writer.writerow([d[0] for d in cur.description])
        writer.writerows(cur)
