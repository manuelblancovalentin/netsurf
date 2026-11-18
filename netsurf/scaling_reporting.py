"""Offline M11 evidence tables and plots; no model imports or new inference."""
import json
from pathlib import Path
import numpy as np
from .scaling_study import verify_directory, VERSION, PROTOCOL, _seal_directory
from .manifests import digest_object, sha256_file
from .interaction_attribution import mean_interval
from .conditional_scaling import paired_summary


def oracle_summary(bundles, output):
    """Independent saved-oracle audits, not pooled populations or paired K effects."""
    rows=[]
    for bundle in bundles:
        p=verify_directory(bundle)
        if p.get("phase")!="selection" or "oracle_comparison" not in p:
            raise ValueError("finalized exact-oracle selection evidence required")
        c=json.loads((Path(bundle)/"manifest.json").read_text())["execution"]
        k=p["candidate_manifest"]["k"]; n=len(p["outside_fields"])
        for method,stats in p["oracle_comparison"]["diagnostics"].items():
            rows.append({"source_payload_sha256":digest_object(p),"checkpoint_sha256":p["checkpoint_sha256"],
                "risk_metric":p["risk_metric"],"k":k,"fields":n,"method":method,**stats,
                "oracle_calls":n*2**k,"linear_calls":n*(k+1),
                "measured_total_evaluation_seconds":c["evaluation_including_apply_restore_seconds"],
                "centering_closure_error":p["oracle_comparison"]["centering_closure_error"]})
    payload={"schema_version":VERSION,"protocol":PROTOCOL,"checkpoint_sha256":"multiple: see rows",
        "rows":rows,"new_inference_calls":0,"limits":[
            "Different K pools/field seeds are not a matched causal K comparison.",
            "Clean absolute-risk error includes the changed outside-background intercept.",
            "Restricted regret is on selection fields only. No method efficacy claim.",
            "Analytic call ratios are not measured end-to-end speedups."]}
    lines=["# M11 exact oracle audit", "", "| K | Fields | Method | Maximum error | Restricted selection regret | Strict reversals | Oracle / linear calls |",
        "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['k']} | {r['fields']} | {r['method']} | {r['max_absolute_error']:.9g} | {r['finite_field_selection_regret']:.9g} | {r['strict_reversals']} | {r['oracle_calls']} / {r['linear_calls']} |")
    lines += ["",*payload["limits"]]
    result=_seal_directory(output,payload,report="\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures=Path(str(output)+"-figures");figures.mkdir(parents=True,exist_ok=False)
    for identity in sorted({r["checkpoint_sha256"] for r in rows}):
        selected=[r for r in rows if r["checkpoint_sha256"]==identity]
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for method in ("clean","outside"):
            data=sorted((r for r in selected if r["method"]==method),key=lambda r:r["k"])
            axes[0].scatter([r["k"] for r in data],[r["max_absolute_error"] for r in data],label=method)
            axes[1].scatter([r["measured_total_evaluation_seconds"] for r in data],
                [r["finite_field_selection_regret"] for r in data],label=method)
        data=[r for r in selected if r["method"]=="outside"]
        for name in ("oracle_calls","linear_calls"):
            axes[2].plot([r["k"] for r in data],[r[name]/r["fields"] for r in data],"o-",label=name)
        axes[0].set(xlabel="K (different pools)",ylabel="Maximum absolute risk error")
        axes[1].set(xlabel="Measured oracle evaluation seconds",ylabel="Finite-selection-field restricted regret")
        for ax in axes[:2]:ax.set_yscale("symlog",linthresh=1e-10)
        axes[2].set(xlabel="K",ylabel="Exact cohort calls per outside field",yscale="log")
        for ax in axes:ax.legend()
        fig.suptitle(f"{selected[0]['risk_metric']} | {identity[:12]} | engineering, not efficacy")
        fig.tight_layout();fig.savefig(figures/f"oracle_{identity[:12]}.png",dpi=160);plt.close(fig)
    (figures/"source.json").write_text(json.dumps({"payload_sha256":digest_object(payload),
        "figures":{p.name:sha256_file(p) for p in figures.glob("*.png")}},indent=2))
    return result


def analyze(plan_path, selection_registry, final_root, output, *, plots=True):
    plan=verify_directory(Path(plan_path).parent,frozen=True)
    registry=verify_directory(Path(selection_registry).parent,frozen=True)
    if registry["plan_sha256"]!=sha256_file(plan_path):raise ValueError("registry/plan mismatch")
    finals={}
    for path in Path(final_root).rglob("FINALIZED"):
        p=verify_directory(path.parent)
        if p.get("phase")!="final" or p.get("plan_sha256")!=registry["plan_sha256"]:continue
        key=(p["pool"],p["arm"])
        if key in finals:raise ValueError("duplicate final records cannot inflate replication")
        finals[key]=(p,path.parent)
    if set(finals)!={(s["pool"],s["arm"]) for s in registry["selections"]}:
        raise ValueError("analysis requires ALL prespecified final pool/arm records")
    rows=[]; per_group={}
    for item in registry["selections"]:
        sel=verify_directory(item["bundle"]); final,path=finals[(item["pool"],item["arm"])]
        if digest_object(sel)!=item["payload_sha256"] or final["selection_payload_sha256"]!=item["payload_sha256"] or sel["frozen_masks"]!=final["frozen_masks"]:
            raise ValueError("frozen mask/source mismatch")
        costs=json.loads((Path(item["bundle"])/"manifest.json").read_text())["execution"]
        k=sel["candidate_manifest"]["k"]; p=plan["arms"][item["arm"]]["logical_ber"]
        oracle=sel.get("oracle_comparison",{})
        rows.append({"pool":item["pool"],"arm":item["arm"],"k":k,"ber":p,
            "lambda":plan["arms"][item["arm"]]["expected_applied_flips"],
            "outside_error":oracle.get("diagnostics",{}).get("outside",{}).get("max_absolute_error"),
            "selection_regret":oracle.get("diagnostics",{}).get("outside",{}).get("finite_field_selection_regret"),
            "authoritative_calls":costs["authoritative_calls"],"seconds":costs["evaluation_including_apply_restore_seconds"],
            "initialization_seconds":costs["initialization_seconds"],"peak_rss_kib":costs["peak_rss_kib"],
            "selection_payload_sha256":digest_object(sel),"final_payload_sha256":digest_object(final),
            "paired_final_comparisons":final["paired_comparisons"]})
        group=per_group.setdefault((k,item["arm"]),[])
        group.append((final["mask_order"],np.mean(final["field_risks"],axis=0)))
    groups=[]
    for (k,arm),data in sorted(per_group.items()):
        names=data[0][0]
        if any(n!=names for n,_ in data):raise ValueError("inconsistent frozen comparator family")
        stats=paired_summary([v for _,v in data],names,plan["request"]["statistics"]["confidence"])
        for s in stats:
            s["independent_pools"]=s.pop("paired_fields")
            s["conditions_on"]="fixed cohort/model and prespecified pool-selection protocol; independently derived pool/field streams"
        groups.append({"k":k,"arm":arm,"equal_pool_comparisons":stats,
            "warning":"pool t intervals are approximate; pool and field counts must not be multiplied"})
    payload={"schema_version":VERSION,"protocol":PROTOCOL,"checkpoint_sha256":plan["checkpoint_sha256"],
        "plan_sha256":registry["plan_sha256"],"new_inference_calls":0,"rows":rows,"groups":groups,
        "limitations":["All prescribed pools retained, including null/adverse.",
            "No exact optimum or approximation-error estimate is claimed for K>12.",
            "Never combine raw CE and AUC across architectures.",
            "Independent pool uncertainty is distinct from final fault fields, input bootstrap and selection stability."]}
    lines=["# M11 scaling evidence","", "| Pool | BER | Expected flips | Calls | Inference-inclusive s | Max outside error | Selection regret |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['pool']} | {r['ber']} | {r['lambda']:.4g} | {r['authoritative_calls']} | {r['seconds']:.4g} | {r['outside_error']} | {r['selection_regret']} |")
    lines += ["", "Null denotes an unmeasured quantity, not zero. Exact-oracle columns describe the finite selection fields, not held-out regret.",
        "", "## Equal-pool paired summaries", "", "```json",json.dumps(groups,indent=2),"```", "", *payload["limitations"]]
    # Plots are derived separately, so finalized scientific evidence is never edited.
    result=_seal_directory(output,payload,report="\n".join(lines)+"\n")
    if plots: plot_bundle(output,Path(str(output)+"-figures"))
    return result


def plot_bundle(bundle,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    payload=verify_directory(bundle); output=Path(output); output.mkdir(parents=True,exist_ok=False)
    rows=payload["rows"]
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for arm in sorted({r["arm"] for r in rows}):
        selected=[r for r in rows if r["arm"]==arm]
        axes[0].scatter([r["k"] for r in selected],[r["seconds"] for r in selected],label=f"BER {selected[0]['ber']:.4g}")
        measured=[r for r in selected if r["outside_error"] is not None]
        axes[1].scatter([r["k"] for r in measured],[r["outside_error"] for r in measured],label=f"BER {selected[0]['ber']:.4g}")
        axes[2].scatter([r["seconds"] for r in measured],[r["selection_regret"] for r in measured],label=f"BER {selected[0]['ber']:.4g}")
    axes[0].set(xlabel="Candidate bits K",ylabel="Measured inference/apply/restore seconds")
    axes[1].set(xlabel="Candidate bits K (exact oracle only)",ylabel="Max absolute risk-prediction error")
    axes[2].set(xlabel="Measured inference/apply/restore seconds",ylabel="Finite-selection-field restricted regret")
    for ax in axes:ax.legend()
    fig.suptitle("M11 bounded evidence; not a superiority or generalization claim")
    fig.tight_layout();fig.savefig(output/"cost_error.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,max(3,len(rows)*.4)))
    for i,row in enumerate(rows):
        contrast=next(c for c in row["paired_final_comparisons"] if c["target"]=="original_S5" and c["baseline"]=="outside_additive")
        # Flip the saved paired contrast to the prespecified outside-minus-S5 direction.
        mean=-contrast["mean_difference"];ci=contrast["simultaneous_interval"]
        interval=None if ci is None else [-ci[1],-ci[0]]
        ax.errorbar(mean,i,xerr=None if interval is None else [[mean-interval[0]],[interval[1]-mean]],fmt="o",color="tab:blue")
    ax.set_yticks(range(len(rows)),[f"{r['pool']} / p={r['ber']:.4g}" for r in rows])
    ax.axvline(0,color="gray",lw=1);ax.set_xlabel("Paired final risk: outside-additive minus original S5")
    ax.set_title("Whole-field paired Bonferroni intervals; fixed masks/cohort")
    fig.tight_layout();fig.savefig(output/"paired_final.png",dpi=160);plt.close(fig)
    (output/"source.json").write_text(json.dumps({"payload_sha256":digest_object(payload),"figures":{
        name:sha256_file(output/name) for name in ("cost_error.png","paired_final.png")}},indent=2))
