"""Offline M11 evidence tables and plots; no model imports or new inference."""
import json
from pathlib import Path
import numpy as np
from .scaling_study import verify_directory, VERSION, PROTOCOL, _seal_directory
from .manifests import digest_object, sha256_file
from .interaction_attribution import mean_interval
from .conditional_scaling import paired_summary


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
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for arm in sorted({r["arm"] for r in rows}):
        selected=[r for r in rows if r["arm"]==arm]
        axes[0].scatter([r["k"] for r in selected],[r["seconds"] for r in selected],label=f"BER {selected[0]['ber']:.4g}")
        measured=[r for r in selected if r["outside_error"] is not None]
        axes[1].scatter([r["k"] for r in measured],[r["outside_error"] for r in measured],label=f"BER {selected[0]['ber']:.4g}")
    axes[0].set(xlabel="Candidate bits K",ylabel="Measured inference/apply/restore seconds")
    axes[1].set(xlabel="Candidate bits K (exact oracle only)",ylabel="Max absolute risk-prediction error")
    for ax in axes:ax.legend()
    fig.suptitle("M11 bounded evidence; not a superiority or generalization claim")
    fig.tight_layout();fig.savefig(output/"cost_error.png",dpi=160);plt.close(fig)
    (output/"source.json").write_text(json.dumps({"payload_sha256":digest_object(payload),"figure_sha256":sha256_file(output/"cost_error.png")},indent=2))
