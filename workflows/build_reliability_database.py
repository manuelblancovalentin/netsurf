"""Build and plot the read-only NetSuRF reliability evidence database."""

from __future__ import annotations

import argparse
import json
import sqlite3
import tomllib
from pathlib import Path

from netsurf.reliability_database import PRIMARY_STUDIES, SCHEMA_VERSION, build_database, export_csv
from netsurf.manifests import sha256_file

COLORS={"no_protection":"black","random_global":"#7f7f7f","msb_global":"#d62728",
        "numerical_jump_global":"#9467bd","gradient_global":"#1f77b4",
        "per_example_absolute_global":"#2ca02c","per_example_positive_global":"#ff7f0e",
        "fkeras_paper":"#8c564b","empirical_isolated":"#17becf","full_protection":"black"}
MODEL_LABELS={"resnet8_cifar10_int8_full":"ResNet-8 / CIFAR-10 (INT8 kernels, 1,000 images)",
              "kws_speechcommands_int8_full":"DS-CNN / Speech Commands (INT8 kernels, 1,000 clips)",
              "toycar_int8_full_pilot64":"ToyCar autoencoder (INT8 kernels, 64 complete files)",
              "kws_speechcommands_int8_fkeras128":"DS-CNN / Speech Commands (separate 128-clip FKeras cohort)"}
RESTRICTED_LABELS={"distilbert_fp32_classifier1024_original":"DistilBERT / SST-2 (original 1,024-bit classifier subset)",
                   "distilbert_fp32_classifier1024_replication":"DistilBERT / SST-2 (independent 1,024-bit subset study)",
                   "pythia160m_fp32_targeted96":"Pythia-160M (96 FP32 stored bits)",
                   "pythia160m_bf16fp32_targeted96":"Pythia-160M (96 BF16 stored bits; FP32 computation)",
                   "qwen3_bf16_targeted96":"Qwen3-0.6B (96 BF16 stored bits)"}

INTERIOR_BUDGETS=(0.0625,0.125,0.1875,0.25)


def primary_plots(database: Path, directory: Path) -> None:
    """One four-panel plot per primary cohort; missing cells remain visible gaps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=False)
    with sqlite3.connect(database) as db:
        db.row_factory=sqlite3.Row
        names={r[0]:r[1] for r in db.execute("SELECT method_id,display_name FROM methods")}
        for sid in PRIMARY_STUDIES:
            rows=list(db.execute("SELECT * FROM primary_conditions WHERE study_id=?",(sid,)))
            if not rows:
                raise ValueError(f"missing primary study: {sid}")
            task_name=("cross-entropy (nats)" if rows[0]["task_loss_name"] == "categorical_crossentropy"
                       else "1 − file-level ROC-AUC" if rows[0]["task_loss_name"] == "one_minus_file_roc_auc"
                       else rows[0]["task_loss_name"])
            bers=sorted({r["ber"] for r in rows})
            fig,axes=plt.subplots(1,4,figsize=(15.5,4.3),sharey=True,constrained_layout=True)
            for ax,budget in zip(axes,INTERIOR_BUDGETS):
                condition=[r for r in rows if abs(r["coverage"]-budget)<1e-10]
                methods=sorted({r["method_id"] for r in condition if r["method_id"] not in ("no_protection","full_protection","fkeras_repository")})
                for method in methods:
                    rr=sorted((r for r in condition if r["method_id"]==method),key=lambda r:r["ber"])
                    # Lines are drawn only when every intervening planned BER is measured.
                    for r in rr:
                        ax.errorbar(r["ber"],r["loss_degradation"],
                                    yerr=[[r["mean_loss"]-r["loss_ci_low"]],[r["loss_ci_high"]-r["mean_loss"]]],
                                    fmt="o",markersize=3,capsize=2,color=COLORS.get(method),label=names[method] if r is rr[0] else None)
                    if [r["ber"] for r in rr]==bers:
                        ax.plot([r["ber"] for r in rr],[r["loss_degradation"] for r in rr],
                                linewidth=1,color=COLORS.get(method))
                for control,style,label in (("no_protection","--","Unprotected"),("full_protection",":","Full protection")):
                    rr=sorted((r for r in rows if r["method_id"]==control),key=lambda r:r["ber"])
                    if len(rr)==len(bers):
                        ax.plot([r["ber"] for r in rr],[r["loss_degradation"] for r in rr],
                                linestyle=style,linewidth=1.4,color="black",label=label)
                        if control=="no_protection":
                            ax.errorbar([r["ber"] for r in rr],[r["loss_degradation"] for r in rr],
                                        yerr=[[r["mean_loss"]-r["loss_ci_low"] for r in rr],
                                              [r["loss_ci_high"]-r["mean_loss"] for r in rr]],
                                        fmt="none",ecolor="black",elinewidth=.6,alpha=.65)
                if not methods:
                    ax.text(.5,.5,"Measurements unavailable\nfor this budget",transform=ax.transAxes,
                            ha="center",va="center",fontsize=10,bbox={"facecolor":"white","alpha":.85,"edgecolor":"none"})
                ax.set_xscale("log")
                ax.set_xticks(bers)
                ax.set_xticklabels([f"{b:g}" for b in bers],rotation=55,ha="right",fontsize=8)
                ax.set_xlim(min(bers)*.75,max(bers)*1.3)
                ax.set_title(f"{budget:.2%} protected",fontsize=10)
                ax.set_xlabel("Logical bit-flip probability")
                ax.grid(alpha=.18)
            axes[0].set_ylabel(f"Task-risk degradation ({task_name})")
            handles={label:handle for ax in axes for handle,label in zip(*ax.get_legend_handles_labels())}
            fig.legend(list(handles.values()),list(handles),loc="upper center",bbox_to_anchor=(.5,1.12),ncol=4,fontsize=8)
            fig.suptitle(f"{MODEL_LABELS[sid]} | {rows[0]['field_count']} fault fields per condition | mean ± 95% CI",fontsize=11,y=1.22)
            fig.savefig(directory/f"{sid}_loss_vs_ber.png",dpi=180,bbox_inches="tight")
            plt.close(fig)


def primary_dashboard(database: Path, output: Path) -> None:
    import csv
    methods=("random_global","msb_global","numerical_jump_global","gradient_global",
             "per_example_absolute_global","per_example_positive_global","fkeras_paper",
             "priorifi_derived","empirical_isolated","exponent_first")
    with sqlite3.connect(database) as db:
        names=dict(db.execute("SELECT method_id,display_name FROM methods"))
        records=[]
        for method in methods:
            for sid in PRIMARY_STUDIES:
                budgets=sorted({round(r[0],6) for r in db.execute(
                    "SELECT coverage FROM primary_conditions WHERE study_id=? AND method_id=?",(sid,method))
                    if any(abs(r[0]-b)<1e-10 for b in INTERIOR_BUDGETS)})
                if budgets:
                    status="MEASURED_PARTIAL" if len(budgets)<4 else "MEASURED_COMPLETE"
                elif method=="priorifi_derived": status="BLOCKED_IMPLEMENTATION"
                elif method=="empirical_isolated": status="BLOCKED_FULL_UNIVERSE_SCORING"
                elif method=="exponent_first": status="INCOMPATIBLE_INT8"
                else: status="MISSING"
                records.append((names[method],method,sid,status,";".join(f"{b:.4f}" for b in budgets),
                                ";".join(f"{b:.4f}" for b in INTERIOR_BUDGETS if b not in budgets)))
    with (output/"method_model_completeness.csv").open("x",newline="") as f:
        writer=csv.writer(f);writer.writerow(("method_name","method_id","study_id","status","measured_interior_budgets","missing_interior_budgets"));writer.writerows(records)
    lines=["# Primary method × model completeness", "", "Statuses apply only to the three primary cohorts, not to other studies.", "",
           "| Method | ResNet-8 | DS-CNN KWS | ToyCar |", "|---|---|---|---|"]
    for method in methods:
        cells=[]
        for record in records:
            if record[1]!=method: continue
            cells.append(("measured " + ", ".join(f"{100*float(b):g}%" for b in record[4].split(";") if b)) if record[4] else record[3].lower().replace("_"," "))
        lines.append(f"| {names[method]} | {' | '.join(cells)} |")
    lines.extend(("", "All six measured ranking methods lack 6.25% and 18.75% primary-cohort cells. Published-form FKeras is measured only on a separate 128-clip KWS cohort and does not fill the 1,000-clip cells. PrioriFI-derived allocation has no faithful validated ranking-to-mask implementation. Exact isolated-bit full-universe masks are unavailable; sampled candidate scores are not substitutes.",""))
    (output/"method_model_dashboard.md").write_text("\n".join(lines))


def primary_readme(database: Path, output: Path) -> None:
    with sqlite3.connect(database) as db:
        lines=["# Primary INT8 reliability database", "", "This is a derived, checksum-verified product; source result bundles and the v1 database archive were not changed. The default comparative table is `primary_conditions.csv` / SQL view `primary_conditions`, containing only the three exact primary study IDs and payload hashes below. No engineering, restricted, or FKeras-128 row enters it.", ""]
        for sid,hash_value in PRIMARY_STUDIES.items():
            n,fields=db.execute("SELECT count(*),min(field_count) FROM primary_conditions WHERE study_id=?",(sid,)).fetchone()
            bers=[r[0] for r in db.execute("SELECT DISTINCT ber FROM primary_conditions WHERE study_id=? ORDER BY ber",(sid,))]
            methods=[r[0] for r in db.execute("SELECT DISTINCT method_id FROM primary_conditions WHERE study_id=? AND coverage>0 AND coverage<1 ORDER BY method_id",(sid,))]
            lines.extend((f"## {MODEL_LABELS[sid]}", "", f"- Source payload SHA-256: `{hash_value}`.", f"- {n} measured conditions; {fields} paired proposed fault fields per condition.", f"- Measured BERs: {', '.join(f'{b:g}' for b in bers)}.", "- Measured budgets: 0%, 12.5%, 25%, 100%; 6.25% and 18.75% unavailable.", f"- Measured interior methods: {', '.join(methods)}.", ""))
        assert db.execute("SELECT count(*) FROM primary_conditions").fetchone()[0]==280, "primary row count changed"
        lines.extend(("## Separation and interpretation", "", "The 128-clip FKeras KWS cohort is in `fkeras_cohort_conditions.csv`; engineering pilots are in `engineering_conditions.csv`; targeted FP32/BF16 studies are in `restricted_conditions_long.csv`. The broad `all_conditions.csv` is archival, not the default comparative table. Compare neither different cohorts nor different task-risk units as paired results.", "", "Each BER panel shows the actual unprotected and fully protected controls. Missing partial-tier panels are explicitly marked unavailable; no values are interpolated. Error bars are Student-t 95% intervals over independently proposed fault fields conditional on the frozen model, inputs and masks. The primary raw fields and exact masks remain queryable through `trial_observations`, `protection_masks` and checksum-verified source artifacts.", "", "See `method_model_dashboard.md` for the full method ledger and `primary_coverage.csv` for every requested cell/status.", "", "## Frozen follow-up plans — not executed", "", "Three validated TOMLs and their source/configuration hashes and rough resource estimates are in `plans/plan_receipt.json`. They repeat the original evaluation cohorts with new independent fault seeds; they are exploratory matched grids, not untouched-input confirmation. The KWS plan contains published-form FKeras only, on the same 1,000-clip cohort as its six comparators. FKeras eigensolving cost is not captured by the ratio estimate; measure it before production execution. Do not launch while the M11 jobs compete for resources without researcher authorization. PrioriFI is not in these plans: a faithful feedback implementation and explicit frozen ranking-to-mask contract are prerequisites.", "", "Validate each with `/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf validate <plan.toml>`. After independent resource approval, run the same path with `netsurf run` in the foreground. New INT8 transformer grids require separately hash-bound clean-quality and exact-fault asset gates; no such rows are inferred from the restricted FP32/BF16 results.", ""))
    (output/"PRIMARY_README.md").write_text("\n".join(lines))


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"frozen source configuration changed; expected one {old!r}")
    return source.replace(old,new,1)


def freeze_primary_plans(sources: Path, output: Path) -> None:
    """Derive hash-bound, unexecuted matched grids from exact finalized source TOMLs."""
    from netsurf.study_config import load_study

    receipt=json.loads(sources.read_text())
    directory=output/"plans"
    directory.mkdir(parents=True,exist_ok=False)
    specs={
        "resnet8_cifar10_int8_full": ("resnet8_partial_tier_matched_v1", 12019,3584,2.4),
        "kws_speechcommands_int8_full": ("kws_partial_tier_fkeras_matched_v1",2701,3584,.669),
        "toycar_int8_full_pilot64": ("toycar_partial_tier_matched_v1",1288,448,3.8),
    }
    plans=[]
    for item in receipt["studies"]:
        sid=item["id"]
        if sid not in specs: continue
        name,source_seconds,source_evaluations,source_gib=specs[sid]
        source_path=Path(item["path"])/"config.toml"
        old=source_path.read_text()
        # The source config is already checksum-verified by build_database.
        doc=tomllib.loads(old)
        current=old
        current="\n".join(line for line in current.splitlines() if not line.lstrip().startswith("#"))+"\n"
        current=_replace_once(current,f'name = "{doc["experiment"]["name"]}"',f'name = "{name}"')
        current=_replace_once(current,'coverages = [0.125, 0.25]',
                              'coverages = [0.0625, 0.125, 0.1875, 0.25]')
        current=_replace_once(current,f'fault_seed = {doc["execution"]["fault_seed"]}',
                              f'fault_seed = {doc["execution"]["fault_seed"]+2000000}')
        current=_replace_once(current,f'permutation_seed = {doc["execution"]["permutation_seed"]}',
                              f'permutation_seed = {doc["execution"]["permutation_seed"]+2000000}')
        for field,value in (("description", "Exploratory matched whole-kernel INT8 reliability grid on the pinned evaluation cohort; new independent proposed fault fields."),
                            ("objective", "Measure all six established static rankings at four interior budgets and the historical BER grid, with paired zero/full controls; not a new confirmation cohort.")):
            current=_replace_once(current,f'{field} = {json.dumps(doc["experiment"][field])}',f'{field} = {json.dumps(value)}')
        if sid=="kws_speechcommands_int8_full":
            current=_replace_once(current,'schema_version = "3.4.0"','schema_version = "3.5.0"')
            fkeras=('[[estimators]]\nname = "fkeras_paper"\n'
                    'implementation_version = "fkeras-paper-continuous-kws/v1"\n'
                    'seed = 970201\n'
                    'hessian = {eigenpairs = 2, max_iterations = 100, max_products = 300, tolerance = 0.00001, ordering = "largest_magnitude_signed"}\n\n'
                    '[[policies]]\nname = "fkeras_paper"\nestimator = "fkeras_paper"\n'
                    'priority = ["bit_significance_desc", "estimator_score_desc", "stable_identity"]\n'
                    'prespecified = true\n\n')
            current=_replace_once(current,'[protection]\n',fkeras+'[protection]\n')
            current=current.replace("all six established static rankings", "all six established static rankings plus published-form FKeras")
        path=directory/f"{name}.toml"
        path.write_text(current)
        validated=load_study(path,check_inputs=True)
        methods=len(validated.policies)
        bers=len(validated.protection.probabilities)
        fields=validated.replications.count*validated.execution.trials
        evaluations=bers*fields*(2+methods*len(validated.protection.coverages))
        plans.append({"name":name,"study_id":sid,"source_configuration":str(source_path),
                      "source_configuration_sha256":sha256_file(source_path),
                      "source_scientific_payload_sha256":item["scientific_payload_sha256"],
                      "configuration":str(path.relative_to(output)),"configuration_sha256":sha256_file(path),
                      "schema_version":validated.schema_version,"status":"VALIDATED_NOT_EXECUTED",
                      "methods":methods,"bers":bers,"interior_budgets":len(validated.protection.coverages),
                      "independent_fault_fields_per_ber":fields,"policy_budget_evaluations":evaluations,
                      "evaluation_units_per_policy_evaluation":validated.samples.evaluation_count,
                      "rough_elapsed_hours_from_source_ratio":round(source_seconds*evaluations/source_evaluations/3600,2),
                      "rough_output_gib_from_source_ratio":round(source_gib*evaluations/source_evaluations,2),
                      "estimate_excludes":"new FKeras ranking/eigensolving, host contention, and setup variation"})
    if len(plans)!=3:
        raise ValueError("missing one or more primary source plans")
    (directory/"plan_receipt.json").write_text(json.dumps({"schema_version":"1.0.0","plans":plans},indent=2,sort_keys=True)+"\n")


def plots(database: Path, directory: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as db:
        db.row_factory=sqlite3.Row
        studies=list(db.execute("SELECT s.study_id,m.task_loss_name FROM studies s JOIN models m USING(model_id) WHERE s.study_kind!='engineering_whole_kernel'"))
        names={r[0]:r[1] for r in db.execute("SELECT method_id,display_name FROM methods")}
        for study in studies:
            sid=study["study_id"]
            rows=list(db.execute("SELECT * FROM aggregated_conditions WHERE study_id=? ORDER BY ber,coverage,method_id",(sid,)))
            if not rows:
                continue
            budgets=sorted({r["coverage"] for r in rows})
            bers=sorted({r["ber"] for r in rows})
            fig,axes=plt.subplots(1,len(budgets),figsize=(4.3*len(budgets),4.1),sharey=True,squeeze=False)
            for ax,coverage in zip(axes[0],budgets):
                selected=[r for r in rows if r["coverage"]==coverage or r["method_id"]=="no_protection"]
                for method in sorted({r["method_id"] for r in selected}):
                    if method=="fkeras_repository":
                        continue
                    rr=[r for r in selected if r["method_id"]==method]
                    ax.errorbar([r["ber"] for r in rr],[r["loss_degradation"] for r in rr],
                                yerr=[[r["mean_loss"]-r["loss_ci_low"] for r in rr],[r["loss_ci_high"]-r["mean_loss"] for r in rr]],
                                label=names[method],marker="o",capsize=2,linewidth=1,color=COLORS.get(method))
                ax.set_xscale("log")
                ax.set_title(f"Protected: {coverage:.2%}")
                ax.set_xlabel("Logical bit-flip probability")
                ax.grid(alpha=.2)
            axes[0][0].set_ylabel(f"Task-risk degradation ({study['task_loss_name']})")
            entries={label:handle for ax in axes[0] for handle,label in zip(*ax.get_legend_handles_labels())}
            labels=list(entries);handles=[entries[label] for label in labels]
            fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,0),ncol=min(4,len(labels)))
            fig.suptitle(f"{MODEL_LABELS.get(sid,sid)}: loss degradation vs BER",y=1.02)
            fig.tight_layout(rect=(0,.08,1,1))
            fig.savefig(directory/f"{sid}_loss_vs_ber.png",dpi=160,bbox_inches="tight")
            plt.close(fig)
            plot_bers=[b for b in bers if any(abs(b-x)<1e-12 for x in (1e-4,1e-3,1e-2,1e-1))] or bers
            fig,axes=plt.subplots(1,len(plot_bers),figsize=(4.2*len(plot_bers),4.1),sharey=True,squeeze=False)
            for ax,ber in zip(axes[0],plot_bers):
                selected=[r for r in rows if r["ber"]==ber]
                for method in sorted({r["method_id"] for r in selected if r["method_id"] not in ("no_protection","full_protection","fkeras_repository")}):
                    rr=[r for r in selected if r["method_id"]==method]
                    ax.errorbar([r["coverage"] for r in rr],[r["loss_degradation"] for r in rr],
                                yerr=[[r["mean_loss"]-r["loss_ci_low"] for r in rr],[r["loss_ci_high"]-r["mean_loss"] for r in rr]],
                                marker="o",capsize=2,label=names[method],linewidth=1,color=COLORS.get(method))
                control=next((r for r in selected if r["method_id"]=="no_protection"),None)
                if control:
                    ax.axhline(control["loss_degradation"],color="black",linestyle="--",label="Unprotected")
                    ax.errorbar([0.0],[control["loss_degradation"]],
                                yerr=[[control["mean_loss"]-control["loss_ci_low"]],[control["loss_ci_high"]-control["mean_loss"]]],
                                color="black",marker="o",capsize=2)
                full=next((r for r in selected if r["method_id"]=="full_protection"),None)
                if full:
                    ax.scatter([1.0],[full["loss_degradation"]],marker="s",color="black",label="Full protection")
                ax.set_title(f"BER {ber:g}")
                ax.set_xlabel("Protected eligible-bit fraction")
                ax.set_xlim(-.025,1.025)
                ax.set_xticks(budgets)
                ax.tick_params(axis="x",labelrotation=45)
                ax.grid(alpha=.2)
            axes[0][0].set_ylabel(f"Task-risk degradation ({study['task_loss_name']})")
            entries={label:handle for ax in axes[0] for handle,label in zip(*ax.get_legend_handles_labels())}
            labels=list(entries);handles=[entries[label] for label in labels]
            fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,0),ncol=min(4,len(labels)))
            fig.suptitle(f"{MODEL_LABELS.get(sid,sid)}: loss degradation vs budget",y=1.02)
            fig.tight_layout(rect=(0,.08,1,1))
            fig.savefig(directory/f"{sid}_loss_vs_budget.png",dpi=160,bbox_inches="tight")
            plt.close(fig)
        restricted=list(db.execute("SELECT study_id,task_loss_name,eligible_bits FROM restricted_studies"))
        restricted_names={"no_protection":"Unprotected","random_global":"Random","msb_global":"Significance", "empirical_isolated":"Exact isolated-bit","full_protection":"Full protection"}
        for study in restricted:
            sid=study["study_id"]
            rows=list(db.execute("SELECT * FROM restricted_conditions WHERE study_id=? ORDER BY ber,coverage,method",(sid,)))
            budgets=sorted({r["coverage"] for r in rows})
            bers=sorted({r["ber"] for r in rows})
            fig,axes=plt.subplots(1,len(budgets),figsize=(4.2*len(budgets),4),sharey=True,squeeze=False)
            for ax,budget in zip(axes[0],budgets):
                selected=[r for r in rows if r["coverage"]==budget or r["method"]=="no_protection"]
                for method in sorted({r["method"] for r in selected}):
                    rr=[r for r in selected if r["method"]==method]
                    ax.errorbar([r["ber"] for r in rr],[r["loss_degradation"] for r in rr],
                                yerr=[[r["mean_loss"]-r["loss_ci_low"] for r in rr],[r["loss_ci_high"]-r["mean_loss"] for r in rr]],
                                marker="o",capsize=2,label=restricted_names.get(method,method),color=COLORS.get(method))
                ax.set_xscale("log")
                ax.set_title(f"{budget:.2%} of {study['eligible_bits']} selected bits")
                ax.set_xlabel("Bernoulli probability within restricted universe")
                ax.grid(alpha=.2)
            axes[0][0].set_ylabel(f"Task-loss degradation ({study['task_loss_name']})")
            entries={label:handle for ax in axes[0] for handle,label in zip(*ax.get_legend_handles_labels())}
            labels=list(entries);handles=[entries[label] for label in labels]
            fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,0),ncol=min(4,len(labels)))
            fig.suptitle(f"{RESTRICTED_LABELS.get(sid,sid)}: RESTRICTED-universe loss vs BER",y=1.02)
            fig.tight_layout(rect=(0,.08,1,1))
            fig.savefig(directory/f"{sid}_restricted_loss_vs_ber.png",dpi=160,bbox_inches="tight")
            plt.close(fig)
            fig,axes=plt.subplots(1,len(bers),figsize=(4.2*len(bers),4),sharey=True,squeeze=False)
            for ax,ber in zip(axes[0],bers):
                selected=[r for r in rows if r["ber"]==ber]
                for method in sorted({r["method"] for r in selected if r["method"] not in ("no_protection","full_protection")}):
                    rr=[r for r in selected if r["method"]==method]
                    ax.errorbar([r["coverage"] for r in rr],[r["loss_degradation"] for r in rr],
                                yerr=[[r["mean_loss"]-r["loss_ci_low"] for r in rr],[r["loss_ci_high"]-r["mean_loss"] for r in rr]],
                                marker="o",capsize=2,label=restricted_names.get(method,method),color=COLORS.get(method))
                control=next((r for r in selected if r["method"]=="no_protection"),None)
                if control:
                    ax.axhline(control["loss_degradation"],color="black",linestyle="--",label="Unprotected")
                    ax.errorbar([0.0],[control["loss_degradation"]],
                                yerr=[[control["mean_loss"]-control["loss_ci_low"]],[control["loss_ci_high"]-control["mean_loss"]]],
                                color="black",marker="o",capsize=2)
                full=next((r for r in selected if r["method"]=="full_protection"),None)
                if full: ax.scatter([1.0],[full["loss_degradation"]],marker="s",color="black",label="Full protection")
                ax.set_title(f"BER {ber:g}")
                ax.set_xlabel("Protected fraction of restricted bits")
                ax.set_xlim(-.025,1.025)
                ax.set_xticks(budgets)
                ax.tick_params(axis="x",labelrotation=45)
                ax.grid(alpha=.2)
            axes[0][0].set_ylabel(f"Task-loss degradation ({study['task_loss_name']})")
            entries={label:handle for ax in axes[0] for handle,label in zip(*ax.get_legend_handles_labels())}
            labels=list(entries);handles=[entries[label] for label in labels]
            fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,0),ncol=min(4,len(labels)))
            fig.suptitle(f"{RESTRICTED_LABELS.get(sid,sid)}: RESTRICTED-universe loss vs budget",y=1.02)
            fig.tight_layout(rect=(0,.08,1,1))
            fig.savefig(directory/f"{sid}_restricted_loss_vs_budget.png",dpi=160,bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources",type=Path,default=Path("config/reliability_sources_v1.json"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--verify",action="store_true",help="verify an existing derived database product")
    args=parser.parse_args()
    if args.verify:
        product=args.output
        checks=json.loads((product/"checksums.json").read_text())
        assert checks["schema_version"]==SCHEMA_VERSION
        assert checks["sources_sha256"]==sha256_file(args.sources)
        for name,digest in checks["files"].items():
            assert sha256_file(product/name)==digest, name
        assert (product/"FINALIZED").read_text().strip()==f"reliability_database_sha256={checks['files']['reliability_v2.sqlite']}"
        with sqlite3.connect(product/"reliability_v2.sqlite") as connection:
            assert connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0]==SCHEMA_VERSION
            assert connection.execute("PRAGMA integrity_check").fetchone()[0]=="ok"
            assert connection.execute("SELECT count(*) FROM primary_conditions").fetchone()[0]==280
        print("Verified",product)
        return
    args.output.mkdir(parents=True,exist_ok=False)
    db=args.output/"reliability_v2.sqlite"
    build_database(args.sources,db)
    export_csv(db,args.output/"primary_conditions.csv","primary_conditions")
    export_csv(db,args.output/"primary_coverage.csv","primary_coverage")
    export_csv(db,args.output/"all_conditions.csv","reliability_grid")
    export_csv(db,args.output/"fkeras_cohort_conditions.csv","fkeras_cohort_conditions")
    export_csv(db,args.output/"engineering_conditions.csv","engineering_conditions")
    export_csv(db,args.output/"coverage_matrix.csv","coverage_matrix")
    export_csv(db,args.output/"restricted_conditions_long.csv","restricted_grid")
    export_csv(db,args.output/"restricted_coverage_matrix.csv","restricted_coverage_matrix")
    export_csv(db,args.output/"planned_model_methods.csv","planned_model_methods")
    primary_plots(db,args.output/"plots")
    primary_dashboard(db,args.output)
    primary_readme(db,args.output)
    freeze_primary_plans(args.sources,args.output)
    files={str(p.relative_to(args.output)):sha256_file(p) for p in sorted(args.output.rglob("*")) if p.is_file()}
    (args.output/"checksums.json").write_text(json.dumps({"schema_version":SCHEMA_VERSION,"sources_sha256":sha256_file(args.sources),"workflow_sha256":sha256_file(Path(__file__)),"files":files},sort_keys=True,indent=2)+"\n")
    (args.output/"FINALIZED").write_text(f"reliability_database_sha256={files['reliability_v2.sqlite']}\n")
    print(db)


if __name__=="__main__":
    main()
