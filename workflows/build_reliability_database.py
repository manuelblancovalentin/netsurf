"""Build and plot the read-only NetSuRF reliability evidence database."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from netsurf.reliability_database import build_database, export_csv
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
        assert checks["schema_version"]=="1.0.0"
        assert checks["sources_sha256"]==sha256_file(args.sources)
        for name,digest in checks["files"].items():
            assert sha256_file(product/name)==digest, name
        assert (product/"FINALIZED").read_text().strip()==f"reliability_database_sha256={checks['files']['reliability_v1.sqlite']}"
        with sqlite3.connect(product/"reliability_v1.sqlite") as connection:
            assert connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0]=="1.0.0"
            assert connection.execute("PRAGMA integrity_check").fetchone()[0]=="ok"
        print("Verified",product)
        return
    args.output.mkdir(parents=True,exist_ok=True)
    db=args.output/"reliability_v1.sqlite"
    build_database(args.sources,db)
    export_csv(db,args.output/"conditions_long.csv","reliability_grid")
    export_csv(db,args.output/"coverage_matrix.csv","coverage_matrix")
    export_csv(db,args.output/"restricted_conditions_long.csv","restricted_grid")
    export_csv(db,args.output/"restricted_coverage_matrix.csv","restricted_coverage_matrix")
    export_csv(db,args.output/"planned_model_methods.csv","planned_model_methods")
    plots(db,args.output/"plots")
    files={str(p.relative_to(args.output)):sha256_file(p) for p in sorted(args.output.rglob("*")) if p.is_file()}
    (args.output/"checksums.json").write_text(json.dumps({"schema_version":"1.0.0","sources_sha256":sha256_file(args.sources),"workflow_sha256":sha256_file(Path(__file__)),"files":files},sort_keys=True,indent=2)+"\n")
    (args.output/"FINALIZED").write_text(f"reliability_database_sha256={files['reliability_v1.sqlite']}\n")
    print(db)


if __name__=="__main__":
    main()
