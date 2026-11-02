"""Render engineering-curve previews from the read-only reconstruction audit CSV.

No new inference. These are descriptive fixed-cohort fault-field intervals, not
simultaneous bands or uncertainty over training/data populations/random masks.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {"msb_global": "MSB-first", "numerical_jump_global": "Numerical jump",
          "gradient_global": "Global first-order", "per_example_absolute_global": "Per-example absolute",
          "per_example_positive_global": "Per-example positive", "random_global": "Random"}
COLORS = ["#000000", "#E69F00", "#0072B2", "#009E73", "#CC79A7", "#777777"]
MODELS = {"resnet": "ResNet-8 / CIFAR-10", "kws": "DS-CNN / Speech Commands", "toycar": "ToyCar autoencoder"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audit", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    with (args.audit / "static_curves.csv").open() as f:
        rows = list(csv.DictReader(f))
    out = args.output or args.audit / "previews"
    out.mkdir(exist_ok=False)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    for model, title in MODELS.items():
        data = [r for r in rows if r["benchmark"] == model]
        n = data[0]["count"]
        metric = "File-level ROC-AUC" if model == "toycar" else "Classification accuracy"
        for mode in ("ber", "budget"):
            fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
            for ax, condition in zip(axs, [.125, .25] if mode == "ber" else [.001, .01]):
                for (policy, label), color in zip(LABELS.items(), COLORS):
                    selected = [r for r in data if r["policy"] == policy and
                                float(r["coverage" if mode == "ber" else "ber"]) == condition]
                    if mode == "budget":
                        selected += [r for r in data if float(r["ber"]) == condition and
                                     r["policy"] == "no_protection"]
                    selected.sort(key=lambda r: float(r[mode == "ber" and "ber" or "coverage"]))
                    x = [float(r["ber"] if mode == "ber" else r["coverage"])* (1 if mode == "ber" else 100) for r in selected]
                    y = [float(r["mean"]) for r in selected]
                    lo = [max(0, float(r["ci_low"])) for r in selected]
                    hi = [min(1, float(r["ci_high"])) for r in selected]
                    ax.errorbar(x, y, yerr=[[a-b for a,b in zip(y,lo)], [b-a for a,b in zip(y,hi)]],
                                fmt="o--", markersize=3, linewidth=.9, capsize=2, color=color, label=label)
                if mode == "ber":
                    ref = sorted([r for r in data if r["policy"] == "no_protection"], key=lambda r: float(r["ber"]))
                    y = [float(r["mean"]) for r in ref]
                    ax.errorbar([float(r["ber"]) for r in ref], y,
                                yerr=[[a-max(0,float(r["ci_low"])) for a,r in zip(y,ref)],
                                      [min(1,float(r["ci_high"]))-a for a,r in zip(y,ref)]],
                                fmt="x:", color="#D55E00", capsize=2, linewidth=.9, label="Unprotected")
                    ax.set_xscale("log")
                    ax.set_xlabel("Logical per-bit flip probability")
                    ax.set_title(f"{100*condition:g}% protected bits")
                    if max(float(r["ber"]) for r in ref) > .02:
                        ax.axvspan(.02,.12,color="gray",alpha=.07)
                        ax.text(.98,.03,"High-BER stress",transform=ax.transAxes,ha="right",fontsize=7,color="#555555")
                else:
                    ax.set_xlabel("Protected fraction of eligible bits (%)")
                    ax.set_title(f"Logical BER = {condition:g}")
                    # Full protection is a separate deterministic control, not
                    # a spurious interpolated point on the 0--25% budget curve.
                    ax.set_xlim(-1, 26)
                    ax.set_xticks([0, 6.25, 12.5, 18.75, 25])
                clean = float(next(r["mean"] for r in data if r["policy"] == "full_protection"))
                ax.axhline(clean,color="#666666",ls="-.",lw=.8,label="Clean / full protection")
                ax.set_ylabel(metric); ax.set_ylim(0, 1.02)
                ax.grid(axis="y",alpha=.2)
            handles, labels = axs[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc="lower center",ncol=4,fontsize=8,frameon=False)
            fig.suptitle(f"{title} — {n} paired fields per BER; fixed evaluation cohort",fontsize=12)
            fig.subplots_adjust(bottom=.36,top=.82,wspace=.26)
            fig.text(.5,.20,"Descriptive 95% t intervals, intersected with [0, 1]; dashed lines are guides, not fitted models.",ha="center",fontsize=8)
            fig.text(.5,.15,"Fixed cohort and realized masks; random-mask variability is nested by replication, not by trial.",ha="center",fontsize=8)
            for suffix in ("png", "pdf"):
                fig.savefig(out / f"{model}_{mode}.{suffix}",dpi=180)
            plt.close(fig)


if __name__ == "__main__":
    main()
