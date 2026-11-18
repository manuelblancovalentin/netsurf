#!/usr/bin/env python3
"""Numbers cited in the paper from Run Q (calibration size of the gradient rankings), offline.

Input:  data/tecs_runQ/analysis.json, a byte copy of the finalized snapshot
        workspace/runs/paper1_runQ_calibration_size_v1/analysis_final_2026-11-12.json (hash in provenance.json).
Output: data/tecs_runQ/summary.json.  Accuracy differences are in percentage points.
"""
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/tecs_runQ"
BERS = (1e-3, 3e-3, 1e-2, 3e-2)
SIZES = (128, 256, 512, 1024)


def pts(x):
    return round(100 * x, 2)


def mask_stability(run_dir):
    """Overlap of each ranking's n-input mask with its 1,024-input mask at 12.5% (from the write-once Run Q masks)."""
    import numpy as np
    out, hashes = {}, {}
    for d in sorted(Path(run_dir).glob("*/rep*/masks.npz")):
        hashes[str(d.relative_to(run_dir))] = hashlib.sha256(d.read_bytes()).hexdigest()
        m = np.load(d)
        name = d.parent.parent.name
        for kind in ("abs", "pos", "global"):
            ref = np.unpackbits(m[f"{kind}1024@0.125000"], bitorder="little").astype(bool)
            for n in (128, 256, 512):
                x = np.unpackbits(m[f"{kind}{n}@0.125000"], bitorder="little").astype(bool)
                out.setdefault(name, {}).setdefault(f"{kind}{n}", []).append(float((x & ref).sum() / ref.sum()))
    means = {name: {k: round(sum(v) / len(v), 4) for k, v in r.items()} for name, r in out.items()}
    span = {k: [min(means[n][k] for n in means), max(means[n][k] for n in means)] for k in next(iter(means.values()))}
    (DATA / "mask_stability.json").write_text(json.dumps(dict(budget=0.125, per_checkpoint_mean=means, range_over_checkpoints=span,
                                                              masks_sha256=hashes), indent=1) + "\n")


def main():
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--masks":
        mask_stability(sys.argv[2])
    raw = (DATA / "analysis.json").read_bytes()
    prov = json.loads((DATA / "provenance.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == prov["analysis_snapshot_sha256"], "analysis.json is not the finalized snapshot"
    a = json.loads(raw)
    ck = a["checkpoints"]
    usable = [(name, c) for name, r in ck.items() for c in r["cells"] if c["decision_useful"]]

    def resolved(c, tag, sign):
        return c is not None and c.get(tag, 1.0) < 0.05 and sign * c["diff"] > 0

    order = []
    for ber in BERS:
        for n in (64, 512, 1024):
            cs = [c["c"][f"pos{n}-abs{n}"] for _, c in usable if c["ber"] == ber]
            if not cs:
                continue
            tag = "p_holm_q1_ref64" if n == 64 else "p_holm_q1"
            order.append(dict(ber=ber, images=n, cells=len(cs), positive_ahead=sum(x["diff"] > 0 for x in cs),
                              median=pts(median(x["diff"] for x in cs)), low=pts(min(x["diff"] for x in cs)),
                              high=pts(max(x["diff"] for x in cs)),
                              resolved_ahead=sum(resolved(x, tag, +1) for x in cs),
                              resolved_behind=sum(resolved(x, tag, -1) for x in cs)))
    both = {}
    for name, r in ck.items():
        cs = [c["c"]["pos1024-abs1024"] for c in r["cells"] if c["decision_useful"]]
        both[name] = dict(ahead=sum(resolved(x, "p_holm_q1", +1) for x in cs),
                          behind=sum(resolved(x, "p_holm_q1", -1) for x in cs),
                          largest_lead=pts(max(x["diff"] for x in cs)))
    size_abs = {}
    for n in SIZES:
        cs = [c["c"][f"abs{n}-abs64"] for _, c in usable]
        size_abs[n] = dict(cells=len(cs), median=pts(median(x["diff"] for x in cs)), low=pts(min(x["diff"] for x in cs)),
                           high=pts(max(x["diff"] for x in cs)), resolved=sum(x.get("p_holm_q2", 1) < 0.05 for x in cs),
                           resolved_half_point=sum(x.get("p_holm_q2", 1) < 0.05 and x["diff"] >= 0.005 for x in cs))
    curves = {}
    for kind in ("abs", "pos", "global"):
        for ber in (1e-3, 3e-3, 1e-2):
            curves[f"{kind}@{ber:g}"] = {n: pts(median(c["c"][f"{kind}{n}-{kind}64"]["diff"] for _, c in usable if c["ber"] == ber))
                                         for n in SIZES}
    d = [c["c"]["abs1024-abs512"]["diff"] for _, c in usable]
    lead = {}
    for group, names in (("well_trained", [n for n in ck if n != "undertrained_e100"]), ("undertrained", ["undertrained_e100"])):
        for n in (512, 1024):
            cs = [c["c"][f"pos{n}-abs{n}"]["diff"] for name, c in usable if name in names]
            lead[f"{group}_{n}"] = dict(positive_largest_lead=pts(max(cs)), positive_largest_loss=pts(-min(cs)))
    pooled = {k: {kk: {x: pts(y) for x, y in vv.items()} for kk, vv in v["contrasts"].items()
                  if kk in ("pos64-abs64", "pos512-abs512", "pos1024-abs1024", "abs512-abs64", "abs1024-abs64", "abs1024-abs512")}
              for k, v in a["seeds_pooled"].items()}
    above = {}
    for name, r in ck.items():
        for c in r["cells"]:
            if c["ber"] == 1e-3 and c["budget"] == 0.25:
                above[name] = {k: pts(c["mean"][k] - c["clean"]) for k in ("abs64", "abs1024", "pos64", "pos512", "pos1024")}
    out = dict(source_sha256=prov["analysis_snapshot_sha256"], q1_default_holds=a["q1_default_holds"],
               accuracy_minus_clean_ber1e3_budget25=above,
               q1_overtakes=[dict(o, diff=pts(o["diff"])) for o in a["q1_overtakes"]], q1_family=a["q1_family"],
               q2_calibration_size_matters=a["q2_calibration_size_matters"], q2_family=a["q2_family"],
               replications={n: len(r["replications"]) for n, r in ck.items()},
               usable_cells=len(usable), order_by_ber_and_size=order, both_directions_at_1024=both,
               absolute_size_gain=size_abs, abs1024_minus_abs512=dict(median=pts(median(d)), low=pts(min(d)), high=pts(max(d))),
               size_curves_median=curves, lead_and_loss=lead, seeds_pooled=pooled)
    (DATA / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("q1_default_holds", "q2_calibration_size_matters", "usable_cells", "lead_and_loss",
                                          "abs1024_minus_abs512")}, indent=1))


if __name__ == "__main__":
    main()
