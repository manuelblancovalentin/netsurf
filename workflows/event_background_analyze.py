#!/usr/bin/env python3
"""Merge the per-layer outputs of event_background_check.py and evaluate the four statements of the frozen plan
(review/EVENT_BACKGROUND_CHECK_PLAN_2026-11-17.md).  No free choice: thresholds and the figure layer are the plan's.

    event_background_analyze.py --runs DIR --out DIR
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

SHARE_RANGE, STANDS_OUT, FEW_EVENTS, KS_LIMIT = 0.02, 3.0, 0.1, 0.05


def statements(rows, figure_layer):
    """{statement: result} from the merged summary rows (dicts with float fields)."""
    groups = {}
    for r in rows:
        groups.setdefault((r["layer"], r["mask"]), []).append(r)
    s1, s3, s4 = {}, {}, {}
    for (layer, mask), g in sorted(groups.items()):
        g = sorted(g, key=lambda r: r["ber"])
        if not any(r["events_mean"] > 0 for r in g):                 # every top bit of the layer is protected
            continue
        key = f"layer{layer}|{mask}"
        shares = [r["event_share_measured"] for r in g]
        exact = [r["event_share_exact"] for r in g]
        s1[key] = dict(measured_min=min(shares), measured_max=max(shares), exact_min=min(exact), exact_max=max(exact),
                       holds=bool(max(shares) - min(shares) < SHARE_RANGE))
        sparse = [r for r in g if r["events_mean"] < FEW_EVENTS and np.isfinite(r["event_over_background_median"])]
        s3[key] = dict(rates=[r["ber"] for r in sparse], medians=[r["event_over_background_median"] for r in sparse],
                       holds=bool(sparse) and all(r["event_over_background_median"] > STANDS_OUT for r in sparse))
        ks = [r["ks_top"] for r in g]
        s4[key] = dict(ks_event_part=ks, falls=bool(all(b <= a + 1e-12 for a, b in zip(ks, ks[1:]))), last=ks[-1])
    fig = s4[f"layer{figure_layer}|unprotected"]
    return dict(
        grow_together=dict(per_layer_mask=s1, holds_everywhere=all(v["holds"] for v in s1.values())),
        one_event_stands_out=dict(per_layer_mask=s3, holds_everywhere=all(v["holds"] for v in s3.values())),
        many_events_bell_curve=dict(per_layer_mask=s4, falls_everywhere=all(v["falls"] for v in s4.values()),
                                    figure_layer_last=fig["last"], figure_layer_holds=bool(fig["falls"] and fig["last"] < KS_LIMIT)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    files = sorted(a.runs.glob("layer*/summary.csv"))
    done = sorted(a.runs.glob("layer*/DONE"))
    if len(files) != 9 or len(done) != 9:
        raise SystemExit(f"expected 9 finished layers, found {len(files)} summaries and {len(done)} DONE")
    text, rows = [], []
    for f in files:
        for r in csv.DictReader(open(f)):
            text.append(r)
            rows.append({k: (v if k in ("kernel", "mask") else float(v)) for k, v in r.items()})
    for r in rows:
        r["layer"] = int(r["layer"])
    layers = [json.loads((f.parent / "layers.json").read_text())[0] for f in files]
    figure_layer = max(layers, key=lambda L: (L["fan_in"], -L["layer"]))["layer"]
    res = statements(rows, figure_layer)
    a.out.mkdir(parents=True, exist_ok=False)
    with open(a.out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(text[0])); w.writeheader(); w.writerows(text)
    out = dict(figure_layer=figure_layer, layers=layers, statements=res,
               thresholds=dict(share_range=SHARE_RANGE, stands_out=STANDS_OUT, few_events=FEW_EVENTS, ks_limit=KS_LIMIT),
               inputs_sha256=hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest(),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.out / "statements.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(figure_layer=figure_layer, grow_together=res["grow_together"]["holds_everywhere"],
                          one_event_stands_out=res["one_event_stands_out"]["holds_everywhere"],
                          bell_curve_falls_everywhere=res["many_events_bell_curve"]["falls_everywhere"],
                          figure_layer_last_ks=res["many_events_bell_curve"]["figure_layer_last"],
                          figure_layer_holds=res["many_events_bell_curve"]["figure_layer_holds"]), indent=1))


if __name__ == "__main__":
    main()
