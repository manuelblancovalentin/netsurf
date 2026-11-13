#!/usr/bin/env python3
"""Run R, Stage 2 confirmatory analysis (plan Section 5, Stage 2 addendum, amendments R-A4 and R-A5, analysis note A2).
Committed before any Stage 2 outcome is read; run once, after every phase has ended.  Write-once.

Inputs: workspace/runs/paper1_runR_stage2_eval_v1/{main,mc}/<checkpoint>/rep<r>/field<f>/ (every job DONE), and the
frozen R-A5 choices paper1_runR_v2_picks_v1/picks.json (sha256 checked).

Arms in a cell (checkpoint, budget, rate): the six static arms (abs64, abs1024, pos1024, squared1024, fkeras, msb),
the analytic rate-aware mask built for the tested rate (`ra_analytic`, v1), the MC-corrected mask (`ra_mc`), and the
selected mask (`v2`, R-A5: in each replication the accuracy of the mask chosen in picks.json, one of ra_analytic,
abs1024, pos1024, on the same fields).

Usable cell: the best arm's mean accuracy is within 0.05 of the checkpoint's clean accuracy (mean over replications of
each replication's window clean accuracy).  Paired test: for two arms, the per-field difference (replication x field,
up to 72 pairs), two-sided pooled-field sign-flip test on the differences in images (exact integer arithmetic),
100,000 assignments, seed 20261123, columns sized to the largest cell.  Violation: mean difference >= 0.5 point and Holm p < 0.05.

- R2a: reference ra_analytic; other arms the six static arms and ra_mc; Holm over the checkpoint's usable cells x
  these seven arms.  Holds if no checkpoint has a violation.
- R2b: the same test with reference abs1024 (others: the five other static arms, ra_analytic, ra_mc) and separately
  with reference pos1024; holds if each fails (has a violation) on at least seven of the nine checkpoints.
- R2c: ra_mc - ra_analytic per usable cell, Holm over the checkpoint's usable cells; resolved ahead / behind:
  Holm p < 0.05 with positive / negative mean.  Kept if resolved ahead in at least one cell on at least three
  checkpoints and resolved behind in none.
- R2a' (R-A5): reference v2; other arms the six static arms; Holm over the checkpoint's usable cells x six arms.
- Descriptive: largest gap of every arm per checkpoint (best arm mean minus own mean over usable cells; biased upward,
  the best arm is chosen on the same fields); v2 - v1 per usable cell; R-A5 choices in usable cells; the misjudged-rate
  table for pinned and under-trained (gap of the analytic mask built for p_a and tested at p_b against the best arm
  of the cell (p_b, budget), mean over budgets where usable).

  python runR_stage2_analyze.py [--eval-root DIR --out DIR --picks FILE --picks-sha HEX]   (synthetic tests only)
  python runR_stage2_analyze.py --interim-main --out DIR   (descriptive look at the main arms; not the confirmatory run)
Output: workspace/runs/paper1_runR_stage2_analysis_v1/{summary.json, cells.csv, cross_rate.csv}.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from runP_phase0 import PAPER, sha256_file
from runR_stage1 import NAMES, reps_for
from runR_stage2 import CROSS, EVAL_OUT, FIELDS, FRACTIONS, RATES, STATIC

PICKS = PAPER / "workspace/runs/paper1_runR_v2_picks_v1/picks.json"
PICKS_SHA = "8c1a864f5310903d941d2d2e4eae51c5479c1aed25b0d78fa680eda347618335"
OUTD = PAPER / "workspace/runs/paper1_runR_stage2_analysis_v1"
MARGIN, GAP, ALPHA, WINDOW = 0.05, 0.5, 0.05, 1000
SIGNS = np.random.default_rng(20261123).choice((-1.0, 1.0), size=(100_000, max(len(reps_for(n)) for n in NAMES) * FIELDS))


def signflip_p(counts):
    """Exact on integer counts (accuracy differences in images of the 1,000-image window): |sum of signed counts|."""
    c = np.asarray(counts, dtype=np.float64)
    if not c.any():
        return 1.0
    null = np.abs(SIGNS[:, : c.size] @ c)
    return float((1 + (null >= abs(c.sum())).sum()) / (SIGNS.shape[0] + 1))


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    adj, running = np.empty(ps.size), 0.0
    for rank, i in enumerate(np.argsort(ps, kind="stable")):
        running = max(running, (ps.size - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def load(root, parts=("main", "mc")):
    """acc[(name, budget, rate, arm)][(rep, field)]; cross[(name, budget, p_a, p_b)][(rep, field)]; clean[name][rep]."""
    acc, cross, clean, missing = defaultdict(dict), defaultdict(dict), defaultdict(dict), []
    for part in parts:
        for name in NAMES:
            for rep in reps_for(name):
                for f in range(FIELDS):
                    d = root / part / name / f"rep{rep}" / f"field{f}"
                    if not (d / "DONE").exists():
                        missing.append(f"{part}/{name}/rep{rep}/field{f}"); continue
                    if part == "main" and f == 0:
                        clean[name][rep] = json.loads((d / "field.json").read_text())["clean_accuracy"]
                    with open(d / "results.csv") as fh:
                        for row in csv.DictReader(fh):
                            b, p = round(float(row["budget"]), 6), round(float(row["ber"]), 8)
                            a, v = row["arm"], float(row["accuracy"])
                            if row["built_for"] == "":
                                acc[(name, b, p, a)][(rep, f)] = v
                                continue
                            pa = round(float(row["built_for"]), 8)
                            if pa == p:
                                acc[(name, b, p, a)][(rep, f)] = v
                            elif a == "ra_analytic":
                                cross[(name, b, pa, p)][(rep, f)] = v
    return acc, cross, clean, missing


def add_v2(acc, picks):
    for name in NAMES:
        for b in FRACTIONS:
            for p in RATES:
                k = lambda a: acc[(name, round(b, 6), round(p, 8), a)]  # noqa: E731
                acc[(name, round(b, 6), round(p, 8), "v2")] = {
                    (r, f): k(picks[f"resnet8/{name}/rep{r}/{b:.6f}/{p:g}"])[(r, f)] for (r, f) in k("ra_analytic")}


def compare(acc, name, cells, ref, others):
    """Paired tests of every arm in `others` against `ref` over `cells`; Holm over the whole family."""
    out = []
    for (b, p) in cells:
        R = acc[(name, b, p, ref)]
        keys = sorted(R)
        for a in others:
            A = acc[(name, b, p, a)]
            if sorted(A) != keys:
                raise SystemExit(f"{name} {b} {p}: {a} and {ref} have different fields")
            n = np.rint(np.array([A[k] - R[k] for k in keys]) * WINDOW)
            out.append(dict(checkpoint=name, budget=b, ber=p, ref=ref, arm=a, diff=float(100 * n.mean() / WINDOW),
                            fields=int(n.size), p_raw=signflip_p(n)))
    for c, h in zip(out, holm([c["p_raw"] for c in out])):
        c["p_holm"] = float(h)
        c["violation"] = bool(c["diff"] >= GAP and h < ALPHA)
    return out


def main(root=EVAL_OUT, outd=OUTD, picks_path=PICKS, picks_sha=PICKS_SHA, interim=False):
    """interim=True: descriptive look at the main arms only (no MC-corrected mask, no R2c), never into OUTD."""
    if interim and outd == OUTD:
        raise SystemExit("an interim look cannot write the confirmatory output")
    mc = () if interim else ("ra_mc",)
    ARMS = STATIC + ("ra_analytic",) + mc + ("v2",)
    if hashlib.sha256(picks_path.read_bytes()).hexdigest() != picks_sha:
        raise SystemExit("picks.json does not match its frozen hash")
    picks = json.loads(picks_path.read_text())["picks"]
    acc, cross, clean, missing = load(root, ("main",) if interim else ("main", "mc"))
    if missing:
        raise SystemExit(f"{len(missing)} Stage 2 jobs are not finished, e.g. {missing[:3]}")
    outd.mkdir(exist_ok=False)
    add_v2(acc, picks)
    mean = lambda d: float(np.mean(list(d.values())))  # noqa: E731
    allcells, per_ckpt, cross_rows = [], {}, []
    rules = dict(R2a=[], R2b_abs=[], R2b_pos=[], R2c=[], R2a_prime=[])
    for name in NAMES:
        c0 = float(np.mean(list(clean[name].values())))
        cells = [(round(b, 6), round(p, 8)) for b in FRACTIONS for p in RATES
                 if max(mean(acc[(name, round(b, 6), round(p, 8), a)]) for a in ARMS) >= c0 - MARGIN]
        r2a = compare(acc, name, cells, "ra_analytic", STATIC + mc)
        r2b_abs = compare(acc, name, cells, "abs1024", tuple(a for a in STATIC if a != "abs1024") + ("ra_analytic",) + mc)
        r2b_pos = compare(acc, name, cells, "pos1024", tuple(a for a in STATIC if a != "pos1024") + ("ra_analytic",) + mc)
        r2c = compare(acc, name, cells, "ra_analytic", mc)
        r2ap = compare(acc, name, cells, "v2", STATIC)
        for k, v in zip(rules, (r2a, r2b_abs, r2b_pos, r2c, r2ap)):
            rules[k] += v
        allcells += r2a + r2b_abs + r2b_pos + r2ap
        gaps = defaultdict(list)
        for (b, p) in cells:
            means = {a: mean(acc[(name, b, p, a)]) for a in ARMS}
            best = max(means.values())
            for a in ARMS:
                gaps[a].append(100 * (best - means[a]))
        v2v1 = [100 * (mean(acc[(name, b, p, "v2")]) - mean(acc[(name, b, p, "ra_analytic")])) for (b, p) in cells]
        kept = Counter(picks[f"resnet8/{name}/rep{r}/{b:.6f}/{p:g}"] for (b, p) in cells for r in reps_for(name))
        per_ckpt[name] = dict(
            clean_accuracy=c0, usable_cells=len(cells),
            R2a_violations=sum(c["violation"] for c in r2a), R2b_abs_fails=any(c["violation"] for c in r2b_abs),
            R2b_pos_fails=any(c["violation"] for c in r2b_pos),
            R2c_ahead=sum(c["p_holm"] < ALPHA and c["diff"] > 0 for c in r2c),
            R2c_behind=sum(c["p_holm"] < ALPHA and c["diff"] < 0 for c in r2c),
            R2a_prime_violations=sum(c["violation"] for c in r2ap),
            largest_gap={a: max(v) if v else None for a, v in gaps.items()},
            v2_minus_v1=dict(median=float(np.median(v2v1)) if v2v1 else None, min=min(v2v1, default=None),
                             max=max(v2v1, default=None)),
            v2_choices_in_usable_cells=dict(kept))
        if name in CROSS:
            usable = set(cells)
            for pa in RATES:
                for pb in RATES:
                    if pa == pb:
                        continue
                    g = []
                    for b in FRACTIONS:
                        key = (round(b, 6), round(pb, 8))
                        if key not in usable:
                            continue
                        best = max(mean(acc[(name, *key, a)]) for a in ARMS)
                        g.append(100 * (best - mean(cross[(name, round(b, 6), round(pa, 8), round(pb, 8))])))
                    cross_rows.append(dict(checkpoint=name, built_for=pa, tested_at=pb, budgets=len(g),
                                           mean_gap=float(np.mean(g)) if g else None))
    viol = lambda key: [{k: c[k] for k in ("checkpoint", "budget", "ber", "arm", "diff", "p_holm")}  # noqa: E731
                        for c in rules[key] if c["violation"]]
    fails_abs = sum(v["R2b_abs_fails"] for v in per_ckpt.values())
    fails_pos = sum(v["R2b_pos_fails"] for v in per_ckpt.values())
    ahead_ckpts = sum(v["R2c_ahead"] > 0 for v in per_ckpt.values())
    behind = sum(v["R2c_behind"] for v in per_ckpt.values())
    summary = dict(
        status=("INTERIM descriptive look, main arms only (no MC-corrected mask); not the confirmatory analysis"
                if interim else "Run R Stage 2, confirmatory analysis by the frozen plan (Section 5, addendum, R-A4, R-A5, A2)"),
        picks_sha256=picks_sha, script_sha256=sha256_file(Path(__file__)),
        R2a=dict(holds=not viol("R2a"), violations=viol("R2a")),
        R2b=dict(holds=fails_abs >= 7 and fails_pos >= 7, abs1024_fails_on=fails_abs, pos1024_fails_on=fails_pos,
                 of=len(NAMES)),
        R2c=None if interim else dict(kept=ahead_ckpts >= 3 and behind == 0, checkpoints_with_resolved_gain=ahead_ckpts,
                                      cells_resolved_behind=behind),
        R2a_prime=dict(holds=not viol("R2a_prime"), violations=viol("R2a_prime")),
        claim_rate_aware_v1=bool(not viol("R2a") and fails_abs >= 7 and fails_pos >= 7),
        per_checkpoint=per_ckpt)
    (outd / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    for fname, rows in (("cells.csv", allcells + rules["R2c"]), ("cross_rate.csv", cross_rows)):
        if not rows:
            continue
        with open(outd / fname, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_checkpoint"}, indent=1, default=float))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path, default=EVAL_OUT)
    ap.add_argument("--out", type=Path, default=OUTD)
    ap.add_argument("--picks", type=Path, default=PICKS)
    ap.add_argument("--picks-sha", default=PICKS_SHA)
    ap.add_argument("--interim-main", action="store_true", help="descriptive look at the main arms before the MC phase ends")
    a = ap.parse_args()
    main(a.eval_root, a.out, a.picks, a.picks_sha, a.interim_main)
