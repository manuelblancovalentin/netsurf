#!/usr/bin/env python3
"""DS-CNN calibration-size test (pre-registration review/KWS_CALIBRATION_SIZE_PREREG_2026-11-15.md, 1be5878).

  build --rep R       CPU.  Calibration sets of 4,096 and 8,192 training clips for replication R: the Stage 4 set
                      (clips 1024 R .. 1024 R + 1023) plus the first 3,072 / all 7,168 clips of the block starting at
                      6144 + 7168 R.  Per-example gradients from the graph companion; pos and abs scores by the Stage 4
                      formula (runQ_calibration.scores); masks at 12.5% and 25%.  <out>/rep<R>/{masks.npz, stats.npz,
                      build.json}.
  evaluate --rep R --field F   The Stage 4 protocol (engine check, TFLite reference kernels, 800-clip window R, field
                      seed [20261119, R, F]) for pos/abs at 4,096 and 8,192 on rates 3e-4 .. 3e-3.
                      <out>/rep<R>/field<F>/results.csv.
  analyze             Primary pos-8192 minus abs-8192; secondaries; D1 per size.  Pure NumPy, run once.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SIZES = (4096, 8192)
FRACTIONS = (0.125, 0.25)
RATES = (3e-4, 1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3)
ARMS = [f"{k}{n}" for n in SIZES for k in ("pos", "abs")]
BLOCK0, BLOCK = 6144, 7168


def log(**kw):
    print(json.dumps(dict(t=time.strftime("%H:%M:%S"), **kw)), flush=True)


def cmd_build(a):
    from netsurf.mean_variance import top_budget
    from runP_phase12 import pack
    from runQ_calibration import scores
    from runR_kws import CAL_COUNT, KwsModel
    from runR_stage1 import bkey
    from runR_stage4_kws import calibration, load_static
    from runG_gradtrust import GenModel
    t0 = time.time()
    k = KwsModel(a.threads)
    m = GenModel(k.adapter._require_loaded().reference_content, a.threads)
    out = a.out / f"rep{a.rep}"
    out.mkdir(parents=True, exist_ok=False)
    x0, y0 = calibration(k, a.rep)
    xb, yb, ib = k.clips("ranking", BLOCK, BLOCK0 + BLOCK * a.rep)
    assert ib[0] == BLOCK0 + BLOCK * a.rep and len(ib) == BLOCK
    x0, y0, xb, yb = np.asarray(x0), np.asarray(y0), np.asarray(xb), np.asarray(yb)
    masks, stats, info = {}, {}, dict(rep=a.rep, stage4_clips=[CAL_COUNT * a.rep, CAL_COUNT * a.rep + CAL_COUNT - 1],
                                      block=[int(ib[0]), int(ib[-1])], sizes={})
    for n in SIZES:
        extra = n - CAL_COUNT
        x, y = np.concatenate([x0, xb[:extra]]), np.concatenate([y0, yb[:extra]])
        st = m.gradient_stats(x, y, batch=a.grad_batch)
        # Stage 4 formula on the streamed moments: abs = mean|g| |J|; pos = mean(g+) J (J > 0), mean(g-) (-J) (J < 0)
        S = dict(abs=st["gabs"][:, None] * np.abs(m.jumps),
                 pos=np.where(m.jumps > 0, st["gpos"][:, None] * m.jumps, st["gneg"][:, None] * (-m.jumps)))
        stats[f"g@{n}"], stats[f"se@{n}"] = st["g"], st["se"]
        info["sizes"][str(n)] = {}
        for f in FRACTIONS:
            c = int(np.floor(f * m.n * 8))
            stored = load_static(k, a.rep, f)
            for kind in ("pos", "abs"):
                mk = top_budget(S[kind], c)
                masks[f"{kind}{n}@{bkey(f)}"] = pack(mk)
                info["sizes"][str(n)][f"{kind}@{f:g}"] = dict(bits=int(mk.sum()),
                                                              differing_from_stored=int((mk & ~stored[f"{kind}1024"]).sum()))
        log(stage="built", rep=a.rep, size=n, done=SIZES.index(n) + 1, total=len(SIZES), seconds=round(time.time() - t0))
    # the scoring formula above must equal runQ_calibration.scores on the Stage 4 set (checked on its 1,024 clips)
    st0 = m.gradient_stats(x0, y0, batch=a.grad_batch)
    G = m.comp.per_example_gradients(x0[:64], y0[:64])
    G = np.concatenate([np.asarray(G[nm]).reshape(64, -1) for nm in m.names], axis=1).astype(np.float64)
    ref = scores(G, m.jumps)
    s64 = m.gradient_stats(x0[:64], y0[:64], batch=64)
    info["formula_check_max_rel"] = dict(
        abs=float(np.max(np.abs(s64["gabs"][:, None] * np.abs(m.jumps) - ref["abs"])) / max(float(np.abs(ref["abs"]).max()), 1e-30)),
        pos=float(np.max(np.abs(np.where(m.jumps > 0, s64["gpos"][:, None] * m.jumps, s64["gneg"][:, None] * (-m.jumps)) - ref["pos"]))
                  / max(float(np.abs(ref["pos"]).max()), 1e-30)))
    stats["g@1024"], stats["se@1024"] = st0["g"], st0["se"]
    np.savez(out / "masks.npz", **masks)
    np.savez(out / "stats.npz", **stats)
    info["seconds"] = round(time.time() - t0)
    (out / "build.json").write_text(json.dumps(info, indent=1))


def cmd_evaluate(a):
    from runP_phase12 import Replayer, unpack
    from runR_kws import OUT as PREP, KwsModel
    from runR_stage1 import bkey
    from runR_stage4_kws import EVAL_SEED, WINDOW, load_static
    m = KwsModel(a.threads)
    t0 = time.time()
    z = np.load(a.masks / f"rep{a.rep}" / "masks.npz")
    out = a.out / f"rep{a.rep}" / f"field{a.field}"
    out.mkdir(parents=True, exist_ok=False)
    rp = Replayer(m)
    ref = json.loads((PREP / "engine_ref.json").read_text())
    xr, yr, ir = m.clips("evaluation", len(ref["evaluation_indices"]), ref["evaluation_indices"][0])
    assert list(ir) == list(ref["evaluation_indices"])
    acc, _, _, _ = rp.evaluate(rp.flips(ref["events"]), load_static(m, 0, 0.125)["msb"], xr, yr)
    check = dict(replay_accuracy=acc, saved_accuracy=ref["accuracy"], match=acc == ref["accuracy"])
    (out / "check.json").write_text(json.dumps(check))
    if not check["match"]:
        raise SystemExit(f"engine check failed: {check}")
    xe, ye, ie = m.clips("evaluation", WINDOW, WINDOW * a.rep)
    assert ie[0] == WINDOW * a.rep and len(ie) == WINDOW
    u = np.random.default_rng([EVAL_SEED, a.rep, a.field]).random((m.n, 8))
    jobs = [(p, f, arm) for p in RATES for f in FRACTIONS for arm in ARMS]
    rows = []
    for i, (p, f, arm) in enumerate(jobs):
        acc, ce, prop, app = rp.evaluate(u < p, unpack(z[f"{arm}@{bkey(f)}"], m.n), xe, ye)
        rows.append(dict(rep=a.rep, field=a.field, ber=p, budget=f, arm=arm, accuracy=acc, cross_entropy=ce,
                         proposed=prop, applied=app))
        el = time.time() - t0
        log(done=i + 1, total=len(jobs), seconds=round(el), eta_minutes=round(el / (i + 1) * (len(jobs) - i - 1) / 60, 1))
    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "DONE").write_text(json.dumps(dict(seconds=round(time.time() - t0), evaluations=len(rows))))


def compare(acc, reps, a_arm, b_arm, clean, window=800, margin=0.05):
    from runG_gradtrust_analyze import ALPHA, MIN_GAIN, holm, signflip_p
    cells = []
    for f in FRACTIONS:
        for p in RATES:
            A = {(r, k): acc[(r, a_arm, f, p, k)] for r in reps for k in range(8)}
            B = {(r, k): acc[(r, b_arm, f, p, k)] for r in reps for k in range(8)}
            if max(np.mean(list(A.values())), np.mean(list(B.values()))) < clean - margin:
                continue
            keys = sorted(A)
            n = np.rint(np.array([A[q] - B[q] for q in keys]) * window)
            cells.append(dict(budget=f, ber=p, diff=float(100 * n.mean() / window), p_raw=signflip_p(n)))
    for c, h in zip(cells, holm([c["p_raw"] for c in cells]) if cells else []):
        c.update(p_holm=float(h), ahead=bool(h < ALPHA and c["diff"] >= MIN_GAIN), behind=bool(h < ALPHA and c["diff"] <= -MIN_GAIN))
    return cells


def cmd_analyze(a):
    from runR_partial_search import RUNS
    reps = range(6)
    acc = {}
    for r in reps:
        for k in range(8):
            for root in (RUNS / "paper1_runR_stage4_kws_eval_v1", a.out):
                for row in csv.DictReader(open(root / f"rep{r}/field{k}/results.csv")):
                    acc[(r, row["arm"], float(row["budget"]), float(row["ber"]), k)] = float(row["accuracy"])
    # clean accuracy as the Stage 4 analysis defines it: mean over replications of each window's clean accuracy
    clean = float(np.mean([np.mean([json.loads((RUNS / f"paper1_runR_stage4_kws_eval_v1/rep{r}/field{k}/field.json")
                                               .read_text())["clean_accuracy"] for k in range(8)]) for r in reps]))
    res = dict(status="pre-registered analysis (review/KWS_CALIBRATION_SIZE_PREREG_2026-11-15.md)", clean=clean)
    res["primary_pos8192_minus_abs8192"] = compare(acc, reps, "pos8192", "abs8192", clean)
    res["signed_ranking_helps"] = any(c["ahead"] for c in res["primary_pos8192_minus_abs8192"])
    for name, x, y in (("pos4096_minus_abs4096", "pos4096", "abs4096"), ("pos8192_minus_pos1024", "pos8192", "pos1024"),
                       ("abs8192_minus_abs1024", "abs8192", "abs1024")):
        res[name] = compare(acc, reps, x, y, clean)
    d1 = {}
    for n in (1024,) + SIZES:
        G = np.stack([np.load(a.masks / f"rep{r}" / "stats.npz")[f"g@{n}"] for r in reps])
        d1[str(n)] = float(np.median(np.corrcoef(G)[np.triu_indices(len(reps), 1)]))
    res["D1"] = d1
    res["build"] = {r: json.loads((a.masks / f"rep{r}" / "build.json").read_text()) for r in reps}
    print(json.dumps(dict(D1=d1, signed_ranking_helps=res["signed_ranking_helps"]), indent=1))
    for k in ("primary_pos8192_minus_abs8192", "pos4096_minus_abs4096", "pos8192_minus_pos1024", "abs8192_minus_abs1024"):
        print(k, " ".join(f"{c['budget']:g}/{c['ber']:g}:{c['diff']:+.2f}{'*' if c['ahead'] or c['behind'] else ''}" for c in res[k]))
    a.analysis.mkdir(parents=True, exist_ok=False)
    (a.analysis / "summary.json").write_text(json.dumps(res, indent=1))


def main():
    from runR_partial_search import RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("build", "evaluate", "analyze"))
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--field", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--grad-batch", type=int, default=256)
    ap.add_argument("--masks", type=Path, default=RUNS / "paper1_kws_calsize_masks_v1")
    ap.add_argument("--out", type=Path, default=RUNS / "paper1_kws_calsize_eval_v1")
    ap.add_argument("--analysis", type=Path, default=RUNS / "paper1_kws_calsize_analysis_v1")
    a = ap.parse_args()
    if a.command == "build":
        if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
            raise SystemExit('run on CPU: CUDA_VISIBLE_DEVICES=""')
        a.out = a.masks
    {"build": cmd_build, "evaluate": cmd_evaluate, "analyze": cmd_analyze}[a.command](a)


if __name__ == "__main__":
    main()
