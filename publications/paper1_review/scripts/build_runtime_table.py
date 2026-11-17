#!/usr/bin/env python3
"""Run times of the paper's studies, from the completion records the jobs wrote (no new measurement).

Every job of the main comparison, the fault-rate study and the model check wrote a DONE record with its wall-clock
seconds and the number of evaluations (one evaluation = one condition on one fault field over the whole evaluation
cohort) or network passes.  The jobs ran concurrently on a shared CPU server, so the times are upper bounds on an idle
machine.  Writes data/tecs_runtimes.json (with a hash of every record read) and prints the rows of the appendix table.

    build_runtime_table.py [--runs DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = HERE.parents[2] / "workspace" / "runs"

GROUPS = (("Jet MLP", "jet-"), ("MNIST MLP", "mnist-"), ("VWW MobileNet", "vww"), ("ResNet-8", "resnet8-"), ("ResNet-8", "pinned"),
          ("ResNet-8", "pruned_"), ("ResNet-8", "retrained_"), ("ResNet-8", "undertrained_"), ("DS-CNN", "dscnn"),
          ("ToyCar", "toycar"), ("DistilBERT", "distilbert"), ("Pythia-160M", "pythia"), ("Qwen3-0.6B", "qwen3"))
ORDER = ("Jet MLP", "MNIST MLP", "VWW MobileNet", "ResNet-8", "DS-CNN", "ToyCar", "DistilBERT", "Pythia-160M", "Qwen3-0.6B")


def group(name: str) -> str:
    for g, prefix in GROUPS:
        if name.startswith(prefix):
            return g
    raise SystemExit(f"no model group for {name}")


def collect(dirs, pattern, units_key, sha, fixed=None):
    """{model group: [(seconds, units)]} over every record matching ``pattern`` in ``dirs``."""
    out = defaultdict(list)
    for d in dirs:
        for f in sorted((RUNS / d).glob(pattern)):
            raw = f.read_bytes()
            sha.update(raw)
            rec = json.loads(raw)
            top = f.relative_to(RUNS / d).parts[0]
            top = f.relative_to(RUNS / d).parts[1] if top == "blur" else top
            out[fixed or group(top)].append((float(rec["seconds"]), float(rec[units_key])))
    return out


def summarize(by):
    rows = {}
    for g in ORDER:
        if g not in by:
            continue
        v = np.asarray(by[g])
        rows[g] = dict(jobs=int(len(v)), units_per_job=float(np.median(v[:, 1])), median_job_seconds=float(np.median(v[:, 0])),
                       seconds_per_unit=float(v[:, 0].sum() / v[:, 1].sum()), total_hours=float(v[:, 0].sum() / 3600))
    return rows


def fmt_s(x):
    return f"{x:.2f}" if x < 1 else (f"{x:.1f}" if x < 10 else f"{x:,.0f}".replace(",", "{,}"))


def fmt_h(x):
    return f"{x:.2f}" if x < 1 else (f"{x:.1f}" if x < 100 else f"{x:,.0f}".replace(",", "{,}"))


def main():
    global RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=RUNS)
    RUNS = ap.parse_args().runs
    sha = hashlib.sha256()
    studies = {
        "main_comparison": summarize(_merge(
            collect(["paper1_runT_tflite_v1"], "*/rep*/draw*/DONE*", "evaluations", sha),
            collect(["paper1_runT_w8a8_v1", "paper1_runT_w8a8_more_v1"], "*/draw*/ber*/DONE*", "evaluations", sha))),
        "fault_rate_study": summarize(_merge(
            collect(["paper1_runR_stage1_crossover_v1"], "*/rep*/field*/DONE", "evaluations", sha),
            _merge(collect(["paper1_gradtrust_faults_v1"], "*/rep*/field*/DONE", "evaluations", sha),
                   collect(["paper1_runR_stage4_kws_eval_v1"], "rep*/field*/DONE", "evaluations", sha, fixed="DS-CNN")))),
        "model_check": summarize(collect(["paper1_blurconf_v1"], "blur/*/DONE", "forwards", sha)),
    }
    fk = defaultdict(list)
    for f in sorted((RUNS / "paper1_runT_tflite_v1").glob("*/rep*/fkeras.json")) + sorted((RUNS / "paper1_runT_w8a8_v1").glob("*/fkeras.json")) \
            + sorted((RUNS / "paper1_runT_w8a8_v1").glob("*/fkeras_quest/fkeras.json")):
        raw = f.read_bytes(); sha.update(raw)
        rec = json.loads(raw)
        fk[group(f.relative_to(f.parents[3] if f.parent.name.startswith("rep") else RUNS / "paper1_runT_w8a8_v1").parts[1 if f.parent.name.startswith("rep") else 0])].append(
            (float(rec["seconds"]), rec.get("device", "cpu")))
    fk = {g: dict(builds=len(fk[g]), median_seconds=float(np.median([x[0] for x in fk[g]])), device=sorted({x[1] for x in fk[g]}))
          for g in ORDER if g in fk}
    out = dict(studies=studies, fkeras_score_builds=fk, records_sha256=sha.hexdigest(),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               note="wall-clock seconds of concurrent jobs on a shared CPU server; unit = evaluation (main comparison, "
                    "fault-rate study) or network pass (model check)")
    (ROOT / "data" / "tecs_runtimes.json").write_text(json.dumps(out, indent=1) + "\n")
    for study, rows in studies.items():
        print(f"% {study}")
        for g, r in rows.items():
            jobs = f"{r['jobs']:,}".replace(",", "{,}")
            print(f"{g} & {jobs} & {r['units_per_job']:.0f} & {fmt_s(r['median_job_seconds'])} & "
                  f"{fmt_s(r['seconds_per_unit'])} & {fmt_h(r['total_hours'])}\\\\")
    print("% fkeras score builds:", json.dumps(fk))
    print("% total hours:", {s: round(sum(r["total_hours"] for r in rows.values()), 1) for s, rows in studies.items()})


def _merge(a, b):
    for k, v in b.items():
        a[k].extend(v)
    return a


if __name__ == "__main__":
    main()
