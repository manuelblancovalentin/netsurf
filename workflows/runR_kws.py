#!/usr/bin/env python3
"""Run R, Stage 4 preparation for the MLPerf Tiny keyword-spotting DS-CNN (INT8 TFLite, pinned checkpoint).

  verify [--count N]
      The gradient companion (MlperfKwsDscnnDifferentiableCompanion: TFLite constants, requantization with a
      straight-through gradient) against the TFLite reference interpreter on N test clips of the clean model:
      agreement of the INT8 output codes and of the predicted class.  Also checks that the engine's bit jumps are
      one scale times the two's-complement code jumps, as for ResNet-8.
  masks --rep R
      abs1024 and pos1024 at the eight Stage 1 budgets from 1,024 training clips (offset 1024 * R), Run Q's scoring
      code.
  predict --rep R
      The predicted objective J_p of both masks on 512 validation clips (0-511) over the Run PQ grid, as
      runR_predict.py does for ResNet-8.  No fault outcome is read.
  summarize --reps ...
      p* per budget, mean over replications, same definition as Stage 1 (max p 3e-2) -> predictions.json.

Output (write-once): <out>/verify.json, <out>/masks/rep<R>/, <out>/predict/rep<R>.json, <out>/predictions.json.
Default <out>: workspace/runs/paper1_runR_kws_prep_v1.  Interpreter: the TensorFlow 2.14.1 reference environment.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from runP_phase0 import PAPER, Model, code_jumps, sha256_file
from site_paths import DATA, LOCKS, WT
from runP_phase12 import pack, unpack
from runPQ_crossover_theory import PGRID
from runQ_calibration import scores
from runR_predict import crossover
from runR_stage1 import FRACTIONS, bkey
from netsurf.mean_variance import bit_second_moments, displacement_moments, mask_sha256, top_budget, variance_term

CHECKPOINT = DATA / "models/models--mlcommons--tiny/blobs/aeea436800704fce17b17292e4412630ad856e9d777c044c64ef748a880bd0ae"
DATASET = DATA / "datasets/speech_commands/0.0.3"
LOCK = LOCKS / "mlperf-tiny/kws.lock.json"
OUT = PAPER / "workspace/runs/paper1_runR_kws_prep_v1"
CAL_COUNT, HELD_COUNT, HELD_OFFSET = 1024, 512, 0


class KwsModel(Model):
    """Same interface as Run P's ResNet-8 `Model`, for the DS-CNN."""

    def __init__(self, threads: int):  # noqa: D107  (does not call Model.__init__)
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        from netsurf.adapters.mlperf_tiny_multibench import MlperfTinyKwsDscnnAdapter
        from netsurf.mlperf_multibench_companion import MlperfKwsDscnnDifferentiableCompanion
        self.tf = tf
        self.adapter = MlperfTinyKwsDscnnAdapter(CHECKPOINT, DATASET, asset_lock=LOCK, split="full_test_4890",
                                                 injectable_parameters=("kernels",))
        self.adapter.load()
        storage = self.adapter._require_loaded()
        values = storage.values()
        self.names = list(values)
        self.shapes = {n: values[n].shape for n in self.names}
        sizes = [values[n].size for n in self.names]
        self.starts = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self.codes = np.concatenate([np.asarray(values[n]).reshape(-1) for n in self.names]).astype(np.int8)
        jumps = self.adapter.bit_perturbations()
        self.jumps = np.concatenate([np.asarray(jumps[n]).reshape(-1, 8) for n in self.names]).astype(np.float64)
        cj = code_jumps(self.codes).astype(np.float64)
        if not np.array_equal(np.sign(self.jumps), np.sign(cj)):
            raise AssertionError("engine jump signs differ from two's-complement code jumps")
        ratio = self.jumps / cj
        self.per_channel_scales = not np.allclose(ratio, ratio[:, :1], rtol=1e-6)
        if self.per_channel_scales:
            raise AssertionError("engine jumps are not one scale times the code jumps within a word")
        self.comp = MlperfKwsDscnnDifferentiableCompanion(storage.reference_content)
        if set(self.comp.variables) != set(self.names):
            raise AssertionError(f"companion kernel names differ from adapter names: {sorted(self.comp.variables)} vs {self.names}")
        self.clean = {n: self.comp.variables[n].numpy().copy() for n in self.names}
        self.n = int(self.starts[-1])
        self.sha256 = storage.sha256

    def clips(self, role, count, offset):
        x, y, part = self.adapter.sample_partition(role, count, offset=offset)
        return np.asarray(x), np.asarray(y), tuple(int(i) for i in part.indices)


FKERAS_SEED = 970201
FKERAS_OPTIONS = dict(eigenpairs=2, max_iterations=100, tolerance=1e-5, max_products=300)
V14_BUNDLE = WT / ("paper1-runA-tflite-confirm/workspace/"
                   "runs/paper1_runA2_kws_confirm12_v1/20261111T163724.628443Z-dd9a9dab")
V14_POLICIES = {"abs64": "per_example_absolute_global", "msb": "msb_global", "fkeras": "fkeras_paper"}


def static_masks(m, x, y, fractions):
    """abs64, significance-first and FKeras masks with the engine's own ordering code (as in the v14 runs):
    abs64 and msb ordered by score then stable identity; FKeras by bit significance, score, stable identity.
    Count per budget: floor(f * bits), the engine's rule."""
    from netsurf.ordering import compose_ordering
    from netsurf.ranking import Ranking
    from netsurf.fkeras_estimator import fkeras_family
    from netsurf.protection import ProtectionPolicy
    ids = m.adapter.bit_identities()
    specs = m.adapter.parameter_specs()
    row = {n: int(m.starts[i]) for i, n in enumerate(m.names)}
    S = scores(m.per_example(x, y), m.jumps)["abs"]
    abs_scores = {b: float(S[row[b.parameter] + b.flat_index, b.bit_index]) for b in ids}
    msb_scores = {b: float(b.bit_index) for b in ids}
    t0 = time.time()
    fk = fkeras_family(m.adapter, x, y, ids, seed=FKERAS_SEED, options=FKERAS_OPTIONS, requested=["fkeras_paper"])
    fk_seconds = time.time() - t0
    specs_by_arm = {"abs64": (abs_scores, ("estimator_score_desc", "stable_identity")),
                    "msb": (msb_scores, ("estimator_score_desc", "stable_identity")),
                    "fkeras": (dict(fk["fkeras_paper"].scores), ("bit_significance_desc", "estimator_score_desc", "stable_identity"))}
    out = {}
    for arm, (sc, criteria) in specs_by_arm.items():
        ordering = compose_ordering(arm, Ranking(arm, sc, {}), ids, specs, criteria)
        for f in fractions:
            pm = ProtectionPolicy.ordered(ordering, f)
            mk = np.zeros((m.n, 8), dtype=bool)
            for b in pm.protected:
                mk[row[b.parameter] + b.flat_index, b.bit_index] = True
            out[f"{arm}@{bkey(f)}"] = mk
    return out, fk_seconds


def parity(out, threads):
    """Rebuild the v14 DS-CNN masks of Run A2 replication 1 (training clips 405-468) and compare bit for bit."""
    t0 = time.time()
    m = KwsModel(threads)
    x, y, idx = m.clips("ranking", 64, 405)
    assert idx[0] == 405
    fr = (0.0625, 0.125, 0.1875, 0.25)
    mine, fk_seconds = static_masks(m, x, y, fr)
    data = json.loads((V14_BUNDLE / "artifacts/protection_masks.json").read_text())["replication_1"]
    row = {n: int(m.starts[i]) for i, n in enumerate(m.names)}
    res = {}
    for arm, pol in V14_POLICIES.items():
        for f in fr:
            ref = np.zeros((m.n, 8), dtype=bool)
            for p, fi, bi in data[f"{pol}@{bkey(f)}"]:
                ref[row[p] + fi, bi] = True
            got = mine[f"{arm}@{bkey(f)}"]
            res[f"{arm}@{bkey(f)}"] = dict(identical=bool(np.array_equal(got, ref)), differing=int((got != ref).sum()),
                                           count=int(got.sum()), v14_count=int(ref.sum()))
    rec = dict(masks=res, all_identical=all(v["identical"] for v in res.values()), fkeras_seconds=round(fk_seconds),
               seconds=round(time.time() - t0), script_sha256=sha256_file(Path(__file__)))
    write_once(out / "parity_v14.json", rec)
    print(json.dumps(rec, indent=0), flush=True)


STAGE4_FRACTIONS = (0.125, 0.1875, 0.25, 0.375)
ENGINE_TRIAL = dict(replication="replication_1", policy="ber=0.003/msb_global@0.125000",
                    schedule_sha256="e448a8a29657348ebacd2279b4bc726bc5f7e5b47e9d063da759f57d4ff87212")


def static(out, rep, threads):
    """abs64 (first 64 clips of the draw), significance-first and FKeras at the four Stage 4 budgets."""
    t0 = time.time()
    m = KwsModel(threads)
    x, y, idx = m.clips("ranking", 64, CAL_COUNT * rep)
    mk, fk_seconds = static_masks(m, x, y, STAGE4_FRACTIONS)
    d = out / "static" / f"rep{rep}"
    d.mkdir(parents=True, exist_ok=False)
    np.savez(d / "static.npz", **{k: pack(v) for k, v in mk.items()})
    write_once(d / "static.json", dict(masks={k: dict(count=int(v.sum()), sha256=mask_sha256(v)) for k, v in mk.items()},
                                       calibration_first=idx[0], calibration_count=len(idx), fkeras_seconds=round(fk_seconds),
                                       seconds=round(time.time() - t0), script_sha256=sha256_file(Path(__file__))))
    print(json.dumps(dict(rep=rep, seconds=round(time.time() - t0))), flush=True)


def engine_ref(out):
    """Extract one saved v14 DS-CNN trial (schedule, expected accuracy, evaluation clips) for the engine check."""
    B = V14_BUNDLE
    sched = next(s for s in json.loads((B / "artifacts/fault_schedules.json").read_text())[ENGINE_TRIAL["replication"]]
                 if s["sha256"] == ENGINE_TRIAL["schedule_sha256"])
    import csv
    with open(B / "measurements/trials.csv") as fh:
        row = next(r for r in csv.DictReader(fh) if r["replication"] == ENGINE_TRIAL["replication"]
                   and r["policy"] == ENGINE_TRIAL["policy"])
    parts = json.loads((B / "artifacts/sample_partitions.json").read_text())[ENGINE_TRIAL["replication"]]
    write_once(out / "engine_ref.json", dict(**ENGINE_TRIAL, events=sched["events"], accuracy=float(row["accuracy"]),
                                             evaluation_indices=parts["evaluation"]["indices"],
                                             source=str(B)))
    print("engine reference:", row["accuracy"], len(sched["events"]), "events")


def write_once(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"{path} exists (write-once)")
    path.write_text(json.dumps(obj, indent=1, default=float))


def verify(out, count, threads):
    t0 = time.time()
    m = KwsModel(threads)
    x, y, idx = m.clips("evaluation", count, 0)
    ref_codes = np.asarray(m.adapter.predict_encoded_outputs(x))
    comp_codes = m.comp.encoded_outputs(x)
    ref_cls, comp_cls = ref_codes.argmax(1), comp_codes.argmax(1)
    loss, grad = m.loss_and_grad(x[:64], y[:64])
    rec = dict(clips=count, first_index=idx[0], weights=m.n, layers=len(m.names), names=m.names,
               output_codes_identical=float((ref_codes == comp_codes).all(axis=1).mean()),
               max_output_code_difference=int(np.abs(ref_codes.astype(int) - comp_codes.astype(int)).max()),
               class_agreement=float((ref_cls == comp_cls).mean()),
               reference_accuracy=float((ref_cls == y).mean()), companion_accuracy=float((comp_cls == y).mean()),
               companion_loss_64=loss, gradient_norm_64=float(np.linalg.norm(grad)),
               model_sha256=m.sha256, seconds=round(time.time() - t0), script_sha256=sha256_file(Path(__file__)))
    write_once(out / "verify.json", rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "names"}), flush=True)


def masks(out, rep, threads):
    t0 = time.time()
    m = KwsModel(threads)
    x, y, idx = m.clips("ranking", CAL_COUNT, CAL_COUNT * rep)
    t1 = time.time()
    S = scores(m.per_example(x, y), m.jumps)
    t_grad = time.time() - t1
    total = m.n * 8
    arrays, info = {}, {}
    for f in FRACTIONS:
        count = int(round(f * total))
        for a in ("abs", "pos"):
            k = f"{a}1024@{bkey(f)}"
            mk = top_budget(S[a], count)
            arrays[k] = pack(mk)
            info[k] = dict(count=count, sha256=mask_sha256(mk))
    d = out / "masks" / f"rep{rep}"
    d.mkdir(parents=True, exist_ok=False)
    np.savez(d / "masks.npz", **arrays)
    write_once(d / "masks.json", dict(masks=info, calibration_first=idx[0], calibration_count=len(idx),
                                      per_example_seconds=round(t_grad), seconds=round(time.time() - t0),
                                      script_sha256=sha256_file(Path(__file__))))
    print(json.dumps(dict(rep=rep, per_example_seconds=round(t_grad), seconds=round(time.time() - t0))), flush=True)


def predict(out, rep, threads):
    t0 = time.time()
    m = KwsModel(threads)
    xh, yh, ih = m.clips("empirical_validation", HELD_COUNT, HELD_OFFSET)
    G = m.per_example(xh, yh)
    m2 = bit_second_moments(m.jumps, (G * G).mean(0))
    del G
    m.set_displacement(np.zeros(m.n))
    clean = m.loss_acc(xh, yh)[0]
    packed = np.load(out / "masks" / f"rep{rep}" / "masks.npz")
    rows = []
    for a in ("abs1024", "pos1024"):
        for f in FRACTIONS:
            mk = unpack(packed[f"{a}@{bkey(f)}"], m.n)
            for p in PGRID:
                mu, _ = displacement_moments(m.jumps, mk, float(p))
                m.set_displacement(mu)
                loss, _ = m.loss_acc(xh, yh)
                rows.append(dict(arm=a, budget=f, p=float(p), J=loss - clean + variance_term(m2, mk, float(p))))
    m.set_displacement(np.zeros(m.n))
    write_once(out / "predict" / f"rep{rep}.json", dict(rep=rep, heldout_first=ih[0], heldout_count=len(ih),
                                                        clean_loss=clean, rows=rows, seconds=round(time.time() - t0),
                                                        script_sha256=sha256_file(Path(__file__))))
    print(json.dumps(dict(rep=rep, seconds=round(time.time() - t0))), flush=True)


def summarize(out, reps, name="predictions.json"):
    pg = np.array(PGRID, dtype=float)
    J = defaultdict(list)
    for rep in reps:
        for r in json.loads((out / "predict" / f"rep{rep}.json").read_text())["rows"]:
            J[(r["arm"], r["budget"], r["p"])].append(r["J"])
    pred = {}
    for f in FRACTIONS:
        D = np.array([np.mean(J[("pos1024", f, float(p))]) - np.mean(J[("abs1024", f, float(p))]) for p in pg])
        ps, how = crossover(pg, D)
        pred[bkey(f)] = dict(p_star=ps, how=how, curve=[float(v) for v in D])
    write_once(out / name, dict(status="DS-CNN predicted switch; no fault outcome read", reps=list(reps),
                                              pgrid=list(map(float, pg)), predictions=pred,
                                              script_sha256=sha256_file(Path(__file__))))
    for b, v in pred.items():
        print(b, f"{v['p_star']:.2e}", v["how"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--threads", type=int, default=1)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify"); v.add_argument("--count", type=int, default=1000)
    sub.add_parser("parity")
    sub.add_parser("engine_ref")
    st = sub.add_parser("static"); st.add_argument("--rep", type=int, required=True)
    for c in ("masks", "predict"):
        s = sub.add_parser(c); s.add_argument("--rep", type=int, required=True)
    s = sub.add_parser("summarize"); s.add_argument("--reps", type=int, nargs="+", required=True)
    s.add_argument("--name", default="predictions.json")
    a = ap.parse_args()
    if a.cmd == "verify":
        verify(a.out, a.count, a.threads)
    elif a.cmd == "parity":
        parity(a.out, a.threads)
    elif a.cmd == "engine_ref":
        engine_ref(a.out)
    elif a.cmd == "static":
        static(a.out, a.rep, a.threads)
    elif a.cmd == "masks":
        masks(a.out, a.rep, a.threads)
    elif a.cmd == "predict":
        predict(a.out, a.rep, a.threads)
    else:
        summarize(a.out, a.reps, a.name)
