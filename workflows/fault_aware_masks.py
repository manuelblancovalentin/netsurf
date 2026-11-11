#!/usr/bin/env python3
"""Run F (EXPLORATORY): fault-rate-aware protection masks for DistilBERT W8A8, tested against the best
fixed ranking on the saved fault fields of the finalized wide-BER study.

Question.  Does a mask rebuilt for a specific fault rate beat the per-example absolute-gradient mask, which
is built once on the fault-free model?  All choices below were fixed before any candidate was evaluated.

Candidates, for a target bit-error probability p and budget B (start: sealed absolute-gradient mask):
  repeat T = 2 times:
    draw K = 4 new fault fields at p (seeds never used by the study), keep only faults on unprotected bits
    of the current mask, and on each faulted model compute the same decoded-weight FP32 gradients as the
    study's ranking code on the same 64 calibration sentences; average over sentences and fields:
        g = mean gradient,  a = mean absolute gradient,  f = mean squared gradient   (per stored weight)
    re-rank all stored bits with their exact XOR jump d (decoded units, fault-free codes):
        C1 "conditional absolute gradient"   a * |d|
        C2 "two-moment benefit"              p * d * g + 0.5 * p * (1 - p) * d**2 * f
    and protect the top B bits.
  C1 and C2 follow separate trajectories.
Targets: p in {1e-2, 3e-2, 1e-1}, B in {12.5%, 25%}: 6 cells, 12 masks.

Evaluation: each mask on every saved field (24 per BER) at p in {3e-3, 1e-2, 3e-2, 1e-1} with the exact
integer executable and the frozen 512-sentence cohort, paired with the finalized outcomes of the six fixed
rankings on the same fields.  Primary contrast: candidate built for (p, B), evaluated at (p, B), minus the
fixed absolute-gradient mask at B; 12 contrasts, Holm.  A gain "matters" if it is at least one accuracy
point with a 95% interval excluding zero.

Gates.  build: gradients on the fault-free model must equal the saved selection moments, and re-ranking with
them must reproduce the sealed absolute-gradient mask.  eval: one finalized observation is replayed per shard
and must match bit for bit before any candidate is evaluated.

Read-only inputs: frozen snapshot, sealed masks and moments, saved proposed fields, finalized observations.
The 64 calibration sentences are the study's selection cohort; no evaluation sentence enters `build`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

W = Path("/projects/netsurf/manuelbv/workspace/implementation-worktrees")
CODE_ROOT = W / "w8a8-ber-stress"
MANIFEST = Path("/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json")
SEL = W / "transformer-integration/workspace/runs/distilbert_sst2_w8a8_reliability_v2_run1/selection"
SOURCE_RUN = W / "w8a8-ber-stress/workspace/runs/w8a8_wide_ber_matched_v1/distilbert_sst2_int8"
TARGET_BERS = (1e-2, 3e-2, 1e-1)
FRACTIONS = (0.125, 0.25)
TEST_BERS = (3e-3, 1e-2, 3e-2, 1e-1)
T_ITER, K_FIELDS, SEED = 2, 4, 20261112
CANDIDATES = ("c1_conditional_absolute", "c2_two_moment")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_once(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.flush(); os.fsync(f.fileno())


def imports():
    sys.path.insert(0, str(CODE_ROOT))
    from netsurf import w8a8_wide_ber_campaign as wide
    from netsurf.w8a8_campaign import _evaluation, _risk_record
    from netsurf.w8a8_dense_protection import Protection, injected_packed, prepared_protection_bytes
    from netsurf.w8a8_faults import W8A8FaultAdapter
    from netsurf.w8a8_rank import PackedMask
    return wide, _evaluation, _risk_record, Protection, injected_packed, prepared_protection_bytes, W8A8FaultAdapter, PackedMask


def top_bits(score: np.ndarray, budget: int) -> np.ndarray:
    """Boolean selection of exactly `budget` largest scores; ties at the threshold go to the lowest ordinals."""
    n = score.size
    thr = np.partition(score, n - budget)[n - budget]
    sel = score > thr
    need = budget - int(sel.sum())
    if need:
        ties = np.flatnonzero(score == thr)[:need]
        sel[ties] = True
    assert int(sel.sum()) == budget
    return sel


def cell_dir(out: Path, p: float, frac: float) -> Path:
    return out / "build" / f"p{p:g}_f{frac:g}"


def build(out: Path, p: float, frac: float) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    wide, _, _, _, _, _, Adapter, _ = imports()
    plan, asset = wide.validate(MANIFEST, SOURCE_RUN, verify_masks=False)
    dest = cell_dir(out, p, frac)
    dest.mkdir(parents=True, exist_ok=False)
    adapter = Adapter(asset); adapter.load()
    bits = plan["eligible_bits"]; budget = int(round(frac * bits))
    entries = adapter.universe.entries
    qmods = [adapter.modules[e["module"]] for e in entries]
    names = [e["module"] for e in entries]
    clean = [q.qweight.reshape(-1).numpy().copy() for q in qmods]
    sizes = [c.size for c in clean]; starts = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    assert int(starts[-1]) * 8 == bits
    # exact XOR jumps in decoded units, fault-free codes: (codes, 8) float32 per module
    jumps = []
    for q, c in zip(qmods, clean):
        u = c.view(np.uint8); scale = np.repeat(q.scale.numpy().astype(np.float32), q.qweight.shape[1])
        j = np.empty((c.size, 8), dtype=np.float32)
        for bit in range(8):
            j[:, bit] = (np.bitwise_xor(u, np.uint8(1 << bit)).view(np.int8).astype(np.int16) - c.astype(np.int16)) * scale
        jumps.append(j)

    # the study's surrogate: FP32 Linear layers holding decoded stored codes; same sentences, same loss
    rows = adapter.sst2_cohorts()["calibration"][:64]
    examples = [(adapter.tokenizer([r.text], return_tensors="pt", padding="max_length", truncation=True, max_length=128), r.label) for r in rows]
    layers = []
    for name, q in zip(names, qmods):
        layer = nn.Linear(q.in_features, q.out_features, bias=q.bias is not None)
        with torch.no_grad():
            if q.bias is not None:
                layer.bias.copy_(q.bias)
        parent, _, leaf = name.rpartition(".")
        setattr(adapter.model.get_submodule(parent) if parent else adapter.model, leaf, layer)
        layers.append(layer)
    for prm in adapter.model.parameters():
        prm.requires_grad_(False)
    for layer in layers:
        layer.weight.requires_grad_(True)
    adapter.model.eval()

    def moments():
        """mean g, mean |g|, mean g^2 over the 64 sentences at the codes currently stored in the modules."""
        with torch.no_grad():
            for layer, q in zip(layers, qmods):
                layer.weight.copy_(q.qweight.float() * q.scale[:, None])
        acc = [[torch.zeros_like(l.weight) for _ in range(3)] for l in layers]
        for encoded, label in examples:
            loss = F.cross_entropy(adapter.model(**encoded).logits, torch.tensor([label], dtype=torch.long))
            assert bool(torch.isfinite(loss))
            loss.backward()
            for layer, (sg, sa, s2) in zip(layers, acc):
                g = layer.weight.grad
                sg += g; sa += g.abs(); s2 += g * g
                layer.weight.grad = None
        n = len(examples)
        return [[(x / n).reshape(-1).numpy() for x in triple] for triple in acc]

    def unpack(path: Path) -> list[np.ndarray]:
        raw = np.fromfile(path, dtype=np.uint8)
        assert raw.size * 8 == bits
        return [np.unpackbits(raw[s:e, None], axis=1, bitorder="little").astype(bool) for s, e in zip(starts[:-1], starts[1:])]

    def rank(kind: str, mom) -> np.ndarray:
        score = np.empty(bits, dtype=np.float32)
        for i, (j, (g, a, f)) in enumerate(zip(jumps, mom)):
            if kind == "abs":
                v = a[:, None] * np.abs(j)
            else:
                v = p * j * g[:, None] + 0.5 * p * (1.0 - p) * j * j * f[:, None]
            score[8 * starts[i]: 8 * starts[i + 1]] = v.reshape(-1)
        return top_bits(score, budget)

    started = time.time(); gates = {}
    # Gate 1: fault-free gradients equal the saved selection moments
    mom0 = moments()
    saved = json.loads((SEL / "moments/manifest.json").read_text())["tensors"]
    worst = 0.0
    for name, (g, a, _) in zip(names, mom0):
        sg = np.fromfile(SEL / "moments" / saved[name]["mean_g"], dtype="<f4"); sa = np.fromfile(SEL / "moments" / saved[name]["mean_abs_g"], dtype="<f4")
        worst = max(worst, float(np.max(np.abs(g - sg)) / (np.max(np.abs(sg)) + 1e-30)), float(np.max(np.abs(a - sa)) / (np.max(np.abs(sa)) + 1e-30)))
    gates["max_relative_difference_to_saved_moments"] = worst
    assert worst < 1e-4, f"fault-free gradients differ from the saved selection moments ({worst})"
    # Gate 2: re-ranking reproduces the sealed absolute-gradient mask
    sealed_path = SEL / "masks/per_example_absolute" / f"per_example_absolute_b{budget}.bits"
    sealed = np.concatenate([m.reshape(-1) for m in unpack(sealed_path)])
    mine = rank("abs", mom0)
    gates["bits_differing_from_sealed_absolute_mask"] = int(np.count_nonzero(mine != sealed))
    assert gates["bits_differing_from_sealed_absolute_mask"] <= 2e-6 * bits, gates
    del mine

    rng_base = SEED + 1000 * TARGET_BERS.index(p) + 100 * FRACTIONS.index(frac)
    result = {}
    for c_index, cand in enumerate(CANDIDATES):
        current = sealed.copy(); trace = []
        for t in range(T_ITER):
            sums = None
            for k in range(K_FIELDS):
                rng = np.random.default_rng(rng_base + 10 * t + k)          # same fields for both candidates
                applied = 0; xors = []
                for i, (q, c) in enumerate(zip(qmods, clean)):
                    prot = current[8 * starts[i]: 8 * starts[i + 1]].reshape(-1, 8)
                    xb = np.empty(c.size, dtype=np.uint8)
                    for s in range(0, c.size, 1 << 21):
                        e = min(s + (1 << 21), c.size)
                        flips = (rng.random((e - s, 8)) < p) & ~prot[s:e]
                        applied += int(flips.sum())
                        xb[s:e] = np.packbits(flips, axis=1, bitorder="little")[:, 0]
                    q.qweight.reshape(-1).numpy().view(np.uint8)[:] ^= xb
                    xors.append(xb)
                m = moments()
                for q, xb, c in zip(qmods, xors, clean):
                    view = q.qweight.reshape(-1).numpy()
                    view.view(np.uint8)[:] ^= xb
                    assert np.array_equal(view, c), "stored codes were not restored"
                sums = m if sums is None else [[x + y for x, y in zip(a3, b3)] for a3, b3 in zip(sums, m)]
                print(json.dumps(dict(p=p, fraction=frac, candidate=cand, iteration=t + 1, field=k, applied_faults=applied,
                                      seconds=round(time.time() - started))), flush=True)
            mom = [[x / K_FIELDS for x in triple] for triple in sums]
            new = rank("abs" if c_index == 0 else "two_moment", mom)
            trace.append(dict(iteration=t + 1, bits_changed_from_previous=int(np.count_nonzero(new != current)) // 2,
                              bits_changed_from_sealed_absolute=int(np.count_nonzero(new != sealed)) // 2))
            current = new
        path = dest / f"{cand}.bits"
        with open(path, "xb") as f:
            f.write(np.packbits(current.reshape(-1, 8), axis=1, bitorder="little").tobytes())
        per_bit = current.reshape(-1, 8).mean(axis=0)
        result[cand] = dict(path=str(path), sha256=sha256(path), budget=budget, trace=trace,
                            protected_fraction_by_bit_position=[float(x) for x in per_bit])
    sealed_bit = sealed.reshape(-1, 8).mean(axis=0)
    write_once(dest / "manifest.json", dict(schema="paper1-runF-fault-aware-build/1.0", status="EXPLORATORY", target_ber=p, fraction=frac,
               budget=budget, iterations=T_ITER, fields_per_iteration=K_FIELDS, seed_base=rng_base, gates=gates, masks=result,
               sealed_absolute_mask=dict(path=str(sealed_path), sha256=sha256(sealed_path), protected_fraction_by_bit_position=[float(x) for x in sealed_bit]),
               calibration_source_indices=[r.source_index for r in rows], script_sha256=sha256(Path(__file__)), seconds=time.time() - started))
    print("done", flush=True)


def evaluate(out: Path, shard: int, shards: int) -> None:
    wide, _evaluation, _risk_record, Protection, injected_packed, prepared_protection_bytes, Adapter, PackedMask = imports()
    plan, asset = wide.validate(MANIFEST, SOURCE_RUN, verify_masks=False)
    bits = plan["eligible_bits"]
    adapter = Adapter(asset); adapter.load()
    run = _evaluation(adapter, {"target": plan["target"], "final": plan["final_cohort"]})
    assert _risk_record(run()) == json.loads((SOURCE_RUN / "clean/shard_00.json").read_text()), "clean record differs"
    masks = []
    for p in TARGET_BERS:
        for frac in FRACTIONS:
            man = json.loads((cell_dir(out, p, frac) / "manifest.json").read_text())
            for cand in CANDIDATES:
                item = man["masks"][cand]
                digest = PackedMask(Path(item["path"]), bits, item["budget"], item["sha256"]).digest
                prot = Protection("packed", bits, item["budget"], digest, Path(item["path"]), item["sha256"])
                masks.append((f"{cand}_p{p:g}_f{frac:g}", p, frac, cand, prot, prepared_protection_bytes(prot)))
    arms = [plan["bers"].index(b) for b in TEST_BERS]
    fields = list(range(shard, plan["fields_per_ber"] if isinstance(plan["fields_per_ber"], int) else len(plan["field_seeds"][0]), shards))
    # gate: replay one finalized observation bit for bit
    ref = next(r for r in plan["policies"] if r["name"].startswith("per_example_absolute_b") and r["budget"] == int(round(0.125 * bits)))
    ref_prot = wide._protection(ref, bits); src = SOURCE_RUN / "fields" / f"arm_{arms[1]:02d}" / f"field_{fields[0]:03d}"
    meta = json.loads((src / "field.json").read_text()); proposed = np.memmap(src / "proposed.u8", dtype=np.uint8, mode="r")
    with injected_packed(adapter, proposed, meta, ref_prot, prepared_protection_bytes(ref_prot)) as events:
        risk = run()
    saved = json.loads((src / f"{ref['name']}.json").read_text())
    assert _risk_record(risk) == saved["risk"] and dict(events) == saved["events"], "replay gate failed; nothing evaluated"
    del proposed
    for index in fields:
        for arm in arms:
            src = SOURCE_RUN / "fields" / f"arm_{arm:02d}" / f"field_{index:03d}"
            meta = json.loads((src / "field.json").read_text())
            assert meta["seed"] == plan["field_seeds"][arm][index] and meta["ber"] == plan["bers"][arm]
            proposed = np.memmap(src / "proposed.u8", dtype=np.uint8, mode="r")
            for name, p, frac, cand, prot, prepared in masks:
                target = out / "eval" / f"arm_{arm:02d}" / f"field_{index:03d}" / f"{name}.json"
                if target.exists():
                    continue
                started = time.monotonic()
                with injected_packed(adapter, proposed, meta, prot, prepared) as events:
                    risk = run()
                write_once(target, dict(mask=name, candidate=cand, target_ber=p, fraction=frac, budget=prot.budget, mask_sha256=prot.file_sha256,
                                        test_ber=plan["bers"][arm], arm=arm, field_index=index, field_sha256=meta["sha256"],
                                        events=dict(events), risk=_risk_record(risk), seconds=time.monotonic() - started))
            del proposed
            print(json.dumps(dict(shard=shard, field=index, test_ber=plan["bers"][arm])), flush=True)
    assert _risk_record(run()) == json.loads((SOURCE_RUN / "clean/shard_00.json").read_text()), "model not restored"
    print("done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--out", type=Path, required=True); b.add_argument("--ber", type=float, required=True); b.add_argument("--fraction", type=float, required=True)
    e = sub.add_parser("eval"); e.add_argument("--out", type=Path, required=True); e.add_argument("--shard", type=int, required=True); e.add_argument("--shards", type=int, required=True)
    args = ap.parse_args()
    if args.cmd == "build":
        assert args.ber in TARGET_BERS and args.fraction in FRACTIONS
        build(args.out, args.ber, args.fraction)
    else:
        evaluate(args.out, args.shard, args.shards)


if __name__ == "__main__":
    main()
