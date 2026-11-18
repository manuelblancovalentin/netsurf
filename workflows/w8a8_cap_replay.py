#!/usr/bin/env python3
"""Run B: deterministic replay of a finalized W8A8 wide-BER language-model grid,
recording finite-token NLL and 30-nat cap statistics that the source run hashed
but did not save.

Nothing in the source run is written.  Its frozen plan, saved proposed fault
fields (``fields/arm_XX/field_XXX/proposed.u8``), sealed masks and evaluation
cohort are read-only inputs.  No new effect is estimated: the estimand is, for
every existing (BER, field, policy) observation, the per-token loss statistics of
that same observation.

Acceptance gate (per observation): the replayed risk record -- including the
SHA-256 of the per-token (target logit, logsumexp) statistics -- and the event
counts must equal the finalized observation bit for bit.  A mismatch is recorded,
never silently accepted; more than ``MAX_MISMATCH`` mismatches abort the shard.

The script imports the ``netsurf`` package of the worktree that produced the
source run (``--code-root``); ``wide.validate`` then verifies that those source
files still match the hashes recorded in the frozen source plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "netsurf-w8a8-cap-replay/v1"
MANIFEST = Path("/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json")
CAP = 30.0
MAX_MISMATCH = 3
FIELDS_PER_BER = 24


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _write_once(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        f.write(_canonical(obj) + b"\n")
        f.flush()
        os.fsync(f.fileno())


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _digest(obj) -> str:
    return hashlib.sha256(_canonical(obj)).hexdigest()


def _import(code_root: Path):
    sys.path.insert(0, str(code_root))
    from netsurf import w8a8_wide_ber_campaign as wide          # noqa: E402
    from netsurf.w8a8_campaign import _evaluation, _risk_record   # noqa: E402
    from netsurf.w8a8_dense_protection import injected_packed, prepared_protection_bytes  # noqa: E402
    from netsurf.w8a8_faults import W8A8FaultAdapter             # noqa: E402
    if Path(wide.__file__).resolve().parents[1] != code_root.resolve():
        raise RuntimeError("netsurf was not imported from the requested code root")
    return wide, _evaluation, _risk_record, injected_packed, prepared_protection_bytes, W8A8FaultAdapter


def diagnostic(risk) -> dict:
    """Finite-token loss statistics; separates cap saturation from nonfinite failures."""
    finite = [float(item[1] - item[0]) for item in risk.logits if item is not None]
    if len(finite) + risk.failed_units != risk.total_units or not all(map(math.isfinite, finite)):
        raise ValueError("risk sufficient statistics are incomplete or nonfinite")
    finite.sort()
    n = len(finite)
    capped = sum(v >= CAP for v in finite)
    q = lambda f: finite[min(n - 1, int(f * n))] if n else None
    capped_mean = (sum(min(max(0.0, v), CAP) for v in finite) + CAP * risk.failed_units) / risk.total_units
    return {"finite_units": n, "finite_nll_at_or_above_30_count": capped,
            "finite_nll_uncapped_mean": sum(finite) / n if n else None,
            "finite_nll_uncapped_median": q(0.5), "finite_nll_uncapped_p95": q(0.95),
            "finite_nll_uncapped_max": finite[-1] if n else None,
            "cap_fraction_of_finite": capped / n if n else None,
            "cap_or_failure_fraction": (capped + risk.failed_units) / risk.total_units,
            "recomputed_capped_mean_nll": capped_mean}


def freeze(code_root: Path, source: Path, out: Path, shards: int) -> dict:
    if out.exists():
        raise FileExistsError(out)
    if not (source / "FINALIZED").is_file():
        raise ValueError("source run is not finalized")
    source_plan = _read(source / "plan.json")
    if source_plan["target"] not in ("pythia160m_int8", "qwen3_0_6b_int8"):
        raise ValueError("cap replay is defined for the two language models only")
    plan = {
        "schema": SCHEMA,
        "purpose": "diagnostic replay; no new effect estimate, no new fault field, no new mask",
        "estimand": "per-observation finite-token NLL statistics, fraction of finite token losses at or above "
                    "the 30-nat cap, and failed fraction, for every saved (BER, field, policy) observation",
        "target": source_plan["target"],
        "source_run": str(source), "source_plan_sha256": source_plan["plan_sha256"],
        "source_plan_file_sha256": _sha(source / "plan.json"),
        "source_finalized_marker": (source / "FINALIZED").read_text().strip(),
        "code_root": str(code_root),
        "code_source_files_sha256": {name: _sha(code_root / name) for name in source_plan["source_files_sha256"]},
        "replay_script_sha256": _sha(Path(__file__).resolve()),
        "bers": source_plan["bers"], "fields_per_ber": source_plan["fields_per_ber"],
        "policies_replayed": [p["name"] for p in source_plan["policies"] if p["kind"] != "full"],
        "final_cohort": source_plan["final_cohort"],
        "execution": {"shards": shards, "threads_per_shard": 4,
                      "field_assignment": "field index i belongs to shard i mod shards; BER arms interleaved"},
        "acceptance_gate": "replayed risk record (metrics, failed units, per-token statistics SHA-256) and event "
                           "counts equal the finalized observation exactly; mismatches are recorded and more than "
                           f"{MAX_MISMATCH} abort the shard",
        "counts": {"observations": len(source_plan["bers"]) * source_plan["fields_per_ber"]
                   * sum(p["kind"] != "full" for p in source_plan["policies"])},
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
    }
    if plan["code_source_files_sha256"] != source_plan["source_files_sha256"]:
        raise ValueError("code root no longer matches the source plan's recorded source hashes")
    plan["plan_sha256"] = _digest(plan)
    _write_once(out / "plan.json", plan)
    (out / "FROZEN").write_text(_sha(out / "plan.json") + "\n")
    return plan


def run_shard(code_root: Path, out: Path, shard: int, smoke: int = 0) -> dict:
    plan = _read(out / "plan.json")
    if _sha(out / "plan.json") != (out / "FROZEN").read_text().strip() or plan["schema"] != SCHEMA:
        raise ValueError("frozen replay plan differs")
    if _sha(Path(__file__).resolve()) != plan["replay_script_sha256"] or str(code_root) != plan["code_root"]:
        raise ValueError("replay script or code root differs from the frozen replay plan")
    shards = plan["execution"]["shards"]
    if not 0 <= shard < shards:
        raise ValueError("invalid shard")
    source = Path(plan["source_run"])
    wide, _evaluation, _risk_record, injected_packed, prepared_protection_bytes, Adapter = _import(code_root)
    source_plan, asset = wide.validate(MANIFEST, source, verify_masks=not smoke)
    if source_plan["plan_sha256"] != plan["source_plan_sha256"]:
        raise ValueError("source plan changed")
    adapter = Adapter(asset)
    adapter.load()
    evaluate = _evaluation(adapter, {"target": source_plan["target"], "final": source_plan["final_cohort"]})
    saved_clean = _read(source / "clean" / "shard_00.json")
    clean = evaluate()
    clean_record = _risk_record(clean)
    tag = f"shard_{shard:02d}" if not smoke else "smoke"
    if clean_record != saved_clean:
        _write_once(out / "clean_drift" / f"{tag}_start.json", {"expected": saved_clean, "observed": clean_record})
        raise RuntimeError("clean inference does not reproduce the finalized clean record; nothing replayed")
    clean_diag = diagnostic(clean)
    clean_path = out / "clean" / f"{tag}.json"
    if not clean_path.exists():
        _write_once(clean_path, {"risk": clean_record, "diagnostic": clean_diag})
    policies = [(rec, wide._protection(rec, source_plan["eligible_bits"]))
                for rec in source_plan["policies"] if rec["kind"] != "full"]
    if smoke:
        policies = policies[:smoke]
    protected = {rec["name"]: prepared_protection_bytes(prot) for rec, prot in policies}
    import numpy as np
    mismatches = done = 0
    indices = [0] if smoke else list(range(shard, FIELDS_PER_BER, shards))
    arms = [0] if smoke else list(range(len(source_plan["bers"])))
    for index in indices:
        for arm in arms:
            src_dir = source / "fields" / f"arm_{arm:02d}" / f"field_{index:03d}"
            dst_dir = out / ("smoke" if smoke else "fields") / f"arm_{arm:02d}" / f"field_{index:03d}"
            if (dst_dir / "field_done.json").exists():
                continue
            meta = _read(src_dir / "field.json")
            if (meta["seed"] != source_plan["field_seeds"][arm][index] or meta["ber"] != source_plan["bers"][arm]
                    or (src_dir / "proposed.u8").stat().st_size != source_plan["eligible_bits"] // 8):
                raise ValueError("saved proposed field differs from the frozen source schedule")
            proposed = np.memmap(src_dir / "proposed.u8", dtype=np.uint8, mode="r")
            field_mismatch = 0
            for rec, prot in policies:
                target_path = dst_dir / f"{rec['name']}.json"
                if target_path.exists():
                    continue
                saved = _read(src_dir / f"{rec['name']}.json")
                started = time.monotonic()
                with injected_packed(adapter, proposed, meta, prot, protected[rec["name"]]) as events:
                    risk = evaluate()
                record = _risk_record(risk)
                match = record == saved["risk"] and dict(events) == saved["events"]
                field_mismatch += not match
                obs = {"replay_plan_sha256": plan["plan_sha256"], "source_plan_sha256": source_plan["plan_sha256"],
                       "arm": arm, "field_index": index, "ber": source_plan["bers"][arm],
                       "field_seed": meta["seed"], "field_sha256": meta["sha256"], "policy": rec["name"],
                       "mask_sha256": prot.digest, "budget_bits": prot.budget, "events": dict(events),
                       "risk": record, "replay_bitwise_match": bool(match),
                       "saved_mean_nll": saved["risk"]["metrics"]["mean_nll"],
                       "diagnostic": diagnostic(risk), "elapsed_seconds": time.monotonic() - started}
                obs["payload_sha256"] = _digest(obs)
                _write_once(target_path, obs)
            del proposed
            mismatches += field_mismatch
            _write_once(dst_dir / "field_done.json", {"arm": arm, "field_index": index,
                                                      "observations": len(policies), "mismatches": field_mismatch})
            done += 1
            print(json.dumps({"target": plan["target"], "shard": shard, "ber": source_plan["bers"][arm],
                              "field": index, "fields_done_this_shard": done, "mismatches": mismatches,
                              "utc": datetime.now(timezone.utc).isoformat()}), flush=True)
            if mismatches > MAX_MISMATCH:
                raise RuntimeError("too many replay mismatches; shard aborted, nothing promoted")
    end_record = _risk_record(evaluate())
    if end_record != saved_clean:
        raise RuntimeError("model did not restore to the exact clean output after the shard")
    _write_once(out / "clean" / f"{tag}_end_{int(time.time())}.json", {"risk": end_record})
    return {"shard": shard, "fields_done": done, "mismatches": mismatches}


def status(out: Path) -> dict:
    plan = _read(out / "plan.json")
    files = list((out / "fields").glob("arm_*/field_*/*.json")) if (out / "fields").exists() else []
    obs = [p for p in files if p.name != "field_done.json"]
    done = [p for p in files if p.name == "field_done.json"]
    mism = sum(_read(p)["mismatches"] for p in done)
    return {"target": plan["target"], "observations_saved": len(obs),
            "observations_planned": plan["counts"]["observations"],
            "fields_done": len(done), "fields_planned": len(plan["bers"]) * plan["fields_per_ber"],
            "mismatches_in_completed_fields": mism}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze"); f.add_argument("--code-root", type=Path, required=True)
    f.add_argument("--source-run", type=Path, required=True); f.add_argument("--out", type=Path, required=True)
    f.add_argument("--shards", type=int, required=True)
    r = sub.add_parser("run-shard"); r.add_argument("--code-root", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True); r.add_argument("--shard", type=int, required=True)
    r.add_argument("--smoke", type=int, default=0, help="replay only BER arm 0, field 0, first N policies")
    s = sub.add_parser("status"); s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "freeze":
        plan = freeze(a.code_root, a.source_run, a.out, a.shards)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "counts": plan["counts"]}))
    elif a.cmd == "run-shard":
        print(json.dumps(run_shard(a.code_root, a.out, a.shard, a.smoke)))
    else:
        print(json.dumps(status(a.out)))


if __name__ == "__main__":
    main()
