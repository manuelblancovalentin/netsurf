#!/usr/bin/env python3
"""Move the Run P/Q/R workflows to another machine (for example Quest) without changing what they compute.

``lists`` needs only the standard library; ``relocate`` imports the netsurf package, so it needs numpy (run it in the
installed environment).  Layout on the other machine, under one folder SITE:

    SITE/data/...      the files of the two MLPerf Tiny asset locks (models, CIFAR-10, Speech Commands), same relative
                       paths as under /data here, symbolic links kept (the locks resolve snapshot links to blobs)
    SITE/wt/...        the run outputs the workflows read (masks, DS-CNN preparation, retrained ResNet-8), same
                       relative paths as under the worktree folder here; the code itself is a clone of a git bundle
                       at SITE/wt/<branch worktree name>
    SITE/locks/mlperf-tiny/{ic,kws}.lock.json   the locks with their machine-local ``resolved`` paths rewritten

and the workflows read it through NETSURF_SITE_DATA=SITE/data, NETSURF_SITE_WT=SITE/wt,
NETSURF_SITE_LOCKS=SITE/locks (``site_paths.py``).

  lists  OUT_DIR            (here) data_files.txt, wt_files.txt (paths relative to the two roots, for
                            ``rsync --files-from``), SHA256SUMS (relative to SITE, for ``sha256sum -c``) and the two
                            locks as they are here.  Lock-listed files take their SHA-256 from the lock; the rest are
                            hashed.
  relocate LOCK OUT --site SITE [--verify]
                            (there) rewrite the ``resolved`` paths of LOCK from /data to SITE/data and write OUT.  The
                            lock digest excludes ``resolved``, so it must be unchanged; with --verify every listed file
                            is size- and SHA-256-checked through the relocated paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from site_paths import DATA, LOCKS, WT  # noqa: E402

LOCK_NAMES = {"ic": "mlperf-tiny/ic", "kws": "mlperf-tiny/kws"}
# Run outputs the workflows read (relative to WT).  Masks of Stage 1/2 ResNet-8, DS-CNN preparation (masks, static
# masks, engine reference), retrained ResNet-8 checkpoints.
WT_DIRS = (
    "paper1-tecs-revision-v1/workspace/runs/paper1_runR_masks_v1",
    "paper1-tecs-revision-v1/workspace/runs/paper1_runR_kws_prep_v1",
    "paper1-tecs-revision-v1/workspace/runs/paper1_runD_resnet8_retrain_v1",
)
DIRECT_DATA = (  # read by path in runP_phase0.py / runR_kws.py, besides the lock-resolved files
    "models/models--mlcommons--tiny/blobs/3c002613d1b2475eb51dd78dfb85a546c8ae658dee71cf6ade43b022fe205415",
    "models/models--mlcommons--tiny/blobs/aeea436800704fce17b17292e4412630ad856e9d777c044c64ef748a880bd0ae",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel_data(p: Path) -> str:
    return str(Path(p).relative_to(DATA))


def lists(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    data, sums = {}, {}
    for short in LOCK_NAMES:
        src = LOCKS / f"mlperf-tiny/{short}.lock.json"
        doc = json.loads(src.read_text())
        (out / f"{short}.lock.json").write_text(src.read_text())
        paths = doc["resolved"]["file_paths"]
        for e in doc["files"]:
            p = Path(paths[f"{e['asset']}|{e['path']}"])
            data[rel_data(p)] = e["sha256"]
            if p.is_symlink():                                   # keep the link and copy its target too
                target = p.resolve()
                data[rel_data(target)] = e["sha256"]
    for r in DIRECT_DATA:
        data.setdefault(r, sha256(DATA / r))
    for r, h in data.items():
        sums[f"data/{r}"] = h
    wt = []
    for d in WT_DIRS:
        for p in sorted((WT / d).rglob("*")):
            if p.is_file():
                r = str(p.relative_to(WT))
                wt.append(r)
                sums[f"wt/{r}"] = sha256(p)
    (out / "data_files.txt").write_text("".join(f"{r}\n" for r in sorted(data)))
    (out / "wt_files.txt").write_text("".join(f"{r}\n" for r in wt))
    (out / "SHA256SUMS").write_text("".join(f"{h}  {r}\n" for r, h in sorted(sums.items())))
    total = sum((DATA / r).stat().st_size for r in data if not (DATA / r).is_symlink()) + \
        sum((WT / r).stat().st_size for r in wt)
    print(json.dumps(dict(out=str(out), data_files=len(data), wt_files=len(wt), bytes=total)))


def relocate(lock: Path, out: Path, site: Path, verify: bool) -> None:
    sys.path.insert(0, str(HERE.parent))
    from netsurf.asset_lock import VerifiedAssetLock
    doc = json.loads(Path(lock).read_text())
    old, new = "/data/", str(Path(site) / "data") + "/"

    def move(v):
        if isinstance(v, str):
            if v == old.rstrip("/") or v.startswith(old):
                return new + v[len(old):] if v != old.rstrip("/") else new.rstrip("/")
            raise SystemExit(f"resolved path outside /data: {v}")
        if isinstance(v, dict):
            return {k: move(x) for k, x in v.items()}
        raise SystemExit(f"unexpected resolved entry {v!r}")
    before = doc["digest"]
    doc["resolved"] = move(doc["resolved"])
    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists (write-once)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    loaded = VerifiedAssetLock.load(out, verify_files=verify, expected_benchmark_id=doc["benchmark"]["id"])
    assert loaded.digest == before, (loaded.digest, before)
    print(json.dumps(dict(out=str(out), digest=before, files=len(loaded.resolved_files), verified=verify)))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("lists"); a.add_argument("out", type=Path)
    b = sub.add_parser("relocate"); b.add_argument("lock", type=Path); b.add_argument("out", type=Path)
    b.add_argument("--site", type=Path, required=True); b.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.cmd == "lists":
        lists(args.out)
    else:
        relocate(args.lock, args.out, args.site, args.verify)


if __name__ == "__main__":
    main()
