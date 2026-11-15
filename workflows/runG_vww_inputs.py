#!/usr/bin/env python3
"""Gradient-trust plan (review/GRADIENT_TRUST_PLAN_2026-11-15.md): calibration draws and evaluation windows for the
MLPerf Tiny Visual Wake Words model (vww_96_int8.tflite, sha256 597a384c…, kona-assets lock mlperf-tiny/vww).

Data: /data/datasets/visualwakewords/{non_person,person}/*.jpg (labels 0, 1; 96x96 RGB, train and val combined).
Split as the MLPerf training script does (Keras flow_from_directory, validation_split 0.1): per class, the files in
lexicographic order; the first int(0.1 n) are validation, the rest training.  The 1,000 official evaluation images
(vww01/y_labels.csv) and the 100 quantization-calibration images are excluded everywhere.
  calibration draws (6 x 1,024): training files ordered by sha256 of "<class>/<file>", the first 6,144, in blocks.
  evaluation windows (6 x 1,000): validation files in the same order, the first 6,000, in blocks.
Decoding: tf.io.decode_jpeg (TensorFlow 2.14.1); encoding to the model input: clip(rint((u8 / 255) / s + z)) with the
model's input scale and zero point (equal to u8 - 128).  Write-once output: <out>/{cal<r>,ev<r>}.npz (x int8, y int64,
files), indices.json, card.json (paths relative to the card; the model as data:<path under NETSURF_SITE_DATA>).

    runG_vww_inputs.py <out>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from site_paths import DATA  # noqa: E402

ROOT = DATA / "datasets/visualwakewords"
SNAP = "models/models--mlcommons--tiny/snapshots/4addd0fa08d216e20637637874e084895f289da4"
MODEL = f"{SNAP}/benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite"
SHA = "597a384c8c2c8a1276f04702f25013b7838f2f814f1ca7c174d295b73e3d6b7b"
EVAL_LABELS = f"{SNAP}/benchmark/evaluation/datasets/vww01/y_labels.csv"
CAL_LIST = f"{SNAP}/benchmark/training/visual_wake_words/calibration_data.txt"
CLASSES = ("non_person", "person")
DRAWS, DRAW, WINDOWS, WINDOW = 6, 1024, 6, 1000


def coco_id(name: str) -> str:
    return name.rsplit("_", 1)[-1].split(".")[0].lstrip("0")


def main(out: Path):
    import tensorflow as tf
    excluded = {coco_id(l.split(",")[0]) for l in (DATA / EVAL_LABELS).read_text().splitlines() if l.strip()}
    excluded |= {coco_id(l.strip()) for l in (DATA / CAL_LIST).read_text().splitlines() if l.strip().endswith(".jpg")}
    train, val = [], []
    for c in CLASSES:
        files = sorted(p.name for p in (ROOT / c).glob("*.jpg"))
        cut = int(0.1 * len(files))
        for k, f in enumerate(files):
            if coco_id(f) in excluded:
                continue
            (val if k < cut else train).append(f"{c}/{f}")
    key = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731
    train.sort(key=key)
    val.sort(key=key)
    content = (DATA / MODEL).read_bytes()
    if hashlib.sha256(content).hexdigest() != SHA:
        raise SystemExit("VWW model bytes differ from the locked sha256")
    it = tf.lite.Interpreter(model_content=content)
    it.allocate_tensors()
    scale, zp = it.get_input_details()[0]["quantization"]
    out.mkdir(parents=True, exist_ok=False)

    def encode(files):
        xs = []
        for f in files:
            u8 = tf.io.decode_jpeg(tf.io.read_file(str(ROOT / f)), channels=3).numpy()
            if u8.shape != (96, 96, 3):
                raise SystemExit(f"{f}: shape {u8.shape}")
            xs.append(np.clip(np.rint((u8.astype(np.float32) / np.float32(255.0)) / np.float32(scale) + zp),
                              -128, 127).astype(np.int8))
        y = np.array([CLASSES.index(f.split("/")[0]) for f in files], dtype=np.int64)
        return np.stack(xs), y

    card = dict(id="vww", tflite=f"data:{MODEL}", sha256=SHA, exploratory=False, calibration=[], evaluation=[])
    for r in range(DRAWS):
        files = train[r * DRAW:(r + 1) * DRAW]
        x, y = encode(files)
        np.savez(out / f"cal{r}.npz", x=x, y=y, files=np.array(files))
        card["calibration"].append(f"cal{r}.npz")
        print(json.dumps(dict(stage="calibration", draw=r, person_share=float(y.mean()), done=r + 1, total=DRAWS + WINDOWS)), flush=True)
    for r in range(WINDOWS):
        files = val[r * WINDOW:(r + 1) * WINDOW]
        x, y = encode(files)
        np.savez(out / f"ev{r}.npz", x=x, y=y, files=np.array(files))
        card["evaluation"].append(f"ev{r}.npz")
        print(json.dumps(dict(stage="evaluation", window=r, person_share=float(y.mean()), done=DRAWS + r + 1, total=DRAWS + WINDOWS)), flush=True)
    (out / "indices.json").write_text(json.dumps(dict(
        rule="see runG_vww_inputs.py docstring", training_candidates=len(train), validation_candidates=len(val),
        excluded_ids=len(excluded), decoder=f"tf.io.decode_jpeg, TensorFlow {tf.__version__}", input_scale=float(scale),
        input_zero_point=int(zp)), indent=1))
    (out / "card.json").write_text(json.dumps(card, indent=1))
    sums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("*.npz"))}
    (out / "SHA256SUMS").write_text("".join(f"{v}  {k}\n" for k, v in sums.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    main(ap.parse_args().out)
