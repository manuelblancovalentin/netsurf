#!/usr/bin/env python3
"""Gradient-trust plan (review/GRADIENT_TRUST_PLAN_2026-11-15.md, amendment A1): checkpoint cards for the kona-assets
jet and MNIST series (kona-assets commits 678ba0c, 705043d; NETSURF_GRADIENT_TRUST_FILES_2026-11-15.tsv).

For each model: the INT8 TFLite file (copied next to the card), its six calibration draws (the family's float32
calibration inputs, encoded with this model's own input scale and zero point: clip(rint(x / s + z))), and its six
evaluation windows (the model's own <model>.eval_int8.npz, already encoded).  Every source file is checked against the
sha256 in the kona-assets file list; the encoded calibration inputs are re-checked against the model's input contract.
Write-once: <out>/<model>/{model.tflite, cal<r>.npz, ev<r>.npz, card.json, SHA256SUMS}.

    runG_kona_inputs.py <out>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from site_paths import DATA  # noqa: E402

TSV = Path("/projects/kona-assets/manuelbv/NETSURF_GRADIENT_TRUST_FILES_2026-11-15.tsv")
SERIES = {"jet": ["jet-e1", "jet-e3", "jet-e10", "jet-e30", "jet-e100"],
          "mnist": ["mnist-tcad", "mnist-e1", "mnist-e3", "mnist-e10", "mnist-e30"]}
EXPLORATORY = {"mnist-tcad"}          # amendment A1


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(out: Path):
    import tensorflow as tf
    files = {}
    for row in csv.DictReader(open(TSV), delimiter="\t"):
        files[(row["item"], row["path"])] = (DATA / row["copy_to"], row["sha256"])

    def get(item, name):
        p, h = files[(item, name)]
        if sha(p) != h:
            raise SystemExit(f"{item}/{name}: sha256 differs from the kona-assets list")
        return p
    out.mkdir(parents=True, exist_ok=True)
    for fam, models in SERIES.items():
        item = f"netsurf-gt-{fam}-mlp"
        cal = np.load(get(item, f"{fam}-calibration-inputs.npz"))
        xc, yc = np.asarray(cal["x"]), np.asarray(cal["y"]).astype(np.int64)
        assert xc.shape[:2] == (6, 1024) and yc.shape == (6, 1024), (xc.shape, yc.shape)
        for mid in models:
            d = out / mid
            d.mkdir(parents=False, exist_ok=False)
            src = get(item, f"{mid}.tflite")
            content = src.read_bytes()
            (d / "model.tflite").write_bytes(content)
            it = tf.lite.Interpreter(model_content=content)
            it.allocate_tensors()
            i = it.get_input_details()[0]
            s, z = i["quantization"]
            ev = np.load(get(item, f"{mid}.eval_int8.npz"))
            if not (np.isclose(float(ev["input_scale"]), s) and int(ev["input_zero_point"]) == z):
                raise SystemExit(f"{mid}: evaluation inputs were encoded with another scale")
            card = dict(id=mid, tflite="model.tflite", sha256=hashlib.sha256(content).hexdigest(),
                        exploratory=mid in EXPLORATORY, family=fam, source=str(src), calibration=[], evaluation=[])
            for r in range(6):
                x8 = np.clip(np.rint(xc[r] / np.float32(s) + z), -128, 127).astype(np.int8)
                if tuple(x8.shape[1:]) != tuple(i["shape"][1:]):
                    raise SystemExit(f"{mid}: input shape {x8.shape} vs {i['shape']}")
                np.savez(d / f"cal{r}.npz", x=x8, y=yc[r])
                card["calibration"].append(f"cal{r}.npz")
                np.savez(d / f"ev{r}.npz", x=np.asarray(ev["x_int8"][r], np.int8), y=np.asarray(ev["y"][r]).astype(np.int64))
                card["evaluation"].append(f"ev{r}.npz")
            (d / "card.json").write_text(json.dumps(card, indent=1))
            (d / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.name}\n" for p in sorted(d.iterdir()) if p.name != "SHA256SUMS"))
            print(json.dumps(dict(model=mid, input_scale=float(s), input_zero_point=int(z),
                                  saturated_share=float(np.mean(np.abs(np.clip(np.rint(xc / np.float32(s) + z), -129, 128)) > 127.5)))), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    main(ap.parse_args().out)
