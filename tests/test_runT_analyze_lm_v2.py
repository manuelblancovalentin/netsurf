"""The 16-draw language-model analysis pools draws from two run directories and refuses gaps or duplicates."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows"))
import runT_analyze_lm_v2 as V  # noqa: E402


def write(base, draws, gain=0.3, fk=True):
    rng = np.random.default_rng(draws[0])
    for d in draws:
        for i, p in enumerate((1e-4, 1e-3)):
            out = base / f"draw{d}" / f"ber{i}"
            out.mkdir(parents=True)
            with open(out / "results.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["condition", "ber", "metric_mean_nll"])
                w.writeheader()
                n = rng.normal(0, 0.01)
                w.writerow(dict(condition="tmr_abs@0.125", ber=p, metric_mean_nll=3.0 + n))
                w.writerow(dict(condition="tmr_msb@0.125", ber=p, metric_mean_nll=3.0 + gain * i + n))
                w.writerow(dict(condition="unprotected", ber=p, metric_mean_nll=3.4))
            (out / "DONE").write_text(json.dumps(dict(clean=dict(mean_nll=3.0))))
            if fk:
                (out / "DONE_fkeras").write_text("{}")


def test_pools_two_directories(tmp_path):
    write(tmp_path / "a" / "qwen3_0_6b_int8", list(range(4)))
    write(tmp_path / "b" / "qwen3_0_6b_int8", list(range(4, 16)))
    spec = V.SPEC["qwen3_0_6b_int8"]
    assert spec["draws"] == 16 and spec["margin"] == 0.5 and spec["min_diff"] == 0.05
    acc, clean = V.load([tmp_path / "a" / "qwen3_0_6b_int8", tmp_path / "b" / "qwen3_0_6b_int8"], spec)
    assert clean == -3.0 and len(acc[("tmr_abs@0.125", 1e-3)]) == 16
    V.A.MARGIN, V.A.MIN_DIFF = spec["margin"], spec["min_diff"]
    signs = np.random.default_rng(V.A.SEED).choice((-1.0, 1.0), size=(20000, 16))
    _, _, rows = V.A.analyze(acc, clean, signs)
    r = [x for x in rows if x["right"] == "tmr_msb@0.125" and x["ber"] == 1e-3][0]
    assert r["diff"] > 0.25 and r["p_raw"] < 1e-3                # 16 concordant draws reach a small p


def test_refuses_duplicates_and_gaps(tmp_path):
    write(tmp_path / "a" / "pythia160m_int8", list(range(8)))
    write(tmp_path / "b" / "pythia160m_int8", list(range(7, 15)))
    with pytest.raises(SystemExit):
        V.load([tmp_path / "a" / "pythia160m_int8", tmp_path / "b" / "pythia160m_int8"], V.SPEC["pythia160m_int8"])
    write(tmp_path / "c" / "pythia160m_int8", list(range(8)))
    write(tmp_path / "d" / "pythia160m_int8", list(range(9, 17)))
    with pytest.raises(SystemExit):
        V.load([tmp_path / "c" / "pythia160m_int8", tmp_path / "d" / "pythia160m_int8"], V.SPEC["pythia160m_int8"])


def test_requires_fkeras_conditions(tmp_path):
    write(tmp_path / "a" / "pythia160m_int8", list(range(16)), fk=False)
    with pytest.raises(SystemExit):
        V.load([tmp_path / "a" / "pythia160m_int8"], V.SPEC["pythia160m_int8"])
