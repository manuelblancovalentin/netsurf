"""Post-hoc fully connected layer and Ranger clip used by runT_ranger.py."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows"))
import runT_ranger as R  # noqa: E402


def test_fc_logits_match_float_reference():
    rng = np.random.default_rng(0)
    x = rng.integers(-128, 128, size=(3, 16)).astype(np.int8)
    w = rng.integers(-127, 128, size=(10, 16)).astype(np.int8)
    b = rng.integers(-1000, 1000, size=10).astype(np.int32)
    zx, sx, sw = -128, 0.05, rng.uniform(0.001, 0.01, size=10)
    ref = ((x.astype(float) - zx) * sx) @ (w.astype(float) * sw[:, None]).T + b * sx * sw
    assert np.allclose(R.fc_logits(x, zx, sx, w, sw, b), ref)


def test_clip_only_caps_large_codes():
    x = np.array([[-128, -100, -83, -50, 127]], dtype=np.int8)
    assert R.ranger_clip(x, -128, -83).tolist() == [[-128, -100, -83, -83, -83]]
