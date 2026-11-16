"""The Qwen GPU FKeras Lanczos solver agrees with the ARPACK solver of netsurf.fkeras_baseline on a test operator."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("scipy")


def test_lanczos_matches_arpack():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("qgpu", root / "workflows/runT_fkeras_qwen_gpu.py")
    sys.path.insert(0, str(root / "workflows/pkg"))
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    from netsurf.fkeras_baseline import leading_hessian_eigenpairs
    rng = np.random.default_rng(1)
    n = 3000
    A = rng.normal(size=(n, 40))
    H = A @ np.diag(rng.normal(size=40) * 5) @ A.T / n + np.diag(rng.normal(size=n) * 0.01)
    H = (H + H.T) / 2
    v1, d1 = g.lanczos_eigenpairs(lambda x: H @ x, n, count=2, seed=970201, tolerance=1e-5, max_products=300)
    v2, d2 = leading_hessian_eigenpairs(lambda x: H @ x, n, count=2, seed=970201, tolerance=1e-5, max_products=300,
                                        max_iterations=100)
    assert np.allclose(d1["eigenvalues"], d2.eigenvalues, rtol=1e-10)
    assert all(abs(a @ b) > 1 - 1e-10 for a, b in zip(v1, v2))
    assert all(x[np.argmax(np.abs(x))] >= 0 for x in v1)
