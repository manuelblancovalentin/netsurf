"""FKeras Eq. (3) and pinned repository scores; no integer-program Hessian."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from .identities import BitIdentity


@dataclass(frozen=True)
class EigenpairDiagnostics:
    eigenvalues: tuple[float, ...]
    residual_norms: tuple[float, ...]
    iterations: tuple[int, ...]
    converged: tuple[bool, ...]
    normalized_residuals: tuple[float, ...] = ()
    orthogonality_error: float = 0.0
    symmetry_error: float = 0.0
    hvp_calls: int = 0
    solver: str = ""
    ordering: str = "largest_magnitude_signed"


def leading_hessian_eigenpairs(
    hessian_vector_product: Callable[[np.ndarray], np.ndarray], dimension: int,
    *, count: int, seed: int, max_iterations: int = 100, tolerance: float = 1e-7,
    max_products: int = 1000,
) -> tuple[np.ndarray, EigenpairDiagnostics]:
    """Residual-checked symmetric Lanczos (dense reference for n <= 64).

    LM matches the magnitude ordering of power iteration, not largest algebraic
    eigenvalue. Negative eigenvalues retain their sign. A seeded start, canonical
    signs and ordering make repeated runs reproducible in one fixed runtime.
    Degenerate cut-off eigenspaces are not uniquely defined. Nonconvergence,
    nonsymmetric products and exhausted work budgets fail closed.
    """

    if not 1 <= count <= dimension:
        raise ValueError("eigenpair count must lie in [1, dimension]")
    if not 0 < tolerance < 1 or max_iterations < 1 or max_products < 1:
        raise ValueError("positive work limits and tolerance in (0,1) required")
    generator = np.random.default_rng(seed)
    calls = 0

    def product(vector):
        nonlocal calls
        if calls >= max_products:
            raise ValueError("FKeras HVP work budget exhausted; no ranking accepted")
        calls += 1
        value = np.asarray(hessian_vector_product(np.asarray(vector)), dtype=np.float64)
        if value.shape != (dimension,) or not np.all(np.isfinite(value)):
            raise ValueError("HVP must return one finite vector of the requested dimension")
        return value

    u, v = generator.normal(size=(2, dimension))
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    hu, hv = product(u), product(v)
    symmetry = float(abs(u @ hv - v @ hu) / max(np.linalg.norm(hu), np.linalg.norm(hv), 1e-30))
    if symmetry > max(tolerance, 1e-10):
        raise ValueError(f"HVP symmetry check failed: {symmetry}")
    if dimension <= 64:
        matrix = np.column_stack([product(np.eye(dimension)[i]) for i in range(dimension)])
        dense_symmetry = float(np.linalg.norm(matrix - matrix.T) / max(np.linalg.norm(matrix), 1e-30))
        symmetry = max(symmetry, dense_symmetry)
        if symmetry > max(tolerance, 1e-10):
            raise ValueError(f"dense Hessian symmetry check failed: {symmetry}")
        values, columns = np.linalg.eigh((matrix + matrix.T) / 2)
        solver = "numpy-eigh-dense-reference/v1"
    else:
        if count >= dimension:
            raise ValueError("full Hessian is bounded to dimension <= 64")
        try:
            from scipy.sparse.linalg import LinearOperator, eigsh, ArpackNoConvergence
        except ImportError as error:
            raise ValueError("FKeras requires scipy>=1.11,<2; researcher must install it") from error
        try:
            values, columns = eigsh(
                LinearOperator((dimension, dimension), matvec=product, dtype=np.float64),
                k=count, which="LM", v0=v, tol=tolerance * 0.1,
                maxiter=max_iterations, ncv=min(dimension, max(20, 2 * count + 1)),
            )
        except ArpackNoConvergence as error:
            raise ValueError("FKeras eigensolver did not converge; no ranking accepted") from error
        solver = "scipy-arpack-symmetric-lanczos/v1"
    order = sorted(range(len(values)), key=lambda i: (-abs(values[i]), -values[i]))[:count]
    vectors = columns[:, order].T.copy()
    values = values[order]
    for vector in vectors:
        if vector[np.argmax(np.abs(vector))] < 0:
            vector *= -1
    products = np.stack([product(vector) for vector in vectors])
    residuals = np.linalg.norm(products - values[:, None] * vectors, axis=1)
    # Scale-free backward error. Exact zero H with zero residual is accepted.
    denominators = np.maximum(np.linalg.norm(products, axis=1) + np.abs(values), 1e-30)
    normalized = residuals / denominators
    orthogonality = float(np.linalg.norm(vectors @ vectors.T - np.eye(count), ord=2))
    if np.any(normalized > tolerance) or orthogonality > max(tolerance, 1e-10):
        raise ValueError(f"FKeras eigenpair validation failed: residual={normalized}, orthogonality={orthogonality}")
    return vectors, EigenpairDiagnostics(
        tuple(map(float, values)), tuple(map(float, residuals)), (),
        tuple(True for _ in values), tuple(map(float, normalized)), orthogonality,
        symmetry, calls, solver,
    )


def principal_weight_hessian_score(
    parameters: np.ndarray, eigenvectors: np.ndarray, eigenvalues: np.ndarray,
) -> np.ndarray:
    """Return the signed paper-form H' parameter score."""

    theta = np.asarray(parameters, dtype=np.float64).reshape(-1)
    vectors = np.asarray(eigenvectors, dtype=np.float64)
    values = np.asarray(eigenvalues, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != theta.size or values.shape != (vectors.shape[0],):
        raise ValueError("incompatible parameter/eigenpair shapes")
    if not all(np.all(np.isfinite(value)) for value in (theta, vectors, values)):
        raise ValueError("FKeras inputs must be finite")
    return np.sum(values[:, None] * (vectors @ theta)[:, None] * vectors, axis=0)


FKERAS_SOURCE_COMMIT = "68e775c77670c9a1aab4e1b5295b523347d97013"


def mask_equivalence(masks):
    """Compare actual saved mask sets at matching cardinalities, not scores."""
    from itertools import combinations
    groups = {}
    for name, members in masks.items():
        groups.setdefault(len(members), []).append((name, frozenset(tuple(b) for b in members)))
    return [{"left": a, "right": b, "protected_bits": count,
             "same_mask": left == right, "symmetric_difference_bits": len(left ^ right)}
            for count, entries in sorted(groups.items())
            for (a, left), (b, right) in combinations(entries, 2)]


def repository_weight_hessian_score(parameters, eigenvectors, eigenvalues):
    """Pinned do_sum_hessian_rank: sum_i abs(lambda_i**2 (v_i.theta) v_i).

    Absolute value is taken BEFORE summation. Not abs(sum(lambda**2 ...)).
    """
    principal_weight_hessian_score(parameters, eigenvectors, eigenvalues)  # validate
    theta = np.asarray(parameters, dtype=np.float64).reshape(-1)
    vectors, values = np.asarray(eigenvectors), np.asarray(eigenvalues)
    return np.sum(np.abs(values[:, None] ** 2 * (vectors @ theta)[:, None] * vectors), axis=0)


def fkeras_msb_tier_order(
    identities: tuple[BitIdentity, ...], word_scores: Mapping[tuple[str, int], float],
) -> tuple[BitIdentity, ...]:
    """Order by word-relative MSB tier, then absolute FKeras word score."""

    missing = {(value.parameter, value.flat_index) for value in identities} - set(word_scores)
    if missing:
        raise KeyError(f"missing FKeras word score: {min(missing)}")
    return tuple(sorted(
        identities,
        key=lambda value: (
            -value.bit_index,
            -abs(float(word_scores[(value.parameter, value.flat_index)])),
            value,
        ),
    ))
