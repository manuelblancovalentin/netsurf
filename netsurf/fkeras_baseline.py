"""Faithful paper-form FKeras principal weight-Hessian baseline.

This module implements H' = sum_i lambda_i (v_i^T theta) v_i from the paper.
It intentionally does not implement a diagonal Hessian, trace estimator, or
the public repository's apparent extra eigenvalue multiplication.
"""

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


def leading_hessian_eigenpairs(
    hessian_vector_product: Callable[[np.ndarray], np.ndarray], dimension: int,
    *, count: int, seed: int, max_iterations: int = 100, tolerance: float = 1e-7,
) -> tuple[np.ndarray, EigenpairDiagnostics]:
    """Deterministic orthogonalized power iteration on a symmetric Hessian."""

    if not 1 <= count <= dimension:
        raise ValueError("eigenpair count must lie in [1, dimension]")
    generator = np.random.default_rng(seed)
    vectors: list[np.ndarray] = []
    eigenvalues: list[float] = []
    residuals: list[float] = []
    iterations: list[int] = []
    converged: list[bool] = []
    for _index in range(count):
        vector = generator.normal(size=dimension)
        for previous in vectors:
            vector -= np.dot(previous, vector) * previous
        vector /= np.linalg.norm(vector)
        done = False
        for iteration in range(1, max_iterations + 1):
            product = np.asarray(hessian_vector_product(vector), dtype=np.float64)
            if product.shape != (dimension,) or not np.all(np.isfinite(product)):
                raise ValueError("HVP must return one finite vector of the requested dimension")
            for previous in vectors:
                product -= np.dot(previous, product) * previous
            norm = float(np.linalg.norm(product))
            if norm == 0.0:
                raise ValueError("power iteration reached a zero deflated HVP")
            updated = product / norm
            if np.dot(updated, vector) < 0.0:
                updated = -updated
            if np.linalg.norm(updated - vector) <= tolerance:
                vector, done = updated, True
                break
            vector = updated
        product = np.asarray(hessian_vector_product(vector), dtype=np.float64)
        eigenvalue = float(np.dot(vector, product))
        residual = float(np.linalg.norm(product - eigenvalue * vector))
        vectors.append(vector)
        eigenvalues.append(eigenvalue)
        residuals.append(residual)
        iterations.append(iteration)
        converged.append(done or residual <= tolerance)
    return np.stack(vectors), EigenpairDiagnostics(
        tuple(eigenvalues), tuple(residuals), tuple(iterations), tuple(converged)
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
    return np.sum(values[:, None] * (vectors @ theta)[:, None] * vectors, axis=0)


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
