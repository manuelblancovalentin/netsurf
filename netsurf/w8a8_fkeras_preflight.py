"""Bounded full-qweight FKeras feasibility primitives, never fault inference.

The Hessian here belongs to a continuous decoded-weight companion.  The
authoritative W8A8 executable still uses stored INT8 codes and torch._int_mm.
No parameter subset may be reported as a full-population FKeras preflight.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class Projection:
    codes: int
    eligible_bits: int
    lanczos_ncv: int
    eigenpairs: int
    float64_lanczos_basis_bytes: int
    float64_eigenvectors_bytes: int
    float32_decoded_weights_bytes: int
    float32_hvp_vectors_bytes: int
    float32_scores_bytes: int
    four_packed_masks_bytes: int
    eigensolver_numeric_peak_bytes: int
    score_mask_numeric_peak_bytes: int
    numeric_peak_projection_bytes: int


def project_full_population(codes: int, *, eigenpairs: int = 8,
                            lanczos_ncv: int = 20) -> Projection:
    """Explicit numeric-array lower bound; excludes activations and allocator overhead."""
    if codes <= 0 or not 1 <= eigenpairs < lanczos_ncv <= codes:
        raise ValueError("invalid full-population eigensolver dimensions")
    basis = codes * lanczos_ncv * 8
    vectors = codes * eigenpairs * 8
    weights = codes * 4
    hvp = codes * 2 * 4
    scores = codes * 4
    masks = codes * 4  # four bitsets, each (8*codes)/8 bytes
    eigensolver_peak = basis + vectors + weights + hvp
    score_mask_peak = vectors + weights + scores + masks
    return Projection(codes, 8 * codes, lanczos_ncv, eigenpairs, basis, vectors,
                      weights, hvp, scores, masks, eigensolver_peak,
                      score_mask_peak, max(eigensolver_peak, score_mask_peak))


def validate_full_mapping(eligible_entries: Sequence[dict], weight_shapes: dict[str, tuple[int, ...]],
                          eligible_bits: int) -> tuple[tuple[str, int, int], ...]:
    """Return canonical code offsets only if every eligible qweight is mapped once."""
    names = [entry["module"] for entry in eligible_entries]
    if len(names) != len(set(names)) or set(names) != set(weight_shapes):
        raise ValueError("continuous surrogate omits or duplicates eligible qweights")
    mapping = []
    offset = 0
    for entry in eligible_entries:
        name = entry["module"]
        shape = tuple(int(x) for x in entry["shape"])
        if shape != tuple(weight_shapes[name]) or any(x <= 0 for x in shape):
            raise ValueError("continuous surrogate qweight shape differs")
        size = math.prod(shape)
        mapping.append((name, offset, size))
        offset += size
    if 8 * offset != eligible_bits:
        raise ValueError("continuous surrogate covers a different eligible bit population")
    return tuple(mapping)


def _dot64(a: np.ndarray, b: np.ndarray, *, chunk: int = 1_048_576) -> float:
    return math.fsum(float(np.dot(a[i:i+chunk].astype(np.float64),
                                 b[i:i+chunk].astype(np.float64)))
                     for i in range(0, len(a), chunk))


def symmetry_probe(hvp: Callable[[np.ndarray], np.ndarray], dimension: int,
                   *, seed: int = 970302) -> dict:
    """Two-product matrix-free symmetry check over the complete vector."""
    if dimension < 2:
        raise ValueError("at least two continuous parameters required")
    rng = np.random.default_rng(seed)
    directions = []
    for _ in range(2):
        direction = np.empty(dimension, dtype=np.float32)
        for start in range(0, dimension, 1_048_576):
            end = min(start + 1_048_576, dimension)
            direction[start:end] = rng.standard_normal(end - start, dtype=np.float32)
        direction /= math.sqrt(_dot64(direction, direction))
        directions.append(direction)
    u, v = directions
    hu = np.asarray(hvp(u), dtype=np.float32)
    hv = np.asarray(hvp(v), dtype=np.float32)
    if hu.shape != (dimension,) or hv.shape != (dimension,) or not (
        np.isfinite(hu).all() and np.isfinite(hv).all()
    ):
        raise ValueError("nonfinite or incomplete symmetry HVP")
    left, right = _dot64(u, hv), _dot64(v, hu)
    error = abs(left - right) / max(
        math.sqrt(_dot64(hu, hu)), math.sqrt(_dot64(hv, hv)), 1e-30)
    return {"relative_symmetry_error": error, "left": left, "right": right,
            "hvp_calls": 2}


def bounded_eigenpair_probe(hvp: Callable[[np.ndarray], np.ndarray], dimension: int,
                            *, seed: int = 970301, max_products: int = 6,
                            tolerance: float = 1e-3) -> dict:
    """Bounded power iteration with measured full-vector Rayleigh residuals.

    One converged leading-magnitude candidate only probes feasibility; it does
    not supply the published top-k score or a protection policy.  Float64
    reductions are chunked so a billion-code model avoids extra dense copies.
    """
    if dimension < 2 or max_products < 2 or not 0 < tolerance < 1:
        raise ValueError("invalid bounded eigenpair preflight dimensions or limits")
    rng = np.random.default_rng(seed)
    current = np.empty(dimension, dtype=np.float32)
    for start in range(0, dimension, 1_048_576):
        end = min(start + 1_048_576, dimension)
        current[start:end] = rng.standard_normal(end - start, dtype=np.float32)
    norm = math.sqrt(_dot64(current, current))
    current /= norm
    history = []
    for call in range(1, max_products + 1):
        product = np.asarray(hvp(current), dtype=np.float32)
        if product.shape != (dimension,) or not np.isfinite(product).all():
            raise ValueError("nonfinite or incomplete full-population HVP")
        value = _dot64(current, product)
        residual_square = 0.0
        product_square = 0.0
        for start in range(0, dimension, 1_048_576):
            end = min(start + 1_048_576, dimension)
            v = current[start:end].astype(np.float64)
            h = product[start:end].astype(np.float64)
            difference = h - value * v
            residual_square += float(np.dot(difference, difference))
            product_square += float(np.dot(h, h))
        residual = math.sqrt(residual_square)
        normalized = residual / max(math.sqrt(product_square) + abs(value), 1e-30)
        history.append({"hvp_call": call, "candidate_eigenvalue": value,
                        "residual_norm": residual, "normalized_residual": normalized})
        if call >= 2 and normalized <= tolerance:
            return {"candidate_eigenvalue": value, "residual_norm": residual,
                    "normalized_residual": normalized, "hvp_calls": call,
                    "converged": True, "history": history,
                    "interpretation": "one residual-validated eigenpair; published top-k score not constructed"}
        norm = math.sqrt(product_square)
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("zero or invalid Hessian product; eigenpair unresolved")
        product /= norm
        current = product
    return {"candidate_eigenvalue": value, "residual_norm": residual,
            "normalized_residual": normalized, "hvp_calls": max_products,
            "converged": False, "history": history,
            "interpretation": "bounded probe did not converge; no FKeras score"}


def tier_first_bit_order(word_scores: np.ndarray, *, bits_per_word: int = 8) -> np.ndarray:
    """Small-fixture reference: MSB tier precedes all lower tiers, regardless of score."""
    scores = np.asarray(word_scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all() or (scores < 0).any() or bits_per_word < 2:
        raise ValueError("finite nonnegative word scores and bit width required")
    words = np.arange(len(scores), dtype=np.int64)
    within = np.lexsort((words, -scores))
    return np.concatenate([bits_per_word * within + bit
                           for bit in range(bits_per_word - 1, -1, -1)])


def tier_first_score_keys(word_scores: np.ndarray) -> np.ndarray:
    """Packed uint64 keys for the existing exact streaming top-budget engine.

    The high 32 bits are the INT8 bit tier (7 first), and the low 32 bits are
    the nonnegative FP32 word-score encoding.  Ordinal breaks equal-score ties.
    This is a bit-ordering primitive, not a Hessian or a measured mask.
    """
    scores = np.asarray(word_scores, dtype=np.float32)
    if scores.ndim != 1 or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("finite nonnegative FKeras word scores required")
    values = scores.view(np.uint32).astype(np.uint64)
    keys = np.empty((len(scores), 8), dtype=np.uint64)
    for bit in range(8):
        keys[:, bit] = (np.uint64(bit) << np.uint64(32)) | values
    return keys.reshape(-1)
