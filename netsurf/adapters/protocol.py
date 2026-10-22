"""Framework-neutral adapter contract consumed by the experiment engine."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from ..identities import BitIdentity, ParameterSpec


class EvaluationResult(Protocol):
    accuracy: float
    categorical_crossentropy: float
    sample_count: int


class BenchmarkAdapter(Protocol):
    """Minimum interface for the currently supported weight-fault protocol."""

    def load(self) -> None: ...

    def dataset(
        self, count: int, *, role: str = "evaluation"
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def values(self) -> dict[str, np.ndarray]: ...

    def parameter_specs(self) -> tuple[ParameterSpec, ...]: ...

    def bit_identities(self) -> Sequence[BitIdentity]: ...

    def temporary_values(
        self, values: Mapping[str, np.ndarray]
    ) -> AbstractContextManager[None]: ...

    def quantized_values(self) -> dict[str, np.ndarray]: ...

    def evaluate(
        self, inputs: np.ndarray, labels: np.ndarray, *, batch_size: int = 256
    ) -> EvaluationResult: ...

    def gradients(self, inputs: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]: ...

    def polar_impacts(
        self, inputs: np.ndarray, *, activation_source: str
    ) -> dict[str, np.ndarray]: ...
