"""NetSuRF: deterministic, side-effect-free scientific primitives.

Importing this package performs no filesystem, database, model, or environment
initialization. Legacy TensorFlow integration is isolated in
``netsurf.adapters`` and imported only by callers that request it.
"""

from .faults import FaultEvent, FaultModel, FaultSchedule, InjectionReport
from .identities import BitIdentity, ParameterSpec
from .protection import ProtectionMask, ProtectionPolicy
from .quantization import BitCodec, FixedPointFormat
from .ranking import Ranking, SusceptibilityEstimator

__all__ = [
    "BitCodec",
    "BitIdentity",
    "FaultEvent",
    "FaultModel",
    "FaultSchedule",
    "FixedPointFormat",
    "InjectionReport",
    "ParameterSpec",
    "ProtectionMask",
    "ProtectionPolicy",
    "Ranking",
    "SusceptibilityEstimator",
]

__version__ = "2.0.0"
