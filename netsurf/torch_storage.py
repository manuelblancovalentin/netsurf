"""Exact, transactional mutation of CPU PyTorch parameter storage.

Torch is imported only when a storage object is constructed.  The eligible
universe is contiguous ``nn.Linear.weight`` parameters of one observed dtype;
all other model state is explicitly outside this initial fault model.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from typing import Iterator, Mapping

from .faults import FaultSchedule, InjectionReport
from .faults import FaultEvent, FaultModel
from .identities import BitIdentity
from .protection import ProtectionMask
from .storage_bit_codec import StorageBitCodec


@dataclass(frozen=True)
class TorchParameterDescriptor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    byte_sha256: str

    @property
    def scalars(self) -> int:
        result = 1
        for axis in self.shape:
            result *= axis
        return result


class TorchStoredLinearWeights:
    def __init__(self, model: object, precision: str):
        import torch

        self.model = model
        self.codec = StorageBitCodec(precision)
        required = torch.float32 if precision == "FP32" else torch.bfloat16
        if getattr(model, "training", True):
            raise ValueError("model must be in eval mode before storage inspection")
        modules = dict(model.named_modules())
        all_parameters = list(model.named_parameters(remove_duplicate=False))
        pointers: dict[int, str] = {}
        for name, parameter in all_parameters:
            if parameter.device.type != "cpu":
                raise ValueError(f"CPU stored-weight backend only: {name}")
            pointer = parameter.untyped_storage().data_ptr()
            if pointer in pointers:
                raise ValueError(f"aliased/tied parameter storage: {pointers[pointer]} and {name}")
            pointers[pointer] = name
        selected = {}
        for name, parameter in all_parameters:
            parent, _, leaf = name.rpartition(".")
            if leaf != "weight" or not isinstance(modules.get(parent), torch.nn.Linear):
                continue
            if parameter.dtype != required:
                raise ValueError(f"eligible {name} dtype {parameter.dtype} differs from {required}")
            if not parameter.is_contiguous() or parameter.storage_offset() != 0:
                raise ValueError(f"eligible {name} is a view or noncontiguous")
            if parameter.numel() * self.codec.byte_width != parameter.untyped_storage().nbytes():
                raise ValueError(f"eligible {name} shares storage with an extra region")
            selected[name] = parameter
        if not selected:
            raise ValueError("no verified dense/linear projection weights")
        self.parameters = dict(sorted(selected.items()))
        self.descriptors = tuple(TorchParameterDescriptor(name, tuple(value.shape),
            str(value.dtype), hashlib.sha256(self._bytes(value)).hexdigest())
            for name, value in self.parameters.items())
        self.bit_count = sum(item.scalars * self.codec.width for item in self.descriptors)

    @staticmethod
    def _byte_view(parameter):
        import torch
        return parameter.detach().view(torch.uint8).reshape(-1)

    @classmethod
    def _bytes(cls, parameter) -> bytes:
        return cls._byte_view(parameter).numpy().tobytes()

    def hashes(self) -> dict[str, str]:
        return {name: hashlib.sha256(self._bytes(value)).hexdigest()
                for name, value in self.parameters.items()}

    def _validate_identity(self, identity: BitIdentity) -> None:
        parameter = self.parameters.get(identity.parameter)
        if parameter is None:
            raise ValueError(f"fault bit outside eligible linear-weight population: {identity.key()}")
        if not 0 <= identity.flat_index < parameter.numel() or not 0 <= identity.bit_index < self.codec.width:
            raise ValueError(f"fault bit outside stored word: {identity.key()}")

    def bit_at(self, ordinal: int) -> BitIdentity:
        if not 0 <= ordinal < self.bit_count:
            raise IndexError(ordinal)
        for item in self.descriptors:
            extent = item.scalars * self.codec.width
            if ordinal < extent:
                return BitIdentity(item.name, ordinal // self.codec.width,
                                   ordinal % self.codec.width)
            ordinal -= extent
        raise RuntimeError("bit population accounting failed")

    @contextmanager
    def injected(self, schedule: FaultSchedule, mask: ProtectionMask) -> Iterator[InjectionReport]:
        import torch
        if schedule.model.target != "weight" or schedule.model.persistence != "persistent_trial":
            raise ValueError("persistent weight schedule required")
        if schedule.universe_size != self.bit_count or mask.universe_size != self.bit_count:
            raise ValueError("schedule/mask population differs from inspected storage")
        if len(mask.protected) > mask.budget:
            raise ValueError("protection budget exceeded")
        for bit in mask.protected:
            self._validate_identity(bit)
        if len({event.bit for event in schedule.events}) != len(schedule.events):
            raise ValueError("duplicate persistent events are not Bernoulli bit draws")
        for event in schedule.events:
            if event.inference_index is not None:
                raise ValueError("persistent event must have no inference index")
            self._validate_identity(event.bit)
        original = {name: self._bytes(parameter) for name, parameter in self.parameters.items()}
        pointers = {name: parameter.untyped_storage().data_ptr() for name, parameter in self.parameters.items()}
        applied = []
        suppressed = 0
        try:
            with torch.no_grad():
                for event in schedule.events:
                    bit = event.bit
                    if bit in mask.protected:
                        suppressed += 1
                        continue
                    parameter = self.parameters[bit.parameter]
                    position = bit.flat_index * self.codec.byte_width + bit.bit_index // 8
                    view = self._byte_view(parameter)
                    view[position] = int(view[position]) ^ (1 << (bit.bit_index % 8))
                    applied.append(bit)
            for name, parameter in self.parameters.items():
                if parameter.untyped_storage().data_ptr() != pointers[name]:
                    raise RuntimeError(f"parameter storage changed before inference: {name}")
            report = InjectionReport(schedule.digest(), len(schedule.events), len(applied),
                                     suppressed, len(set(applied)))
            yield report
            for name, parameter in self.parameters.items():
                if parameter.untyped_storage().data_ptr() != pointers[name]:
                    raise RuntimeError(f"parameter storage changed during inference: {name}")
        finally:
            with torch.no_grad():
                for name, raw in original.items():
                    self._byte_view(self.parameters[name]).copy_(torch.frombuffer(bytearray(raw), dtype=torch.uint8))
            if self.hashes() != {name: hashlib.sha256(raw).hexdigest() for name, raw in original.items()}:
                raise RuntimeError("byte-level PyTorch parameter restoration failed")


def expected_faults(eligible_bits: int, protected_bits: int, ber: float) -> dict[str, float]:
    if not 0 <= protected_bits <= eligible_bits or not 0 <= ber <= 1:
        raise ValueError("invalid bit population, protection budget, or BER")
    return {"proposed": eligible_bits * ber, "applied": (eligible_bits - protected_bits) * ber,
            "suppressed": protected_bits * ber}


def sparse_bernoulli_schedule(storage: TorchStoredLinearWeights, *, ber: float,
                              seed: int, inference_count: int,
                              max_events: int = 10000) -> FaultSchedule:
    """Exact independent Bernoulli draws via geometric success gaps.

    This avoids allocating a vector of all eligible bits.  The event cap is a
    feasibility gate, not a truncation: exceeding it fails the run.
    """
    import numpy as np
    if not 0 <= ber <= 1 or inference_count < 1 or max_events < 0:
        raise ValueError("invalid Bernoulli schedule configuration")
    law = FaultModel("weight", "cpu_pytorch_linear_weight_storage", "persistent_trial",
                     "independent_bernoulli", ber)
    events = []
    if ber:
        generator = np.random.default_rng(seed)
        ordinal = -1
        while True:
            ordinal += int(generator.geometric(ber))
            if ordinal >= storage.bit_count:
                break
            if len(events) >= max_events:
                raise ValueError("Bernoulli event count exceeds prespecified feasibility cap")
            events.append(FaultEvent(None, storage.bit_at(ordinal)))
    return FaultSchedule(law, seed, tuple(events), storage.bit_count, inference_count)
