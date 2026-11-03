"""Exact BF16 storage faults, with explicitly widened FP32 execution.

The private BF16 words are the fault state. FP32 parameters are a computation
cache, never a second fault population. Widening is the exact bit mapping
``u16 -> u32 << 16``, including signed zero, subnormals, infinity and NaN
payloads. This is NOT native BF16 arithmetic.
"""
from contextlib import contextmanager
import hashlib
import sys

from .storage_bit_codec import StorageBitCodec
from .torch_storage import TorchStoredLinearWeights, TorchParameterDescriptor


def widen_bf16_bytes(raw: bytes) -> bytes:
    if len(raw) % 2:
        raise ValueError("BF16 storage must contain complete 16-bit words")
    import numpy as np
    return (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).tobytes()


class BF16StorageFP32Compute(TorchStoredLinearWeights):
    """Transactional XOR on verified BF16 words and exact FP32 consumption.

    ``model`` must already be loaded in FP32 using the validated reference
    loading path. ``stored_tensors`` are read from the pinned BF16 checkpoint,
    not obtained by rounding potentially different live FP32 weights.
    """
    execution_contract = "bf16-storage-fp32-compute/v1"

    def __init__(self, model, stored_tensors, *, eligible_bits=None):
        import torch
        if sys.byteorder != "little":
            raise ValueError("little-endian storage backend required")
        # Reuse alias, layout and Linear eligibility checks, not FP32 faults.
        super().__init__(model, "FP32")
        self.runtime_parameters = self.parameters
        self.parameters = {}
        self.codec = StorageBitCodec("BF16")
        for name, live in self.runtime_parameters.items():
            source = stored_tensors.get(name)
            if source is None or source.dtype != torch.bfloat16 or source.shape != live.shape:
                raise ValueError(f"missing/mismatched checkpoint BF16 tensor: {name}")
            if self._bytes(live) != widen_bf16_bytes(self._bytes(source)):
                raise ValueError(f"runtime is not exact BF16-to-FP32 widening: {name}")
            self.parameters[name] = source.detach().clone().contiguous()
        self.descriptors = tuple(TorchParameterDescriptor(name, tuple(value.shape),
            str(value.dtype), hashlib.sha256(self._bytes(value)).hexdigest())
            for name, value in self.parameters.items())
        self._eligible_bits = None
        self._eligible_set = None
        self.bit_count = sum(item.scalars * 16 for item in self.descriptors)
        if eligible_bits is not None:
            if not eligible_bits or len(set(eligible_bits)) != len(eligible_bits):
                raise ValueError("restricted population must be nonempty and unique")
            for bit in eligible_bits:
                self._validate_stored_coordinate(bit)
            self._eligible_bits = tuple(eligible_bits)
            self._eligible_set = frozenset(eligible_bits)
            self.bit_count = len(eligible_bits)
        self._active = False
        self._clean_stored = self.hashes()
        self._clean_runtime = self.runtime_hashes()

    def runtime_hashes(self):
        return {name: hashlib.sha256(self._bytes(value)).hexdigest()
                for name, value in self.runtime_parameters.items()}

    @contextmanager
    def injected(self, schedule, mask):
        import torch
        if self._active:
            raise RuntimeError("nested storage transactions are unsupported")
        if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
            raise RuntimeError("autocast violates the FP32 computation contract")
        if self.hashes() != self._clean_stored or self.runtime_hashes() != self._clean_runtime:
            raise RuntimeError("stored or computation state changed outside a fault transaction")
        self._active = True
        original = {}
        pointers = {name: p.untyped_storage().data_ptr() for name, p in self.runtime_parameters.items()}
        try:
            with super().injected(schedule, mask) as report:
                words = sorted({(e.bit.parameter, e.bit.flat_index) for e in schedule.events
                                if e.bit not in mask.protected})
                with torch.no_grad():
                    for name, index in words:
                        live = self._byte_view(self.runtime_parameters[name])[4 * index:4 * index + 4]
                        original[(name, index)] = live.clone()
                        word = self._byte_view(self.parameters[name])[2 * index:2 * index + 2].numpy().tobytes()
                        expected = widen_bf16_bytes(word)
                        live.copy_(torch.frombuffer(bytearray(expected), dtype=torch.uint8))
                        if live.numpy().tobytes() != expected:
                            raise RuntimeError("BF16 XOR state did not reach FP32 inference")
                yield report
        finally:
            with torch.no_grad():
                for (name, index), raw in original.items():
                    self._byte_view(self.runtime_parameters[name])[4 * index:4 * index + 4].copy_(raw)
            self._active = False
            if any(p.dtype != torch.float32 or p.untyped_storage().data_ptr() != pointers[name]
                   for name, p in self.runtime_parameters.items()):
                raise RuntimeError("FP32 parameter representation changed during inference")
            if self.hashes() != self._clean_stored or self.runtime_hashes() != self._clean_runtime:
                raise RuntimeError("BF16/FP32 byte restoration failed")
