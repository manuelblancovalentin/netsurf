"""TFLite constant-buffer inspection and exact private-copy XOR patching.

TensorFlow is imported lazily only for inference. Flatbuffer parsing uses the
schema shipped with the installed TensorFlow runtime and never imports legacy
NetSuRF, QKeras, or tensorflow-model-optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np

from .affine_quantization import AffineQuantization, affine_bit_transition
from .faults import FaultSchedule, InjectionReport
from .identities import BitIdentity, ParameterSpec
from .protection import ProtectionMask


def _schema():
    from tensorflow.lite.python import schema_py_generated

    return schema_py_generated


@dataclass(frozen=True)
class TFLiteTensorDescriptor:
    parameter: str
    tensor_index: int
    buffer_index: int
    original_name: str
    role: str
    shape: tuple[int, ...]
    dtype: str
    bit_width: int
    quantization: AffineQuantization

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    def to_dict(self) -> dict[str, object]:
        return {
            "parameter": self.parameter,
            "tensor_index": self.tensor_index,
            "buffer_index": self.buffer_index,
            "original_name": self.original_name,
            "role": self.role,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "bit_width": self.bit_width,
            "quantization": self.quantization.to_dict(),
        }


class TFLiteModelStorage:
    """Immutable reference bytes plus explicit faultable constant buffers."""

    def __init__(self, model_content: bytes, *, expected_sha256: str | None = None):
        self.reference_content = bytes(model_content)
        self.sha256 = hashlib.sha256(self.reference_content).hexdigest()
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError(
                f"TFLite checkpoint SHA-256 mismatch: expected {expected_sha256}, observed {self.sha256}"
            )
        self.descriptors = self._inspect(self.reference_content)
        self._by_parameter = {value.parameter: value for value in self.descriptors}
        if len(self._by_parameter) != len(self.descriptors):
            raise RuntimeError("TFLite parameter identities are not unique")

    @classmethod
    def from_path(cls, path: Path, *, expected_sha256: str | None = None) -> "TFLiteModelStorage":
        return cls(Path(path).read_bytes(), expected_sha256=expected_sha256)

    @staticmethod
    def _inspect(model_content: bytes) -> tuple[TFLiteTensorDescriptor, ...]:
        schema = _schema()
        model = schema.Model.GetRootAsModel(model_content, 0)
        if model.SubgraphsLength() != 1:
            raise ValueError("only single-subgraph TFLite models are supported")
        subgraph = model.Subgraphs(0)
        roles: dict[int, str] = {}
        supported_ops = {
            schema.BuiltinOperator.CONV_2D,
            schema.BuiltinOperator.DEPTHWISE_CONV_2D,
            schema.BuiltinOperator.FULLY_CONNECTED,
        }
        for operation_index in range(subgraph.OperatorsLength()):
            operation = subgraph.Operators(operation_index)
            opcode = model.OperatorCodes(operation.OpcodeIndex()).BuiltinCode()
            if opcode not in supported_ops or operation.InputsLength() < 2:
                continue
            inputs = operation.InputsAsNumpy()
            for position, role in ((1, "kernel"), (2, "bias")):
                if position >= len(inputs) or int(inputs[position]) < 0:
                    continue
                tensor_index = int(inputs[position])
                previous = roles.get(tensor_index)
                if previous is not None and previous != role:
                    raise ValueError(f"tensor {tensor_index} is used as both {previous} and {role}")
                roles[tensor_index] = role

        result: list[TFLiteTensorDescriptor] = []
        for tensor_index, role in sorted(roles.items()):
            tensor = subgraph.Tensors(tensor_index)
            buffer_index = int(tensor.Buffer())
            data = model.Buffers(buffer_index).DataAsNumpy()
            if isinstance(data, int) or data.size == 0:
                raise ValueError(f"parameter tensor {tensor_index} has no immutable constant buffer")
            if role == "kernel" and tensor.Type() == schema.TensorType.INT8:
                dtype, width = "int8", 8
            elif role == "bias" and tensor.Type() == schema.TensorType.INT32:
                dtype, width = "int32", 32
            else:
                # Unsupported numerical state is excluded, never coerced.
                continue
            quantization = tensor.Quantization()
            if quantization is None or quantization.ScaleLength() == 0:
                raise ValueError(f"parameter tensor {tensor_index} lacks affine metadata")
            scales = tuple(float(value) for value in quantization.ScaleAsNumpy())
            zero_points = tuple(int(value) for value in quantization.ZeroPointAsNumpy())
            original_name = tensor.Name().decode("utf-8")
            result.append(
                TFLiteTensorDescriptor(
                    parameter=f"tflite_tensor_{tensor_index:04d}/{role}",
                    tensor_index=tensor_index,
                    buffer_index=buffer_index,
                    original_name=original_name,
                    role=role,
                    shape=tuple(int(value) for value in tensor.ShapeAsNumpy()),
                    dtype=dtype,
                    bit_width=width,
                    quantization=AffineQuantization(
                        scales, zero_points, int(quantization.QuantizedDimension())
                    ),
                )
            )
        if not result:
            raise ValueError("no supported INT8 kernels or INT32 biases were found")
        return tuple(result)

    def _model(self, content):
        return _schema().Model.GetRootAsModel(content, 0)

    @staticmethod
    def _decode_buffer(raw: np.ndarray, descriptor: TFLiteTensorDescriptor) -> np.ndarray:
        if descriptor.dtype == "int8":
            values = np.asarray(raw, dtype=np.uint8).view(np.int8)
        else:
            values = np.frombuffer(np.asarray(raw, dtype=np.uint8).tobytes(), dtype="<i4")
        if values.size != descriptor.size:
            raise ValueError(
                f"buffer size mismatch for {descriptor.parameter}: {values.size} != {descriptor.size}"
            )
        return values.reshape(descriptor.shape).copy()

    def values(self, content: bytes | bytearray | None = None) -> dict[str, np.ndarray]:
        source = self.reference_content if content is None else content
        model = self._model(source)
        return {
            descriptor.parameter: self._decode_buffer(
                model.Buffers(descriptor.buffer_index).DataAsNumpy(), descriptor
            )
            for descriptor in self.descriptors
        }

    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return tuple(ParameterSpec(value.parameter, value.shape) for value in self.descriptors)

    def bit_identities(self) -> tuple[BitIdentity, ...]:
        return tuple(
            BitIdentity(descriptor.parameter, flat_index, bit_index)
            for descriptor in self.descriptors
            for flat_index in range(descriptor.size)
            for bit_index in range(descriptor.bit_width)
        )

    def model_content_from_values(self, values: Mapping[str, np.ndarray]) -> bytes:
        if set(values) != set(self._by_parameter):
            raise ValueError("TFLite state keys differ from the eligible constant-buffer population")
        mutable = bytearray(self.reference_content)
        model = self._model(mutable)
        for parameter, descriptor in self._by_parameter.items():
            value = np.asarray(values[parameter])
            expected_dtype = np.dtype(descriptor.dtype)
            if value.dtype != expected_dtype or tuple(value.shape) != descriptor.shape:
                raise ValueError(
                    f"state mismatch for {parameter}: {value.dtype}/{value.shape} != "
                    f"{expected_dtype}/{descriptor.shape}"
                )
            encoded = np.ascontiguousarray(
                value.astype("<i4", copy=False) if descriptor.dtype == "int32" else value
            ).tobytes()
            destination = model.Buffers(descriptor.buffer_index).DataAsNumpy()
            if isinstance(destination, int) or destination.size != len(encoded):
                raise ValueError(f"flatbuffer storage size mismatch for {parameter}")
            destination[:] = np.frombuffer(encoded, dtype=np.uint8)
        return bytes(mutable)

    def bit_transition(
        self, values: Mapping[str, np.ndarray], identity: BitIdentity
    ) -> tuple[int, int, float, float, dict[str, np.ndarray]]:
        descriptor = self._by_parameter[identity.parameter]
        original, flipped, original_value, flipped_value, changed_tensor = affine_bit_transition(
            values[identity.parameter],
            descriptor.quantization,
            identity.flat_index,
            identity.bit_index,
        )
        changed = {name: np.asarray(value).copy() for name, value in values.items()}
        changed[identity.parameter] = changed_tensor
        return original, flipped, original_value, flipped_value, changed

    def apply_schedule(
        self,
        schedule: FaultSchedule,
        protection: ProtectionMask,
        *,
        inference_index: int | None = None,
    ) -> tuple[bytes, InjectionReport]:
        if schedule.model.persistence == "persistent_trial":
            if inference_index is not None:
                raise ValueError("persistent schedules do not take an inference index")
            events = schedule.events
        elif schedule.model.persistence == "per_inference":
            if inference_index is None:
                raise ValueError("per-inference schedules require an inference index")
            events = tuple(
                event for event in schedule.events if event.inference_index == inference_index
            )
        else:
            raise ValueError(f"unsupported persistence: {schedule.model.persistence}")

        mutable = bytearray(self.reference_content)
        model = self._model(mutable)
        applied: list[BitIdentity] = []
        suppressed = 0
        for event in events:
            identity = event.bit
            if identity in protection.protected:
                suppressed += 1
                continue
            descriptor = self._by_parameter.get(identity.parameter)
            if descriptor is None:
                raise KeyError(f"unknown TFLite fault target: {identity.parameter}")
            if not 0 <= identity.flat_index < descriptor.size:
                raise IndexError(identity)
            if not 0 <= identity.bit_index < descriptor.bit_width:
                raise IndexError(identity)
            destination = model.Buffers(descriptor.buffer_index).DataAsNumpy()
            byte_index = identity.flat_index * (descriptor.bit_width // 8) + identity.bit_index // 8
            destination[byte_index] ^= np.uint8(1 << (identity.bit_index % 8))
            applied.append(identity)
        return bytes(mutable), InjectionReport(
            schedule.digest(), len(events), len(applied), suppressed, len(set(applied))
        )

    def manifest(self) -> dict[str, object]:
        values = self.values()
        return {
            "storage_format": "TFLite immutable constant buffers",
            "reference_sha256": self.sha256,
            "eligible_tensor_count": len(self.descriptors),
            "eligible_scalar_count": sum(value.size for value in self.descriptors),
            "eligible_bit_count": len(self.bit_identities()),
            "zero_valued_encoded_scalars": sum(
                int(np.count_nonzero(values[value.parameter] == 0)) for value in self.descriptors
            ),
            "roles": {
                role: sum(1 for value in self.descriptors if value.role == role)
                for role in ("kernel", "bias")
            },
            "tensors": [value.to_dict() for value in self.descriptors],
            "excluded": [
                "activation buffers",
                "tensor scales",
                "tensor zero points",
                "operator/control metadata",
                "non-parameter constants",
                "unsupported constant dtypes",
            ],
        }
