"""Differentiable companion for the pinned MLPerf Tiny ResNet-8 INT8 graph.

The companion is deliberately architecture-specific.  Its parameters are the
dequantized values of the immutable TFLite INT8 kernel buffers, in their exact
TFLite tensor order.  The authoritative clean and faulted inference path
remains ``tf.lite.Interpreter`` with ``BUILTIN_REF``; this module supplies only
a straight-through gradient approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .tflite_storage import TFLiteModelStorage, TFLiteTensorDescriptor


_KERNEL_TENSORS = (8, 9, 10, 11, 12, 13, 14, 15, 16, 7)
_BIAS_TENSORS = (3, 4, 17, 5, 18, 19, 6, 20, 21, 1)
_KERAS_LAYERS = (
    "conv2d", "conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4",
    "conv2d_5", "conv2d_6", "conv2d_7", "conv2d_8", "dense",
)
_BATCH_NORMALIZATION_LAYERS = (
    "batch_normalization", "batch_normalization_1", "batch_normalization_2",
    "batch_normalization_3", "batch_normalization_4", None,
    "batch_normalization_5", "batch_normalization_6", None, None,
)


@dataclass(frozen=True)
class CompanionCorrespondence:
    """Measured clean-output correspondence to authoritative TFLite output."""

    sample_count: int
    exact_code_fraction: float
    exact_output_vector_fraction: float
    class_agreement: float
    probability_mae: float
    probability_max_error: float
    authoritative_accuracy: float | None
    companion_accuracy: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return dict(self.__dict__)


def affine_bit_perturbations(
    values: Mapping[str, np.ndarray],
    descriptors: tuple[TFLiteTensorDescriptor, ...],
) -> dict[str, np.ndarray]:
    """Return exact decoded XOR jumps with a final least-to-most bit axis."""

    by_parameter = {value.parameter: value for value in descriptors}
    if set(values) != set(by_parameter):
        raise ValueError("affine delta values and descriptor populations differ")
    result: dict[str, np.ndarray] = {}
    for parameter, encoded in values.items():
        descriptor = by_parameter[parameter]
        source = np.ascontiguousarray(encoded)
        if descriptor.bit_width != source.dtype.itemsize * 8:
            raise ValueError(f"stored width mismatch for {parameter}")
        unsigned = source.view(np.dtype(f"uint{descriptor.bit_width}"))
        changes = []
        for bit_index in range(descriptor.bit_width):
            flipped = np.bitwise_xor(
                unsigned, np.array(1 << bit_index, dtype=unsigned.dtype)
            ).view(source.dtype)
            changes.append(
                descriptor.quantization.dequantize(flipped)
                - descriptor.quantization.dequantize(source)
            )
        result[parameter] = np.stack(changes, axis=-1)
    return result


class MlperfResnet8DifferentiableCompanion:
    """Straight-through graph derived only from the frozen TFLite constants.

    Forward convolutions use dequantized TFLite kernels and biases.  Every
    activation boundary is requantized with the TFLite scale and zero point;
    ``stop_gradient(quantized - value) + value`` preserves the quantized
    forward value while defining an identity derivative through rounding and
    clipping.  Integer multiplier rounding inside TFLite is not reproduced,
    so output correspondence must be measured rather than assumed.

    Estimator loss is sparse cross-entropy on the companion's requantized
    pre-softmax logits.  Exact empirical damage continues to use the adapter's
    registered integer-softmax loss.
    """

    implementation_version = "mlperf-resnet8-tflite-constants-ste/v1"
    loss_version = "requantized-logit-sparse-crossentropy/v1"
    numerical_dtype = "float32"

    def __init__(self, model_content: bytes):
        import tensorflow as tf

        self.tf = tf
        self.storage = TFLiteModelStorage(
            model_content, eligible_roles=("kernel", "bias")
        )
        kernels = {value.tensor_index: value for value in self.storage.descriptors if value.role == "kernel"}
        biases = {value.tensor_index: value for value in self.storage.descriptors if value.role == "bias"}
        if tuple(_KERNEL_TENSORS) != tuple(index for index in _KERNEL_TENSORS if index in kernels):
            raise ValueError("TFLite graph does not match the pinned ResNet-8 kernel topology")
        if tuple(_BIAS_TENSORS) != tuple(index for index in _BIAS_TENSORS if index in biases):
            raise ValueError("TFLite graph does not match the pinned ResNet-8 bias topology")

        encoded = self.storage.values()
        self.kernel_descriptors = tuple(kernels[index] for index in _KERNEL_TENSORS)
        self.bias_descriptors = tuple(biases[index] for index in _BIAS_TENSORS)
        self.variables = {
            value.parameter: tf.Variable(
                value.quantization.dequantize(encoded[value.parameter]).astype(self.numerical_dtype),
                trainable=True,
                name=f"netsurf_{value.tensor_index}_kernel",
            )
            for value in self.kernel_descriptors
        }
        self.biases = {
            value.tensor_index: tf.constant(
                value.quantization.dequantize(encoded[value.parameter]).astype(self.numerical_dtype)
            )
            for value in self.bias_descriptors
        }

        interpreter = tf.lite.Interpreter(
            model_content=model_content,
            experimental_delegates=[],
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
            num_threads=1,
        )
        interpreter.allocate_tensors()
        self.tensor_details = {int(value["index"]): value for value in interpreter.get_tensor_details()}
        expected_shapes = {
            0: (1, 32, 32, 3), 22: (1, 32, 32, 16), 23: (1, 32, 32, 16),
            24: (1, 32, 32, 16), 25: (1, 32, 32, 16), 26: (1, 16, 16, 32),
            27: (1, 16, 16, 32), 28: (1, 16, 16, 32), 29: (1, 16, 16, 32),
            30: (1, 8, 8, 64), 31: (1, 8, 8, 64), 32: (1, 8, 8, 64),
            33: (1, 8, 8, 64), 34: (1, 1, 1, 64), 35: (1, 64),
            36: (1, 10), 37: (1, 10),
        }
        for index, shape in expected_shapes.items():
            observed = tuple(int(value) for value in self.tensor_details[index]["shape"])
            if observed != shape:
                raise ValueError(f"unexpected TFLite tensor {index} shape: {observed} != {shape}")

    def _quantization(self, tensor_index: int) -> tuple[float, float]:
        metadata = self.tensor_details[tensor_index]["quantization_parameters"]
        scales = np.asarray(metadata["scales"])
        zero_points = np.asarray(metadata["zero_points"])
        if scales.shape != (1,) or zero_points.shape != (1,):
            raise ValueError(f"activation tensor {tensor_index} is not per-tensor affine")
        return float(scales[0]), float(zero_points[0])

    def _requantize_ste(self, values, tensor_index: int):
        tf = self.tf
        scale, zero_point = self._quantization(tensor_index)
        quantized = (
            tf.clip_by_value(tf.round(values / scale + zero_point), -128.0, 127.0)
            - zero_point
        ) * scale
        return values + tf.stop_gradient(quantized - values)

    def _kernel(self, tensor_index: int):
        descriptor = next(
            value for value in self.kernel_descriptors if value.tensor_index == tensor_index
        )
        return self.variables[descriptor.parameter]

    def _conv(self, inputs, kernel_index: int, bias_index: int, stride: int, output_index: int, relu: bool):
        tf = self.tf
        # TFLite kernels are OHWI; TensorFlow conv2d consumes HWIO.
        kernel = tf.transpose(self._kernel(kernel_index), (1, 2, 3, 0))
        values = tf.nn.conv2d(
            inputs, kernel, strides=(1, stride, stride, 1), padding="SAME"
        ) + self.biases[bias_index]
        if relu:
            values = tf.nn.relu(values)
        return self._requantize_ste(values, output_index)

    def _add_relu(self, left, right, output_index: int):
        return self._requantize_ste(self.tf.nn.relu(left + right), output_index)

    def forward(self, encoded_inputs: np.ndarray):
        """Return requantized logits and output probabilities."""

        tf = self.tf
        values = tf.convert_to_tensor(encoded_inputs, dtype=tf.int8)
        if values.shape.rank != 4 or tuple(values.shape[1:]) != (32, 32, 3):
            raise ValueError("companion inputs must be int8 NHWC (N,32,32,3)")
        scale, zero_point = self._quantization(0)
        values = (tf.cast(values, self.numerical_dtype) - zero_point) * scale

        residual = self._conv(values, 8, 3, 1, 22, True)
        branch = self._conv(residual, 9, 4, 1, 23, True)
        branch = self._conv(branch, 10, 17, 1, 24, False)
        values = self._add_relu(residual, branch, 25)

        branch = self._conv(values, 11, 5, 2, 26, True)
        branch = self._conv(branch, 12, 18, 1, 27, False)
        residual = self._conv(values, 13, 19, 2, 28, False)
        values = self._add_relu(residual, branch, 29)

        branch = self._conv(values, 14, 6, 2, 30, True)
        branch = self._conv(branch, 15, 20, 1, 31, False)
        residual = self._conv(values, 16, 21, 2, 32, False)
        values = self._add_relu(residual, branch, 33)

        values = self._requantize_ste(
            tf.reduce_mean(values, axis=(1, 2), keepdims=True), 34
        )
        values = tf.reshape(values, (-1, 64))
        logits = tf.linalg.matmul(values, self._kernel(7), transpose_b=True) + self.biases[1]
        logits = self._requantize_ste(logits, 36)
        probabilities = self._requantize_ste(tf.nn.softmax(logits), 37)
        return logits, probabilities

    def losses(self, encoded_inputs: np.ndarray, labels: np.ndarray):
        logits, _ = self.forward(encoded_inputs)
        labels_tensor = self.tf.convert_to_tensor(labels, dtype=self.tf.int64)
        return self.tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels_tensor, logits=logits
        )

    def gradients(self, encoded_inputs: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
        tf = self.tf
        ordered = tuple(self.variables)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(self.losses(encoded_inputs, labels))
        gradients = tape.gradient(loss, [self.variables[name] for name in ordered])
        if any(value is None for value in gradients):
            raise RuntimeError("companion gradient is disconnected from a TFLite kernel")
        return {name: value.numpy() for name, value in zip(ordered, gradients)}

    def per_example_gradients(self, encoded_inputs: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
        """Compute per-example gradients without TensorFlow's pfor retracing path."""

        ordered = tuple(self.variables)
        rows = {name: [] for name in ordered}
        for sample, label in zip(np.asarray(encoded_inputs), np.asarray(labels)):
            gradients = self.gradients(sample[None, ...], np.asarray([label]))
            for name in ordered:
                rows[name].append(gradients[name])
        return {name: np.stack(values, axis=0) for name, values in rows.items()}

    def encoded_probabilities(self, encoded_inputs: np.ndarray) -> np.ndarray:
        _, probabilities = self.forward(encoded_inputs)
        scale, zero_point = self._quantization(37)
        return np.clip(
            np.rint(probabilities.numpy() / scale + zero_point), -128, 127
        ).astype(np.int8)

    def manifest(self) -> dict[str, object]:
        return {
            "implementation_version": self.implementation_version,
            "loss_version": self.loss_version,
            "parameter_source": "dequantized immutable TFLite INT8 kernel buffers",
            "kernel_tensor_indices": list(_KERNEL_TENSORS),
            "bias_source": "dequantized immutable TFLite INT32 bias buffers; not fault eligible in M9",
            "straight_through_rule": "identity derivative through activation round and clip",
            "authoritative_inference": False,
            "limitations": [
                "does not reproduce TFLite fixed-point multiplier rounding",
                "gradient loss uses requantized pre-softmax logits",
                "must be validated against exact TFLite empirical bit damage",
            ],
        }

    def keras_kernel_correspondence(self, hdf5_checkpoint) -> tuple[dict[str, object], ...]:
        """Compare TFLite kernels with the official HDF5 graph after BN folding.

        This is a provenance diagnostic only.  The HDF5 values are never used
        by the companion forward or gradient path.
        """

        model = self.tf.keras.models.load_model(hdf5_checkpoint, compile=False)
        rows = []
        for descriptor, layer_name, batch_norm_name in zip(
            self.kernel_descriptors, _KERAS_LAYERS, _BATCH_NORMALIZATION_LAYERS
        ):
            expected = np.asarray(model.get_layer(layer_name).get_weights()[0], dtype=np.float64)
            folding = "none"
            if batch_norm_name is not None:
                layer = model.get_layer(batch_norm_name)
                gamma, _beta, _mean, variance = layer.get_weights()
                factor = gamma / np.sqrt(variance + layer.epsilon)
                expected = expected * factor.reshape((1, 1, 1, -1))
                folding = f"{batch_norm_name}: kernel * gamma/sqrt(moving_variance+epsilon)"
            if layer_name == "dense":
                expected = expected.T
                layout = "Keras IO -> TFLite OI"
            else:
                expected = np.transpose(expected, (3, 0, 1, 2))
                layout = "Keras HWIO -> TFLite OHWI"
            observed = self.variables[descriptor.parameter].numpy().astype(np.float64)
            error = observed - expected
            rows.append(
                {
                    "tensor_index": descriptor.tensor_index,
                    "parameter": descriptor.parameter,
                    "keras_layer": layer_name,
                    "layout_transform": layout,
                    "batch_normalization_folding": folding,
                    "scalar_count": descriptor.size,
                    "maximum_absolute_error": float(np.max(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(error * error))),
                    "relative_l2_error": float(np.linalg.norm(error) / np.linalg.norm(expected)),
                    "pearson_correlation": float(
                        np.corrcoef(observed.reshape(-1), expected.reshape(-1))[0, 1]
                    ),
                }
            )
        return tuple(rows)


class MlperfResnet8ContinuousCompanion(MlperfResnet8DifferentiableCompanion):
    """Separate float64 decoded-weight loss model for a classical FKeras HVP.

    Activation rounding and clipping are absent.  The authoritative faulted
    inference remains TFLite BUILTIN_REF; this is not its integer Hessian.
    """

    implementation_version = "mlperf-resnet8-continuous-float64/v1"
    loss_version = "continuous-logit-sparse-crossentropy/v1"
    numerical_dtype = "float64"

    def _requantize_ste(self, values, tensor_index: int):
        return values

    def manifest(self) -> dict[str, object]:
        return {
            "implementation_version": self.implementation_version,
            "loss_version": self.loss_version,
            "parameter_source": "exact affine decoded frozen INT8 kernel codes",
            "kernel_tensor_indices": list(_KERNEL_TENSORS),
            "bias_source": "fixed decoded INT32 constants; neither differentiated nor fault eligible",
            "activation_quantization": "removed in this separate continuous extension",
            "dtype": "float64",
            "authoritative_inference": False,
            "limitations": ["not a Hessian of integer TFLite inference or the STE derivative operator", "ReLU kinks have no classical Hessian"],
        }
