"""Architecture-checked differentiable companions for pinned MLPerf Tiny models.

The classes in this module are ranking surrogates.  They reconstruct their
parameters exclusively from immutable TFLite constants and use straight-
through activation requantization.  Exact clean and faulted measurements stay
on TensorFlow Lite ``BUILTIN_REF``.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .tflite_storage import TFLiteModelStorage


class _PinnedSequentialCompanion:
    implementation_version = ""
    loss_version = ""
    input_shape: tuple[int, ...]
    kernel_indices: tuple[int, ...]
    bias_indices: tuple[int, ...]

    def __init__(self, model_content: bytes):
        import tensorflow as tf

        self.tf = tf
        self.storage = TFLiteModelStorage(
            model_content, eligible_roles=("kernel", "bias")
        )
        descriptors = self.storage.descriptors
        self._kernel_descriptors = {
            value.tensor_index: value for value in descriptors if value.role == "kernel"
        }
        self._bias_descriptors = {
            value.tensor_index: value for value in descriptors if value.role == "bias"
        }
        if set(self._kernel_descriptors) != set(self.kernel_indices):
            raise ValueError("TFLite kernels do not match the pinned companion topology")
        if set(self._bias_descriptors) != set(self.bias_indices):
            raise ValueError("TFLite biases do not match the pinned companion topology")
        encoded = self.storage.values()
        self.variables = {
            descriptor.parameter: tf.Variable(
                descriptor.quantization.dequantize(encoded[descriptor.parameter]).astype(np.float32),
                trainable=True,
                name=f"netsurf_{descriptor.tensor_index}_kernel",
            )
            for descriptor in (
                self._kernel_descriptors[index] for index in self.kernel_indices
            )
        }
        self.biases = {
            index: tf.constant(
                self._bias_descriptors[index].quantization.dequantize(
                    encoded[self._bias_descriptors[index].parameter]
                ).astype(np.float32)
            )
            for index in self.bias_indices
        }
        interpreter = tf.lite.Interpreter(
            model_content=model_content,
            experimental_delegates=[],
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
            num_threads=1,
        )
        interpreter.allocate_tensors()
        self.tensor_details = {
            int(value["index"]): value for value in interpreter.get_tensor_details()
        }

    def _quantization(self, tensor_index: int) -> tuple[float, float]:
        metadata = self.tensor_details[tensor_index]["quantization_parameters"]
        scales = np.asarray(metadata["scales"])
        zero_points = np.asarray(metadata["zero_points"])
        if scales.shape != (1,) or zero_points.shape != (1,):
            raise ValueError(f"activation tensor {tensor_index} is not per-tensor affine")
        return float(scales[0]), float(zero_points[0])

    def _decode_input(self, encoded_inputs: np.ndarray):
        values = self.tf.convert_to_tensor(encoded_inputs, dtype=self.tf.int8)
        if values.shape.rank != len(self.input_shape) + 1 or tuple(values.shape[1:]) != self.input_shape:
            raise ValueError(
                f"companion inputs must be int8 with trailing shape {self.input_shape}"
            )
        scale, zero_point = self._quantization(0)
        return (self.tf.cast(values, self.tf.float32) - zero_point) * scale

    def _requantize_ste(self, values, tensor_index: int):
        scale, zero_point = self._quantization(tensor_index)
        quantized = (
            self.tf.clip_by_value(
                self.tf.round(values / scale + zero_point), -128.0, 127.0
            )
            - zero_point
        ) * scale
        return values + self.tf.stop_gradient(quantized - values)

    def _kernel(self, tensor_index: int):
        return self.variables[self._kernel_descriptors[tensor_index].parameter]

    def gradients(self, encoded_inputs: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
        ordered = tuple(self.variables)
        with self.tf.GradientTape() as tape:
            loss = self.tf.reduce_mean(self.losses(encoded_inputs, labels))
        gradients = tape.gradient(loss, [self.variables[name] for name in ordered])
        if any(value is None for value in gradients):
            raise RuntimeError("companion gradient is disconnected from a TFLite kernel")
        return {name: value.numpy() for name, value in zip(ordered, gradients)}

    def per_example_gradients(
        self, encoded_inputs: np.ndarray, labels: np.ndarray
    ) -> dict[str, np.ndarray]:
        ordered = tuple(self.variables)
        rows = {name: [] for name in ordered}
        for sample, label in zip(np.asarray(encoded_inputs), np.asarray(labels)):
            gradients = self.gradients(sample[None, ...], label[None, ...])
            for name in ordered:
                rows[name].append(gradients[name])
        return {name: np.stack(values, axis=0) for name, values in rows.items()}

    def manifest(self) -> dict[str, object]:
        return {
            "implementation_version": self.implementation_version,
            "loss_version": self.loss_version,
            "parameter_source": "dequantized immutable TFLite INT8 kernel buffers",
            "kernel_tensor_indices": list(self.kernel_indices),
            "bias_source": "dequantized immutable INT32 constants; excluded from M9.3 fault population",
            "straight_through_rule": "identity derivative through activation round and clip",
            "authoritative_inference": False,
            "limitations": [
                "does not reproduce integer multiplier rounding inside TFLite kernels",
                "supplies ranking derivatives only",
                "requires measured correspondence and exact-bit empirical validation",
            ],
        }


class MlperfKwsDscnnDifferentiableCompanion(_PinnedSequentialCompanion):
    """Pinned DS-CNN: initial convolution, four depthwise/pointwise blocks."""

    implementation_version = "mlperf-kws-dscnn-tflite-constants-ste/v1"
    loss_version = "requantized-logit-sparse-crossentropy/v1"
    input_shape = (49, 10, 1)
    kernel_indices = (17, 5, 18, 8, 19, 11, 20, 14, 21, 16)
    bias_indices = (3, 4, 6, 7, 9, 10, 12, 13, 15, 1)

    def _conv(self, values, kernel_index: int, bias_index: int, output_index: int, *, stride: int = 1, depthwise: bool = False):
        if depthwise:
            # TFLite depthwise constants are 1HWO; TensorFlow consumes HWIM.
            kernel = self.tf.transpose(self._kernel(kernel_index), (1, 2, 3, 0))
            result = self.tf.nn.depthwise_conv2d(
                values, kernel, strides=(1, stride, stride, 1), padding="SAME"
            )
        else:
            kernel = self.tf.transpose(self._kernel(kernel_index), (1, 2, 3, 0))
            result = self.tf.nn.conv2d(
                values, kernel, strides=(1, stride, stride, 1), padding="SAME"
            )
        result = self.tf.nn.relu(result + self.biases[bias_index])
        return self._requantize_ste(result, output_index)

    def forward(self, encoded_inputs: np.ndarray):
        values = self._decode_input(encoded_inputs)
        values = self._conv(values, 17, 3, 22, stride=2)
        for kernel, bias, point_kernel, point_bias, depth_output, point_output in (
            (5, 4, 18, 6, 23, 24),
            (8, 7, 19, 9, 25, 26),
            (11, 10, 20, 12, 27, 28),
            (14, 13, 21, 15, 29, 30),
        ):
            values = self._conv(values, kernel, bias, depth_output, depthwise=True)
            values = self._conv(values, point_kernel, point_bias, point_output)
        values = self._requantize_ste(
            self.tf.reduce_mean(values, axis=(1, 2), keepdims=True), 31
        )
        values = self.tf.reshape(values, (-1, 64))
        logits = self.tf.linalg.matmul(values, self._kernel(16), transpose_b=True) + self.biases[1]
        logits = self._requantize_ste(logits, 33)
        probabilities = self._requantize_ste(self.tf.nn.softmax(logits), 34)
        return logits, probabilities

    def losses(self, encoded_inputs: np.ndarray, labels: np.ndarray):
        logits, _ = self.forward(encoded_inputs)
        labels = self.tf.convert_to_tensor(np.asarray(labels).reshape(-1), dtype=self.tf.int64)
        return self.tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    def encoded_outputs(self, encoded_inputs: np.ndarray) -> np.ndarray:
        _, probabilities = self.forward(encoded_inputs)
        scale, zero_point = self._quantization(34)
        return np.clip(
            np.rint(probabilities.numpy() / scale + zero_point), -128, 127
        ).astype(np.int8)


class MlperfToyCarDifferentiableCompanion(_PinnedSequentialCompanion):
    """Pinned ten-layer dense ToyCar autoencoder."""

    implementation_version = "mlperf-toycar-dense-ae-tflite-constants-ste/v1"
    loss_version = "float-target-mean-squared-reconstruction/v1"
    input_shape = (640,)
    kernel_indices = tuple(range(11, 21))
    bias_indices = tuple(range(1, 11))

    def forward(self, encoded_inputs: np.ndarray):
        values = self._decode_input(encoded_inputs)
        for position, (kernel_index, bias_index, output_index) in enumerate(
            zip(self.kernel_indices, self.bias_indices, range(21, 31))
        ):
            values = self.tf.linalg.matmul(values, self._kernel(kernel_index), transpose_b=True)
            values = values + self.biases[bias_index]
            if position < len(self.kernel_indices) - 1:
                values = self.tf.nn.relu(values)
            values = self._requantize_ste(values, output_index)
        return values

    @staticmethod
    def reconstruction_targets(labels: np.ndarray) -> np.ndarray:
        values = np.asarray(labels)
        if values.ndim != 2 or values.shape[1] < 640:
            raise ValueError("ToyCar labels must include 640 float reconstruction targets")
        return values[:, :640]

    def losses(self, encoded_inputs: np.ndarray, labels: np.ndarray):
        outputs = self.forward(encoded_inputs)
        targets = self.tf.convert_to_tensor(self.reconstruction_targets(labels), dtype=self.tf.float32)
        return self.tf.reduce_mean(self.tf.square(outputs - targets), axis=1)

    def encoded_outputs(self, encoded_inputs: np.ndarray) -> np.ndarray:
        outputs = self.forward(encoded_inputs)
        scale, zero_point = self._quantization(30)
        return np.clip(
            np.rint(outputs.numpy() / scale + zero_point), -128, 127
        ).astype(np.int8)
