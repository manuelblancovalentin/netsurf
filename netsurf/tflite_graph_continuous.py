"""Continuous float64 version of the generic TFLite graph companion, for Hessian-based scores (FKeras).

Same graph walk as ``TFLiteGraphCompanion``, with decoded kernels and biases in float64 and NO activation rounding or
clipping, as in the pinned continuous companions of the DS-CNN, ResNet-8 and ToyCar (``Mlperf*ContinuousCompanion``).
The loss is piecewise smooth (ReLU kinks remain).  It never computes faulted outcomes.
"""
from __future__ import annotations

import numpy as np

from .tflite_graph_companion import TFLiteGraphCompanion


class TFLiteGraphContinuousCompanion(TFLiteGraphCompanion):
    implementation_version = "tflite-graph-continuous-float64/v1"
    loss_version = "continuous-logit-sparse-crossentropy/v1"
    numerical_dtype = "float64"

    def _decode_input(self, encoded_inputs):
        idx = self.inputs[0]
        shape = self.tensors[idx]["shape"][1:]
        values = self.tf.convert_to_tensor(np.asarray(encoded_inputs), dtype=self.tf.int8)
        if tuple(values.shape[1:]) != tuple(shape):
            raise ValueError(f"companion inputs must be int8 with trailing shape {shape}")
        scale, zp = self._qparams(idx)
        return (self.tf.cast(values, self.tf.float64) - zp) * scale

    def _requantize_ste(self, values, tensor_index):
        return values


class GraphHessian:
    """Hessian-vector products of the mean continuous loss over fixed calibration inputs, flattened in ``names`` order."""

    def __init__(self, model_content: bytes, names, inputs, labels, batch: int = 256):
        self.companion = TFLiteGraphContinuousCompanion(model_content)
        tf = self.companion.tf
        self.names = tuple(names)
        if set(self.names) != set(self.companion.variables):
            raise ValueError("Hessian names must be exactly the companion kernels")
        self.variables = tuple(self.companion.variables[n] for n in self.names)
        self.shapes = tuple(tuple(v.shape) for v in self.variables)
        self.sizes = tuple(int(np.prod(s)) for s in self.shapes)
        self.theta = np.concatenate([v.numpy().reshape(-1) for v in self.variables])
        x, y = np.asarray(inputs), np.asarray(labels).reshape(-1)
        self.batches = [(x[s:s + batch], y[s:s + batch]) for s in range(0, len(y), batch)]
        self.count = len(y)

        def make(xb, yb):                   # one traced function per fixed calibration batch (inputs are constants)
            @tf.function(input_signature=[tf.TensorSpec((len(self.theta),), tf.float64)], autograph=False)
            def hvp_batch(flat):
                parts = [tf.reshape(v, s) for v, s in zip(tf.split(flat, self.sizes), self.shapes)]
                with tf.GradientTape() as outer:
                    with tf.GradientTape() as inner:
                        loss = tf.reduce_sum(self.companion.losses(xb, yb))
                    g = inner.gradient(loss, self.variables)
                    prod = tf.add_n([tf.reduce_sum(a * b) for a, b in zip(g, parts)])
                r = outer.gradient(prod, self.variables)
                return tf.concat([tf.reshape(v, (-1,)) for v in r], axis=0)
            return hvp_batch

        self._hvps = [make(xb, yb) for xb, yb in self.batches]

    def __call__(self, vector):
        v = self.companion.tf.convert_to_tensor(np.asarray(vector, dtype=np.float64))
        total = np.zeros(len(self.theta))
        for f in self._hvps:
            total += f(v).numpy()
        return total / self.count
