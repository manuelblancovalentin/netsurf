"""Straight-through differentiable companion built by walking an INT8 TFLite graph (gradient-trust plan,
review/GRADIENT_TRUST_PLAN_2026-11-15.md).

The pinned companions (ResNet-8, DS-CNN, ToyCar) are written by hand for one topology each.  This one reads the
operators of any single-subgraph, full-integer INT8 TFLite model built from the supported operators below and builds
the same kind of forward pass: dequantized kernels (trainable variables, one per INT8 kernel tensor, named as
``TFLiteModelStorage`` names them), dequantized INT32 biases (constants), and every activation tensor requantized with
its own scale and zero point through ``value + stop_gradient(quantized - value)`` (identity derivative through
rounding and clipping).  Integer multiplier rounding inside TFLite is not reproduced; agreement with the reference
interpreter must be measured, never assumed.

Supported operators: QUANTIZE, DEQUANTIZE (graph boundaries), FULLY_CONNECTED, CONV_2D, DEPTHWISE_CONV_2D,
AVERAGE_POOL_2D, MAX_POOL_2D, MEAN (over height and width), ADD, RESHAPE, SOFTMAX; fused NONE, RELU, RELU6.
Anything else raises.  The logits are the input of the final SOFTMAX; the loss is sparse cross-entropy on them (the
pinned companions' loss).  Side effect free; TensorFlow is imported when an instance is built.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .tflite_storage import TFLiteModelStorage, _schema


class UnsupportedGraph(ValueError):
    pass


class TFLiteGraphCompanion:
    implementation_version = "tflite-graph-ste/v1"
    loss_version = "requantized-logit-sparse-crossentropy/v1"
    numerical_dtype = "float32"

    def __init__(self, model_content: bytes):
        import tensorflow as tf

        self.tf = tf
        self.model_content = bytes(model_content)
        self.storage = TFLiteModelStorage(self.model_content, eligible_roles=("kernel", "bias"))
        encoded = self.storage.values()
        by_index = {d.tensor_index: d for d in self.storage.descriptors}
        self.variables = {}
        self.constants = {}
        for d in self.storage.descriptors:
            value = d.quantization.dequantize(encoded[d.parameter]).astype(self.numerical_dtype)
            if d.role == "kernel":
                self.variables[d.parameter] = tf.Variable(value, trainable=True, name=f"netsurf_{d.tensor_index}_kernel")
            else:
                self.constants[d.tensor_index] = tf.constant(value)
        self._kernel_name = {i: d.parameter for i, d in by_index.items() if d.role == "kernel"}
        self._parse()

    # ---- graph -------------------------------------------------------------------------------------------------
    def _parse(self):
        schema = _schema()
        model = schema.Model.GetRootAsModel(self.model_content, 0)
        if model.SubgraphsLength() != 1:
            raise UnsupportedGraph("only single-subgraph models are supported")
        sg = model.Subgraphs(0)
        B = schema.BuiltinOperator
        names = {v: k for k, v in vars(B).items() if not k.startswith("_")}
        self.tensors = {}
        for i in range(sg.TensorsLength()):
            t = sg.Tensors(i)
            q = t.Quantization()
            scales = q.ScaleAsNumpy() if q is not None and q.ScaleLength() else None
            zps = q.ZeroPointAsNumpy() if q is not None and q.ZeroPointLength() else None
            buf = model.Buffers(t.Buffer()).DataAsNumpy()
            self.tensors[i] = dict(shape=tuple(int(s) for s in t.ShapeAsNumpy()) if t.ShapeLength() else (),
                                   type=int(t.Type()), scales=None if scales is None else np.asarray(scales, np.float64),
                                   zero_points=None if zps is None else np.asarray(zps, np.int64),
                                   data=None if isinstance(buf, int) or buf.size == 0 else np.asarray(buf, np.uint8))
        self.inputs = [int(v) for v in sg.InputsAsNumpy()]
        self.outputs = [int(v) for v in sg.OutputsAsNumpy()]
        if len(self.inputs) != 1 or len(self.outputs) != 1:
            raise UnsupportedGraph("one input and one output are required")
        self.ops = []
        for k in range(sg.OperatorsLength()):
            op = sg.Operators(k)
            code = model.OperatorCodes(op.OpcodeIndex())
            opcode = int(code.BuiltinCode())
            dep = int(code.DeprecatedBuiltinCode())
            if opcode == B.ADD and dep != opcode:
                opcode = dep
            name = names.get(opcode, str(opcode))
            ins = [int(v) for v in op.InputsAsNumpy()]
            outs = [int(v) for v in op.OutputsAsNumpy()]
            opts = self._options(schema, name, op)
            self.ops.append(dict(name=name, inputs=ins, outputs=outs, options=opts))
        softmax = [o for o in self.ops if o["name"] == "SOFTMAX"]
        if len(softmax) != 1:
            raise UnsupportedGraph("exactly one SOFTMAX is required (classification models)")
        self.logits_tensor = softmax[0]["inputs"][0]

    @staticmethod
    def _options(schema, name, op):
        table = op.BuiltinOptions()

        def load(cls):
            o = cls()
            o.Init(table.Bytes, table.Pos)
            return o
        if name == "CONV_2D":
            o = load(schema.Conv2DOptions)
            if o.DilationWFactor() != 1 or o.DilationHFactor() != 1:
                raise UnsupportedGraph("dilated convolutions are not supported")
            return dict(padding=o.Padding(), stride=(o.StrideH(), o.StrideW()), act=o.FusedActivationFunction())
        if name == "DEPTHWISE_CONV_2D":
            o = load(schema.DepthwiseConv2DOptions)
            if o.DilationWFactor() != 1 or o.DilationHFactor() != 1:
                raise UnsupportedGraph("dilated convolutions are not supported")
            return dict(padding=o.Padding(), stride=(o.StrideH(), o.StrideW()), act=o.FusedActivationFunction(),
                        multiplier=o.DepthMultiplier())
        if name == "FULLY_CONNECTED":
            o = load(schema.FullyConnectedOptions)
            return dict(act=o.FusedActivationFunction(), keep_dims=bool(o.KeepNumDims()))
        if name in ("AVERAGE_POOL_2D", "MAX_POOL_2D"):
            o = load(schema.Pool2DOptions)
            return dict(padding=o.Padding(), stride=(o.StrideH(), o.StrideW()),
                        filter=(o.FilterHeight(), o.FilterWidth()), act=o.FusedActivationFunction())
        if name == "ADD":
            o = load(schema.AddOptions)
            return dict(act=o.FusedActivationFunction())
        if name == "MEAN":
            o = load(schema.ReducerOptions)
            return dict(keep_dims=bool(o.KeepDims()))
        if name == "SOFTMAX":
            o = load(schema.SoftmaxOptions)
            return dict(beta=o.Beta())
        if name in ("RESHAPE", "QUANTIZE", "DEQUANTIZE"):
            return {}
        raise UnsupportedGraph(f"operator {name} is not supported")

    # ---- numerics ----------------------------------------------------------------------------------------------
    def _qparams(self, tensor_index):
        t = self.tensors[tensor_index]
        if t["scales"] is None or t["scales"].shape != (1,):
            raise UnsupportedGraph(f"activation tensor {tensor_index} is not per-tensor affine")
        return float(t["scales"][0]), float(t["zero_points"][0])

    def _requantize_ste(self, values, tensor_index):
        t = self.tensors[tensor_index]
        if t["type"] != _schema().TensorType.INT8:
            return values
        scale, zp = self._qparams(tensor_index)
        q = (self.tf.clip_by_value(self.tf.round(values / scale + zp), -128.0, 127.0) - zp) * scale
        return values + self.tf.stop_gradient(q - values)

    def _act(self, values, code):
        A = _schema().ActivationFunctionType
        if code == A.NONE:
            return values
        if code == A.RELU:
            return self.tf.nn.relu(values)
        if code == A.RELU6:
            return self.tf.clip_by_value(values, 0.0, 6.0)
        raise UnsupportedGraph(f"fused activation {code} is not supported")

    def _const(self, tensor_index):
        if tensor_index in self.constants:
            return self.constants[tensor_index]
        t = self.tensors[tensor_index]
        if t["data"] is None:
            return None
        if t["type"] == _schema().TensorType.INT32:
            return np.frombuffer(t["data"].tobytes(), dtype="<i4").reshape(t["shape"])
        raise UnsupportedGraph(f"unexpected constant tensor {tensor_index}")

    def _kernel(self, tensor_index):
        if tensor_index not in self._kernel_name:
            raise UnsupportedGraph(f"tensor {tensor_index} is not an INT8 kernel")
        return self.variables[self._kernel_name[tensor_index]]

    @staticmethod
    def _pad(code):
        return "SAME" if code == _schema().Padding.SAME else "VALID"

    def _decode_input(self, encoded_inputs):
        idx = self.inputs[0]
        shape = self.tensors[idx]["shape"][1:]
        values = self.tf.convert_to_tensor(np.asarray(encoded_inputs), dtype=self.tf.int8)
        if tuple(values.shape[1:]) != tuple(shape):
            raise ValueError(f"companion inputs must be int8 with trailing shape {shape}")
        scale, zp = self._qparams(idx)
        return (self.tf.cast(values, self.tf.float32) - zp) * scale

    def forward(self, encoded_inputs):
        """(logits, probabilities): the requantized input and output of the final SOFTMAX."""
        tf = self.tf
        env = {self.inputs[0]: self._decode_input(encoded_inputs)}
        for op in self.ops:
            name, ins, out, o = op["name"], op["inputs"], op["outputs"][0], op["options"]
            x = env[ins[0]]
            if name in ("QUANTIZE", "DEQUANTIZE", "RESHAPE"):
                if name == "RESHAPE":
                    shape = self.tensors[out]["shape"]
                    x = tf.reshape(x, (-1, *shape[1:]))
                y = x
            elif name == "FULLY_CONNECTED":
                w = self._kernel(ins[1])
                y = tf.linalg.matmul(tf.reshape(x, (-1, w.shape[1])), w, transpose_b=True)
                if len(ins) > 2 and ins[2] >= 0:
                    y = y + self._const(ins[2])
                y = self._act(y, o["act"])
            elif name == "CONV_2D":
                w = tf.transpose(self._kernel(ins[1]), (1, 2, 3, 0))
                y = tf.nn.conv2d(x, w, strides=(1, *o["stride"], 1), padding=self._pad(o["padding"]))
                if len(ins) > 2 and ins[2] >= 0:
                    y = y + self._const(ins[2])
                y = self._act(y, o["act"])
            elif name == "DEPTHWISE_CONV_2D":
                k = self._kernel(ins[1])                         # 1, H, W, C * multiplier
                c = int(x.shape[-1])
                m = int(k.shape[-1]) // c
                w = tf.reshape(k[0], (k.shape[1], k.shape[2], c, m))
                y = tf.nn.depthwise_conv2d(x, w, strides=(1, *o["stride"], 1), padding=self._pad(o["padding"]))
                if len(ins) > 2 and ins[2] >= 0:
                    y = y + self._const(ins[2])
                y = self._act(y, o["act"])
            elif name in ("AVERAGE_POOL_2D", "MAX_POOL_2D"):
                pool = tf.nn.avg_pool2d if name == "AVERAGE_POOL_2D" else tf.nn.max_pool2d
                y = self._act(pool(x, o["filter"], (1, *o["stride"], 1), self._pad(o["padding"])), o["act"])
            elif name == "MEAN":
                axes = tuple(int(a) for a in np.asarray(self._const(ins[1])).reshape(-1))
                y = tf.reduce_mean(x, axis=axes, keepdims=o["keep_dims"])
            elif name == "ADD":
                y = self._act(x + env[ins[1]], o["act"])
            elif name == "SOFTMAX":
                y = tf.nn.softmax(x * o["beta"])
            else:
                raise UnsupportedGraph(name)
            env[out] = self._requantize_ste(y, out)
        return env[self.logits_tensor], env[self.outputs[0]]

    def losses(self, encoded_inputs, labels):
        logits, _ = self.forward(encoded_inputs)
        labels = self.tf.convert_to_tensor(np.asarray(labels).reshape(-1), dtype=self.tf.int64)
        return self.tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    def encoded_outputs(self, encoded_inputs):
        _, p = self.forward(encoded_inputs)
        scale, zp = self._qparams(self.outputs[0])
        return np.clip(np.rint(p.numpy() / scale + zp), -128, 127).astype(np.int8)

    def per_example_gradients(self, encoded_inputs, labels, batch: int = 256) -> dict[str, np.ndarray]:
        """Per-example gradients of the loss for every kernel, vectorized over a batch (jacobian of the per-example
        losses); identical in value to one tape per example."""
        tf = self.tf
        ordered = tuple(self.variables)
        out = {n: [] for n in ordered}
        x_all, y_all = np.asarray(encoded_inputs), np.asarray(labels)
        for s in range(0, len(y_all), batch):
            with tf.GradientTape() as tape:
                losses = self.losses(x_all[s:s + batch], y_all[s:s + batch])
            jac = tape.jacobian(losses, [self.variables[n] for n in ordered], experimental_use_pfor=True)
            for n, j in zip(ordered, jac):
                out[n].append(j.numpy())
        return {n: np.concatenate(v, axis=0) for n, v in out.items()}

    def manifest(self) -> dict[str, object]:
        return dict(implementation_version=self.implementation_version, loss_version=self.loss_version,
                    operators=[o["name"] for o in self.ops], kernels=list(self.variables),
                    straight_through_rule="identity derivative through activation round and clip",
                    authoritative_inference=False,
                    limitations=["does not reproduce integer multiplier rounding inside TFLite kernels",
                                 "agreement with the reference interpreter must be measured"])


def set_kernels(companion: TFLiteGraphCompanion, values: Mapping[str, np.ndarray]) -> None:
    for name, v in values.items():
        companion.variables[name].assign(np.asarray(v, dtype=np.float32))
