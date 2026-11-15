"""Deterministic tests of netsurf.tflite_graph_companion (gradient-trust plan, review/GRADIENT_TRUST_PLAN_2026-11-15.md).

A small Keras model (convolution, depthwise convolution, pooling, dense layers, softmax) is converted to a full-integer
INT8 TFLite model inside the test; the companion must agree with the reference interpreter on most outputs, expose one
variable per INT8 kernel, and give batched per-example gradients equal to one tape per example.  The real-model check
(identical to the pinned DS-CNN and ResNet-8 companions) is workflows/runG_companion_check.py.
"""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from netsurf.tflite_graph_companion import TFLiteGraphCompanion, UnsupportedGraph  # noqa: E402


def _model(conv=True):
    tf.keras.utils.set_random_seed(7)
    if conv:
        inp = tf.keras.Input((8, 8, 2), batch_size=1)   # static batch, as the MLPerf models
        x = tf.keras.layers.Conv2D(4, 3, strides=2, padding="same", activation="relu")(inp)
        x = tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu")(x)
        x = tf.keras.layers.AveragePooling2D(2)(x)
        x = tf.keras.layers.Reshape((16,))(x)          # fixed shape: Flatten adds SHAPE ops under newer TF
    else:
        inp = tf.keras.Input((6,), batch_size=1)
        x = inp
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    out = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    rng = np.random.default_rng(0)
    xs = rng.normal(size=(200, *inp.shape[1:])).astype(np.float32)
    ys = rng.integers(0, 3, size=200)
    model.compile("adam", "sparse_categorical_crossentropy")
    model.fit(xs, ys, epochs=3, verbose=0, batch_size=1)
    conv_ = tf.lite.TFLiteConverter.from_keras_model(model)
    conv_.optimizations = [tf.lite.Optimize.DEFAULT]
    conv_.representative_dataset = lambda: ([v[None]] for v in xs[:64])
    conv_.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv_.inference_input_type = tf.int8
    conv_.inference_output_type = tf.int8
    return conv_.convert(), xs, ys


def _reference(content, x8):
    it = tf.lite.Interpreter(model_content=content,
                             experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    out = []
    for s in x8:
        it.set_tensor(i["index"], s[None])
        it.invoke()
        out.append(it.get_tensor(o["index"])[0])
    return np.array(out), i["quantization"]


@pytest.mark.parametrize("conv", [True, False])
def test_companion_agrees_with_reference_and_gradients_match_single_tapes(conv):
    content, xs, ys = _model(conv)
    g = TFLiteGraphCompanion(content)
    _, (scale, zp) = _reference(content, np.zeros((1, *xs.shape[1:]), np.int8))
    x8 = np.clip(np.rint(xs[:64] / scale + zp), -128, 127).astype(np.int8)
    ref, _ = _reference(content, x8)
    enc = g.encoded_outputs(x8)
    assert np.mean(np.abs(enc.astype(int) - ref.astype(int)) <= 1) >= 0.95
    assert np.mean(enc.argmax(1) == ref.argmax(1)) >= 0.9
    assert len(g.variables) == (4 if conv else 2)
    pe = g.per_example_gradients(x8[:6], ys[:6], batch=4)
    for k in range(6):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(g.losses(x8[k:k + 1], ys[k:k + 1]))
        grads = tape.gradient(loss, list(g.variables.values()))
        for name, gr in zip(g.variables, grads):
            np.testing.assert_allclose(pe[name][k], gr.numpy(), rtol=1e-5, atol=1e-6)


def test_unsupported_operator_is_refused():
    tf.keras.utils.set_random_seed(1)
    inp = tf.keras.Input((4,))
    x = tf.keras.layers.Dense(4)(inp)
    out = tf.keras.layers.Softmax()(tf.keras.layers.Multiply()([x, x]))
    model = tf.keras.Model(inp, out)
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = lambda: ([np.ones((1, 4), np.float32) * k] for k in range(8))
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.int8
    c.inference_output_type = tf.int8
    with pytest.raises(UnsupportedGraph):
        TFLiteGraphCompanion(c.convert())
