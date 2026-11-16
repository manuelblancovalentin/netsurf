"""The continuous generic companion and its Hessian against finite differences on a tiny TFLite MLP."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")


def _tiny_model():
    rng = np.random.default_rng(0)
    m = tf.keras.Sequential([tf.keras.Input((6,), batch_size=1), tf.keras.layers.Dense(5, activation="relu"),
                             tf.keras.layers.Dense(3)])
    x = rng.normal(size=(64, 6)).astype(np.float32)

    def rep():
        for i in range(64):
            yield [x[i:i + 1]]
    c = tf.lite.TFLiteConverter.from_keras_model(m)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = rep
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.int8
    c.inference_output_type = tf.int8
    return c.convert()


def test_continuous_hessian_matches_finite_differences_and_is_symmetric():
    from netsurf.tflite_graph_continuous import GraphHessian, TFLiteGraphContinuousCompanion
    content = _tiny_model()
    comp = TFLiteGraphContinuousCompanion(content)
    names = sorted(comp.variables)
    rng = np.random.default_rng(1)
    x = rng.integers(-128, 128, size=(40, 6)).astype(np.int8)
    y = rng.integers(0, 3, size=40)
    H = GraphHessian(content, names, x, y, batch=16)
    assert H.companion.variables[names[0]].dtype == tf.float64
    n = len(H.theta)
    u, v = rng.normal(size=n), rng.normal(size=n)
    assert abs(u @ H(v) - v @ H(u)) < 1e-8 * (1 + abs(u @ H(v)))

    def grad(theta):
        off = 0
        for var, s, k in zip(H.variables, H.shapes, H.sizes):
            var.assign(theta[off:off + k].reshape(s)); off += k
        with tf.GradientTape() as t:
            loss = tf.reduce_mean(H.companion.losses(x, y))
        g = t.gradient(loss, H.variables)
        return np.concatenate([gi.numpy().reshape(-1) for gi in g])
    eps = 1e-5
    fd = (grad(H.theta + eps * v) - grad(H.theta - eps * v)) / (2 * eps)
    grad(H.theta)                                               # restore
    hv = H(v)
    assert np.linalg.norm(fd - hv) <= 1e-4 * (1 + np.linalg.norm(hv))
