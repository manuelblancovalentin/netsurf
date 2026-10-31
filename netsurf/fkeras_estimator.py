"""Architecture-specific FKeras construction, shared by both scoring variants."""
from dataclasses import asdict
import hashlib
from time import perf_counter

import numpy as np

from .estimators_m3 import EstimatorOutput
from .fkeras_baseline import (
    FKERAS_SOURCE_COMMIT, leading_hessian_eigenpairs,
    principal_weight_hessian_score, repository_weight_hessian_score,
)


class ContinuousKwsHessian:
    """One traced HVP for one immutable calibration set; canonical flat mapping."""

    def __init__(self, adapter, inputs, labels):
        from .mlperf_multibench_companion import MlperfKwsContinuousCompanion
        if adapter.benchmark_id != "mlperf-tiny/kws":
            raise ValueError("M97 continuous FKeras is validated only for pinned KWS")
        if adapter.storage is None:
            adapter.load()
        if adapter._model_content != adapter.storage.reference_content:
            raise ValueError("FKeras construction requires the clean pinned kernel state")
        self.companion = MlperfKwsContinuousCompanion(adapter.storage.reference_content)
        self.names = tuple(sorted(self.companion.variables))
        self.variables = tuple(self.companion.variables[name] for name in self.names)
        self.shapes = tuple(tuple(v.shape) for v in self.variables)
        self.sizes = tuple(int(np.prod(shape)) for shape in self.shapes)
        self.theta = np.concatenate([v.numpy().reshape(-1) for v in self.variables])
        self.inputs, self.labels = np.asarray(inputs).copy(), np.asarray(labels).copy()
        tf = self.companion.tf

        @tf.function(input_signature=[tf.TensorSpec((len(self.theta),), tf.float64)], autograph=False)
        def hvp(flat):
            parts = [tf.reshape(v, shape) for v, shape in zip(tf.split(flat, self.sizes), self.shapes)]
            with tf.GradientTape() as outer:
                with tf.GradientTape() as inner:
                    loss = tf.reduce_mean(self.companion.losses(self.inputs, self.labels))
                gradient = inner.gradient(loss, self.variables)
                product = tf.add_n([tf.reduce_sum(g * v) for g, v in zip(gradient, parts)])
            result = outer.gradient(product, self.variables)
            return tf.concat([tf.reshape(v, (-1,)) for v in result], axis=0)

        self._hvp = hvp
        self.mapping = []
        offset = 0
        for name, shape, size in zip(self.names, self.shapes, self.sizes):
            descriptor = next(d for d in adapter.storage.descriptors if d.parameter == name)
            decoded = descriptor.quantization.dequantize(adapter.values()[name])
            np.testing.assert_array_equal(decoded, self.companion.variables[name].numpy())
            self.mapping.append({**descriptor.to_dict(), "flat_offset": offset, "flat_size": size})
            offset += size

    def __call__(self, vector):
        return self._hvp(np.asarray(vector, dtype=np.float64)).numpy()

    def gradient_at(self, theta):
        """Finite-difference diagnostic only; always restores companion state."""
        offset = 0
        try:
            for variable, shape, size in zip(self.variables, self.shapes, self.sizes):
                variable.assign(np.asarray(theta)[offset:offset + size].reshape(shape))
                offset += size
            values = self.companion.gradients(self.inputs, self.labels)
            return np.concatenate([values[name].reshape(-1) for name in self.names])
        finally:
            offset = 0
            for variable, shape, size in zip(self.variables, self.shapes, self.sizes):
                variable.assign(self.theta[offset:offset + size].reshape(shape))
                offset += size


def fkeras_family(adapter, inputs, labels, identities, *, seed, options, requested):
    started = perf_counter()
    operator = ContinuousKwsHessian(adapter, inputs, labels)
    vectors, diagnostics = leading_hessian_eigenpairs(
        operator, len(operator.theta), count=options["eigenpairs"], seed=seed,
        max_iterations=options["max_iterations"], tolerance=options["tolerance"],
        max_products=options["max_products"],
    )
    values = np.asarray(diagnostics.eigenvalues)
    metadata = {
        "source_commit": FKERAS_SOURCE_COMMIT,
        "source_method": "HessianMetrics.do_sum_hessian_rank(strategy=sum)",
        "companion": operator.companion.manifest(),
        "eigenpairs": asdict(diagnostics), "seed": seed, "options": options,
        "parameter_mapping": operator.mapping,
        "parameter_sha256": hashlib.sha256(operator.theta.tobytes()).hexdigest(),
        "eigenvectors_sha256": hashlib.sha256(vectors.tobytes()).hexdigest(),
        "eigenvectors_flat": vectors.tolist(),
        "eigenvector_sign_rule": "largest absolute component nonnegative",
        "negative_eigenvalues": "retained; LM descending magnitude, algebraic value breaks ties",
        "ordering_requirement": "bit_significance_desc, estimator_score_desc, stable_identity",
        "runtime_accounting": "one shared eigensolve; wall time repeated, do not sum variants",
        "scope": "kernel-only architecture adaptation, not execution of upstream QKeras package",
    }
    offsets = {row["parameter"]: row["flat_offset"] for row in operator.mapping}
    functions = {"fkeras_paper": principal_weight_hessian_score, "fkeras_repository": repository_weight_hessian_score}
    outputs = {}
    for name in requested:
        word_scores = functions[name](operator.theta, vectors, values)
        scores = {b: abs(float(word_scores[offsets[b.parameter] + b.flat_index])) for b in identities}
        if not all(np.isfinite(value) for value in scores.values()):
            raise ValueError("FKeras produced nonfinite sensitivity scores")
        expression = ("abs(sum_i lambda_i*(v_i.T theta)*v_i)" if name == "fkeras_paper"
                      else "sum_i abs(lambda_i**2*(v_i.T theta)*v_i)")
        outputs[name] = EstimatorOutput(
            name, name.replace("_", "-") + "-continuous-kws/v1", scores, None,
            perf_counter() - started, {**metadata, "expression": expression,
                "raw_parameter_score": word_scores.tolist(),
                "warning": "parameter sensitivity, not signed finite-bit loss prediction; strict tiers ignore affine scale differences"},
        )
    shared_elapsed = perf_counter() - started
    from dataclasses import replace
    return {name: replace(value, runtime_seconds=shared_elapsed) for name, value in outputs.items()}
