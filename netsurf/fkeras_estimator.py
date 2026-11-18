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


class ContinuousTFLiteHessian:
    """One continuous-loss HVP for immutable calibration; canonical flat mapping."""

    def __init__(self, adapter, inputs, labels):
        from .mlperf_multibench_companion import (
            MlperfKwsContinuousCompanion, MlperfToyCarContinuousCompanion,
        )
        from .mlperf_companion import MlperfResnet8ContinuousCompanion
        companions = {
            "mlperf-tiny/kws": MlperfKwsContinuousCompanion,
            "mlperf-tiny/ad": MlperfToyCarContinuousCompanion,
            "mlperf-tiny/ic": MlperfResnet8ContinuousCompanion,
        }
        if adapter.benchmark_id not in companions:
            raise ValueError("continuous FKeras requires a pinned, mapped TFLite companion")
        if adapter.storage is None:
            adapter.load()
        if adapter._model_content != adapter.storage.reference_content:
            raise ValueError("FKeras construction requires the clean pinned kernel state")
        self.companion = companions[adapter.benchmark_id](adapter.storage.reference_content)
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
        if offset != len(self.theta):
            raise ValueError("FKeras parameter mapping does not cover the full kernel population")
        eligible = tuple(adapter.bit_identities())
        word_bits = {}
        for bit in eligible:
            if not 0 <= bit.bit_index < 8:
                raise ValueError("FKeras requires INT8 kernel bits")
            word = (bit.parameter, bit.flat_index)
            word_bits[word] = word_bits.get(word, 0) | (1 << bit.bit_index)
        expected_words = {
            (name, index) for name, size in zip(self.names, self.sizes)
            for index in range(size)
        }
        if (len(eligible) != len(self.theta) * 8
                or len(word_bits) != len(self.theta)
                or set(word_bits) != expected_words
                or any(bits != 255 for bits in word_bits.values())):
            raise ValueError("FKeras requires exactly all eight bits of every mapped INT8 kernel word")

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


class ContinuousKwsHessian(ContinuousTFLiteHessian):
    """Backward-compatible KWS-only name retained for M97 numerical tests."""

    def __init__(self, adapter, inputs, labels):
        if adapter.benchmark_id != "mlperf-tiny/kws":
            raise ValueError("ContinuousKwsHessian is validated only for pinned KWS")
        super().__init__(adapter, inputs, labels)


def fkeras_family(adapter, inputs, labels, identities, *, seed, options, requested):
    if adapter.benchmark_id != "mlperf-tiny/kws" and "fkeras_repository" in requested:
        raise ValueError("repository-form FKeras is retained only for the pinned KWS historical comparison")
    started = perf_counter()
    operator = ContinuousTFLiteHessian(adapter, inputs, labels)
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
    variant = {"mlperf-tiny/kws": "kws", "mlperf-tiny/ic": "resnet8", "mlperf-tiny/ad": "toycar"}[adapter.benchmark_id]
    outputs = {}
    for name in requested:
        word_scores = functions[name](operator.theta, vectors, values)
        scores = {b: abs(float(word_scores[offsets[b.parameter] + b.flat_index])) for b in identities}
        if not all(np.isfinite(value) for value in scores.values()):
            raise ValueError("FKeras produced nonfinite sensitivity scores")
        expression = ("abs(sum_i lambda_i*(v_i.T theta)*v_i)" if name == "fkeras_paper"
                      else "sum_i abs(lambda_i**2*(v_i.T theta)*v_i)")
        outputs[name] = EstimatorOutput(
            name, name.replace("_", "-") + f"-continuous-{variant}/v1", scores, None,
            perf_counter() - started, {**metadata, "expression": expression,
                **({"source_method": "published signed Hessian projection",
                    "source_reference": "FKeras ACM JATS 2024 Eq. (3); pinned repository retained for provenance"}
                   if adapter.benchmark_id != "mlperf-tiny/kws" else {}),
                "raw_parameter_score": word_scores.tolist(),
                "warning": "parameter sensitivity, not signed finite-bit loss prediction; strict tiers ignore affine scale differences"},
        )
    shared_elapsed = perf_counter() - started
    from dataclasses import replace
    return {name: replace(value, runtime_seconds=shared_elapsed) for name, value in outputs.items()}
