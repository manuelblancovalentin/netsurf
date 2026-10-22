from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np

from netsurf.experiment_config import load_experiment, template_text
from netsurf.experiment_engine import execute_experiment
from netsurf.identities import ParameterSpec, enumerate_bits


@dataclass(frozen=True)
class _Evaluation:
    accuracy: float
    categorical_crossentropy: float
    sample_count: int


class _Adapter:
    def __init__(self, codec):
        self.codec = codec
        self.state = {"weight": np.array([0.25, -0.5], dtype=np.float32)}

    def load(self):
        pass

    def dataset(self, count):
        return np.zeros((count, 1), dtype=np.float32), np.zeros(count, dtype=np.int64)

    def values(self):
        return {name: value.copy() for name, value in self.state.items()}

    def parameter_specs(self):
        return (ParameterSpec("weight", (2,)),)

    def bit_identities(self):
        return enumerate_bits(self.parameter_specs(), 6)

    @contextmanager
    def temporary_values(self, values):
        before = self.values()
        self.state = {name: np.asarray(value).copy() for name, value in values.items()}
        try:
            yield
        finally:
            self.state = before

    def quantized_values(self):
        return {name: self.codec.quantize(value).astype(value.dtype) for name, value in self.state.items()}

    def evaluate(self, inputs, labels, *, batch_size=256):
        loss = float(np.sum(np.abs(self.state["weight"])))
        return _Evaluation(max(0.0, 1.0 - loss / 10.0), loss, len(labels))

    def gradients(self, inputs, labels):
        return {"weight": np.array([0.5, -0.25], dtype=np.float32)}

    def polar_impacts(self, inputs, *, activation_source):
        raise AssertionError("smoke template does not request qPolarGrad")


def test_engine_is_adapter_driven_and_deterministic():
    from netsurf.quantization import BitCodec, FixedPointFormat

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint, dataset = root / "checkpoint", root / "dataset"
        checkpoint.write_bytes(b"c")
        dataset.write_bytes(b"d")
        text = template_text("smoke").replace(
            '/ml/experiments/netsurf/legacy/benchmarks/mnist_fnn_lite/q6_2_1/pruned_0.125_159569_qflatten_dense10u_qapplyalpha_quantized_relu_dense10u_qsoftmax/models/pruned_0.125_159569_qflatten_dense10u_qapplyalpha_quantized_relu_dense10u_qsoftmax.20250414_152629.keras', str(checkpoint)
        ).replace('/ml/datasets/mnist/mnist.npz', str(dataset))
        config_path = root / "experiment.toml"
        config_path.write_text(text, encoding="utf-8")
        config = load_experiment(config_path)
        codec = BitCodec(FixedPointFormat.parse("q<6,2,1>"))
        first = execute_experiment(config, adapter=_Adapter(codec))
        second = execute_experiment(config, adapter=_Adapter(codec))
        assert first.scientific_payload_sha256 == second.scientific_payload_sha256
        assert first.scientific_payload["checks"]["state_restored"] is True
        assert first.scientific_payload["checks"]["matched_schedules"] is True
        assert len(first.trials) == config.execution.trials * len(config.estimators)
