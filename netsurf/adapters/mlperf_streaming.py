"""Pinned streaming DS-CNN engineering adapter, not the full streaming benchmark.

Feature/event definitions follow MLCommons Tiny 4addd0fa, streaming_wakeword
get_dataset.py:get_lfbe_func and str_ww_util.py:get_true_and_false_detections.
Their Apache-2.0 source provenance is recorded in docs/M97_BENCHMARK_ADAPTER_STATUS.md.
"""
from dataclasses import dataclass
import functools
import hashlib
import json
from pathlib import Path

import numpy as np

from .mlperf_tiny_multibench import _ExactTFLiteKernelAdapter, MLPERF_TINY_COMMIT
from ..asset_lock import sha256_path
from ..tflite_storage import TFLiteModelStorage

MODEL_SHA = "3af8550895ba7d5c584277102b5075c52dcfa63ba9d2b2240f37c4e6abd5dd2b"
WAVE_SHA = "741ab419e46087ce1b738b23ca12c0633e796234f81673c7554629c38a4b708a"
INDEX_SHA = "c1da7de188c118a2abcffedc459469ea250b55a504f63f17110827a1298a6c08"
INPUT_SHA = "706fa76c1897de5732b8c281fa4ce55db81665a2435aa53520322666ecb8b5d8"
OUTPUT_SHA = "cedd76d2fafd974f2788ee206cf64ed8a6f756321719d671a1089b995a8da1ef"


def streaming_features(waveform):
    """Exact source operation ordering, float32 LFBE, not MFCC."""
    import tensorflow as tf
    wave = tf.expand_dims(tf.cast(waveform, tf.float32), 0)
    wave = tf.pad(wave, [[0, 0], [1, 0]])
    wave = tf.squeeze(wave[:, 1:] - (1 - 2**-5) * wave[:, :-1])
    spectrum = tf.signal.stft(wave, 1024, 512, fft_length=None,
        window_fn=functools.partial(tf.signal.hamming_window, periodic=False), pad_end=False)
    magnitude = tf.abs(spectrum)
    power = (1 / 1024) * tf.square(magnitude)
    power = tf.clip_by_value(power, 1e-30, tf.reduce_max(power))
    mel = tf.signal.linear_to_mel_weight_matrix(40, magnitude.shape[-1], 16000, 0.0, 8000.0)
    energies = tf.tensordot(power, mel, 1)
    logmel = 10 * (tf.math.log(energies) / tf.math.log(tf.constant(10, dtype=energies.dtype)))
    logmel = tf.expand_dims(logmel, -2)
    return tf.clip_by_value((logmel + 52 - 32 + 32.0) / 64.0, 0, 1).numpy()


def event_counts(detection, present, *, stride_samples=512, sample_rate=16000):
    """Source left-padding, 0.5s detection delay, 1s suppression/debounce."""
    observed = np.repeat(np.asarray(detection, dtype=bool), stride_samples)
    if len(observed) > len(present):
        raise ValueError("stream output exceeds waveform duration")
    observed = np.pad(observed, (len(present) - len(observed), 0))
    present = np.asarray(present, dtype=np.int8)
    starts, stops = np.flatnonzero(np.diff(present) > 0), np.flatnonzero(np.diff(present) < 0)
    if len(starts) != len(stops) or present[0] or present[-1]:
        raise ValueError("wakeword windows must start and end inside the waveform")
    false = observed.copy()
    tp = 0
    for start, stop in zip(starts, stops):
        tp += int(np.any(observed[start:stop + int(.5 * sample_rate)]))
        false[start:stop + sample_rate] = False
    fp, position = 0, 0
    while position < len(false):
        hits = np.flatnonzero(false[position:])
        if not len(hits):
            break
        position += int(hits[0]) + sample_rate
        fp += 1
    return int(tp), int(fp), int(len(starts) - tp)


@dataclass(frozen=True)
class StreamingEvaluation:
    true_detections: int
    false_positives: int
    false_negatives: int
    sample_count: int
    output_sha256: str

    @property
    def metrics(self):
        return {"true_detections": self.true_detections, "false_positives": self.false_positives,
                "false_negatives": self.false_negatives}


class MlperfStreamingShortAdapter(_ExactTFLiteKernelAdapter):
    """Whole ten-second stream is one persistent fault window.

    No fabricated frame labels, CE, accuracy, training split or combined task
    risk. Static structural rankings can be built; gradient and schema-6 risk
    capabilities explicitly remain unavailable.
    """
    benchmark_id = "mlperf-tiny/sww-short-engineering"
    input_shape, output_shape = (30, 1, 40), (3,)
    model_sha256 = MODEL_SHA

    def __init__(self, model_root):
        root = Path(model_root)
        self.checkpoint = root / "benchmark/training/streaming_wakeword/trained_models/str_ww_ref_model.tflite"
        self.wave_path = root / "benchmark/runner/sww_data_dir/short_wav_easy10s.wav"
        self.index_path = root / "benchmark/runner/sww_data_dir/sww_short_test.json"
        self.storage = self._model_content = self._interpreter = None
        self.asset_lock = self._companion_instance = self._bounded_conformance = None
        self.require_reference_equivalence = True
        self._runtime_provenance = {}

    def load(self):
        import tensorflow as tf
        if tf.__version__ != "2.14.1":
            raise ValueError("streaming reference requires TensorFlow 2.14.1 BUILTIN_REF")
        for path, expected in ((self.checkpoint, MODEL_SHA), (self.wave_path, WAVE_SHA), (self.index_path, INDEX_SHA)):
            if sha256_path(path) != expected:
                raise ValueError(f"streaming pinned artifact mismatch: {path.name}")
        self.storage = TFLiteModelStorage.from_path(self.checkpoint, expected_sha256=MODEL_SHA, eligible_roles=("kernel",))
        if len(self.storage.descriptors) != 9 or sum(d.size for d in self.storage.descriptors) != 46040:
            raise ValueError("streaming kernel population differs from inspected pin")
        self._set_model_content(self.storage.reference_content)
        inputs, _ = self.fixture()
        input_digest = hashlib.sha256(inputs.tobytes()).hexdigest()
        output_digest = hashlib.sha256(self._outputs(inputs).tobytes()).hexdigest()
        if (input_digest, output_digest) != (INPUT_SHA, OUTPUT_SHA):
            raise ValueError("streaming short-fixture exact input/output conformance failed")
        self._bounded_conformance = {
            "scope": "M97 short engineering fixture, not full streaming benchmark",
            "input_sha256": input_digest, "output_sha256": output_digest,
            "source_function_equivalence": "pinned MLCommons LFBE and event functions; M97 integration test",
        }

    def fixture(self):
        from scipy.io import wavfile
        rate, pcm = wavfile.read(self.wave_path)
        if rate != 16000 or pcm.dtype != np.int16:
            raise ValueError("pinned streaming fixture requires signed PCM16 at 16kHz")
        if pcm.ndim > 1:
            pcm = pcm[:, 0]
        features = streaming_features(pcm / 2**15)
        if not np.all(np.isfinite(features)):
            raise ValueError("nonfinite source streaming features")
        scale, zero = self._interpreter.get_input_details()[0]["quantization"]
        # Preserve reference cast semantics, including truncation, not np.rint.
        codes = np.asarray(features / scale + zero, dtype=np.int8)
        windows = np.stack([codes[i:i+30] for i in range(len(codes)-29)])
        config = json.loads(self.index_path.read_text())[0]
        present = np.zeros(int(config["length_sec"] * rate), dtype=np.int8)
        if len(pcm) != len(present):
            raise ValueError("waveform and source event duration differ")
        for start, stop in config["detection_windows"]:
            present[int(start*rate):int(stop*rate)] = 1
        return windows, present

    def evaluate(self, inputs, labels, *, batch_size=256):
        codes = self._outputs(inputs)
        scale, zero = self._interpreter.get_output_details()[0]["quantization"]
        probabilities = (codes.astype(np.float32) - zero) * scale
        tp, fp, fn = event_counts(probabilities[:, 0] > .95, labels)
        return StreamingEvaluation(tp, fp, fn, 1, hashlib.sha256(codes.tobytes()).hexdigest())

    def _companion(self):
        raise NotImplementedError("streaming event-risk gradient/Hessian companion is not validated")

    def sample_partition(self, *args, **kwargs):
        raise NotImplementedError("short engineering fixture has no independent calibration/final partitions")

    def dataset_identity_manifest(self):
        return {"scope": "official ten-second engineering fixture only", "suite_commit": MLPERF_TINY_COMMIT,
                "waveform_sha256": WAVE_SHA, "event_index_sha256": INDEX_SHA,
                "preprocessing": "pinned-get_lfbe_func/v1", "task": "event FP and FN, not frame accuracy"}
