"""Schema-6.8 complete-file adapter; authoritative inference remains M9.3.

Only explicit file selection and capture are added. The inherited evaluation
executes the pinned scoring rule once, with one fault state spanning the cohort.
"""
import hashlib
from dataclasses import asdict
import numpy as np

from .mlperf_tiny_multibench import MlperfTinyToyCarAdapter
from ..cohort_risk import SOURCE, CONTRACT, cohort_risk, score_digest, ordering_diagnostics
from ..manifests import digest_object, sha256_file
from ..samples import SamplePartition, sample_identity_digest


def _bytes_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


class ConditionalToyCarAdapter(MlperfTinyToyCarAdapter):
    def __init__(self, configuration):
        b = configuration.benchmark
        super().__init__(b.checkpoint, b.dataset, asset_lock=b.asset_lock,
            inference_backend=b.inference_backend, split=b.split, injectable_parameters=b.injectable_parameters)
        self.cohorts = configuration.cohorts
        self.cohort_records = {}
        self.score_records = []
        self._capture_cohort = None
        self._fault_identity = None
        self._mask_digest_cache = {}

    def sample_partition(self, role, count, *, offset=0):
        if role not in self.cohorts:
            return super().sample_partition(role, count, offset=offset)
        indices = tuple(self.cohorts[role])
        if count != len(indices) or offset:
            raise ValueError("ToyCar runtime sample request differs from frozen complete-file cohort")
        official = self._official_files()
        if len(official) != 248 or tuple(v[2] for v in official) != (0,) * 140 + (1,) * 108:
            raise ValueError("ToyCar official index ordering/class composition changed")
        selected = [official[i] for i in indices]
        vectors = [self._vectors(path) for _, path, _ in selected]
        if any(v.shape != (196, 640) for v in vectors):
            raise ValueError("incomplete ToyCar file: expected every one of 196 windows")
        targets = np.concatenate(vectors)
        scale, zp = self._interpreter.get_input_details()[0]["quantization"]
        inputs = np.clip(np.round(targets / scale + zp), -128, 127).astype(np.int8)
        labels = np.column_stack((targets,
            np.repeat(np.arange(count, dtype=np.float64), 196),
            np.repeat([v[2] for v in selected], 196)))
        records = []
        for local, (i, path, label) in enumerate(selected):
            windows = inputs[local * 196:(local + 1) * 196]
            records.append({"sample_id": f"{SOURCE}:{i}", "source_split": SOURCE,
                "original_index": i, "file_id": path.name, "file_sha256": sha256_file(path),
                "label": label, "window_count": 196,
                "ordered_window_ids": [f"{SOURCE}:{i}:window:{w}" for w in range(196)],
                "encoded_input_sha256": _bytes_hash(windows),
                "float64_targets_sha256": _bytes_hash(vectors[local])})
        record = {"role": role, "files": records, "file_count": count, "window_count": len(inputs),
            "encoded_input_sha256": _bytes_hash(inputs), "targets_sha256": _bytes_hash(targets),
            "source_index_sha256": sha256_file(self.asset_lock.file(self.index_relative_path))}
        record["cohort_sha256"] = digest_object(record)
        # Identity map uses the actual array, not a hash-based approximate match.
        self.cohort_records[id(inputs)] = record
        return inputs, labels, SamplePartition(role, SOURCE, indices,
            sample_identity_digest(SOURCE, indices, tuple(v[2] for v in selected)),
            "Explicit original file indices; all 196 ordered windows per file; fixed nonadditive ROC-AUC cohort")

    def sample_identity_records(self, inputs, labels, partition):
        return tuple(self.cohort_records[id(inputs)]["files"])

    def partition_diagnostics(self, inputs, labels, partition):
        record = self.cohort_records[id(inputs)]
        return {**record, "class_counts": {str(v): sum(f["label"] == v for f in record["files"]) for v in (0, 1)},
                "source_split": SOURCE, "sample_sha256": partition.sample_sha256}

    def predict_outputs(self, inputs):
        if self._capture_cohort is None:
            return super().predict_outputs(inputs)
        encoded = self._outputs(inputs)
        self._captured_output_sha256 = _bytes_hash(encoded)
        scale, zp = self._interpreter.get_output_details()[0]["quantization"]
        return (encoded.astype(np.float64) - zp) * scale

    def evaluate(self, inputs, labels, *, batch_size=256):
        record = self.cohort_records.get(id(inputs))
        self._capture_cohort = record
        try:
            result = super().evaluate(inputs, labels, batch_size=batch_size)
            if record is not None:
                scores = result.file_scores
                file_labels = [f["label"] for f in record["files"]]
                risk = cohort_risk(file_labels, scores)
                if risk != 1 - result.roc_auc:
                    raise RuntimeError("captured complete-file scores differ from authoritative ROC-AUC")
                self.score_records.append({"cohort_sha256": record["cohort_sha256"],
                    "fault": self._fault_identity, "scores": list(scores),
                    "scores_float64_hex": [float(s).hex() for s in scores],
                    "scores_sha256": score_digest(scores), "int8_outputs_sha256": self._captured_output_sha256,
                    "risk": risk, "file_roc_auc": result.roc_auc,
                    "reconstruction_mse_diagnostic_only": result.reconstruction_mse,
                    "pair_ordering": ordering_diagnostics(file_labels, scores)})
            return result
        finally:
            self._capture_cohort = None

    def evaluate_fault_schedule(self, inputs, labels, schedule, protected, **kwargs):
        key = id(protected)
        if key not in self._mask_digest_cache:
            self._mask_digest_cache[key] = (protected, digest_object([b.key() for b in sorted(protected.protected)]))
        self._fault_identity = {"schedule_sha256": schedule.digest(),
            "protected_mask_sha256": self._mask_digest_cache[key][1]}
        try:
            result, report = super().evaluate_fault_schedule(inputs, labels, schedule, protected, **kwargs)
            self.score_records[-1]["events"] = asdict(report)
            return result, report
        finally:
            self._fault_identity = None

    def cohort_evidence(self):
        cohorts = {v["cohort_sha256"]: v for v in self.cohort_records.values()}
        for key, record in cohorts.items():
            clean = next((r["scores"] for r in self.score_records if r["cohort_sha256"] == key and r["fault"] is None), None)
            for row in self.score_records:
                if row["cohort_sha256"] == key:
                    row["pair_ordering"] = ordering_diagnostics([f["label"] for f in record["files"]], row["scores"], clean)
        return {"schema_version": "1.0.0", "risk_contract": CONTRACT,
                "cohorts": cohorts, "evaluations": self.score_records}
