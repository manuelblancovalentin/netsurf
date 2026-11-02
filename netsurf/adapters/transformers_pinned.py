"""Offline, pinned PyTorch/Transformers task adapters for engineering trials.

Loading is deliberately unavailable until a READY handoff passes byte and
environment verification.  Only CPU eager attention and observed FP32/BF16
linear-weight storage are in scope.  Embeddings, norms, biases and all other
state are excluded and remain unchanged; this is not full-model coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

from ..manifests import canonical_json_bytes
from ..transformer_data import read_sst2_parquet, sst2_fault_partitions
from ..transformer_data import read_wikitext_parquet, wikitext_joined_text
from ..torch_storage import TorchStoredLinearWeights
from ..transformer_assets import AssetBlocked, ReadyTransformerAsset, verify_running_interpreter
from ..transformer_risk import TaskRisk, classification_risk, language_model_risk_from_stats, teacher_forced_windows


@dataclass(frozen=True)
class LabeledText:
    source_index: int
    text: str
    label: int


def read_sst2_jsonl(asset: ReadyTransformerAsset) -> tuple[LabeledText, ...]:
    if asset.target != "distilbert_sst2_fp32" or asset.record.get("dataset_format") != "sst2-labeled-jsonl/v1":
        raise AssetBlocked("SST-2 requires a pinned labeled JSONL mapping")
    examples = []
    for line in asset.dataset_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not isinstance(row.get("source_index"), int) or not isinstance(row.get("text"), str) or row.get("label") not in (0, 1):
            raise AssetBlocked("incomplete SST-2 source/label mapping")
        examples.append(LabeledText(row["source_index"], row["text"], row["label"]))
    if len({x.source_index for x in examples}) != len(examples):
        raise AssetBlocked("duplicate SST-2 source identities")
    partitions = asset.record.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {"engineering", "calibration", "selection", "evaluation"}:
        raise AssetBlocked("engineering, calibration, selection and evaluation partitions must be explicit")
    sets = []
    valid = {x.source_index for x in examples}
    for name in ("engineering", "calibration", "selection", "evaluation"):
        indices = partitions[name]
        if not isinstance(indices, list) or not indices or len(set(indices)) != len(indices) or not set(indices) <= valid:
            raise AssetBlocked(f"invalid SST-2 {name} identities")
        sets.append(set(indices))
    if any(sets[i] & sets[j] for i in range(4) for j in range(i + 1, 4)):
        raise AssetBlocked("SST-2 partitions overlap")
    return tuple(examples)


def read_sst2_cohorts(asset: ReadyTransformerAsset) -> dict[str, tuple[LabeledText, ...]]:
    """Read only the asset-hash-verified rows; keep fault-outcome cohorts separate."""
    if asset.target != "distilbert_sst2_fp32":
        raise AssetBlocked("SST-2 cohort requested for a different target")
    if asset.record.get("dataset_format") == "sst2-labeled-parquet/v1":
        inventory = {item["path"] for item in asset.dataset_files}
        train_file = "sst2/train-00000-of-00001.parquet"
        validation_file = "sst2/validation-00000-of-00001.parquet"
        if not {train_file, validation_file} <= inventory:
            raise AssetBlocked("READY inventory lacks labeled SST-2 train/validation files")
        train = read_sst2_parquet(asset.dataset_path / train_file, "train")
        validation = read_sst2_parquet(asset.dataset_path / validation_file, "validation")
        if len(validation) != 872:
            raise AssetBlocked("SST-2 validation cohort differs from READY reference")
        partitions = sst2_fault_partitions(train, validation)
        return {name: tuple(LabeledText(row.source_index, row.text, row.label) for row in rows)
                for name, rows in partitions.items()}
    examples = read_sst2_jsonl(asset)
    by_id = {row.source_index: row for row in examples}
    return {name: tuple(by_id[index] for index in asset.record["partitions"][name])
            for name in ("engineering", "calibration", "selection", "evaluation")}


class PinnedTransformerAdapter:
    def __init__(self, asset: ReadyTransformerAsset):
        self.asset = asset
        self.model = None
        self.tokenizer = None
        self.storage = None

    def load(self) -> None:
        verify_running_interpreter(self.asset)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(self.asset.backend["threads"])
        torch.use_deterministic_algorithms(True)
        if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
            raise AssetBlocked("autocast would change the consumed numerical representation")
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = False
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.asset.model_root),
            local_files_only=True, trust_remote_code=False, use_fast=True)
        config = AutoConfig.from_pretrained(str(self.asset.model_root),
            local_files_only=True, trust_remote_code=False)
        config._attn_implementation = "eager"
        kind = AutoModelForSequenceClassification if self.asset.target == "distilbert_sst2_fp32" else AutoModelForCausalLM
        self.model = kind.from_pretrained(str(self.asset.model_root), config=config,
            local_files_only=True, trust_remote_code=False, use_safetensors=True,
            torch_dtype="auto")
        self.model.eval()
        if self.model.device.type != "cpu" or getattr(self.model.config, "_attn_implementation", None) != "eager":
            raise AssetBlocked("model loaded with unvalidated device or attention implementation")
        self.storage = TorchStoredLinearWeights(self.model, self.asset.precision)
        self._verify_checkpoint_storage()

    def _verify_checkpoint_storage(self) -> None:
        from safetensors import safe_open

        eligible = set(self.storage.parameters)
        matched = set()
        for item in self.asset.model_files:
            if not item["path"].endswith(".safetensors"):
                continue
            with safe_open(str(self.asset.model_root / item["path"]), framework="pt", device="cpu") as source:
                for name in set(source.keys()) & eligible:
                    if name in matched:
                        raise AssetBlocked(f"duplicate checkpoint tensor {name}")
                    source_tensor = source.get_tensor(name)
                    live_tensor = self.storage.parameters[name]
                    if source_tensor.dtype != live_tensor.dtype or source_tensor.shape != live_tensor.shape:
                        raise AssetBlocked(f"checkpoint/runtime tensor representation changed: {name}")
                    if TorchStoredLinearWeights._bytes(source_tensor) != self.storage._bytes(live_tensor):
                        raise AssetBlocked(f"checkpoint/runtime stored bytes differ: {name}")
                    matched.add(name)
        if matched != eligible:
            raise AssetBlocked(f"checkpoint missing {len(eligible - matched)} eligible runtime tensors")

    def _require_loaded(self) -> None:
        if self.model is None or self.tokenizer is None or self.storage is None:
            raise RuntimeError("load verified pinned model first")

    def evaluate_sst2(self, examples: Sequence[LabeledText], *, batch_size: int = 8,
                      max_length: int = 128) -> TaskRisk:
        self._require_loaded()
        if self.asset.target != "distilbert_sst2_fp32" or batch_size < 1 or max_length < 2:
            raise ValueError("invalid SST-2 evaluation configuration")
        import torch
        rows: list[tuple[float, ...] | None] = []
        with torch.inference_mode():
            for start in range(0, len(examples), batch_size):
                batch = examples[start:start + batch_size]
                encoded = self.tokenizer([x.text for x in batch], return_tensors="pt",
                    padding="max_length", truncation=True, max_length=max_length)
                try:
                    logits = self.model(**encoded).logits.detach().to(torch.float32).cpu().numpy()
                    rows.extend(tuple(float(v) for v in row) for row in logits)
                except RuntimeError:
                    rows.extend([None] * len(batch))
        return classification_risk(rows, [x.label for x in examples])

    def token_sequence(self, *, split: str = "test") -> tuple[int, ...]:
        self._require_loaded()
        if self.asset.target == "distilbert_sst2_fp32":
            raise AssetBlocked("classification target has no causal-LM token sequence")
        format_id = self.asset.record.get("dataset_format")
        if format_id == "wikitext2-raw-parquet/v1":
            if split not in {"validation", "test"}:
                raise AssetBlocked("WikiText-2 calibration and evaluation splits are fixed")
            relative = f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet"
            if relative not in {item["path"] for item in self.asset.dataset_files}:
                raise AssetBlocked("READY inventory lacks requested WikiText-2 split")
            rows = read_wikitext_parquet(self.asset.dataset_path / relative)
            text, text_sha256 = wikitext_joined_text(rows)
            expected_text = (self.asset.record.get("text_sha256") or {}).get(split)
            if expected_text is not None and text_sha256 != expected_text:
                raise AssetBlocked("WikiText-2 joined text differs from READY identity")
        elif format_id == "wikitext2-raw-text/v1" and split == "test":
            text = self.asset.dataset_path.read_text(encoding="utf-8")
        else:
            raise AssetBlocked("pinned WikiText-2 raw text contract required")
        ids = list(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        if format_id == "wikitext2-raw-parquet/v1" and (self.asset.record.get("bos_mode") != "none" or
            self.asset.record.get("eos_mode") != "none"):
            raise AssetBlocked("Kona WikiText-2 protocol forbids BOS/EOS insertion")
        if self.asset.record.get("bos_mode") == "prepend":
            ids.insert(0, self.tokenizer.bos_token_id)
        elif self.asset.record.get("bos_mode") != "none":
            raise AssetBlocked("BOS behavior is not pinned")
        if self.asset.record.get("eos_mode") == "append":
            ids.append(self.tokenizer.eos_token_id)
        elif self.asset.record.get("eos_mode") != "none":
            raise AssetBlocked("EOS behavior is not pinned")
        if any(not isinstance(x, int) or x < 0 for x in ids):
            raise AssetBlocked("invalid token ID sequence")
        if format_id == "wikitext2-raw-parquet/v1":
            import numpy as np
            digest = hashlib.sha256(np.asarray(ids, dtype="<i4").tobytes()).hexdigest()
            expected = (self.asset.record.get("token_sequence_sha256") or {}).get(split)
        else:
            digest = hashlib.sha256(canonical_json_bytes(ids)).hexdigest()
            expected = self.asset.record.get("token_sequence_sha256")
        if expected is not None and digest != expected:
            raise AssetBlocked("token sequence differs from READY identity")
        self.last_token_sequence_identity = {"split": split, "n_tokens": len(ids),
            "token_ids_sha256": digest, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "context": 1024, "stride": 512, "bos": "none", "eos": "none"}
        return tuple(ids)

    def evaluate_language_model(self, token_ids: Sequence[int], *, context: int,
                                stride: int) -> TaskRisk:
        self._require_loaded()
        if self.asset.target == "distilbert_sst2_fp32":
            raise ValueError("classification target has no causal-LM risk")
        import torch
        rows: list[tuple[float, float] | None] = []
        targets: list[int] = []
        with torch.inference_mode():
            for start, end, first in teacher_forced_windows(token_ids, context, stride):
                targets.extend(token_ids[first:end])
                encoded = torch.tensor([token_ids[start:end]], dtype=torch.long)
                try:
                    logits = self.model(input_ids=encoded, use_cache=False).logits[0]
                    for position in range(first, end):
                        values = logits[position - start - 1].detach().to(torch.float32)
                        if not bool(torch.isfinite(values).all()):
                            rows.append((float("nan"), float("nan")))
                        else:
                            rows.append((float(values[token_ids[position]]),
                                         float(torch.logsumexp(values, dim=0))))
                except RuntimeError:
                    rows.extend([None] * (end - first))
        if targets != list(token_ids[1:]):
            raise RuntimeError("teacher-forced window scoring omitted or duplicated tokens")
        return language_model_risk_from_stats(rows)
