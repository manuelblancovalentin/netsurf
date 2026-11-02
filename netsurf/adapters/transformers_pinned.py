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

        torch.set_num_threads(1)
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

    def token_sequence(self) -> tuple[int, ...]:
        self._require_loaded()
        if self.asset.target == "distilbert_sst2_fp32" or self.asset.record.get("dataset_format") != "wikitext2-raw-text/v1":
            raise AssetBlocked("pinned WikiText-2 raw text contract required")
        text = self.asset.dataset_path.read_text(encoding="utf-8")
        ids = list(self.tokenizer(text, add_special_tokens=False)["input_ids"])
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
        digest = hashlib.sha256(canonical_json_bytes(ids)).hexdigest()
        if digest != self.asset.record["token_sequence_sha256"]:
            raise AssetBlocked("token sequence differs from READY identity")
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
