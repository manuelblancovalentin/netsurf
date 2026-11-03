"""Exact stored-code Bernoulli faults for Kona's CPU W8A8 executable.

The universe is the complete declared qweight population.  Global ordinals
avoid Python objects for the billions of eligible coordinates; sparse fields
are sampled by geometric gaps, which is exactly independent Bernoulli sampling.
"""

from __future__ import annotations

from bisect import bisect_right
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
from typing import Protocol

import numpy as np

from .manifests import digest_object
from .w8a8_assets import ReadyW8A8, W8A8AssetBlocked, KONA_ROOT


@dataclass(frozen=True)
class W8A8Coordinate:
    ordinal: int
    module: str
    code_index: int
    bit: int


class W8A8Universe:
    def __init__(self, asset: ReadyW8A8):
        self.entries = tuple(asset.quantization["tensors"])
        self.offsets = [0]
        for item in self.entries:
            self.offsets.append(self.offsets[-1] + 8 * math.prod(item["shape"]))
        if self.offsets[-1] != asset.eligible_bits:
            raise W8A8AssetBlocked("eligible bit count differs from exact tensor census")
        self.bits = self.offsets[-1]
        self.codes = self.bits // 8
        self.digest = digest_object([{k: item[k] for k in ("module", "shape", "qweight_sha256", "scale_sha256")}
                                     for item in self.entries])

    def coordinate(self, ordinal: int) -> W8A8Coordinate:
        if not 0 <= ordinal < self.bits:
            raise ValueError("fault ordinal outside declared INT8 universe")
        index = bisect_right(self.offsets, ordinal) - 1
        local = ordinal - self.offsets[index]
        return W8A8Coordinate(ordinal, self.entries[index]["module"], local // 8, local % 8)

    def ordinal(self, module: str, code_index: int, bit: int) -> int:
        names = [item["module"] for item in self.entries]
        if module not in names:
            raise ValueError("module outside INT8 universe")
        i = names.index(module)
        if not (0 <= code_index < math.prod(self.entries[i]["shape"]) and 0 <= bit < 8):
            raise ValueError("code or bit outside tensor")
        return self.offsets[i] + 8 * code_index + bit


@dataclass(frozen=True)
class W8A8Field:
    seed: int
    ber: float
    universe_bits: int
    ordinals: tuple[int, ...]
    law: str = "independent_bernoulli"

    @property
    def sha256(self) -> str:
        return digest_object({"version": "w8a8-sparse-field/v1", "law": self.law, "seed": self.seed,
                              "ber": self.ber, "universe_bits": self.universe_bits,
                              "ordinals": self.ordinals})


def bernoulli_field(bits: int, ber: float, seed: int, *, max_events: int = 100000) -> W8A8Field:
    if bits <= 0 or not 0 <= ber < 1 or not math.isfinite(ber) or seed < 0 or max_events < 1:
        raise ValueError("invalid sparse Bernoulli field parameters")
    rng = np.random.default_rng(seed)
    selected = []
    current = -1
    if ber:
        while True:
            current += int(rng.geometric(ber))
            if current >= bits:
                break
            selected.append(current)
            if len(selected) > max_events:
                raise ValueError("fault exposure exceeded frozen event bound")
    return W8A8Field(seed, ber, bits, tuple(selected))


class FaultMask(Protocol):
    budget: int
    digest: str
    def contains(self, ordinal: int) -> bool: ...


@dataclass(frozen=True)
class FixedMask:
    """Small engineering mask; production rankers must use compact storage."""
    universe_bits: int
    protected: frozenset[int]

    def __post_init__(self):
        if any(not 0 <= x < self.universe_bits for x in self.protected):
            raise ValueError("protected ordinal outside universe")

    @property
    def budget(self) -> int:
        return len(self.protected)

    @property
    def digest(self) -> str:
        return digest_object({"version": "w8a8-fixed-mask/v1", "bits": self.universe_bits,
                              "protected": sorted(self.protected)})

    def contains(self, ordinal: int) -> bool:
        return ordinal in self.protected


@dataclass(frozen=True)
class MSBMask:
    """Complete-universe bit-tier ranking; ties break by global code ordinal."""
    universe_bits: int
    budget: int

    def __post_init__(self):
        if self.universe_bits % 8 or not 0 <= self.budget <= self.universe_bits:
            raise ValueError("invalid whole-universe MSB budget")

    @property
    def digest(self) -> str:
        return digest_object({"version": "w8a8-msb-mask/v1", "bits": self.universe_bits,
                              "budget": self.budget, "tie_order": "global-code-ordinal"})

    def contains(self, ordinal: int) -> bool:
        if not 0 <= ordinal < self.universe_bits:
            raise ValueError("ordinal outside universe")
        return (7 - ordinal % 8) * (self.universe_bits // 8) + ordinal // 8 < self.budget


@dataclass(frozen=True)
class BoundaryMask:
    universe_bits: int
    full: bool

    @property
    def budget(self) -> int:
        return self.universe_bits if self.full else 0

    @property
    def digest(self) -> str:
        return digest_object({"version": "w8a8-boundary-mask/v1", "bits": self.universe_bits,
                              "full": self.full})

    def contains(self, ordinal: int) -> bool:
        if not 0 <= ordinal < self.universe_bits:
            raise ValueError("ordinal outside universe")
        return self.full


@dataclass(frozen=True)
class W8A8InjectionReceipt:
    field_sha256: str
    mask_sha256: str
    proposed: int
    applied: int
    suppressed: int
    touched_module_hashes_before: dict[str, str]
    touched_module_hashes_after: dict[str, str]
    restored: bool


class W8A8FaultAdapter:
    """Mutates live QLinear.qweight bytes used by torch._int_mm and restores them."""

    def __init__(self, asset: ReadyW8A8):
        self.asset = asset
        self.universe = W8A8Universe(asset)
        self.model = None
        self.tokenizer = None
        self.modules = None
        self.baseline_hashes = None

    def load(self) -> None:
        import os
        import sys
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        source = str(KONA_ROOT / "src")
        if source not in sys.path:
            sys.path.insert(0, source)
        import torch
        import transformers
        import platform
        from transformers import AutoTokenizer
        from kona_assets import int8_torch as execution
        if (torch.__version__ != self.asset.record["environment"]["torch"] or
            transformers.__version__ != self.asset.record["environment"]["packages"]["transformers"] or
            platform.python_version() != self.asset.record["environment"]["python"] or
            torch.cuda.is_available()):
            raise W8A8AssetBlocked("unsupported integer execution backend")
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        kind = "sequence-classification" if self.asset.target == "distilbert_sst2_int8" else "causal-lm"
        model, manifest = execution.load_executable(self.asset.model_root, kind)
        if manifest["tensors_digest"] != self.asset.quantization["tensors_digest"]:
            raise W8A8AssetBlocked("loaded tensor identities differ from READY")
        modules = execution.quantized_modules(model)
        if set(modules) != {x["module"] for x in self.universe.entries}:
            raise W8A8AssetBlocked("actual QLinear module coverage differs from READY")
        state = model.state_dict()
        if any(state[f"{name}.qweight"].data_ptr() != module.qweight.data_ptr() or
               module.qweight.dtype != torch.int8 for name, module in modules.items()):
            raise W8A8AssetBlocked("checkpoint state does not alias the live INT8 inference buffer")
        self.model = model
        self.modules = modules
        self.baseline_hashes = {name: execution.tensor_sha256(module.qweight) for name, module in modules.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.asset.model_root),
                                                       local_files_only=True, trust_remote_code=False, use_fast=True)

    @contextmanager
    def injected(self, field: W8A8Field, mask: FaultMask):
        if self.model is None or self.modules is None or self.baseline_hashes is None:
            raise RuntimeError("load exact INT8 executable first")
        if field.universe_bits != self.universe.bits or mask.budget > self.universe.bits:
            raise ValueError("field or mask population differs from loaded W8A8 model")
        if len(set(field.ordinals)) != len(field.ordinals) or any(
            not 0 <= x < self.universe.bits for x in field.ordinals):
            raise ValueError("invalid duplicate or out-of-universe fault event")
        from kona_assets import int8_torch as execution
        applied = [self.universe.coordinate(x) for x in field.ordinals if not mask.contains(x)]
        touched = sorted({c.module for c in applied})
        before = {name: execution.tensor_sha256(self.modules[name].qweight) for name in touched}
        if any(before[name] != self.baseline_hashes[name] for name in touched):
            raise W8A8AssetBlocked("prior trial left changed INT8 stored bytes")
        done = []
        try:
            for c in applied:
                execution.xor_code(self.modules[c.module], c.code_index, 1 << c.bit)
                done.append(c)
            for name in touched:
                if execution.tensor_sha256(self.modules[name].qweight) == before[name]:
                    raise W8A8AssetBlocked("applied INT8 mutation did not change stored bytes")
            yield {"field_sha256": field.sha256, "mask_sha256": mask.digest,
                   "proposed": len(field.ordinals), "applied": len(applied),
                   "suppressed": len(field.ordinals) - len(applied),
                   "touched_module_hashes_before": before}
        finally:
            for c in reversed(done):
                execution.xor_code(self.modules[c.module], c.code_index, 1 << c.bit)
            after = {name: execution.tensor_sha256(self.modules[name].qweight) for name in touched}
            if before != after or any(after[name] != self.baseline_hashes[name] for name in touched):
                raise W8A8AssetBlocked("INT8 source bytes were not restored after trial")

    def sst2_cohorts(self):
        if self.asset.target != "distilbert_sst2_int8":
            raise ValueError("SST-2 cohort requested for LM")
        from .transformer_data import read_sst2_parquet, sst2_fault_partitions
        root = self.asset.dataset_root / "sst2"
        train = read_sst2_parquet(root / "train-00000-of-00001.parquet", "train")
        validation = read_sst2_parquet(root / "validation-00000-of-00001.parquet", "validation")
        if len(validation) != 872:
            raise W8A8AssetBlocked("labeled validation count changed")
        return sst2_fault_partitions(train, validation)

    def evaluate_sst2(self, rows, *, batch_size: int = 8):
        if self.asset.target != "distilbert_sst2_int8" or self.model is None or self.tokenizer is None:
            raise ValueError("loaded W8A8 SST-2 model required")
        import torch
        from .transformer_risk import classification_risk
        if not rows or batch_size < 1:
            raise ValueError("nonempty cohort and positive batch size required")
        logits = []
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                encoded = self.tokenizer([r.text for r in batch], return_tensors="pt",
                                         truncation=True, padding="max_length", max_length=128)
                try:
                    output = self.model(**encoded).logits.detach().float().cpu().numpy()
                    logits.extend(tuple(float(v) for v in row) for row in output)
                except RuntimeError:
                    logits.extend([None] * len(batch))
        return classification_risk(logits, [r.label for r in rows])

    def token_sequence(self, split: str = "test") -> tuple[int, ...]:
        if self.asset.target == "distilbert_sst2_int8" or self.tokenizer is None:
            raise ValueError("loaded W8A8 causal LM required")
        if split not in {"validation", "test"}:
            raise ValueError("unreviewed WikiText-2 split")
        from .transformer_data import read_wikitext_parquet, wikitext_joined_text
        root = self.asset.dataset_root / "wikitext-2-raw-v1"
        rows = read_wikitext_parquet(root / f"{split}-00000-of-00001.parquet")
        text, text_hash = wikitext_joined_text(rows)
        ids = tuple(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        if not ids or any(type(x) is not int or x < 0 for x in ids):
            raise W8A8AssetBlocked("invalid token sequence")
        self.last_token_identity = {"split": split, "token_ids_sha256": hashlib.sha256(
            np.asarray(ids, dtype="<i4").tobytes()).hexdigest(), "text_sha256": text_hash,
            "bos": "none", "eos": "none", "context": 1024, "stride": 512}
        if split == "test":
            import json
            report = json.loads(open(self.asset.record["reference_results"]["smoke_report"]).read())
            reference = report.get("dataset", {})
            if (self.last_token_identity["token_ids_sha256"] != reference.get("token_ids_sha256") or
                text_hash != reference.get("text_sha256") or len(ids) != reference.get("n_tokens")):
                raise W8A8AssetBlocked("tokenizer output differs from pinned clean reference")
        return ids

    def evaluate_lm(self, ids, *, context: int = 1024, stride: int = 512):
        if self.asset.target == "distilbert_sst2_int8" or self.model is None:
            raise ValueError("loaded W8A8 causal LM required")
        import torch
        from .transformer_risk import teacher_forced_windows, language_model_risk_from_stats
        stats = []
        with torch.inference_mode():
            for start, end, first in teacher_forced_windows(ids, context, stride):
                try:
                    output = self.model(input_ids=torch.tensor([ids[start:end]], dtype=torch.long),
                                        use_cache=False).logits[0]
                    for pos in range(first, end):
                        values = output[pos - start - 1].detach().float()
                        if not bool(torch.isfinite(values).all()):
                            stats.append((float("nan"), float("nan")))
                        else:
                            stats.append((float(values[ids[pos]]), float(torch.logsumexp(values, dim=0))))
                except RuntimeError:
                    stats.extend([None] * (end - first))
        if len(stats) != len(ids) - 1:
            raise RuntimeError("teacher-forced target accounting changed")
        return language_model_risk_from_stats(stats)
