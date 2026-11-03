"""Pinned Pythia BF16 storage with FP32 eager computation, not native BF16."""
from dataclasses import dataclass
import hashlib
import math
import os

from .transformers_pinned import PinnedTransformerAdapter
from ..bf16_fp32_storage import BF16StorageFP32Compute, widen_bf16_bytes
from ..transformer_assets import AssetBlocked, verify_running_interpreter
from ..pythia_assets import CONTRACT, TARGET
from ..torch_storage import TorchStoredLinearWeights
from ..transformer_risk import teacher_forced_windows


@dataclass(frozen=True)
class LanguageRisk:
    """Reference raw NLL plus a separately labeled bounded failure endpoint.

    Nonfinite token NLL is retained as null + an explicit failure count. Raw
    mean/perplexity are unavailable if ANY token failed; no survivor mean.
    Capped NLL is an engineering endpoint, not reference corpus perplexity.
    """
    token_nll: tuple[float | None, ...]
    failed_tokens: int
    clipped_finite_tokens: int
    mean_nll: float | None
    perplexity: float | None
    mean_capped_nll: float
    penalty_nats: float
    token_nll_sha256: str
    scored_token_positions: tuple[int, ...]
    window_identities: tuple[tuple[int, int, int], ...]


def summarize_nll(values, positions, windows, penalty_nats=30.0):
    import numpy as np
    if not values or len(values) != len(positions) or not math.isfinite(penalty_nats) or penalty_nats <= 0:
        raise ValueError("nonempty aligned token losses and finite positive cap required")
    valid = tuple(float(v) if math.isfinite(float(v)) else None for v in values)
    failed = sum(v is None for v in valid)
    if any(v is not None and v < 0 for v in valid):
        raise ValueError("negative token NLL")
    raw = np.asarray(values, dtype="<f8")
    # Hash the raw finite loss vector using the reference float64 reduction.
    # Exceptional tokens use a separate explicit null/position record.
    from ..manifests import digest_object
    digest = hashlib.sha256(raw.tobytes()).hexdigest() if not failed else digest_object(valid)
    mean = float(raw.mean()) if not failed else None
    perplexity = math.exp(mean) if mean is not None and mean <= math.log(float.fromhex('0x1.fffffffffffffp+1023')) else None
    capped = [penalty_nats if v is None else min(v, penalty_nats) for v in valid]
    return LanguageRisk(valid, failed, sum(v is not None and v > penalty_nats for v in valid),
        mean, perplexity, float(np.mean(capped)), penalty_nats, digest, tuple(positions), tuple(windows))


class PythiaBF16StorageAdapter(PinnedTransformerAdapter):
    def load(self):
        if self.asset.target != TARGET or self.asset.record.get("execution", {}).get("contract") != CONTRACT:
            raise AssetBlocked("distinct verified mixed-precision Pythia asset required")
        verify_running_interpreter(self.asset)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from safetensors import safe_open
        torch.set_num_threads(self.asset.backend["threads"])
        torch.use_deterministic_algorithms(True)
        if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
            raise AssetBlocked("autocast is incompatible with FP32 computation")
        self.tokenizer = AutoTokenizer.from_pretrained(self.asset.model_root, local_files_only=True,
                                                       trust_remote_code=False, use_fast=True)
        # Match Kona's direct FP32 loader, including rotary buffers. Loading in
        # BF16 and converting the model afterward is NOT assumed equivalent.
        self.model = AutoModelForCausalLM.from_pretrained(self.asset.model_root, local_files_only=True,
            trust_remote_code=False, use_safetensors=True, dtype=torch.float32, attn_implementation="eager").eval()
        if self.model.config.model_type != "gpt_neox" or self.model.device.type != "cpu":
            raise AssetBlocked("only pinned CPU GPTNeoX/Pythia is supported")
        if self.model.config._attn_implementation != "eager":
            raise AssetBlocked("attention backend changed")
        state = self.model.state_dict()
        stored, matched = {}, set()
        for item in self.asset.model_files:
            if not item["path"].endswith(".safetensors"):
                continue
            with safe_open(str(self.asset.model_root / item["path"]), framework="pt", device="cpu") as checkpoint:
                for name in checkpoint.keys():
                    original = checkpoint.get_tensor(name)
                    live = state.get(name)
                    if name in matched or live is None or live.dtype != torch.float32 or original.dtype != torch.bfloat16 or live.shape != original.shape:
                        raise AssetBlocked(f"checkpoint/runtime BF16 mapping changed: {name}")
                    if TorchStoredLinearWeights._bytes(live) != widen_bf16_bytes(TorchStoredLinearWeights._bytes(original)):
                        raise AssetBlocked(f"checkpoint/runtime exact widening failed: {name}")
                    stored[name] = original
                    matched.add(name)
        if matched != set(dict(self.model.named_parameters())):
            raise AssetBlocked("not all runtime parameters are mapped to checkpoint storage")
        self.storage = BF16StorageFP32Compute(self.model, stored)
        self.mapped_parameter_count = len(matched)

    def evaluate_windows(self, token_ids, *, window_indices, penalty_nats=30.0):
        """Score complete prespecified 1024/512 source windows, preserving context.

        The caller freezes split, full token hash and window indices before
        faults. No split is chosen by outcome. Production is not auto-launched.
        """
        self._require_loaded()
        import torch
        if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
            raise AssetBlocked("autocast violates FP32 computation")
        plan = teacher_forced_windows(token_ids, 1024, 512)
        if not window_indices or list(window_indices) != sorted(set(window_indices)) or any(
                type(i) is not int or i < 0 or i >= len(plan) for i in window_indices):
            raise ValueError("unique increasing valid source window indices required")
        values, positions, windows = [], [], []
        with torch.inference_mode():
            for index in window_indices:
                begin, end, first = plan[index]
                x = torch.tensor([token_ids[begin:end]], dtype=torch.long)
                # Exact reference operation order: log_softmax in FP32,
                # gather, then float64 storage/reduction. Do not substitute
                # a separately rounded logsumexp-minus-target computation.
                logits = self.model(input_ids=x, use_cache=False).logits[0]
                if logits.dtype != torch.float32:
                    raise AssetBlocked("inference stopped using FP32 logits")
                nll = -torch.log_softmax(logits[:-1], dim=-1).gather(1, x[0, 1:, None])[:, 0]
                values.extend(nll[first - begin - 1:].double().cpu().tolist())
                positions.extend(range(first, end))
                windows.append((begin, end, first))
        return summarize_nll(values, positions, windows, penalty_nats)
