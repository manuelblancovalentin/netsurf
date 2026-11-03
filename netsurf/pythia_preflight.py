"""Read-only preparation and readiness CLI. Never launches model inference."""
import argparse
import json
import math
from pathlib import Path
import struct

from .manifests import digest_object, sha256_file
from .pythia_assets import load_pythia_asset
from .transformer_assets import AssetBlocked


def inspect_checkpoint(root, pin_path):
    root = Path(root)
    pin = json.loads(Path(pin_path).read_text())
    for relative, expected in pin["files"].items():
        if sha256_file(root / relative) != expected:
            raise AssetBlocked(f"checkpoint pin mismatch: {relative}")
    with (root / "model.safetensors").open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        if size > 16 * 1024 * 1024:
            raise AssetBlocked("unreasonable safetensors header")
        header = json.loads(handle.read(size))
    metadata = header.pop("__metadata__")
    for key in ("source_repository", "source_revision", "source_file_sha256"):
        if metadata.get(key) != pin["derivation"][key]:
            raise AssetBlocked("derived artifact parent changed")
    if metadata.get("recipe") != "float32 -> bfloat16 round-to-nearest-even":
        raise AssetBlocked("derived artifact recipe changed")
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != "gpt_neox" or config.get("tie_word_embeddings") is not False:
        raise AssetBlocked("unreviewed model topology or tied output storage")
    if any(tensor["dtype"] != "BF16" for tensor in header.values()):
        raise AssetBlocked("checkpoint contains a different stored precision")
    expected = {"embed_out.weight"}
    for i in range(config["num_hidden_layers"]):
        expected.update(f"gpt_neox.layers.{i}.{part}.weight" for part in (
            "attention.query_key_value", "attention.dense", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"))
    if not expected <= set(header):
        raise AssetBlocked("missing GPTNeoX linear weight tensor")
    tensors = [{"name": name, "shape": header[name]["shape"], "stored_dtype": "BF16",
                "words": math.prod(header[name]["shape"])} for name in sorted(expected)]
    total = sum(row["words"] * 16 for row in tensors)
    return {"schema_version": "pythia-storage-inventory/v1", "status": "HEADER_VERIFIED_RUNTIME_PENDING",
        "checkpoint_sha256": pin["files"]["model.safetensors"], "tensors": tensors,
        "tensor_identity_sha256": digest_object(tensors), "eligible_bits": total,
        "eligible_words": total // 16, "eligible_storage_bytes": total // 8,
        "fp32_computation_cache_bytes": total // 4,
        "excluded": "input embeddings, all biases, normalization, rotary buffers, activations and caches",
        "whole_population_ber_for_expected_proposed": {str(k): k / total for k in (1, 4, 16)},
        "limitation": "Header enumeration is not runtime conformance or fault-effectiveness evidence."}


def validate_preparation(plan, inventory):
    if plan.get("schema_version") != "pythia-protection-preparation/v1" or plan.get("status") != "DRAFT_BLOCKED_READY_AND_CONFORMANCE":
        raise ValueError("this is a preparation plan, not an executable frozen campaign")
    if plan["whole_linear_population_bits"] != inventory["eligible_bits"]:
        raise ValueError("whole-network eligible population changed")
    population = plan["fault_support"]
    tensor = next((r for r in inventory["tensors"] if r["name"] == population["tensor"]), None)
    words = population["word_indices"]
    if tensor is None or not words or words != sorted(set(words)) or any(type(w) is not int or w < 0 or w >= tensor["words"] for w in words):
        raise ValueError("invalid targeted stored words")
    if population["bit_indices"] != list(range(16)):
        raise ValueError("all sixteen stored BF16 bits must be explicit")
    n = len(words) * 16
    if population["eligible_bits"] != n or any(type(b) is not int or not 0 <= b <= n for b in plan["budgets"]):
        raise ValueError("wrong targeted bit count or budget")
    lambdas = plan["expected_proposed_faults"]
    if not lambdas or any(not 0 <= v <= n for v in lambdas) or len(set(lambdas)) != len(lambdas):
        raise ValueError("invalid Bernoulli exposure plan")
    for name in ("engineering", "selection", "final"):
        cohort = plan["cohorts"][name]
        if cohort["split"] not in {"validation", "test"} or not cohort["window_indices"]:
            raise ValueError("explicit complete source windows required")
        if cohort["window_indices"] != sorted(set(cohort["window_indices"])) or any(
                type(i) is not int or i < 0 for i in cohort["window_indices"]):
            raise ValueError("invalid window indices")
    # Reject context overlap, not only duplicated scored target identities.
    def spans(role):
        return [(i * 512, i * 512 + 1024) for i in plan["cohorts"][role]["window_indices"]]
    for a, b in (("engineering", "selection"), ("engineering", "final"), ("selection", "final")):
        if plan["cohorts"][a]["split"] == plan["cohorts"][b]["split"] and any(
                max(x, u) < min(y, v) for x, y in spans(a) for u, v in spans(b)):
            raise ValueError("input context overlap between roles")
    fields = plan["fields_per_ber"]
    if type(fields) is not int or fields < 1 or len(set(plan["fault_seeds"])) != len(lambdas) or len(plan["fault_seeds"]) != len(lambdas):
        raise ValueError("independent prespecified fault seeds required")
    if plan["methods"] != ["random", "encoded_significance", "exact_isolated"]:
        raise ValueError("unsupported prepared methods")
    # One clean and N isolated evaluations during selection; full/zero controls
    # shared between methods. No production executable is implied by this count.
    policies = 2 + len(plan["methods"]) * sum(0 < b < n for b in plan["budgets"])
    selection_requests = (n + 1) * len(plan["cohorts"]["selection"]["window_indices"])
    final_requests = (1 + len(lambdas) * fields * policies) * len(plan["cohorts"]["final"]["window_indices"])
    return {"status": plan["status"], "targeted_bits": n, "whole_linear_bits": inventory["eligible_bits"],
        "ber": [v / n for v in lambdas], "expected_applied": {
            str(b): [(n - b) * v / n for v in lambdas] for b in plan["budgets"]},
        "selection_forward_windows": selection_requests, "final_forward_windows": final_requests,
        "total_forward_windows": selection_requests + final_requests,
        "forward_tokens_upper_bound": (selection_requests + final_requests) * 1024,
        "independent_final_fields": len(lambdas) * fields,
        "limitation": "No whole-transformer BER claim; only the declared targeted subset is faultable. Not power justified."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--readiness", type=Path)
    args = parser.parse_args()
    try:
        if args.readiness:
            asset = load_pythia_asset(args.readiness, args.pin)
            print(json.dumps({"target": asset.target, "record_sha256": asset.record_sha256,
                              "status": "ASSETS_VERIFIED_NO_INFERENCE"}, indent=2))
        elif args.checkpoint:
            inventory = inspect_checkpoint(args.checkpoint, args.pin)
            result = validate_preparation(json.loads(args.plan.read_text()), inventory) if args.plan else inventory
            print(json.dumps(result, indent=2))
        else:
            parser.error("provide --readiness or --checkpoint; neither command runs inference")
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Pythia preflight blocked: {error}\n")


if __name__ == "__main__":
    main()
