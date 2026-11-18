"""Foreground commands for transformer readiness and bounded engineering evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import struct

from .manifests import canonical_json_bytes, sha256_file
from .storage_bit_codec import StorageBitCodec
from .transformer_assets import AssetBlocked, load_ready_asset
from .transformer_risk import classification_risk
from .transformer_smoke import read_smoke_receipt, run_distilbert_smoke
from .transformer_export import export_smoke
from .transformer_handoff import audit_handoff


ROOT = Path(__file__).resolve().parents[1]
PORTFOLIO = ROOT / "config/benchmark_expansion_registry.json"


def _fixture(output: Path) -> dict:
    """Small exposed engineering calculation; no model or dataset asset used."""
    codec = StorageBitCodec("FP32")
    original = struct.pack("<ffff", 1.0, -0.5, 0.25, 0.75)
    corrupted = codec.xor(original, 0, 31)
    restored = codec.xor(corrupted, 0, 31)
    if restored != original:
        raise RuntimeError("fixture storage did not restore")
    def score(raw):
        values = struct.unpack("<ffff", raw)
        return [(values[0] * 1.0 + values[1] * 2.0,
                 values[2] * 1.0 + values[3] * 2.0),
                (values[0] * -1.0 + values[1] * 1.0,
                 values[2] * -1.0 + values[3] * 1.0)]
    clean_logits, fault_logits = score(original), score(corrupted)
    payload = {"schema_version": "transformer-fixture/v1", "scope": "synthetic engineering only",
        "codec": codec.version, "precision": codec.precision, "source_indices": [0, 1],
        "original_sha256": __import__("hashlib").sha256(original).hexdigest(),
        "corrupted_sha256": __import__("hashlib").sha256(corrupted).hexdigest(),
        "restored_exactly": restored == original, "fault_identity": "linear.weight[0]#b31",
        "clean": asdict(classification_risk(clean_logits, [0, 1])),
        "faulted": asdict(classification_risk(fault_logits, [0, 1]))}
    if output.exists():
        existing = json.loads(output.read_text())
        if existing != payload:
            raise ValueError("existing fixture output differs; immutable evidence")
        return existing
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json_bytes(payload).decode("utf-8") + "\n")
    output.chmod(0o444)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("target")
    verify.add_argument("manifest", type=Path)
    smoke = commands.add_parser("smoke-distilbert")
    smoke.add_argument("manifest", type=Path)
    smoke.add_argument("output", type=Path)
    fixture = commands.add_parser("fixture")
    fixture.add_argument("output", type=Path)
    aggregate = commands.add_parser("aggregate-smoke")
    aggregate.add_argument("output", type=Path)
    export = commands.add_parser("export-smoke")
    export.add_argument("output", type=Path)
    export.add_argument("manifest", type=Path)
    export.add_argument("--scp-host", default="kona")
    audit = commands.add_parser("audit-readiness")
    audit.add_argument("manifest", type=Path)
    audit.add_argument("output", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "verify":
        try:
            asset = load_ready_asset(arguments.manifest, arguments.target, PORTFOLIO)
        except AssetBlocked as error:
            parser.exit(2, f"asset blocked: {error}\n")
        print(json.dumps({"target": asset.target, "status": "READY", "interpreter": str(asset.interpreter),
                          "manifest_sha256": asset.manifest_sha256,
                          "record_sha256": asset.record_sha256}, sort_keys=True))
    elif arguments.command == "smoke-distilbert":
        try:
            result = run_distilbert_smoke(arguments.manifest, PORTFOLIO, arguments.output, ROOT)
        except AssetBlocked as error:
            parser.exit(2, f"asset blocked: {error}\n")
        print(json.dumps({"output": str(arguments.output.resolve()), "status": result["status"],
                          "eligible_bits": result["eligible_bits"]}, sort_keys=True))
    elif arguments.command == "fixture":
        result = _fixture(arguments.output)
        print(json.dumps({"output": str(arguments.output.resolve()),
                          "clean_loss": result["clean"]["mean_nll"],
                          "faulted_loss": result["faulted"]["mean_nll"]}, sort_keys=True))
    elif arguments.command == "aggregate-smoke":
        result = read_smoke_receipt(arguments.output)
        print(json.dumps({"target": result["target"], "status": result["status"],
                          "clean_loss": result["clean"]["mean_nll"],
                          "one_bit_loss": result["trials"]["one_bit"]["risk"]["mean_nll"],
                          "multi_bit_loss": result["trials"]["multi_bit"]["risk"]["mean_nll"],
                          "protected_loss": result["trials"]["equal_budget_protected"]["risk"]["mean_nll"]}, sort_keys=True))
    elif arguments.command == "export-smoke":
        path = export_smoke(arguments.output, arguments.manifest, ROOT,
                            scp_host=arguments.scp_host)
        print((path / "mac_scp_commands.txt").read_text(), end="")
    elif arguments.command == "audit-readiness":
        report = audit_handoff(arguments.manifest, PORTFOLIO, arguments.output)
        print(json.dumps({"output": str(arguments.output.resolve()),
                          "readiness_sha256": report["readiness_sha256"],
                          "ready": [row["target"] for row in report["targets"] if row["asset_status"] == "READY"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
