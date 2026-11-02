"""Offline compact and full engineering-evidence exports."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

from .manifests import canonical_json_bytes, sha256_file
from .transformer_smoke import read_smoke_receipt


def export_smoke(smoke_dir: Path, readiness_manifest: Path, source_root: Path,
                 *, scp_host: str = "kona") -> Path:
    smoke_dir, readiness_manifest, source_root = (Path(x).resolve() for x in
        (smoke_dir, readiness_manifest, source_root))
    receipt = read_smoke_receipt(smoke_dir)
    if sha256_file(readiness_manifest) != receipt["asset_manifest_sha256"]:
        raise ValueError("readiness manifest changed after the smoke")
    destination = smoke_dir.with_name(smoke_dir.name + "_export")
    if destination.exists():
        checksums = json.loads((destination / "checksums.json").read_text())
        if any(sha256_file(destination / name) != value for name, value in checksums.items()):
            raise ValueError("completed export is corrupt")
        return destination
    staging = destination.with_name("." + destination.name + ".incomplete")
    if staging.exists():
        raise FileExistsError(f"partial export retained: {staging}")
    staging.mkdir()
    summary = {"schema_version": "transformer-engineering-summary/v1", "target": receipt["target"],
        "scope": receipt["status"], "asset_manifest_sha256": receipt["asset_manifest_sha256"],
        "eligible_bits": receipt["eligible_bits"],
        "clean_loss": receipt["clean"]["mean_nll"],
        "trials": {name: {"loss": value["risk"]["mean_nll"],
                          "accuracy": value["risk"]["accuracy"],
                          "failure_fraction": value["risk"]["failed_units"] / value["risk"]["total_units"],
                          "report": value["report"]}
                   for name, value in receipt["trials"].items()}}
    (staging / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    with zipfile.ZipFile(staging / "compact.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(staging / "summary.json", "summary.json")
        archive.write(smoke_dir / "manifest.json", "smoke/manifest.json")
    source_files = [
        "netsurf/storage_bit_codec.py", "netsurf/torch_storage.py", "netsurf/transformer_risk.py",
        "netsurf/transformer_assets.py", "netsurf/adapters/transformers_pinned.py",
        "netsurf/transformer_smoke.py", "netsurf/transformer_baselines.py",
        "config/transformer_target_status.json", "docs/TRANSFORMER_BENCHMARK_PROTOCOL.md",
    ]
    with zipfile.ZipFile(staging / "complete.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(smoke_dir.iterdir()):
            if file.is_file():
                archive.write(file, "smoke/" + file.name)
        archive.write(readiness_manifest, "assets/NETSURF_ASSET_READINESS.json")
        for relative in source_files:
            archive.write(source_root / relative, "source/" + relative)
    commands = "\n".join(
        f"scp {scp_host}:{(destination / name).as_posix()} ~/Downloads/{name}"
        for name in ("compact.zip", "complete.zip")
    ) + "\n"
    (staging / "mac_scp_commands.txt").write_text(commands)
    checksums = {name: sha256_file(staging / name) for name in
                 ("summary.json", "compact.zip", "complete.zip", "mac_scp_commands.txt")}
    (staging / "checksums.json").write_bytes(canonical_json_bytes(checksums) + b"\n")
    for file in staging.iterdir():
        file.chmod(0o444)
    staging.rename(destination)
    return destination
