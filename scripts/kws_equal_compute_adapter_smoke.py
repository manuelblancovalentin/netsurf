"""Two already exposed KWS clips; exact paired fault/restoration receipt only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from netsurf.conditional_damage import _sample_identity_records
from netsurf.equal_compute_kws_campaign import (
    _context, _evaluate, _fields, _mask, _source, _source_config, _storage, _write,
)
from netsurf.manifests import digest_object, sha256_file

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "workspace/runs/kws_equal_compute_adapter_smoke_v1"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("write-once KWS engineering smoke already exists")
    _, candidates = _source()
    config = _source_config()
    storage = _storage(config)
    universe = storage.bit_identities()
    context = _context(config, universe, candidates)
    from netsurf.adapters.registry import create_adapter
    adapter = create_adapter(config, None)
    adapter.load()
    # M11 and earlier KWS work already exposed these original test indices.
    inputs, labels, partition = adapter.sample_partition(
        "conditional_model_selection", 2, offset=2088 - 2024)
    if list(partition.indices) != [2088, 2089]:
        raise ValueError("engineering input identity changed")
    schedules = _fields(universe, [970991], 2)
    schedule = schedules[0]
    original = tuple(bit for bit in candidates if bit in context.original)
    displaced = tuple(sorted((set(original) - {original[0]}) |
                             {next(bit for bit in candidates if bit not in context.original)}))
    before = hashlib.sha256(adapter._model_content).hexdigest()
    results = {}
    for name, selected in (("static_s5", original), ("one_swap", displaced)):
        mask = _mask(name, selected, context)
        results[name] = {"mask_sha256": mask.digest(),
                         "selected_candidate_bits": [bit.key() for bit in sorted(selected)],
                         **_evaluate(adapter, (inputs, labels), schedule, mask)}
        if hashlib.sha256(adapter._model_content).hexdigest() != before:
            raise ValueError("KWS authoritative model bytes not restored")
    result = {"schema_version": "kws-equal-query-adapter-smoke/v1",
              "status": "ENGINEERING_ONLY", "original_test_indices": [2088, 2089],
              "sample_sha256": partition.sample_sha256,
              "identity_records": list(_sample_identity_records(inputs, labels, partition)),
              "checkpoint_sha256": sha256_file(config.benchmark.checkpoint),
              "source_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "smoke_source_sha256": sha256_file(Path(__file__)),
              "schedule_seed": schedule.seed,
              "schedule_sha256": schedule.digest(),
              "events": [event.key() for event in schedule.events],
              "original_model_bytes_sha256": before,
              "restored_model_bytes_sha256": hashlib.sha256(adapter._model_content).hexdigest(),
              "policies": results,
              "limitation": "Two previously exposed clips and one field; adapter receipt, not held-out protection evidence."}
    OUTPUT.mkdir(parents=True)
    _write(OUTPUT / "receipt.json", {**result, "payload_sha256": digest_object(result)})
    with zipfile.ZipFile(OUTPUT / "complete.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(OUTPUT / "receipt.json", "receipt.json")
        archive.write(Path(__file__), "source/scripts/kws_equal_compute_adapter_smoke.py")
    checks = {name: sha256_file(OUTPUT / name) for name in ("receipt.json", "complete.zip")}
    _write(OUTPUT / "checksums.json", checks)
    (OUTPUT / "FINALIZED").write_text(digest_object(checks) + "\n")
    for path in OUTPUT.iterdir():
        path.chmod(0o444)
    print(json.dumps({"status": "ENGINEERING_ONLY_FINALIZED", "payload_sha256": digest_object(result),
                      "zip_sha256": checks["complete.zip"], "events": len(schedule.events)}))


if __name__ == "__main__":
    main()
