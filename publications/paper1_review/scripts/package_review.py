#!/usr/bin/env python3
"""Validate and archive the manuscript; offline, no experiment imports."""
import argparse
import csv
import difflib
import hashlib
import itertools
import json
import re
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SOURCE = ROOT / "source"
SUFFIXES = {".tex", ".bib", ".md", ".csv", ".json", ".py", ".pdf", ".svg", ".png"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def math_checks():
    # Exact arithmetic behind the adapted INT8 delta-matrix illustration.
    def signed(code):
        return code if code < 128 else code - 256
    displayed = [
        (5, "00000101", .3125, [-8, 4, 2, 1, .5, -.25, .125, -.0625]),
        (-6, "11111010", -.375, [8, -4, -2, -1, -.5, .25, -.125, .0625]),
        (15, "00001111", .9375, [-8, 4, 2, 1, -.5, -.25, -.125, -.0625]),
        (-1, "11111111", -.0625, [8, -4, -2, -1, -.5, -.25, -.125, -.0625]),
    ]
    for q, word, weight, expected in displayed:
        code = int(word, 2)
        assert signed(code) == q and q / 16 == weight
        actual = [(signed(code ^ (1 << bit)) - q) / 16 for bit in range(7, -1, -1)]
        assert actual == expected
    assert signed(5 ^ 0b10000100) == -127
    assert (signed(5 ^ 0b10000100) - 5) / 16 == -8.25
    for code in range(256):
        jumps = []
        for bit in range(8):
            coefficient = -(1 << bit) if bit == 7 else 1 << bit
            predicted = (1 - 2 * ((code >> bit) & 1)) * coefficient / 16
            observed = (signed(code ^ (1 << bit)) - signed(code)) / 16
            assert predicted == observed
            jumps.append(predicted)
        for a, b in itertools.combinations(range(8), 2):
            observed = (signed(code ^ (1 << a) ^ (1 << b)) - signed(code)) / 16
            assert observed == jumps[a] + jumps[b]
    # Additivity here concerns decoded word values, not network task loss.
    # Exhaustive expectation of the printed 3-bit counterexample.
    def risk(s):
        return 1.2 * s[0] + s[1] + .1 * s[2] + 3 * s[1] * s[2]
    for p in (.001, .01, 1 / 15, .1, .5, 1.):
        actual = []
        for protected in range(3):
            total = 0.
            other = [i for i in range(3) if i != protected]
            for state in itertools.product((0, 1), repeat=2):
                s = [0, 0, 0]
                for i, v in zip(other, state):
                    s[i] = v
                total += p ** sum(state) * (1 - p) ** (2 - sum(state)) * risk(s)
            actual.append(total)
        expected = [1.1 * p + 3 * p * p, 1.3 * p, 2.2 * p]
        assert max(abs(a - b) for a, b in zip(actual, expected)) < 1e-12
    # Every Boolean-valued response on a 3-bit cube verifies the range bound.
    for values in itertools.product((0., 1.), repeat=8):
        for p in (.01, .1, .5):
            exact = sum(p ** s.bit_count() * (1 - p) ** (3 - s.bit_count())
                        * values[s] for s in range(8))
            first = values[0] + p * sum(values[1 << i] - values[0] for i in range(3))
            assert abs(exact - first) <= 3 * p * (1 - (1 - p) ** 2) + 1e-12
    return {"counterexample_probabilities": 6, "bound_vertex_checks": 768,
            "int8_displayed_matrix_rows": len(displayed),
            "int8_single_flip_checks": 2048, "int8_distinct_two_flip_checks": 7168}


def validate():
    tex = "\n".join(p.read_text() for p in SOURCE.rglob("*.tex"))
    assert not re.search(r"\\TODO|\\TBD|INSERT RESULTS", tex, re.I)
    bib = (SOURCE / "references.bib").read_text()
    keys = re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib)
    assert len(keys) == len(set(keys)), "Duplicate bibliography keys"
    cited = {k.strip() for group in re.findall(r"\\cite\w*\{([^}]+)\}", tex)
             for k in group.split(",")}
    assert not cited - set(keys), sorted(cited - set(keys))
    log = (SOURCE / "main.log").read_text()
    assert "Output written on main.pdf" in log
    assert not re.search(r"undefined|Overfull|^! ", log, re.M)
    required = [
        "review/claim_to_evidence.md", "review/figure_table_manifest.md",
        "review/adversarial_scientific_review.md", "review/venue_and_novelty.md",
        "review/unresolved_experiment_ledger.md", "data/figure_provenance.json",
        "figures/reliability_all_budgets_supplement.pdf",
        "review/compact_revision_notes.md", "review/compact_venue_literature.md",
        "review/compact_figure_provenance.json",
        "review/fourth_revision_notes.md", "data/fkeras_candidate_audit_v1.json",
    ]
    for name in required:
        assert (ROOT / name).is_file(), name
    with (ROOT / "data/reliability_conditions.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1254, len(rows)
    reused = json.loads((ROOT / "review/legacy_figure_reuse.json").read_text())
    assert len(reused["files"]) == 24
    for item in reused["files"]:
        assert digest(Path(item["path"])) == item["sha256"], item["name"]
    compact = json.loads((ROOT / "review/compact_figure_provenance.json").read_text())
    receipts = dict(compact["inputs"])
    receipts.update(compact["outputs"])
    for name in ("generator", "analytic_asset"):
        receipt = compact[name]
        receipts[receipt["path"]] = receipt
    for path, receipt in receipts.items():
        assert digest(ROOT / path) == receipt["sha256"], path
    candidate_audit = json.loads((ROOT / "data/fkeras_candidate_audit_v1.json").read_text())
    candidate_content = candidate_audit["content"]
    encoded = json.dumps(candidate_content, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    assert hashlib.sha256(encoded).hexdigest() == candidate_audit["finalization"]["content_sha256"]
    assert candidate_content["status"] == "FINALIZED"
    candidate_models = {m["model"]: m for m in candidate_content["models"]}
    assert candidate_models["resnet8"]["status"] == "validated"
    assert all(candidate_models[m]["results"] is None for m in ("kws", "toycar"))
    assert all(m["candidate_count"] == 320 for m in candidate_models.values())
    for receipt in candidate_content["implementation_files"]:
        assert digest(Path(receipt["path"])) == receipt["sha256"], receipt["path"]
    return {"cited_keys": sorted(cited), "bibliography_records": len(keys),
            "condition_rows": len(rows), "mathematics": math_checks(),
            "unchanged_original_image_hashes": len(reused["files"]),
            "compact_figure_receipts_verified": len(receipts),
            "fkeras_candidate_audit_content_sha256": candidate_audit["finalization"]["content_sha256"],
            "fkeras_candidate_audit": "ResNet-8 verified; KWS/ToyCar calibration-input join unresolved",
            "latex": "compiled; no undefined references or overfull boxes",
            "expected_class_warning": "ACM reference block omitted for internal review"}


def archive(path, entries):
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for local, name in entries:
            z.write(local, name)
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--original-zip", type=Path, required=True)
    parser.add_argument("--original-pdf", type=Path, required=True)
    args = parser.parse_args()
    checks = validate()
    args.out.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
    shutil.copy2(SOURCE / "main.pdf", args.out / "NetSuRF_internal_review.pdf")
    shutil.copy2(ROOT / "figures/reliability_all_budgets_supplement.pdf",
                 args.out / "NetSuRF_full_grid_supplement.pdf")
    # Diff against supplied project, not an obsolete repository draft.
    diffs = []
    with zipfile.ZipFile(args.original_zip) as original:
        for name in original.namelist():
            if not name.endswith((".tex", ".bib", ".md")):
                continue
            rel = name.split("/", 1)[1] if "/" in name else name
            current = SOURCE / rel
            if current.is_file():
                diffs.extend(difflib.unified_diff(
                    original.read(name).decode("utf-8").splitlines(True),
                    current.read_text().splitlines(True),
                    fromfile="supplied/" + rel, tofile="revised/" + rel))
    (args.out / "source_from_supplied.diff").write_text("".join(diffs))
    entries = []
    for p in sorted(SOURCE.rglob("*")):
        if p.is_file() and p.suffix in {".tex", ".bib", ".md"}:
            entries.append((p, str(p.relative_to(SOURCE))))
    for p in sorted((ROOT / "figures").glob("*.pdf")):
        entries.append((p, "figures/" + p.name))
    archive(args.out / "NetSuRF_Overleaf_source.zip", entries)
    manifest = {
        "schema": "netsurf-manuscript-review/1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": revision, "branch": branch,
        "source_inputs": [{"path": str(p), "sha256": digest(p)}
                          for p in (args.original_zip, args.original_pdf)],
        "validation": checks,
        "files": {str(p.relative_to(ROOT)): digest(p)
                  for p in sorted(ROOT.rglob("*"))
                  if p.is_file() and p.suffix in SUFFIXES and "source/main.pdf" != str(p.relative_to(ROOT))},
        "verification_scope": "Derived data and cited source verification receipts; "
                              "not an independent rehash of every omitted original scientific byte.",
        "experiments_executed": 0,
    }
    (args.out / "delivery_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    complete = [(p, "NetSuRF_review/" + str(p.relative_to(ROOT)))
                for p in sorted(ROOT.rglob("*")) if p.is_file() and p.suffix in SUFFIXES
                and str(p.relative_to(ROOT)) != "source/main.pdf"]
    for p in sorted(args.out.iterdir()):
        if p.suffix != ".zip":
            complete.append((p, "NetSuRF_review/" + p.name))
    archive(args.out / "NetSuRF_complete_review.zip", complete)
    hashes = {p.name: digest(p) for p in sorted(args.out.iterdir()) if p.is_file()}
    (args.out / "checksums.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "checks": checks, "sha256": hashes}, indent=2))


if __name__ == "__main__":
    main()
