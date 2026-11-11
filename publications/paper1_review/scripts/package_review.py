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
    # Non-nested optimum printed in the TECS revision: best single bit {1}, best pair {2,3}.
    def nested(s):
        return s[0] + 1.05 * s[1] + 1.05 * s[2] + s[0] * s[1] + s[0] * s[2]
    def expected(fn, protected, p):
        free = [i for i in range(3) if i not in protected]
        total = 0.
        for state in itertools.product((0, 1), repeat=len(free)):
            s = [0, 0, 0]
            for i, v in zip(free, state):
                s[i] = v
            total += p ** sum(state) * (1 - p) ** (len(free) - sum(state)) * fn(s)
        return total
    singles = [expected(nested, {b}, .1) for b in range(3)]
    pairs = {c: expected(nested, set(c), .1) for c in itertools.combinations(range(3), 2)}
    assert abs(singles[0] - .210) < 1e-12 and abs(singles[1] - .215) < 1e-12 and singles[0] < min(singles[1:])
    assert abs(pairs[(1, 2)] - .100) < 1e-12 and abs(pairs[(0, 1)] - .105) < 1e-12
    assert min(pairs, key=pairs.get) == (1, 2)
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
        "review/TECS_DECISION_MEMO_2026-11-11.md", "review/tecs_revision_notes.md",
        "data/tecs_revision_provenance.json", "data/tecs_audit/summary.json",
        "data/tecs_audit/per_cell_regret.csv", "data/tecs_audit/pairwise_contrasts.csv",
        "data/tecs_audit/conditions_with_quality.csv", "data/tecs_audit/kws_conditional_audit.json",
        "data/tecs_audit/distilbert_residual_drift.json",
        "figures/tecs_task_quality_ber_125.pdf", "figures/tecs_grid_regret_map.pdf",
        "figures/tecs_protection_vs_unprotected_125.pdf",
        "review/RUN_A_B_FROZEN_PLANS_2026-11-11.md", "review/RUN_A_B_RESULTS_2026-11-11.md",
        "data/tecs_runA/summary.json", "data/tecs_runA/absgrad_contrasts.csv", "data/tecs_runA/within_tier_contrasts.csv",
        "data/tecs_runA/fresh_cell_quality.csv", "data/tecs_runB/summary.json", "data/tecs_runB/cap_statistics.csv", "data/tecs_runL2/analysis.json", "figures/tecs_correction_mechanisms.pdf",
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
    tecs_early = json.loads((ROOT / "data/tecs_revision_provenance.json").read_text())
    for name in ("generator", "analytic_asset"):
        receipt = compact[name]
        # The fourth-delivery receipt for the workflow schematic is superseded: its labels were
        # simplified in the TECS revision and the current file is bound in the TECS provenance.
        if receipt["path"] in tecs_early.get("analytic_assets", {}):
            assert digest(ROOT / receipt["path"]) == tecs_early["analytic_assets"][receipt["path"]], receipt["path"]
            continue
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
    tecs = json.loads((ROOT / "data/tecs_revision_provenance.json").read_text())
    groups = (("derived_data", "data/tecs_audit"), ("figures", "figures"), ("scripts", "scripts"),
              ("derived_data_runA", "data/tecs_runA"), ("derived_data_runB", "data/tecs_runB"),
              ("derived_data_runA2", "data/tecs_runA2"), ("derived_data_runL2", "data/tecs_runL2"))
    for group, folder in groups:
        for name, expected_hash in tecs[group].items():
            assert digest(ROOT / folder / name) == expected_hash, f"{folder}/{name} changed after provenance"
    for rel, expected_hash in tecs["frozen_plans"].items():
        assert digest(ROOT / rel) == expected_hash, f"{rel} changed after provenance"
    assert tecs["experiments_executed"] == len(tecs["executed_runs"]) == 6
    assert len(list((ROOT / "data/tecs_runL2").glob("ber_*_draw_*.json"))) == 64
    run_a2 = json.loads((ROOT / "data/tecs_runA2/summary.json").read_text())["models"]["kws"]
    assert run_a2["decision_rule"]["primary/decision_useful_primary"]["confirmed_replication_level"] is True
    assert run_a2["fresh_replications"] == list(range(1, 12))
    # The manuscript states the Run A outcome; bind it to the tracked analysis.
    run_a = json.loads((ROOT / "data/tecs_runA/summary.json").read_text())["models"]
    assert run_a["resnet8"]["decision_rule"]["primary/decision_useful_primary"]["confirmed_replication_level"] is True
    assert run_a["kws"]["decision_rule"]["primary/decision_useful_primary"]["confirmed_replication_level"] is False
    run_b = json.loads((ROOT / "data/tecs_runB/summary.json").read_text())["models"]
    assert all(m["bitwise_mismatches"] == 0 and m["failed_units"] == 0 and m["observations"] == 4800 for m in run_b.values())
    assert set(run_b) == {"pythia", "qwen"}
    return {"cited_keys": sorted(cited), "bibliography_records": len(keys),
            "tecs_revision_receipts_verified": sum(len(tecs[g]) for g, _ in groups),
            "experiments_executed": tecs["experiments_executed"],
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
        "experiments_executed": checks["experiments_executed"],
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
