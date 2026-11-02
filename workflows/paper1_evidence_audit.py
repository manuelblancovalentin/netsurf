"""Offline Paper 1 reconstruction checks; never imports ML or writes source bundles.

Run from any directory with --sources <historical receipts> --output <new directory>.
The earlier draft's receipts locate evidence, but every consumed digest is checked
again against the actual finalized bundle. Output is derived analysis, not a run.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, values):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(values[0]))
        w.writeheader()
        w.writerows(values)


def estimate(values):
    a = np.asarray(values, dtype=float)
    assert len(a) > 1 and np.isfinite(a).all()
    mean = float(a.mean())
    sd = float(a.std(ddof=1))
    half = float(t.ppf(.975, len(a)-1) * sd / np.sqrt(len(a)))
    return dict(count=len(a), mean=mean, sd=sd, ci_low=mean-half, ci_high=mean+half)


def equal(a, b):
    assert np.isclose(float(a), float(b), rtol=1e-10, atol=1e-12), (a, b)


def verify(tag, source):
    p = Path(source["path"])
    assert (p / "FINALIZED").is_file(), p
    inv = read(p / "checksums.json")["files"]
    assert sha(p / "checksums.json") == source["checksums_sha256"]
    required = set(source["verified_files"])
    if tag.startswith("static_"):
        required |= {"config.toml", "measurements/trials.csv", "artifacts/protection_masks.json"}
    actual = {}
    for rel in sorted(required):
        actual[rel] = sha(p / rel)
        assert actual[rel] == inv[rel], (tag, rel)
        if rel in source["verified_files"]:
            assert actual[rel] == source["verified_files"][rel], (tag, rel)
    payload = read(p / "scientific_payload.json")
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if tag == "ToyCar_exposure":
        # The finalized ledger has its payload receipt in FINALIZED, not a manifest.
        assert (p / "FINALIZED").read_text().strip() == f"scientific_payload_sha256={digest}"
        m = {"scientific_payload_sha256": digest}
    else:
        m = read(p / "manifest.json")
    assert digest == m["scientific_payload_sha256"] == source["scientific_payload_sha256"]
    return dict(path=str(p), scientific_payload_sha256=digest,
                checksums_sha256=sha(p / "checksums.json"), verified_files=actual,
                scope="Listed consumed artifacts, not every file in every bundle"), m


def audit_static(tag, path, manifest):
    trials = rows(path / "measurements/trials.csv")
    saved = rows(path / "measurements/protection_results.csv")
    paired = rows(path / "measurements/paired_comparisons.csv")
    toy = tag == "static_toycar"
    metric = "primary_metric_value" if toy else "accuracy"
    loss = "loss_metric_value" if toy else "categorical_crossentropy"
    prefix = "primary_metric" if toy else "accuracy"
    loss_prefix = "loss_metric" if toy else "loss"
    payload = read(path / "scientific_payload.json")
    replications = {r["replication"]["name"]: r for r in payload["replications"]}
    for replication in replications.values():
        partitions = replication["benchmark"]["partitions"]
        identities = [{(part["source_split"], i) for i in part["indices"]}
                      for part in partitions.values()]
        assert all(not a.intersection(b) for i, a in enumerate(identities) for b in identities[i+1:])
    groups = defaultdict(dict)
    events = {}
    for row in trials:
        ber, condition = row["policy"].split("/", 1)
        ber = float(ber.removeprefix("ber="))
        identity = (row["replication"], int(row["trial"]), row["schedule_sha256"])
        assert identity not in groups[(ber, condition)]
        groups[(ber, condition)][identity] = row
        proposed, applied, suppressed = (int(row[k]) for k in ("proposed_events", "applied_events", "suppressed_events"))
        assert proposed == applied + suppressed
        key = (ber, identity)
        assert key not in events or events[key] == proposed
        events[key] = proposed
        if condition == "full_protection":
            assert applied == 0
            clean = replications[row["replication"]]["clean_evaluation"]
            equal(row[metric], clean["roc_auc" if toy else "accuracy"])
            equal(row[loss], clean["reconstruction_mse" if toy else "categorical_crossentropy"])
        if condition == "no_protection":
            assert suppressed == 0
    curves = []
    for row in saved:
        ber = float(row["logical_ber"])
        data = groups[(ber, row["condition"])]
        assert set(data) == set(groups[(ber, "no_protection")])
        e = estimate([float(r[metric]) for r in data.values()])
        le = estimate([float(r[loss]) for r in data.values()])
        equal(e["mean"], row[prefix+"_mean"])
        equal(e["sd"], row[prefix+"_standard_deviation"])
        equal(le["mean"], row[loss_prefix+"_mean"])
        assert e["count"] == int(row["trial_count"]) == int(row["fault_realization_count"])
        populations = {r["benchmark"]["eligible_bits"] for r in replications.values()}
        assert len(populations) == 1
        equal(int(row["protected_bits"]), float(row["coverage"]) * populations.pop())
        if row["condition"] == "full_protection":
            equal(e["sd"], 0)
            equal(le["sd"], 0)
        curves.append(dict(benchmark=tag.removeprefix("static_"), ber=ber,
                           policy=row["policy"], coverage=float(row["coverage"]),
                           protected_bits=int(row["protected_bits"]), metric="file ROC-AUC" if toy else "accuracy",
                           **e, loss_mean=le["mean"], ranking_masks=int(row["ranking_mask_count"]),
                           proposed_mean=float(row["mean_proposed_events"]), applied_mean=float(row["mean_applied_events"])))
    comparisons = []
    for row in paired:
        def condition(name):
            return name if name in ("no_protection", "full_protection") else f"{name}@{float(row['coverage']):.6f}"
        ber = float(row["logical_ber"])
        left = groups[(ber, condition(row["left"]))]
        right = groups[(ber, condition(row["right"]))]
        assert left.keys() == right.keys()
        col = metric if row["metric"] in ("accuracy", "roc_auc") else loss
        e = estimate([float(left[k][col])-float(right[k][col]) for k in sorted(left)])
        equal(e["mean"], row["mean_difference"])
        equal(e["sd"], row["standard_deviation"])
        equal(e["ci_low"], row["ci_low"])
        equal(e["ci_high"], row["ci_high"])
        assert e["count"] == int(row["matched_pairs"])
        comparisons.append(dict(benchmark=tag.removeprefix("static_"), **row))
    execution = manifest["execution"]
    info = dict(benchmark=manifest["benchmark"], scientific_payload_sha256=manifest["scientific_payload_sha256"],
                parallel_execution=execution["parallel_execution"],
                probabilities=execution["protection"]["probabilities"], coverages=execution["protection"]["coverages"],
                trial_rows=len(trials), distinct_fields=len(events), aggregate_conditions=len(saved),
                checked_paired_summaries=len(paired),
                bytes_in_bundle=sum(p.stat().st_size for p in path.rglob("*") if p.is_file()))
    return curves, comparisons, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = read(args.sources)
    receipts, curves, pairs, inventory = {}, [], [], {}
    for tag, source in sources.items():
        receipts[tag], manifest = verify(tag, source)
        if tag.startswith("static_"):
            c, p, info = audit_static(tag, Path(source["path"]), manifest)
            curves.extend(c); pairs.extend(p); inventory[tag] = info
        print(f"Verified {tag}", flush=True)
    write_json(args.output / "verification.json", dict(source_receipt_sha256=sha(args.sources),
               bundles=receipts, bundle_count=len(receipts),
               checked_files=sum(len(r["verified_files"]) for r in receipts.values())))
    write_json(args.output / "static_inventory.json", inventory)
    write_csv(args.output / "static_curves.csv", curves)
    write_csv(args.output / "paired_comparisons_checked.csv", pairs)
    print(json.dumps(dict(output=str(args.output), bundles=len(receipts), curves=len(curves), paired_summaries=len(pairs))))


if __name__ == "__main__":
    main()
