"""Offline, write-once recovery of clean-conformant Pythia wide-BER shards.

The original 24-field/BER run is never edited or called finalized. Shard zero
failed its exact clean restoration gate. This derives a separate 18-field/BER
analysis from shards one through three, whose clean start and end records agree.
No model is loaded; no fault field or ranking mask is constructed.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from netsurf.manifests import digest_object, sha256_file
from netsurf.w8a8_dense_ber_pilot import LAW_VERSION, POPCOUNT
from netsurf.w8a8_wide_ber_campaign import (
    FIELDS_PER_BER, INTERIOR, METHODS, _holm, status, validate,
)


ACCEPTED_SHARDS = (1, 2, 3)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _hash_and_count(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    bits = 0
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
            bits += int(POPCOUNT[np.frombuffer(block, dtype=np.uint8)].sum(dtype=np.uint64))
    return digest.hexdigest(), bits


def clean_subset(root: Path, shards: int) -> dict:
    if shards != 4 or (root / "FINALIZED").exists():
        raise ValueError("this recovery applies only to the unfinalized four-shard run")
    accepted = []
    for shard in ACCEPTED_SHARDS:
        start = _read(root / "clean" / f"shard_{shard:02d}.json")
        end = _read(root / "clean" / f"shard_{shard:02d}_end.json")
        if start != end:
            raise ValueError(f"accepted shard {shard} did not restore exact clean inference")
        accepted.append(start)
    if any(item != accepted[0] for item in accepted):
        raise ValueError("accepted shards disagree on exact clean inference")
    excluded_start = _read(root / "clean" / "shard_00.json")
    if excluded_start == accepted[0] or (root / "clean" / "shard_00_end.json").exists():
        raise ValueError("the recorded shard-zero failure no longer matches this recovery protocol")
    return accepted[0]


def _source_member(root: Path, path: Path, digest: str, receipts: dict[str, str]) -> None:
    receipts[str(path.relative_to(root))] = digest


def _check_excluded_records(root: Path, plan: dict) -> int:
    """Inventory, but never select, all saved shard-zero observation payloads."""
    count = 0
    for arm in range(len(plan["bers"])):
        for index in range(0, FIELDS_PER_BER, 4):
            directory = root / "fields" / f"arm_{arm:02d}" / f"field_{index:03d}"
            if not (directory / "field.json").is_file():
                raise ValueError("excluded field metadata is missing")
            for policy in plan["policies"]:
                item = _read(directory / f"{policy['name']}.json")
                if item.get("payload_sha256") != digest_object({
                    key: value for key, value in item.items() if key != "payload_sha256"
                }):
                    raise ValueError("excluded observation payload digest differs")
                count += 1
    return count


def _audit_rows(root: Path, plan: dict, clean: dict) -> tuple[list[dict], dict[str, str]]:
    rows: list[dict] = []
    receipts: dict[str, str] = {}
    for path in (root / "plan.json", root / "FROZEN.json"):
        _source_member(root, path, sha256_file(path), receipts)
    for shard in range(4):
        path = root / "clean" / f"shard_{shard:02d}.json"
        _source_member(root, path, sha256_file(path), receipts)
        if shard in ACCEPTED_SHARDS:
            end = root / "clean" / f"shard_{shard:02d}_end.json"
            _source_member(root, end, sha256_file(end), receipts)
    for arm, ber in enumerate(plan["bers"]):
        for index in range(FIELDS_PER_BER):
            if index % 4 not in ACCEPTED_SHARDS:
                continue
            directory = root / "fields" / f"arm_{arm:02d}" / f"field_{index:03d}"
            field_path, meta_path = directory / "proposed.u8", directory / "field.json"
            meta = _read(meta_path)
            field_hash, popcount = _hash_and_count(field_path)
            if (meta["ber"] != ber or meta["seed"] != plan["field_seeds"][arm][index]
                or meta["universe_bits"] != plan["eligible_bits"]
                or meta["law_version"] != LAW_VERSION
                or meta["bytes"] != field_path.stat().st_size
                or meta["bytes"] != plan["eligible_bits"] // 8
                or meta["sha256"] != field_hash or meta["proposed"] != popcount):
                raise ValueError(f"saved proposed field differs: BER arm {arm}, field {index}")
            _source_member(root, field_path, field_hash, receipts)
            _source_member(root, meta_path, sha256_file(meta_path), receipts)
            for policy in plan["policies"]:
                item_path = directory / f"{policy['name']}.json"
                item = _read(item_path)
                event, risk = item["events"], item["risk"]
                if (item.get("payload_sha256") != digest_object({
                        key: value for key, value in item.items() if key != "payload_sha256"})
                    or item["plan_sha256"] != plan["plan_sha256"]
                    or item["arm"] != arm or item["field_index"] != index
                    or item["ber"] != ber or item["field_seed"] != meta["seed"]
                    or item["field_sha256"] != field_hash
                    or item["policy"] != policy["name"]
                    or item["mask_sha256"] != policy["mask_digest"]
                    or item["budget_bits"] != policy["budget"]
                    or event["field_sha256"] != field_hash
                    or event["mask_sha256"] != policy["mask_digest"]
                    or event["proposed"] != popcount
                    or event["proposed"] != event["applied"] + event["suppressed"]
                    or risk["total_units"] != plan["counts"]["scored_units_per_evaluation"]
                    or not math.isfinite(risk["metrics"]["mean_nll"])):
                    raise ValueError(f"invalid policy observation: {arm}/{index}/{policy['name']}")
                if policy["kind"] == "full" and (event["applied"] != 0 or risk != clean):
                    raise ValueError("accepted full-protection control differs from clean")
                if policy["kind"] == "none" and event["suppressed"] != 0:
                    raise ValueError("accepted unprotected control suppressed faults")
                rows.append({
                    "target": plan["target"], "ber": ber, "arm": arm,
                    "field_index": index, "field_seed": meta["seed"],
                    "field_sha256": field_hash, "policy": policy["name"],
                    "budget_bits": policy["budget"],
                    "coverage": policy["budget"] / plan["eligible_bits"],
                    "mask_sha256": policy["mask_digest"],
                    "proposed": event["proposed"], "applied": event["applied"],
                    "suppressed": event["suppressed"],
                    "mean_nll_nats": risk["metrics"]["mean_nll"],
                    "failed_units": risk["failed_units"],
                    "total_units": risk["total_units"],
                    "source_observation_sha256": sha256_file(item_path),
                })
                _source_member(root, item_path, rows[-1]["source_observation_sha256"], receipts)
    expected = len(plan["bers"]) * (FIELDS_PER_BER * 3 // 4) * len(plan["policies"])
    if len(rows) != expected:
        raise ValueError("accepted subset is incomplete")
    return rows, receipts


def summarize(rows: list[dict], plan: dict, clean: dict) -> tuple[list[dict], list[dict]]:
    by_condition: dict[tuple[int, str], list[dict]] = {}
    by_field = {}
    for row in rows:
        by_condition.setdefault((row["arm"], row["policy"]), []).append(row)
        by_field[(row["arm"], row["field_index"], row["policy"])] = row
    conditions = []
    for arm, ber in enumerate(plan["bers"]):
        for position, policy in enumerate(plan["policies"]):
            group = by_condition[(arm, policy["name"])]
            values = np.asarray([x["mean_nll_nats"] for x in group], dtype=np.float64)
            if len(values) != 18:
                raise ValueError("accepted subset has incomplete condition")
            rng = np.random.default_rng(1532901 + arm * 100 + position)
            boot = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
            conditions.append({
                "target": plan["target"], "ber": ber, "policy": policy["name"],
                "coverage": policy["budget"] / plan["eligible_bits"],
                "budget_bits": policy["budget"], "field_count": len(values),
                "mean_nll_nats": float(values.mean()),
                "loss_std_nats": float(values.std(ddof=1)),
                "loss_ci95_low": float(np.quantile(boot, .025)),
                "loss_ci95_high": float(np.quantile(boot, .975)),
                "clean_nll_nats": clean["metrics"]["mean_nll"],
                "mean_loss_degradation_nats": float(values.mean() - clean["metrics"]["mean_nll"]),
                "failed_unit_count": sum(x["failed_units"] for x in group),
                "mask_sha256": policy["mask_digest"],
            })
    comparisons = []
    for arm, ber in enumerate(plan["bers"]):
        for fraction in INTERIOR:
            budget = int(plan["eligible_bits"] * fraction)
            reference = f"msb_first_b{budget}"
            for method in METHODS:
                if method == "msb_first":
                    continue
                compared = f"{method}_b{budget}"
                diff = np.asarray([
                    by_field[(arm, i, compared)]["mean_nll_nats"] -
                    by_field[(arm, i, reference)]["mean_nll_nats"]
                    for i in range(FIELDS_PER_BER) if i % 4 in ACCEPTED_SHARDS
                ], dtype=np.float64)
                rng = np.random.default_rng(1533901 + len(comparisons))
                boot = diff[rng.integers(0, len(diff), size=(10000, len(diff)))].mean(axis=1)
                permuted = (diff * rng.choice((-1., 1.), size=(10000, len(diff)))).mean(axis=1)
                observed = float(diff.mean())
                comparisons.append({
                    "target": plan["target"], "ber": ber, "coverage": fraction,
                    "budget_bits": budget, "compared_method": method,
                    "reference_method": "msb_first", "field_count": len(diff),
                    "mean_nll_difference_nats": observed,
                    "ci95_low": float(np.quantile(boot, .025)),
                    "ci95_high": float(np.quantile(boot, .975)),
                    "p_signflip": (1 + int(np.sum(np.abs(permuted) >= abs(observed)))) / 10001,
                    "favorable_fields": int(np.sum(diff < 0)),
                    "adverse_fields": int(np.sum(diff > 0)),
                    "tied_fields": int(np.sum(diff == 0)),
                })
    _holm(comparisons)
    if len(conditions) != 208 or len(comparisons) != 160:
        raise ValueError("recovered condition/comparison grid incomplete")
    return conditions, comparisons


def _plot_ber(out: Path, conditions: list[dict], plan: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {(r["ber"], r["policy"]): r for r in conditions}
    labels = {
        "random": "Random", "msb_first": "MSB-first",
        "numerical_bit_jump": "Numerical jump", "global_first_order": "Global gradient",
        "per_example_absolute": "Absolute gradient", "per_example_positive": "Positive gradient",
    }
    colors = dict(zip(labels, plt.get_cmap("tab10").colors[:6]))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for ax, fraction in zip(axes.flat, INTERIOR):
        budget = int(plan["eligible_bits"] * fraction)
        for method, label in labels.items():
            records = [lookup[(ber, f"{method}_b{budget}")] for ber in plan["bers"]]
            means = np.asarray([r["mean_nll_nats"] for r in records])
            low = np.asarray([r["loss_ci95_low"] for r in records])
            high = np.asarray([r["loss_ci95_high"] for r in records])
            ax.plot(plan["bers"], means, marker="o", ms=3, lw=1.3,
                    color=colors[method], label=label)
            ax.fill_between(plan["bers"], low, high, color=colors[method], alpha=.10)
        control = [lookup[(ber, "unprotected")]["mean_nll_nats"] for ber in plan["bers"]]
        ax.plot(plan["bers"], control, "k--", lw=1.6, label="Unprotected")
        ax.axhline(lookup[(plan["bers"][0], "full_protection")]["mean_nll_nats"],
                   color="0.35", ls=":", lw=1.4, label="Full protection / clean")
        ax.set_xscale("log")
        ax.set_title(f"{100 * fraction:g}% of eligible bits protected")
        ax.grid(alpha=.20)
        ax.set_ylabel("Mean token NLL (nats)")
        ax.set_xlabel("Logical stored-bit BER")
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="center", bbox_to_anchor=(.5, .075),
               ncol=4, fontsize=9)
    fig.suptitle("Pythia INT8: 18 clean-conformant paired fields per BER (derived subset)")
    fig.text(.5, .017, "Original 24-field run failed shard-0 clean conformance; this is not the full frozen study.",
             ha="center", fontsize=9, color="darkred")
    fig.tight_layout(rect=(0, .16, 1, .95))
    fig.savefig(out / "loss_vs_ber.png", dpi=150)
    fig.savefig(out / "loss_vs_ber.svg")
    plt.close(fig)


def _plot_budget(out: Path, conditions: list[dict], plan: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {(r["ber"], r["policy"]): r for r in conditions}
    labels = {
        "random": "Random", "msb_first": "MSB-first",
        "numerical_bit_jump": "Numerical jump", "global_first_order": "Global gradient",
        "per_example_absolute": "Absolute gradient", "per_example_positive": "Positive gradient",
    }
    colors = dict(zip(labels, plt.get_cmap("tab10").colors[:6]))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ax, ber in zip(axes.flat, (1e-4, 1e-3, 1e-2, 1e-1)):
        for method, label in labels.items():
            records = [lookup[(ber, f"{method}_b{int(plan['eligible_bits'] * fraction)}")]
                       for fraction in INTERIOR]
            means = np.asarray([r["mean_nll_nats"] for r in records])
            low = np.asarray([r["loss_ci95_low"] for r in records])
            high = np.asarray([r["loss_ci95_high"] for r in records])
            xs = np.asarray(INTERIOR) * 100
            ax.plot(xs, means, marker="o", ms=4, lw=1.3,
                    color=colors[method], label=label)
            ax.fill_between(xs, low, high, color=colors[method], alpha=.10)
        ax.axhline(lookup[(ber, "unprotected")]["mean_nll_nats"],
                   color="black", ls="--", lw=1.5, label="Unprotected")
        ax.axhline(lookup[(ber, "full_protection")]["mean_nll_nats"],
                   color="0.35", ls=":", lw=1.4, label="Full protection / clean")
        ax.set_title(f"Logical BER {ber:g}")
        ax.set_xticks(np.asarray(INTERIOR) * 100)
        ax.grid(alpha=.20)
        ax.set_ylabel("Mean token NLL (nats)")
        ax.set_xlabel("Eligible bits protected (%)")
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="center", bbox_to_anchor=(.5, .075),
               ncol=4, fontsize=9)
    fig.suptitle("Pythia INT8: protection budget (18 clean-conformant paired fields per BER)")
    fig.text(.5, .017, "Original 24-field run failed shard-0 clean conformance; this is not the full frozen study.",
             ha="center", fontsize=9, color="darkred")
    fig.tight_layout(rect=(0, .16, 1, .95))
    fig.savefig(out / "loss_vs_budget.png", dpi=150)
    fig.savefig(out / "loss_vs_budget.svg")
    plt.close(fig)


def recover(source: Path, destination: Path, ready_manifest: Path) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if destination.exists() or destination == source or source in destination.parents:
        raise ValueError("write-once derived destination must not exist under original run")
    plan, _ = validate(ready_manifest, source, verify_masks=True)
    if plan["target"] != "pythia160m_int8" or status(source) != {
        "target": "pythia160m_int8", "complete_fields": 192, "planned_fields": 192,
        "saved_policy_observations": 4992, "planned_policy_observations": 4992,
    }:
        raise ValueError("original Pythia observation grid is not complete")
    clean = clean_subset(source, plan["execution_shards"])
    excluded_count = _check_excluded_records(source, plan)
    rows, source_receipts = _audit_rows(source, plan, clean)
    conditions, comparisons = summarize(rows, plan, clean)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pythia-recovery-", dir=destination.parent))
    try:
        _write_csv(staging / "trials.csv", rows)
        _write_csv(staging / "conditions.csv", conditions)
        _write_csv(staging / "paired_comparisons.csv", comparisons)
        _write_json(staging / "source_checksums.json", source_receipts)
        _plot_ber(staging, conditions, plan)
        _plot_budget(staging, conditions, plan)
        summary = {
            "schema": "netsurf-w8a8-derived-clean-shard-subset/v1",
            "status": "FINALIZED_DERIVED_VALIDATED_SUBSET_NOT_FROZEN_FULL_STUDY",
            "source_root": str(source), "source_plan_sha256": plan["plan_sha256"],
            "source_frozen_file_sha256": sha256_file(source / "FROZEN.json"),
            "accepted_shards": list(ACCEPTED_SHARDS), "excluded_shard": 0,
            "excluded_reason": "shard-0 initial clean differs and exact end-clean gate failed",
            "excluded_saved_observations": excluded_count,
            "accepted_fields_per_ber": 18, "accepted_fields": 144,
            "trial_observations": len(rows), "condition_rows": len(conditions),
            "paired_comparisons": len(comparisons), "clean_risk": clean,
            "unit": "independent proposed field, paired across all policies",
            "uncertainty": "bootstrap over 18 accepted fields per BER; Holm over 160 paired contrasts",
            "interpretation": "post-failure three-shard subset; excludes shard by clean-conformance status, not method outcome",
            "source_checksums_sha256": sha256_file(staging / "source_checksums.json"),
        }
        _write_json(staging / "analysis.json", summary)
        (staging / "README.md").write_text(
            "# Pythia wide-BER clean-conformant subset\n\n"
            "The original 24-field-per-BER run is **not finalized**. Shard 0 had an "
            "anomalous initial clean result and failed the final exact clean check. "
            "These derived results include only shards 1–3: 18 paired fields per BER, "
            "144 fields and 3,744 saved policy records. No model inference or fault "
            "schedule generation was used to build this directory. The excluded 1,248 "
            "saved policy records remain untouched in the source run. Results are "
            "exploratory because the full prespecified 24-field study did not finalize. "
            "Do not pool this subset with another Pythia study or cite it as the "
            "complete frozen campaign. Source member digests are in source_checksums.json.\n"
        )
        output_hashes = {p.name: sha256_file(p) for p in sorted(staging.iterdir()) if p.is_file()}
        manifest = {
            "schema": "netsurf-derived-recovery-manifest/v1",
            "status": summary["status"],
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "source_plan_sha256": plan["plan_sha256"],
            "output_sha256": output_hashes,
        }
        manifest["payload_sha256"] = digest_object(manifest)
        _write_json(staging / "manifest.json", manifest)
        _write_json(staging / "DERIVED_FINALIZED.json", {
            "manifest_sha256": sha256_file(staging / "manifest.json")
        })
        os.rename(staging, destination)
    except BaseException:
        # Never delete partial evidence automatically. The staging path is unique
        # and cannot overwrite either the source or an existing derived bundle.
        raise
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("ready_manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(recover(args.source, args.destination, args.ready_manifest), indent=2))


if __name__ == "__main__":
    main()
