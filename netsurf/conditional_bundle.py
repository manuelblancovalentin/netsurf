"""Immutable result bundles for schema-6 exact conditional-damage studies."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from . import __version__
from .conditional_damage import ConditionalOutcome
from .manifests import git_provenance, runtime_provenance, sha256_file
from .result_bundle import ResultBundleError, _inventory, _write_bytes, _write_csv, _write_json


def _report(outcome: ConditionalOutcome) -> str:
    payload = outcome.scientific_payload
    interval_label = f"{100.0 * outcome.configuration.execution.confidence:g}% CI"
    lines = [
        f"# {outcome.configuration.name}", "",
        outcome.configuration.description, "",
        "## Scientific question", "", outcome.configuration.objective, "",
        "## Protocol", "",
        "For each prespecified candidate bit, NetSuRF measures the exact encoded-XOR risk "
        "difference `D_b(G) = R(G ∪ {b}) - R(G)`. The candidate is removed from each "
        "common Bernoulli background before the paired comparison. Calibration and held-out "
        "evaluation use disjoint samples and independently seeded background fields.", "",
        "The reported confidence intervals describe background-field variability conditional "
        "on this frozen model, fixed candidate list, fixed sample partition, and logical BER. "
        "They are not uncertainty over models, candidate sampling, or physical radiation.", "",
        "## Conditional-damage estimates", "",
        f"| Split | Logical BER | Candidate | Isolated damage | Conditional mean | {interval_label} | Negative fields |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    measurements = outcome.measurements
    for row in outcome.summaries:
        estimate = row["conditional_damage"]
        matching = [
            value["conditional_damage"] for value in measurements
            if value["split"] == row["split"]
            and value["logical_ber"] == row["logical_ber"]
            and value["candidate"] == row["candidate"]
        ]
        negative = sum(value < 0.0 for value in matching)
        low, high = estimate["confidence_interval"]
        lines.append(
            f"| {row['split']} | {row['logical_ber']:.3g} | `{row['candidate']}` | "
            f"{row['isolated_damage']:.8g} | {estimate['mean']:.8g} | "
            f"[{low:.8g}, {high:.8g}] | {negative}/{len(matching)} |"
        )
    lines.extend(["", "## Rank-order diagnostics", ""])
    for row in outcome.rank_comparisons:
        correlation = row["spearman_average_rank"]
        correlation_text = "undefined (all ties)" if correlation is None else f"{correlation:.6g}"
        lines.append(
            f"- {row['split']}, BER {row['logical_ber']:.3g}: "
            f"{row['strict_rank_reversals']}/{row['strict_comparable_pairs']} strict pairwise "
            f"reversals; average-rank Spearman {correlation_text}."
        )
    if outcome.oracle_summaries:
        lines.extend(["", "## Restricted protection oracle", ""])
        lines.extend([
            "The empirically optimized mask is selected exclusively on the model-selection "
            "partition and its independently seeded outside-background fields. Held-out "
            "selected-minus-isolated differences are paired by outside field. The reported "
            "held-out optimum is post-hoc diagnostic information and is not a valid selected "
            "policy or an unbiased confirmation estimate.", "",
            "| BER | K | Budget | Selected mask | Isolated mask | Held-out selected risk | Selected - isolated | Post-hoc regret |",
            "|---:|---:|---:|---|---|---:|---:|---:|",
        ])
        for row in outcome.oracle_summaries:
            lines.append(
                f"| {row['logical_ber']:.3g} | {row['candidate_count']} | {row['budget']} | "
                f"`{', '.join(row['selected_mask_bits'])}` | "
                f"`{', '.join(row['isolated_mask_bits'])}` | "
                f"{row['held_out_selected_risk']['mean']:.8g} | "
                f"{row['held_out_selected_minus_isolated']['mean']:.8g} | "
                f"{row['posthoc_held_out_regret']['mean']:.8g} |"
            )
    lines.extend([
        "", "## Limitations", "",
        "This bounded candidate study does not estimate a population-wide ranking or establish "
        "causal architectural mechanisms. The restricted oracle selects a mask only within the "
        "explicit candidate subcube; it does not optimize over the full network bit population. "
        "Logical BER is not a calibrated SEU rate, and exact TFLite XOR damage is authoritative; "
        "no float-shadow perturbation is used.", "",
        "## Reproduction", "",
        f"`netsurf conditional run {outcome.configuration.source_path}`", "",
        f"Scientific payload SHA-256: `{outcome.scientific_payload_sha256}`", "",
    ])
    return "\n".join(lines)


def create_conditional_bundle(
    outcome: ConditionalOutcome, *, repository_root: Path, run_id: str | None = None,
) -> Path:
    """Write a complete bundle through staging, then atomically finalize it."""

    now = datetime.now(timezone.utc)
    identifier = run_id or f"{now.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:8]}"
    root = outcome.configuration.output.root / outcome.configuration.name
    destination = root / identifier
    staging = root / f".{identifier}.incomplete"
    root.mkdir(parents=True, exist_ok=True)
    if destination.exists() or staging.exists():
        raise ResultBundleError(f"run directory already exists: {destination}")
    staging.mkdir()
    try:
        _write_bytes(staging / "config.toml", outcome.configuration.source_text.encode("utf-8"))
        _write_json(staging / "resolved_config.json", outcome.configuration.resolved_document())
        _write_json(staging / "scientific_payload.json", outcome.scientific_payload)
        _write_json(staging / "summary.json", {
            "schema_version": outcome.configuration.schema_version,
            "scientific_protocol": outcome.configuration.scientific_protocol,
            "experiment_name": outcome.configuration.name,
            "benchmark_id": outcome.configuration.benchmark.id,
            "risk_metric": outcome.configuration.risk_metric,
            "candidate_count": len(outcome.candidate_records),
            "background_field_count": len(outcome.background_records),
            "conditional_estimates": list(outcome.summaries),
            "rank_comparisons": list(outcome.rank_comparisons),
            "restricted_oracle": list(outcome.oracle_summaries),
            "scientific_payload_sha256": outcome.scientific_payload_sha256,
        })
        _write_csv(staging / "measurements/conditional_damage.csv", [
            "schema_version", "split", "sample_sha256", "logical_ber", "field", "field_seed",
            "candidate", "parameter", "flat_index", "bit_index", "master_selected_candidate",
            "master_schedule_sha256", "background_schedule_sha256", "candidate_schedule_sha256",
            "clean_risk", "isolated_fault_risk", "isolated_damage", "background_risk",
            "background_plus_candidate_risk", "conditional_damage", "background_proposed_events",
            "background_applied_events", "background_suppressed_events", "background_unique_applied_bits",
            "candidate_proposed_events", "candidate_applied_events", "candidate_suppressed_events",
            "candidate_unique_applied_bits", "empty_proposed_events", "isolated_proposed_events",
        ], ({"schema_version": outcome.configuration.schema_version, **row} for row in outcome.measurements))
        _write_csv(staging / "measurements/conditional_summary.csv", [
            "schema_version", "split", "sample_sha256", "logical_ber", "candidate", "parameter",
            "flat_index", "bit_index", "isolated_damage", "conditional_mean", "conditional_std",
            "conditional_se", "ci_low", "ci_high", "background_field_count", "uncertainty_unit",
        ], ({
            "schema_version": outcome.configuration.schema_version, "split": row["split"],
            "sample_sha256": row["sample_sha256"], "logical_ber": row["logical_ber"],
            "candidate": row["candidate"], "parameter": row["parameter"],
            "flat_index": row["flat_index"], "bit_index": row["bit_index"],
            "isolated_damage": row["isolated_damage"],
            "conditional_mean": row["conditional_damage"]["mean"],
            "conditional_std": row["conditional_damage"]["standard_deviation"],
            "conditional_se": row["conditional_damage"]["standard_error"],
            "ci_low": row["conditional_damage"]["confidence_interval"][0],
            "ci_high": row["conditional_damage"]["confidence_interval"][1],
            "background_field_count": row["background_field_count"],
            "uncertainty_unit": row["uncertainty_unit"],
        } for row in outcome.summaries))
        _write_json(staging / "artifacts/candidates.json", list(outcome.candidate_records))
        _write_json(staging / "artifacts/background_fields.json", list(outcome.background_records))
        _write_json(staging / "artifacts/rank_comparisons.json", list(outcome.rank_comparisons))
        if outcome.oracle_response_records:
            _write_csv(staging / "measurements/restricted_oracle_responses.csv", [
                "schema_version", "phase", "logical_ber", "field", "field_seed",
                "candidate_state", "candidate_bits", "outside_background_sha256",
                "risk", "schedule_sha256", "proposed_events", "applied_events",
                "suppressed_events", "unique_applied_bits", "cache_hit",
            ], ({"schema_version": outcome.configuration.schema_version, **row}
                for row in outcome.oracle_response_records))
            _write_json(
                staging / "artifacts/restricted_oracle_backgrounds.json",
                list(outcome.oracle_background_records),
            )
            _write_json(
                staging / "artifacts/restricted_oracle_masks.json",
                list(outcome.oracle_summaries),
            )
        if outcome.configuration.output.report:
            _write_bytes(staging / "report.md", _report(outcome).encode("utf-8"))
        manifest = {
            "schema_version": outcome.configuration.schema_version,
            "run_id": identifier, "status": "complete",
            "source": git_provenance(repository_root.resolve()),
            "runtime": runtime_provenance(),
            "versions": {"netsurf": __version__, "conditional_schema": outcome.configuration.schema_version,
                         "scientific_protocol": outcome.configuration.scientific_protocol},
            "benchmark": {
                **outcome.configuration.resolved_document()["benchmark"],
                "checkpoint_sha256": sha256_file(outcome.configuration.benchmark.checkpoint),
                "asset_lock_sha256": sha256_file(outcome.configuration.benchmark.asset_lock),
            },
            "execution": {
                "runtime_seconds": outcome.runtime_seconds,
                "calibration_background_seed": outcome.configuration.backgrounds.calibration_seed,
                "evaluation_background_seed": outcome.configuration.backgrounds.evaluation_seed,
            },
            "outputs": _inventory(staging),
            "scientific_payload_sha256": outcome.scientific_payload_sha256,
        }
        _write_json(staging / "manifest.json", manifest)
        _write_json(staging / "checksums.json", {
            "schema_version": outcome.configuration.schema_version,
            "files": {row["path"]: row["sha256"] for row in _inventory(staging)},
        })
        _write_bytes(staging / "FINALIZED", (
            f"scientific_payload_sha256={outcome.scientific_payload_sha256}\n"
        ).encode("ascii"))
        staging.rename(destination)
        return destination
    except BaseException:
        # Preserve incomplete evidence for diagnosis; never finalize or overwrite it.
        raise


def verify_conditional_bundle(path: str | Path) -> dict[str, object]:
    root = Path(path).expanduser().resolve()
    if not (root / "FINALIZED").is_file():
        raise ResultBundleError(f"not a finalized conditional bundle: {root}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    checksums = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in {"6.0.0", "6.1.0"}:
        raise ResultBundleError(f"unsupported conditional manifest: {manifest.get('schema_version')!r}")
    for relative, expected in checksums["files"].items():
        actual = sha256_file(root / relative)
        if actual != expected:
            raise ResultBundleError(f"checksum mismatch for {relative}: expected {expected}, got {actual}")
    return manifest
