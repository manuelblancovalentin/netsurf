"""Single-process foreground commands for the frozen targeted DistilBERT study."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import zipfile

from .manifests import canonical_json_bytes, sha256_file
from .transformer_campaign import (aggregate, freeze_distilbert_plan, run_final,
                                   run_selection, validate_targeted_plan)


ROOT = Path(__file__).resolve().parents[1]
PORTFOLIO = ROOT / "config/benchmark_expansion_registry.json"


def _figure(rows: list[dict], horizontal: str) -> str:
    """A dependency-free, labeled engineering figure; CE axes never mix tasks."""
    x_values = sorted({row[horizontal] for row in rows})
    y_values = [row["mean_nll"] for row in rows]
    low, high = min(y_values), max(y_values)
    if high == low:
        high = low + 1e-9
    methods = ("random", "significance", "exact_isolated")
    colors = ("#4978b5", "#a65a2a", "#278453")
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="780" height="450" viewBox="0 0 780 450">',
             '<rect width="780" height="450" fill="white"/>',
             '<text x="35" y="25" font-size="17">DistilBERT targeted 1,024-bit study: mean cross-entropy</text>',
             '<path d="M70 40V370H730" fill="none" stroke="black"/>',
             f'<text x="350" y="430" font-size="14">{horizontal} (BER is per eligible stored bit)</text>',
             '<text x="5" y="45" font-size="13">nats</text>']
    for index, value in enumerate(x_values):
        x = 70 + 660 * index / max(1, len(x_values) - 1)
        lines.append(f'<text x="{x-22:.1f}" y="390" font-size="12">{value:.6g}</text>')
    for index in range(5):
        y = 370 - 330 * index / 4
        value = low + (high - low) * index / 4
        lines.append(f'<text x="10" y="{y+4:.1f}" font-size="12">{value:.3f}</text>')
    fixed = "budget" if horizontal == "ber" else "ber"
    for j, fixed_value in enumerate(sorted({row[fixed] for row in rows})):
        for method, color in zip(methods, colors):
            points = []
            for index, value in enumerate(x_values):
                row = next((row for row in rows if row[horizontal] == value and
                            row[fixed] == fixed_value and row["method"] == method), None)
                if row is None:
                    continue
                x = 70 + 660 * index / max(1, len(x_values) - 1)
                y = 370 - 330 * (row["mean_nll"] - low) / (high - low)
                points.append(f"{x:.1f},{y:.1f}")
            if points:
                opacity = 1 - j * 0.2
                lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-opacity="{opacity:.2f}" stroke-width="2"/>')
                lines.append(f'<text x="500" y="{45+j*48+methods.index(method)*14}" fill="{color}" font-size="12">{method}, {fixed}={fixed_value:.6g}</text>')
    lines.append('</svg>')
    return "\n".join(lines) + "\n"


def export(plan_path: Path, readiness_path: Path, run_root: Path, *, scp_host: str) -> Path:
    plan = json.loads(Path(plan_path).read_text())
    validate_targeted_plan(plan)
    if sha256_file(Path(readiness_path)) != plan["asset_manifest_sha256"]:
        raise ValueError("READY snapshot differs from frozen plan")
    report = aggregate(plan_path, run_root)
    destination = Path(run_root).with_name(Path(run_root).name + "_export")
    if destination.exists():
        checksums = json.loads((destination / "checksums.json").read_text())
        if any(sha256_file(destination / name) != value for name, value in checksums.items()):
            raise ValueError("completed campaign export is corrupt")
        return destination
    stage = destination.with_name("." + destination.name + ".incomplete")
    if stage.exists():
        raise FileExistsError(f"partial campaign export retained: {stage}")
    stage.mkdir()
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=("ber", "budget", "method", "expected_applied_faults", "mean_nll", "accuracy", "failed_clip_fraction"))
    writer.writeheader(); writer.writerows(report["rows"])
    (stage / "table.csv").write_text(stream.getvalue())
    (stage / "ber_loss.svg").write_text(_figure(report["rows"], "ber"))
    (stage / "budget_loss.svg").write_text(_figure(report["rows"], "budget"))
    (stage / "summary.json").write_bytes(canonical_json_bytes({
        "schema_version": "transformer-targeted-summary/v1",
        "plan_sha256": plan["plan_sha256"], "clean": report["clean"],
        "comparisons": report["comparisons"], "scope": report["scope"]}) + b"\n")
    with zipfile.ZipFile(stage / "compact.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("summary.json", "table.csv", "ber_loss.svg", "budget_loss.svg"):
            archive.write(stage / name, name)
    with zipfile.ZipFile(stage / "complete.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(plan_path, "plan.json")
        archive.write(readiness_path, "assets/NETSURF_ASSET_READINESS.json")
        for path in sorted(Path(run_root).rglob("*")):
            if path.is_file():
                archive.write(path, "run/" + str(path.relative_to(run_root)))
        for name in ("summary.json", "table.csv", "ber_loss.svg", "budget_loss.svg"):
            archive.write(stage / name, "analysis/" + name)
        for relative in plan["source_files_sha256"]:
            archive.write(ROOT / relative, "source/" + relative)
    commands = "\n".join(f"scp {scp_host}:{(destination/name).as_posix()} ~/Downloads/{name}"
                         for name in ("compact.zip", "complete.zip")) + "\n"
    (stage / "mac_scp_commands.txt").write_text(commands)
    checksums = {name: sha256_file(stage / name) for name in
                 ("summary.json", "table.csv", "ber_loss.svg", "budget_loss.svg",
                  "compact.zip", "complete.zip", "mac_scp_commands.txt")}
    (stage / "checksums.json").write_bytes(canonical_json_bytes(checksums) + b"\n")
    for path in stage.iterdir():
        path.chmod(0o444)
    stage.rename(destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "validate", "selection", "final", "aggregate", "export"))
    parser.add_argument("plan", type=Path)
    parser.add_argument("readiness", type=Path)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--scp-host", default="kona")
    parser.add_argument("--engineering-receipt", type=Path)
    args = parser.parse_args(argv)
    if args.phase == "freeze":
        if args.engineering_receipt is None:
            parser.error("freeze requires --engineering-receipt")
        plan = freeze_distilbert_plan(args.readiness, PORTFOLIO,
            args.engineering_receipt, args.plan, args.run_root)
        result = {"status": "FROZEN", "plan_sha256": plan["plan_sha256"],
                  "request_counts": plan["costs"]}
    elif args.phase == "validate":
        plan = json.loads(args.plan.read_text())
        validate_targeted_plan(plan)
        if sha256_file(args.readiness) != plan["asset_manifest_sha256"]:
            raise ValueError("READY snapshot differs from frozen plan")
        result = {"status": "FROZEN_VALID", "plan_sha256": plan["plan_sha256"],
                  "request_counts": plan["costs"]}
    elif args.phase == "selection":
        result = run_selection(args.plan, args.readiness, PORTFOLIO, args.run_root)
    elif args.phase == "final":
        result = run_final(args.plan, args.readiness, PORTFOLIO, args.run_root)
    elif args.phase == "aggregate":
        result = aggregate(args.plan, args.run_root)
    else:
        result = {"export": str(export(args.plan, args.readiness, args.run_root,
                                        scp_host=args.scp_host))}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
