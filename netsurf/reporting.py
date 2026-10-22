"""Human-readable scientific reports generated from saved result data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def render_report(summary: Mapping[str, Any], resolved: Mapping[str, Any], *, figure_paths: tuple[str, ...] = ()) -> str:
    experiment, benchmark, fault = resolved["experiment"], resolved["benchmark"], resolved["fault"]
    lines = [
        f"# {experiment['name']}", "", experiment.get("description", ""), "",
        "## Objective", "", experiment.get("objective", "Not specified."), "",
        "## Experimental contract", "",
        f"- Benchmark: `{benchmark['id']}` (`{benchmark['adapter']}`).",
        f"- Dataset split: `{benchmark['split']}`; {benchmark['evaluation_samples']} evaluation samples and {benchmark['ranking_samples']} ranking samples.",
        f"- Quantization: `{benchmark['quantization']}`.",
        f"- Fault model: `{fault['distribution']}` logical `{fault['target']}` faults at probability `{fault['probability']}`, `{fault['persistence']}`.",
        f"- Independent uncertainty unit: fault trial; {summary['trial_count']} trials at confidence `{resolved['execution']['confidence']}`.",
        "- Policy comparisons within a trial use the same fault schedule and are therefore paired.",
        "- Protection is modeled as perfect suppression of selected logical-bit events; this is not a calibrated physical SEU rate or a hardware guarantee.",
        "", "## Baseline", "",
        f"Checkpoint accuracy: `{summary['baseline']['checkpoint']['accuracy']:.6f}`; categorical cross-entropy: `{summary['baseline']['checkpoint']['categorical_crossentropy']:.6f}`.",
        "", "## Measured and derived results", "",
        "Trial rows are measured evaluations. Means and confidence intervals below are derived across independent fault trials.", "",
        "| Policy | Protected bits | Coverage | Accuracy mean | Confidence interval | Cross-entropy mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in summary["protection_results"].items():
        accuracy, loss = result["accuracy"], result["categorical_crossentropy"]
        interval = accuracy["confidence_interval"]
        lines.append(f"| {name} | {result['protected_bits']} | {result['coverage']:.6f} | {accuracy['mean']:.6f} | [{interval[0]:.6f}, {interval[1]:.6f}] | {loss['mean']:.6f} |")
    if figure_paths:
        lines.extend(["", "## Figures", ""])
        for path in figure_paths:
            lines.extend([f"![Accuracy with trial-based confidence intervals]({path})", "", "Accuracy by policy; error bars are confidence intervals over independent persistent-fault trials.", ""])
    lines.extend([
        "## Reproducibility checks", "",
        *[f"- `{name}`: `{value}`." for name, value in summary["checks"].items()], "",
        "## Limitations", "",
        "This report does not establish physical radiation reliability, fabricated-silicon cost, or statistical superiority among policies. `legacy_qpolargrad` retains the recovered historical expression and is not redefined here.", "",
        "## Reproduction", "", "```bash", "netsurf run config.toml", "```", "",
        "The bundled `config.toml` is the exact submitted configuration; resolved paths and executed provenance are recorded separately in `resolved_config.json` and `manifest.json`.", "",
        f"Scientific payload SHA-256: `{summary['scientific_payload_sha256']}`", "",
    ])
    return "\n".join(lines)


def write_report(summary: Mapping[str, Any], resolved: Mapping[str, Any], output: Path, *, figure_paths: tuple[str, ...] = ()) -> Path:
    output = Path(output)
    output.write_text(render_report(summary, resolved, figure_paths=figure_paths), encoding="utf-8")
    return output
