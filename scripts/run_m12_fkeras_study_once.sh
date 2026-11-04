#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 FROZEN_SCHEMA_3_6_CONFIG.toml" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "$script_dir/.." && pwd)"
configuration_path="$(realpath -- "$1")"
python_path="/projects/kona-assets/manuelbv/.venv-mlperf/bin/python"

if [[ ! -x "$python_path" || ! -f "$configuration_path" ]]; then
  echo "Pinned MLPerf interpreter or configuration is unavailable" >&2
  exit 2
fi

cd -- "$repository_root"
"$python_path" -m netsurf validate "$configuration_path"
"$python_path" -c '
import json, sys
from pathlib import Path
from netsurf.api import configuration_operation_plan, load_configuration
config = load_configuration(Path(sys.argv[1]))
if config.schema_version != "3.6.0" or config.benchmark.adapter not in {
    "mlperf_tiny_ic_int8_v1", "mlperf_tiny_ad_int8_v1"
} or {item.name for item in config.estimators if item.name.startswith("fkeras_")} != {"fkeras_paper"}:
    raise SystemExit("Runner accepts only schema-3.6 ResNet-8/ToyCar published-form FKeras studies")
if "config/examples/" in str(config.source_path):
    raise SystemExit("Engineering example is not a frozen production configuration")
plan = configuration_operation_plan(config)
reference = ({"fault_evaluations": 6656, "cohort_size": 1000,
              "wall_hours": 6.2, "bounded_score_peak_mib": 1071388 / 1024}
             if config.benchmark.adapter == "mlperf_tiny_ic_int8_v1" else
             {"fault_evaluations": 832, "cohort_size": 64,
              "wall_hours": 0.7, "bounded_score_peak_mib": 1187208 / 1024})
rough_fault_hours = (reference["wall_hours"] * plan["Fault evaluation"] /
                     reference["fault_evaluations"] *
                     config.samples.evaluation_count / reference["cohort_size"])
print(json.dumps({"experiment": config.name, "planned_operations": plan,
                  "total_operations": sum(plan.values()),
                  "rough_fault_evaluation_hours": round(rough_fault_hours, 2),
                  "historical_reference": reference,
                  "warning": "Timing scales an approximate historical throughput by cohort size; it excludes setup, ranking and bundle writing, is not a resource-admission receipt, and may be inaccurate at different BERs. Peak MiB is bounded score construction only, not whole-run peak. Check host resources before launch."}, indent=2))
' "$configuration_path"

echo "Starting foreground scientific run; this script does not install packages, spawn a background job, or update a database." >&2
"$python_path" -m netsurf run "$configuration_path"
