#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 FROZEN_SCHEMA_3_6_CONFIG.toml EXPECTED_SOURCE_COMMIT" >&2
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
if [[ "$(git rev-parse HEAD)" != "$2" ]]; then
  echo "Source revision differs from the frozen expected commit; refusing the run" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Implementation worktree has source or untracked changes; refusing the run" >&2
  exit 2
fi
"$python_path" -m netsurf validate "$configuration_path"
"$python_path" -c '
import hashlib, json, sys
from pathlib import Path
from netsurf.api import configuration_operation_plan, load_configuration
config = load_configuration(Path(sys.argv[1]))
if config.schema_version != "3.6.0" or config.benchmark.adapter not in {
    "mlperf_tiny_ic_int8_v1", "mlperf_tiny_ad_int8_v1"
} or {item.name for item in config.estimators if item.name.startswith("fkeras_")} != {"fkeras_paper"}:
    raise SystemExit("Runner accepts only schema-3.6 ResNet-8/ToyCar published-form FKeras studies")
if not Path(config.source_path).is_relative_to(Path.cwd() / "config/experiments"):
    raise SystemExit("Runner accepts only a frozen config/experiments TOML in this worktree")
frozen_hashes = {
    "m12_resnet8_fkeras_matched_grid_v1.toml": "237f4697d69d9778cfd2f08510cc5ec4099ce1d1693b854c77906b83e901496a",
    "m12_resnet8_fkeras_paired_precision_v1.toml": "00ab9e7e7697142c3205a4476b64c3ce7efe1a3f17f9ee68c79fefa344a67e46",
    "m12_toycar_fkeras_matched_grid_v1.toml": "f8b59c0b3674f84ffd126fda82b6c03b3f441216b63e846aa95f591fc446d806",
    "m12_toycar_fkeras_paired_precision_v1.toml": "967f4048e939a04b98f0ff81a600e8a450b6ed5bb2ba65ac65b18c7c72614057",
}
source = Path(config.source_path)
if source.name not in frozen_hashes or hashlib.sha256(source.read_bytes()).hexdigest() != frozen_hashes[source.name]:
    raise SystemExit("Configuration is not one of the four hash-frozen M12 FKeras plans")
if any((Path.cwd() / "workspace/runs" / config.name).glob("*/FINALIZED")):
    raise SystemExit("A finalized bundle already exists for this frozen experiment name; refusing a repeat run")
plan = configuration_operation_plan(config)
reference = ({"policy_evaluations": 6656, "cohort_size": 1000,
              "coordinator_elapsed_hours": 43960.399230125826 / 3600,
              "historical_full_run_peak_mib": 5087.98046875,
              "bounded_score_peak_mib": 1071388 / 1024}
             if config.benchmark.adapter == "mlperf_tiny_ic_int8_v1" else
             {"policy_evaluations": 832, "cohort_size": 64,
              "coordinator_elapsed_hours": 3382.9723196299747 / 3600,
              "historical_full_run_peak_mib": 10754.8125,
              "bounded_score_peak_mib": 1187208 / 1024})
policy_evaluations = (len(config.protection.probabilities) *
                      config.execution.trials * config.replications.count *
                      (2 + len(config.policies) * len(config.protection.coverages)))
rough_hours = (reference["coordinator_elapsed_hours"] * policy_evaluations /
                     reference["policy_evaluations"] *
                     config.samples.evaluation_count / reference["cohort_size"])
print(json.dumps({"experiment": config.name, "source_commit": sys.argv[2],
                  "configuration_sha256": frozen_hashes[source.name], "planned_operations": plan,
                  "total_operations": sum(plan.values()),
                  "policy_evaluations": policy_evaluations,
                  "rough_end_to_end_hours_from_historical_ratio": round(rough_hours, 2),
                  "historical_reference": reference,
                  "warning": "Timing scales a finalized historical coordinator wall time by policy evaluations and cohort size; setup, ranking, BER mix and finalization need not scale linearly. The full-run peak is a historical measurement, not a cap for this run. Check live host resources before launch."}, indent=2))
' "$configuration_path" "$2"

echo "Starting foreground scientific run; this script does not install packages, spawn a background job, or update a database." >&2
"$python_path" -m netsurf run "$configuration_path"
