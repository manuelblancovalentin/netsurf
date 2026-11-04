#!/usr/bin/env bash
# Researcher-controlled foreground continuation after finalized Pythia v3.
set -euo pipefail
readonly root=/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration
readonly python=/projects/kona-assets/manuelbv/envs/torch-2.9.1-cpu-py311/bin/python
readonly ready=/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json
readonly pythia_plan="$root/workspace/plans/pythia160m_w8a8_reliability_v3"
readonly pythia_run="$root/workspace/runs/pythia160m_w8a8_reliability_v3_run1"
cd "$root"
export PYTHONPATH="$root" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$python" -m netsurf.w8a8_campaign export "$ready" "$pythia_plan" "$pythia_run" >/dev/null
"$python" - "$pythia_run" <<'PY'
import json
from pathlib import Path
import sys
from netsurf.manifests import sha256_file
run = Path(sys.argv[1])
receipt = run.with_name(run.name + "_runtime_receipt.json")
export = run.with_name(run.name + "_export")
if not receipt.is_file() or json.loads(receipt.read_text()).get("exit_code") != 0:
    raise SystemExit("Pythia v3 runtime has not completed successfully")
if not (export / "FINALIZED").is_file() or not (export / "complete.zip").is_file():
    raise SystemExit("Pythia v3 complete export is not finalized")
print("Pythia finalized ZIP verified; SHA-256=" + sha256_file(export / "complete.zip"), flush=True)
PY
"$python" "$root/scripts/admit_w8a8_lm_v3.py" run qwen
exec "$python" "$root/scripts/report_lm_w8a8_v3.py" \
  "$root/workspace/plans/qwen3_0_6b_w8a8_reliability_v3" \
  "$root/workspace/runs/qwen3_0_6b_w8a8_reliability_v3_run1"
