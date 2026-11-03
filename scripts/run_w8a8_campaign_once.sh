#!/usr/bin/env bash
# Foreground W8A8 scientific campaign; execute after researcher resource approval.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "usage: $0 /absolute/frozen-plan-dir /absolute/new-run-dir" >&2
  exit 2
fi
readonly source_root=/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration
readonly pinned_python=/projects/kona-assets/manuelbv/envs/torch-2.9.1-cpu-py311/bin/python
readonly ready=/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json
if [[ $1 != /* || $2 != /* || ! -x $pinned_python ]]; then
  echo "absolute paths and the pinned interpreter are required" >&2
  exit 2
fi
if pgrep -f 'm11_run_both_matched_lambda.py --worker' >/dev/null; then
  echo "M11 workers are active; defer this CPU-intensive campaign" >&2
  exit 3
fi
cd "$source_root"
export PYTHONPATH="$source_root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$pinned_python" -m netsurf.w8a8_campaign validate "$ready" "$1"
"$pinned_python" -m netsurf.w8a8_campaign selection "$ready" "$1" "$2"
"$pinned_python" -m netsurf.w8a8_campaign seal "$ready" "$1" "$2"
"$pinned_python" -m netsurf.w8a8_campaign final "$ready" "$1" "$2"
"$pinned_python" -m netsurf.w8a8_campaign aggregate "$ready" "$1" "$2"
"$pinned_python" -m netsurf.w8a8_campaign export "$ready" "$1" "$2"
cat "${2}_export/mac_scp_command.txt"
