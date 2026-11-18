#!/usr/bin/env bash
# Foreground KWS equal-query campaign; frozen selection before final fields.
set -euo pipefail
if [[ $# -ne 2 || $1 != /* || $2 != /* ]]; then
  echo "usage: $0 /absolute/frozen-plan.json /absolute/run-dir" >&2
  exit 2
fi
readonly source_root=/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration
readonly pinned_python=/projects/kona-assets/manuelbv/.venv-mlperf/bin/python
if [[ ! -x $pinned_python ]]; then echo "pinned MLPerf interpreter missing" >&2; exit 2; fi
if pgrep -f 'm11_run_both_matched_lambda.py --worker' >/dev/null; then
  echo "M11 scaling workers are active; defer this CPU campaign until they finish" >&2
  exit 3
fi
cd "$source_root"
export PYTHONPATH="$source_root"
"$pinned_python" -m netsurf.equal_compute_kws_campaign validate "$1" "$2"
"$pinned_python" -m netsurf.equal_compute_kws_campaign selection "$1" "$2"
"$pinned_python" -m netsurf.equal_compute_kws_campaign seal "$1" "$2"
"$pinned_python" -m netsurf.equal_compute_kws_campaign final "$1" "$2"
"$pinned_python" -m netsurf.equal_compute_kws_campaign aggregate "$1" "$2"
"$pinned_python" -m netsurf.equal_compute_kws_campaign export "$1" "$2"
cat "${2}_export/mac_scp_command.txt"
