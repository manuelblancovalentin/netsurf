#!/usr/bin/env bash
# Foreground follow-up. Run only after the frozen plan has passed review.
set -euo pipefail
if [[ $# -ne 6 ]]; then
  echo "usage: $0 TARGET /absolute/pilot-complete.zip /absolute/frozen-READY.json /absolute/runtime-registry.json /absolute/plan-dir /absolute/run-dir" >&2
  exit 2
fi
readonly source_root=/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration
readonly pinned_python=/projects/kona-assets/manuelbv/envs/torch-2.9.1-cpu-py311/bin/python
for path in "$2" "$3" "$4" "$5" "$6"; do
  if [[ $path != /* ]]; then echo "absolute paths required" >&2; exit 2; fi
done
if [[ ! -x $pinned_python ]]; then echo "pinned interpreter missing" >&2; exit 2; fi
if pgrep -f 'm11_run_both_matched_lambda.py --worker' >/dev/null; then
  echo "M11 scaling workers are active; defer this CPU campaign until they finish" >&2
  exit 3
fi
cd "$source_root"
export PYTHONPATH="$source_root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$pinned_python" -m netsurf.transformer_lm_exponent_control validate "$1" "$2" "$3" "$4" "$5" "$6"
"$pinned_python" -m netsurf.transformer_lm_exponent_control final "$1" "$2" "$3" "$4" "$5" "$6"
"$pinned_python" -m netsurf.transformer_lm_exponent_control aggregate "$1" "$2" "$3" "$4" "$5" "$6"
"$pinned_python" -m netsurf.transformer_lm_exponent_control export "$1" "$2" "$3" "$4" "$5" "$6"
cat "${6}_export/mac_scp_command.txt"
