#!/usr/bin/env bash
# Foreground one-shot v2 campaign; launch inside a dedicated tmux session.
set -euo pipefail
readonly root=/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration
readonly python=/projects/kona-assets/manuelbv/envs/torch-2.9.1-cpu-py311/bin/python
cd "$root"
export PYTHONPATH="$root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "$python" "$root/scripts/w8a8_concurrent_gate_v2.py" run
