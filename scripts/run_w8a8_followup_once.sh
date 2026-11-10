#!/usr/bin/env bash
set -Eeuo pipefail

repo=/projects/netsurf/manuelbv/workspace/implementation-worktrees/w8a8-followup
inference=/projects/kona-assets/manuelbv/envs/torch-2.9.1-cpu-py311/bin/python
analysis=/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python
ready=/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json
kind="${1:?expected qwen or pythia}"
case "$kind" in
    qwen) root="$repo/workspace/runs/qwen_crossover_p05_p075_v1" ;;
    pythia) root="$repo/workspace/runs/pythia_clean_shard_diagnostic_v1" ;;
    *) echo "unsupported follow-up: $kind" >&2; exit 2 ;;
esac
cd "$repo"
test -x "$inference"
test -x "$analysis"
test -f "$ready"
if test -e "$root/FINALIZED"; then
    echo "$kind already finalized at $root"
    exit 0
fi
if test "$kind" = qwen; then
    if test ! -f "$root/plan.json"; then
        "$inference" -m netsurf.w8a8_followup freeze-qwen "$ready" "$root"
    fi
    "$inference" -m netsurf.w8a8_followup run-qwen "$ready" "$root"
    "$analysis" -m netsurf.w8a8_followup finalize-qwen "$root"
else
    if test ! -f "$root/plan.json"; then
        "$inference" -m netsurf.w8a8_followup freeze-pythia "$ready" "$root" --processes 4 --repeats 2
    fi
    printf '0\n1\n2\n3\n' | xargs -P 4 -I '{}' sh -c '
        if test ! -f "$1/worker_$(printf "%02d" "$2").json"; then
            "$3" -m netsurf.w8a8_followup probe-pythia "$4" "$1" "$2"
        fi
    ' sh "$root" '{}' "$inference" "$ready"
    "$analysis" -m netsurf.w8a8_followup finalize-pythia "$root"
fi
