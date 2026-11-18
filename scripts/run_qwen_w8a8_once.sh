#!/usr/bin/env bash
set -euo pipefail
readonly root=/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration
exec "$root/scripts/run_w8a8_campaign_once.sh" \
  "$root/workspace/plans/qwen3_0_6b_w8a8_reliability_v1" \
  "$root/workspace/runs/qwen3_0_6b_w8a8_reliability_v1_run1"
