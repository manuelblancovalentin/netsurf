#!/usr/bin/env bash
set -euo pipefail
readonly root=/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration
exec "$root/scripts/run_w8a8_campaign_once.sh" \
  "$root/workspace/plans/pythia160m_w8a8_reliability_v1" \
  "$root/workspace/runs/pythia160m_w8a8_reliability_v1_run1"
