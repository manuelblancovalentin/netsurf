#!/usr/bin/env bash
set -euo pipefail
cd /projects/netsurf/manuelbv/workspace/implementation-worktrees/w8a8-ber-stress
mkdir -p workspace/runs
bash scripts/run_w8a8_high_ber_pilot_once.sh 2>&1 | tee workspace/runs/w8a8_high_ber_exposure_pilot_v1.log
