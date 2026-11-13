#!/usr/bin/env bash
# Run R: run one script's jobs in parallel, one process per job, with a cap on concurrent processes.
#   runR_launch.sh <masks|predict|stage1|stage2_bundle|stage2_build|stage2_eval_main|stage2_mc|stage2_eval_mc> <max_parallel>
# Logs: workspace/runs/paper1_runR_<kind>_logs/<job>.log in the paper worktree.  Jobs whose output exists are skipped
# by the scripts themselves (write-once), so a relaunch only runs what is missing.
set -euo pipefail
KIND=$1; P=$2
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/projects/kona-assets/manuelbv/.venv-mlperf/bin/python
RUNS=/projects/netsurf/manuelbv/workspace/implementation-worktrees/paper1-tecs-revision-v1/workspace/runs
LOGS=$RUNS/paper1_runR_${KIND}_logs
mkdir -p "$LOGS"
cd "$HERE"
"$PY" - "$KIND" <<'EOF' > "$LOGS/jobs.txt"
import sys
from runR_stage1 import NAMES, reps_for, FIELDS
kind = sys.argv[1]
for name in NAMES:
    for rep in reps_for(name):
        if kind == "masks":
            print(f"{name}_rep{rep} runR_masks.py --checkpoint {name} --rep {rep} --threads 2")
        elif kind == "predict":
            print(f"{name}_rep{rep} runR_predict.py job --checkpoint {name} --rep {rep} --threads 1")
        elif kind == "stage2_bundle":
            if rep == reps_for(name)[0]:
                print(f"{name} runR_stage2.py bundle --checkpoint {name}")
        elif kind == "stage2_build":
            from runR_stage2 import FRACTIONS as F2
            for b in F2:
                print(f"{name}_rep{rep}_b{b} runR_stage2.py build --checkpoint {name} --rep {rep} --budget {b}")
        elif kind == "stage2_mc":
            from runR_stage2 import FRACTIONS as F2, RATES as R2
            for b in F2:
                for p in R2:
                    print(f"{name}_rep{rep}_b{b}_p{p:g} runR_stage2.py mc --checkpoint {name} --rep {rep} --budget {b} --rate {p:g}")
        elif kind in ("stage2_eval_main", "stage2_eval_mc"):
            arms = kind.rsplit("_", 1)[1]
            for f in range(FIELDS):
                print(f"{name}_rep{rep}_f{f} runR_stage2.py eval --arms {arms} --checkpoint {name} --rep {rep} --field {f}")
        else:
            for f in range(FIELDS):
                print(f"{name}_rep{rep}_f{f} runR_stage1.py job --checkpoint {name} --rep {rep} --field {f} --threads 1")
EOF
export PY LOGS
xargs -P "$P" -L 1 sh -c 'id=$0; "$PY" "$@" > "$LOGS/$id.log" 2>&1 || echo "FAILED $id" >> "$LOGS/failed.txt"' < "$LOGS/jobs.txt"
echo "all jobs ended: $(date)"
if [ -s "$LOGS/failed.txt" ]; then cat "$LOGS/failed.txt"; exit 1; fi
echo "no failures"
