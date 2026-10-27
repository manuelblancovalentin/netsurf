#!/usr/bin/env bash
# Execute inside tmux -CC AFTER `proj activate netsurf`.
# This is a child shell, so a failed benchmark does not close your tmux shell.
set -euo pipefail
mode="${1:-core}"
core=(random msb_first gradient_signed_delta per_sample_positive hutchinson_hessian legacy_qpolargrad)
all=(random msb_first gradient_signed_delta gradient_magnitude per_sample_magnitude per_sample_positive hutchinson_hessian legacy_qpolargrad clean_activation_qpolargrad uncancelled_activation_drift)
case "$mode" in
  smoke) experiments=(mnist_m44_hist_smoke_msb);;
  core) experiments=(); for policy in "${core[@]}"; do experiments+=("mnist_m44_hist_${policy}_main_p01"); done;;
  all) experiments=(); for policy in "${all[@]}"; do experiments+=("mnist_m44_hist_${policy}_main_p01"); done;;
  transfer_core) experiments=(); for policy in "${core[@]}"; do experiments+=("mnist_m44_hist_${policy}_transfer_p01"); done;;
  transfer_all) experiments=(); for policy in "${all[@]}"; do experiments+=("mnist_m44_hist_${policy}_transfer_p01"); done;;
  *) echo "Usage: bash run_m44_historical.sh [smoke|core|all|transfer_core|transfer_all]" >&2; exit 2;;
esac
command -v netsurf >/dev/null || { echo "Run 'proj activate netsurf' inside tmux first" >&2; exit 1; }
[[ -d config/experiments ]] || { echo "Project must already be activated (it changes directory)" >&2; exit 1; }
for experiment in "${experiments[@]}"; do
 config="config/experiments/${experiment}.toml"
 echo "=== $(date -Is): VALIDATE $config ==="
 netsurf validate "$config" --resolved
 echo "=== $(date -Is): RUN $config ==="
 netsurf run "$config"
done
printf 'Completed %d experiment configuration(s).\n' "${#experiments[@]}"
