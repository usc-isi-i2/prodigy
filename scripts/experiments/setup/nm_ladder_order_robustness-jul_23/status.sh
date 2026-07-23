#!/usr/bin/env bash
# One-shot progress report for the order-robustness pipeline.
#
#   ssh tucker "bash /dataMeR1/phil/gfm/prodigy/scripts/experiments/setup/nm_ladder_order_robustness-jul_23/status.sh"
#
# Shows the pipeline phase, then each of the 11 rungs as DONE / running (step) / queued.
# A rung is DONE when its state_dict_40000.ckpt exists -- the matched-40k budget we
# evaluate. Training is configured for 50k and self-terminates, so "38000/50000" is 95%
# of the way to what we actually care about, not 76%.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_ROOT="${STATE_ROOT:-/dataMeR1/phil/gfm/prodigy/state}"

date "+=== %Y-%m-%d %H:%M:%S ==="
cat "${SCRIPT_DIR}/run_logs/pipeline_status.txt" 2>/dev/null || echo "(no status file yet)"
echo

cd "${SCRIPT_DIR}" || exit 1
done_count=0
for cfg in train_ord*.yaml; do
  [[ -f "$cfg" ]] || continue
  prefix="$(awk '/^prefix:/{print $2; exit}' "$cfg")"
  run_dir="$(ls -dt "${STATE_ROOT}/${prefix}_"*/ 2>/dev/null | head -1)"
  if [[ -n "$run_dir" && -f "${run_dir}checkpoint/state_dict_40000.ckpt" ]]; then
    printf "  DONE     %s\n" "$prefix"
    done_count=$((done_count + 1))
  else
    log="$(ls -t "run_logs/${cfg%.yaml}_gpu"*.log 2>/dev/null | head -1)"
    if [[ -n "$log" ]]; then
      step="$(tail -c 400 "$log" | tr '\r' '\n' | grep -o '[0-9]*/50000' | tail -1)"
      printf "  running  %s  %s\n" "$prefix" "${step:-loading graph}"
    else
      printf "  queued   %s\n" "$prefix"
    fi
  fi
done
echo
echo "rungs complete: ${done_count}/11  (40k ckpt = done)"

n_eval=$(ls -d /dataMeR1/phil/gfm/prodigy/log/eval_nm_ladder_ord*_nm_3shot_30way_* 2>/dev/null | wc -l | tr -d ' ')
[[ "${n_eval}" != "0" ]] && echo "eval jobs started: ${n_eval}/88"
for f in nm_ladder_order_robustness.csv nm_ladder_order_robustness_long.csv; do
  [[ -f "${SCRIPT_DIR}/${f}" ]] && echo "RESULT READY: ${f}"
done
