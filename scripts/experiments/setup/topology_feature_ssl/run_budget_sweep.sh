#!/usr/bin/env bash
# Transfer budget sweep: eval B0 and E1 at several pretrain checkpoints on a fast
# benchmark subset, to find where the DOWNSTREAM (transfer) metrics plateau vs
# pretrain step. NM saturates ~30-40k, but NM is the feature-shortcut objective
# this experiment probes — NM plateau != transfer plateau — so we measure transfer
# directly. The result sets the data-driven minimum budget for E2-E4 (and, since
# the reading chain needs matched budget, tells us which B0/B1/E1 checkpoints to
# compare them against).
#
# Fast subset: small graphs only (election2020 79k, twibot20) + reg & cls, so the
# sweep is minutes not hours. Run on Tucker (conda on PATH), ideally after the main
# downstream frees the GPUs.
#   STEPS="20000 40000 60000 120000" GPUS=0,1,2,3 bash .../run_budget_sweep.sh
set -uo pipefail
DIR=scripts/experiments/topology_feature_ssl
STATE_DIR="${STATE_DIR:-/dataMeR1/phil/gfm/prodigy/state}"
LOG_ROOT="${LOG_ROOT:-/dataMeR1/phil/gfm/prodigy/log}"
STEPS="${STEPS:-20000 40000 60000 120000}"
GPUS="${GPUS:-0,1,2,3}"
RUNNER=scripts/eval/eval_ckpts_all_graph_tasks_tucker.py

run_dir_of(){ ls -dt "${STATE_DIR}/tfssl_$1_"*/ 2>/dev/null | head -1; }
mk_list(){  # arm -> model_list_budget_<arm>.txt with <arm>_step<N> rows (existing ckpts)
  local arm=$1 d; d=$(run_dir_of "$arm"); local out="$DIR/model_list_budget_${arm}.txt"; : > "$out"
  for s in $STEPS; do local c="${d}checkpoint/state_dict_${s}.ckpt"; [ -f "$c" ] && echo "${arm}_step${s} $c" >> "$out"; done
  echo "$out"
}
B0L=$(mk_list B0); E1L=$(mk_list E1)
echo "== B0 budget list =="; cat "$B0L"; echo "== E1 budget list =="; cat "$E1L"

COMMON=(--python python3 --data-root /dataMeR1/phil/data
        --datasets election2020,twibot20 --continue-on-error --gpus "$GPUS")
sweep(){  # $1=model_list  $2..=extra (e.g. --structural-features directed3)
  local ml=$1; shift
  python3 "$RUNNER" --model-list "$ml" "${COMMON[@]}" "$@" --tasks reg --shots 10 --reg-transform log1p --reg-targets followers_count,statuses_count,account_age_days
  python3 "$RUNNER" --model-list "$ml" "${COMMON[@]}" "$@" --tasks pl  --shots 10
}
[ -s "$B0L" ] && sweep "$B0L"
[ -s "$E1L" ] && sweep "$E1L" --structural-features directed3

python3 scripts/analysis/benchmark_tasks/parse_benchmark_eval_logs.py --log-root "$LOG_ROOT" --out-dir scripts/experiments/analysis
python3 "$DIR/analyze_budget.py"
echo "BUDGET_SWEEP_DONE"
