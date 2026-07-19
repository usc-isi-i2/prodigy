#!/usr/bin/env bash
# Overnight eval of the directed3_log (input-scaling-fix) arms. Waits until all five
# _log arms have state_dict_40000, then runs the frozen-encoder benchmark
# (regression / classification / static-LP) + capability probes with the correct
# per-arm flags, mirroring run_matched40k_tucker.sh's sweep(). Results land in
# scripts/plotting/{node_regression,node_classification,static_link_prediction}/data
# (model=<arm>_log, alongside the original arms) + capability_probes_directed3log.csv.
#
# Encoder groups (different architectures -> different eval flags):
#   e1  : E1_log                      sage,        directed3_log
#   e2  : E2_log,E4_log,E4r_log       sage_multi,  directed3_log
#   e2b : E2b_log                     sage_multi,  directed3_log, --no-bn-encoder
# (E4/E4r reuse E2's encoder at eval; the extra heads are ignored on load.)
#
# Launch on Tucker in tmux:
#   tmux new-session -d -s d3log_eval 'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
#     bash scripts/experiments/topology_feature_ssl/run_directed3log_overnight.sh \
#     > /tmp/d3log_eval.log 2>&1'
set -uo pipefail

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /dataMeR1/phil/gfm/prodigy

DIR=scripts/experiments/topology_feature_ssl
STATE_DIR="${STATE_DIR:-/dataMeR1/phil/gfm/prodigy/state}"
LOG_ROOT="${LOG_ROOT:-/dataMeR1/phil/gfm/prodigy/log}"
DATA_ROOT="${DATA_ROOT:-/dataMeR1/phil/data}"
GPUS="${GPUS:-0,1,2,3}"
DS=midterm,ukr_rus_twitter,covid19_twitter,twibot20,election2020
REG6=followers_count,friends_count,statuses_count,favourites_count,listed_count,account_age_days
RUNNER=scripts/experiments/eval/eval_ckpts_all_graph_tasks_tucker.py
STEP=40000
ARMS="E1_log E2_log E2b_log E4_log E4r_log"
say(){ echo "[d3log-eval $(date +%m-%d_%H:%M:%S)] $*"; }

run_dir_of(){ ls -dt "${STATE_DIR}/tfssl_$1_"*/ 2>/dev/null | head -1; }
ckpt_at(){ local d; d=$(run_dir_of "$1"); local c="${d}checkpoint/state_dict_$2.ckpt"; [ -f "$c" ] && echo "$c"; }

# 1) wait for all five state_dict_40000 (poll 5 min, up to ~8h)
say "waiting for state_dict_${STEP} of:${ARMS}"
for i in $(seq 1 96); do
  missing=""
  for a in $ARMS; do [ -z "$(ckpt_at "$a" "$STEP")" ] && missing="$missing $a"; done
  [ -z "$missing" ] && { say "all 40k checkpoints present"; break; }
  say "poll $i — still waiting for:$missing"
  sleep 300
done

# 2) per-encoder-group model lists (skip any arm whose ckpt never appeared)
: > "$DIR/ml_log_e1.txt"; : > "$DIR/ml_log_e2.txt"; : > "$DIR/ml_log_e2b.txt"
c=$(ckpt_at E1_log  "$STEP"); [ -n "$c" ] && echo "E1_log $c"  >> "$DIR/ml_log_e1.txt"
for a in E2_log E4_log E4r_log; do c=$(ckpt_at "$a" "$STEP"); [ -n "$c" ] && echo "$a $c" >> "$DIR/ml_log_e2.txt"; done
c=$(ckpt_at E2b_log "$STEP"); [ -n "$c" ] && echo "E2b_log $c" >> "$DIR/ml_log_e2b.txt"
cat "$DIR/ml_log_e1.txt" "$DIR/ml_log_e2.txt" "$DIR/ml_log_e2b.txt" > "$DIR/ml_log_all.txt"
say "model lists:"; cat "$DIR/ml_log_all.txt"

# 3) benchmark sweeps (reg / static-LP / classification) per group
sweep(){ local ml=$1; shift; [ -s "$ml" ] || { say "skip empty $ml"; return 0; }
  python3 "$RUNNER" --model-list "$ml" --data-root "$DATA_ROOT" --datasets "$DS" --continue-on-error --gpus "$GPUS" "$@" --tasks reg --shots 10 --reg-transform log1p --reg-targets "$REG6"
  python3 "$RUNNER" --model-list "$ml" --data-root "$DATA_ROOT" --datasets "$DS" --continue-on-error --gpus "$GPUS" "$@" --tasks slp --shots 0 --slp-n-query 4
  python3 "$RUNNER" --model-list "$ml" --data-root "$DATA_ROOT" --datasets "$DS" --continue-on-error --gpus "$GPUS" "$@" --tasks pl  --shots 10
}
say "STAGE benchmark (reg/slp/pl)"
sweep "$DIR/ml_log_e1.txt"  --structural-features directed3_log
sweep "$DIR/ml_log_e2.txt"  --structural-features directed3_log --gnn-type sage_multi
sweep "$DIR/ml_log_e2b.txt" --structural-features directed3_log --gnn-type sage_multi --no-bn-encoder

# 4) capability probes per group (eval only; final combined parse below)
say "STAGE probes"
STRUCTURAL=directed3_log MODEL_LIST="$DIR/ml_log_e1.txt" bash "$DIR/run_capability_probes.sh" --gpus "$GPUS" || say "probe e1 FAILED"
STRUCTURAL=directed3_log GNN_TYPE=sage_multi MODEL_LIST="$DIR/ml_log_e2.txt" bash "$DIR/run_capability_probes.sh" --gpus "$GPUS" || say "probe e2 FAILED"
STRUCTURAL=directed3_log GNN_TYPE=sage_multi NO_BN_ENCODER=1 MODEL_LIST="$DIR/ml_log_e2b.txt" bash "$DIR/run_capability_probes.sh" --gpus "$GPUS" || say "probe e2b FAILED"

# 5) parse everything (benchmark parser reads all logs; probes parsed over all _log arms)
say "STAGE parse"
python3 scripts/analysis/benchmark_tasks/parse_benchmark_eval_logs.py --log-root "$LOG_ROOT" --out-dir scripts/plotting || say "benchmark parse FAILED"
python3 "$DIR/parse_capability_probes.py" --log-root "$LOG_ROOT" --model-list "$DIR/ml_log_all.txt" \
  --out scripts/plotting/topology_feature_ssl/data/capability_probes_directed3log.csv || say "probe parse FAILED"

say "D3LOG_EVAL_DONE — results in scripts/plotting/{node_regression,node_classification,static_link_prediction}/data (model=*_log) + capability_probes_directed3log.csv"
