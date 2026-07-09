#!/usr/bin/env bash
# Matched-budget eval: B0 / B1 / E1 / E2 all at E2's final step (~40k episodes, the
# budget-sweep regression peak), FULL 6-target profile panel + static-LP + cls +
# 2x2 + capability probes + shot-matched leakage. Arms named <arm>_40k so they do
# NOT collide with the 110k rows. B0/B1 (768) and E1/E2 (771, directed3) run as
# separate sweeps. Run on Tucker after E2 pretrains.
#   GPUS=0,1,2,3 bash scripts/experiments/topology_feature_ssl/run_matched40k_tucker.sh
set -uo pipefail
DIR=scripts/experiments/topology_feature_ssl
STATE_DIR="${STATE_DIR:-/dataMeR1/phil/gfm/prodigy/state}"
LOG_ROOT="${LOG_ROOT:-/dataMeR1/phil/gfm/prodigy/log}"
DATA_ROOT="${DATA_ROOT:-/dataMeR1/phil/data}"
GPUS="${GPUS:-0,1,2,3}"
DS=midterm,ukr_rus_twitter,covid19_twitter,twibot20,election2020
REG6=followers_count,friends_count,statuses_count,favourites_count,listed_count,account_age_days
RUNNER=scripts/experiments/eval/eval_ckpts_all_graph_tasks_tucker.py
say(){ echo "[matched40k $(date +%H:%M:%S)] $*"; }

run_dir_of(){ ls -dt "${STATE_DIR}/tfssl_$1_"*/ 2>/dev/null | head -1; }
final_ckpt(){ local d; d=$(run_dir_of "$1"); [ -z "$d" ] && return 1
  ls "${d}checkpoint/"state_dict_*.ckpt 2>/dev/null | sed -E 's#.*_([0-9]+)\.ckpt#\1 &#' | sort -n | tail -1 | cut -d" " -f2-; }
ckpt_at(){ local d; d=$(run_dir_of "$1"); local c="${d}checkpoint/state_dict_$2.ckpt"; [ -f "$c" ] && echo "$c"; }

# Match everyone to E2's final step (true matched budget).
E2C=$(final_ckpt E2) || { echo "E2 has no checkpoint yet — pretrain first" >&2; exit 1; }
STEP=$(echo "$E2C" | sed -E 's#.*_([0-9]+)\.ckpt#\1#')
say "matching all arms at step ${STEP} (E2 final)"

: > "$DIR/model_list_40k_base.txt"
for a in B0 B1; do c=$(ckpt_at "$a" "$STEP"); [ -n "$c" ] && echo "${a}_40k $c" >> "$DIR/model_list_40k_base.txt"; done
: > "$DIR/model_list_40k_struct.txt"
c=$(ckpt_at E1 "$STEP"); [ -n "$c" ] && echo "E1_40k $c" >> "$DIR/model_list_40k_struct.txt"
echo "E2_40k $E2C" >> "$DIR/model_list_40k_struct.txt"
cat "$DIR/model_list_40k_base.txt" "$DIR/model_list_40k_struct.txt" > "$DIR/model_list_40k_all.txt"
say "model lists:"; cat "$DIR/model_list_40k_all.txt"

sweep(){ local ml=$1; shift
  python3 "$RUNNER" --model-list "$ml" --data-root "$DATA_ROOT" --datasets "$DS" --continue-on-error --gpus "$GPUS" "$@" --tasks reg --shots 10 --reg-transform log1p --reg-targets "$REG6"
  python3 "$RUNNER" --model-list "$ml" --data-root "$DATA_ROOT" --datasets "$DS" --continue-on-error --gpus "$GPUS" "$@" --tasks slp --shots 0 --slp-n-query 4
  python3 "$RUNNER" --model-list "$ml" --data-root "$DATA_ROOT" --datasets "$DS" --continue-on-error --gpus "$GPUS" "$@" --tasks pl --shots 10
}
say "STAGE T1 (full 6-target panel)"
[ -s "$DIR/model_list_40k_base.txt" ]   && sweep "$DIR/model_list_40k_base.txt"
[ -s "$DIR/model_list_40k_struct.txt" ] && sweep "$DIR/model_list_40k_struct.txt" --structural-features directed3

say "STAGE T2 2x2"
MODEL_LIST="$DIR/model_list_40k_base.txt"                       bash "$DIR/run_2x2_ablation.sh" --gpus "$GPUS"
STRUCTURAL=directed3 MODEL_LIST="$DIR/model_list_40k_struct.txt" bash "$DIR/run_2x2_ablation.sh" --gpus "$GPUS"
say "STAGE T3 probes"
MODEL_LIST="$DIR/model_list_40k_base.txt"                       bash "$DIR/run_capability_probes.sh" --gpus "$GPUS"
STRUCTURAL=directed3 MODEL_LIST="$DIR/model_list_40k_struct.txt" bash "$DIR/run_capability_probes.sh" --gpus "$GPUS"

say "leakage (full 6-panel, shot-matched)"
python3 "$DIR/leakage_baseline.py" --data-root "$DATA_ROOT" --targets "$REG6" \
  --out scripts/plotting/topology_feature_ssl/data/leakage_baseline_6panel.csv || say "leakage FAILED"

say "parse + analyze"
python3 scripts/analysis/benchmark_tasks/parse_benchmark_eval_logs.py --log-root "$LOG_ROOT" --out-dir scripts/plotting
python3 "$DIR/parse_2x2.py" --log-root "$LOG_ROOT" --model-list "$DIR/model_list_40k_all.txt" --out scripts/plotting/topology_feature_ssl/data/ablation_2x2_40k.csv || say "2x2 parse FAILED"
python3 "$DIR/parse_capability_probes.py" --log-root "$LOG_ROOT" --model-list "$DIR/model_list_40k_all.txt" --out scripts/plotting/topology_feature_ssl/data/capability_probes_40k.csv || say "probe parse FAILED"
python3 "$DIR/analyze_matched40k.py" || say "analyze FAILED"
say "MATCHED40K_DONE"
