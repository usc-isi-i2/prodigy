#!/usr/bin/env bash
# Downstream for topology_feature_ssl: eval sweep + diagnostics + parse for whichever
# arms have checkpoints. B0/B1 (input 768) and E1 (input 771, directed3) run as
# SEPARATE sweeps. Stages are ordered by importance (T1 benchmark -> T2 2x2 ->
# T3 probes) so partial completion still yields the most important tables.
#
# Assumes checkpoints already exist. Run on Tucker in the prodigy env (conda on PATH).
#   GPUS=0,1,2,3 bash scripts/experiments/topology_feature_ssl/run_downstream_tucker.sh
set -uo pipefail
DIR=scripts/experiments/topology_feature_ssl
STATE_DIR="${STATE_DIR:-/dataMeR1/phil/gfm/prodigy/state}"
LOG_ROOT="${LOG_ROOT:-/dataMeR1/phil/gfm/prodigy/log}"
DATA_ROOT="${DATA_ROOT:-/dataMeR1/phil/data}"
GPUS="${GPUS:-0,1,2,3}"
say() { echo "[downstream $(date +%H:%M:%S)] $*"; }

# --- model lists: base (B0/B1, 768) and structural (E1, 771) ---
STATE_DIR="$STATE_DIR" ARMS="B0 B1" bash "$DIR/make_model_list.sh" && cp "$DIR/model_list.txt" "$DIR/model_list_base.txt" || echo "" > "$DIR/model_list_base.txt"
STATE_DIR="$STATE_DIR" ARMS="E1"    bash "$DIR/make_model_list.sh" && cp "$DIR/model_list.txt" "$DIR/model_list_struct.txt" || echo "" > "$DIR/model_list_struct.txt"
HAVE_BASE=$( [ -s "$DIR/model_list_base.txt" ] && echo 1 || echo 0 )
HAVE_STRUCT=$( [ -s "$DIR/model_list_struct.txt" ] && echo 1 || echo 0 )
say "base arms present=$HAVE_BASE  structural arms present=$HAVE_STRUCT"

# --- leakage baseline (no encoder; cheap) + probe graphs (once) ---
say "leakage baseline"
python3 "$DIR/leakage_baseline.py" --data-root "$DATA_ROOT" || say "leakage FAILED"
python3 "$DIR/make_probe_graphs.py" --out-dir "$DATA_ROOT/synthetic_probes/graphs" || say "probe-graph build FAILED"

# --- STAGE T1: benchmark sweep (reg + slp + pl) ---
say "STAGE T1 benchmark"
[ "$HAVE_BASE" = 1 ]   && MODEL_LIST="$DIR/model_list_base.txt"                        bash "$DIR/run_eval_sweep.sh" --gpus "$GPUS"
[ "$HAVE_STRUCT" = 1 ] && STRUCTURAL=directed3 MODEL_LIST="$DIR/model_list_struct.txt" bash "$DIR/run_eval_sweep.sh" --gpus "$GPUS"

# --- STAGE T2: 2x2 ablation ---
say "STAGE T2 2x2"
[ "$HAVE_BASE" = 1 ]   && MODEL_LIST="$DIR/model_list_base.txt"                        bash "$DIR/run_2x2_ablation.sh" --gpus "$GPUS"
[ "$HAVE_STRUCT" = 1 ] && STRUCTURAL=directed3 MODEL_LIST="$DIR/model_list_struct.txt" bash "$DIR/run_2x2_ablation.sh" --gpus "$GPUS"

# --- STAGE T3: capability probes ---
say "STAGE T3 probes"
[ "$HAVE_BASE" = 1 ]   && MODEL_LIST="$DIR/model_list_base.txt"                        bash "$DIR/run_capability_probes.sh" --gpus "$GPUS"
[ "$HAVE_STRUCT" = 1 ] && STRUCTURAL=directed3 MODEL_LIST="$DIR/model_list_struct.txt" bash "$DIR/run_capability_probes.sh" --gpus "$GPUS"

# --- FINAL authoritative parse over ALL arms (the per-stage parses were per-group) ---
say "final parse (all arms)"
STATE_DIR="$STATE_DIR" ARMS="B0 B1 E1" bash "$DIR/make_model_list.sh"   # -> model_list.txt (all present)
python3 scripts/analysis/benchmark_tasks/parse_benchmark_eval_logs.py --log-root "$LOG_ROOT" --out-dir scripts/plotting || say "benchmark parse FAILED"
python3 "$DIR/parse_2x2.py" --log-root "$LOG_ROOT" --model-list "$DIR/model_list.txt" --out scripts/plotting/topology_feature_ssl/data/ablation_2x2.csv || say "2x2 parse FAILED"
python3 "$DIR/parse_capability_probes.py" --log-root "$LOG_ROOT" --model-list "$DIR/model_list.txt" --out scripts/plotting/topology_feature_ssl/data/capability_probes.csv || say "probe parse FAILED"

# render a glance-able RESULTS.md from the parsed CSVs (numbers without running the notebook)
python3 "$DIR/render_results.py" || say "render FAILED"

say "DOWNSTREAM_DONE"
