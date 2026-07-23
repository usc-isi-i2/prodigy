#!/usr/bin/env bash
# End-to-end autonomous pipeline for the order-robustness ladder.
#
#   gate_wait -> gate_eval -> gate_check -> rungs_train -> rungs_eval -> assemble
#
# Runs entirely ON TUCKER inside tmux so it survives laptop sleep, VPN drops, and the
# controlling session going away. Progress is written to run_logs/pipeline_status.txt as
# a single parseable line, so a watcher only has to read one file.
#
#   tmux new-session -d -s nmlor_pipeline \
#     'export PATH="/home/mhchu/miniconda3/bin:$PATH"; bash .../run_pipeline_tucker.sh'
#
#   TRAIN_GPUS="0 1 2" EVAL_GPUS="0,1,2" ./run_pipeline_tucker.sh
#   SKIP_GATE=1 ./run_pipeline_tucker.sh    # gate already passed; go straight to rungs
#
# The gate is a hard stop: if check_gate.py fails, the 11 rungs are NOT launched, because
# they would all be built on an invalid shortcut. That is the one decision this pipeline
# makes on its own, and it makes it conservatively.
#
# Deliberately NOT `set -e`: every phase failure must be recorded in the status file
# before exiting, otherwise a watcher cannot tell "failed" from "still running".
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/run_logs"
mkdir -p "${LOG_DIR}"

STATUS_FILE="${LOG_DIR}/pipeline_status.txt"
PIPELOG="${LOG_DIR}/pipeline.log"
PY="/home/mhchu/miniconda3/envs/prodigy/bin/python"

TRAIN_GPUS="${TRAIN_GPUS:-0 1 2}"    # GPU 3 left alone: labmates' ollama serves there
EVAL_GPUS="${EVAL_GPUS:-0,1,2}"
# Two different identifiers, do not conflate them:
#   GATE_PREFIX -- the yaml's `prefix:`, used for state/ dirs and eval log dirs
#   GATE_CONFIG -- the config FILENAME, which is what appears in the trainer's cmdline.
# pgrep must match the config name; the prefix never appears in /proc/<pid>/cmdline.
GATE_PREFIX="nm_ladder_gate_ordA_r4"
GATE_CONFIG="train_gate_ordA_r4"
PARTIAL=0

say() { echo "[$(date +%F_%T)] $*" | tee -a "${PIPELOG}"; }
set_status() {
  echo "PHASE=$1 STATUS=$2 TS=$(date +%F_%T) NOTE=${3:-}" > "${STATUS_FILE}"
  say "PHASE=$1 STATUS=$2 ${3:-}"
}
fail() { set_status "$1" FAILED "${2:-}"; say "PIPELINE FAILED at $1"; exit 1; }

say "=== pipeline start (train_gpus='${TRAIN_GPUS}' eval_gpus='${EVAL_GPUS}') ==="

# ---------------------------------------------------------------- phase 1: gate_wait
if [[ "${SKIP_GATE:-0}" != "1" ]]; then
  set_status gate_wait RUNNING
  deadline=$(( $(date +%s) + 10800 ))          # 3h cap
  while :; do
    ckpt="$(ls -d "${REPO_ROOT}/state/${GATE_PREFIX}_"*/checkpoint/state_dict_40000.ckpt 2>/dev/null | head -1)"
    if [[ -n "${ckpt}" ]]; then
      say "gate checkpoint present: ${ckpt}"
      break
    fi
    if ! pgrep -f "run_single_experiment.py.*${GATE_CONFIG}" >/dev/null; then
      fail gate_wait "trainer gone with no 40k checkpoint"
    fi
    if (( $(date +%s) > deadline )); then
      fail gate_wait "exceeded 3h"
    fi
    sleep 120
  done

  # The trainer keeps going to its planned 50k; 40k is the matched budget we evaluate,
  # so stop it here to free the GPU for the rungs.
  if pgrep -f "run_single_experiment.py.*${GATE_CONFIG}" >/dev/null; then
    say "stopping gate trainer (40k reached; 50k planned is surplus)"
    pkill -f "run_single_experiment.py.*${GATE_CONFIG}"
    sleep 20
  fi

  # ------------------------------------------------------------- phase 2: gate_eval
  set_status gate_eval RUNNING
  GATE=1 STATE_DIR="${REPO_ROOT}/state" bash "${SCRIPT_DIR}/make_model_list.sh" \
    >>"${PIPELOG}" 2>&1 || fail gate_eval "make_model_list failed"
  GATE=1 GPUS="${EVAL_GPUS}" bash "${SCRIPT_DIR}/eval_ladder_tucker.sh" \
    >>"${LOG_DIR}/gate_eval.log" 2>&1 || fail gate_eval "eval harness failed"

  # ------------------------------------------------------------ phase 3: gate_check
  set_status gate_check RUNNING
  "${PY}" "${SCRIPT_DIR}/check_gate.py" --log-root "${REPO_ROOT}/log" \
    >>"${LOG_DIR}/gate_check.log" 2>&1
  rc=$?
  cat "${LOG_DIR}/gate_check.log" >>"${PIPELOG}"
  if [[ ${rc} -ne 0 ]]; then
    set_status gate_check GATE_FAILED "check_gate.py exit ${rc}; 11 rungs NOT launched"
    say "the subset shortcut did not reproduce the published rung -- stopping by design."
    say "see ${LOG_DIR}/gate_check.log for the per-column deltas."
    exit 1
  fi
  say "GATE PASSED"
else
  say "SKIP_GATE=1 -- assuming the gate already passed"
fi

# ------------------------------------------------------------- phase 4: rungs_train
set_status rungs_train RUNNING
GPUS="${TRAIN_GPUS}" bash "${SCRIPT_DIR}/run_all_train_tucker.sh" \
  >>"${LOG_DIR}/rungs_train.log" 2>&1
train_rc=$?
if [[ ${train_rc} -ne 0 ]]; then
  say "WARNING: at least one rung reported a failure (rc=${train_rc}); continuing with whatever trained"
  PARTIAL=1
fi

# -------------------------------------------------------------- phase 5: rungs_eval
set_status rungs_eval RUNNING
STATE_DIR="${REPO_ROOT}/state" bash "${SCRIPT_DIR}/make_model_list.sh" >>"${PIPELOG}" 2>&1
ml_rc=$?
n_models=$(wc -l < "${SCRIPT_DIR}/model_list.txt" 2>/dev/null || echo 0)
if [[ "${n_models}" -eq 0 ]]; then
  fail rungs_eval "no trained rungs have a 40k checkpoint"
fi
if [[ ${ml_rc} -ne 0 ]]; then
  say "WARNING: only ${n_models}/11 rungs have 40k checkpoints; evaluating the partial set"
  PARTIAL=1
fi

GPUS="${EVAL_GPUS}" bash "${SCRIPT_DIR}/eval_ladder_tucker.sh" \
  >>"${LOG_DIR}/rungs_eval.log" 2>&1 || fail rungs_eval "eval harness failed"

# ---------------------------------------------------------------- phase 6: assemble
set_status assemble RUNNING
asm_args=(--log-root "${REPO_ROOT}/log")
[[ "${PARTIAL}" == "1" ]] && asm_args+=(--allow-partial)
"${PY}" "${SCRIPT_DIR}/assemble_order_table.py" "${asm_args[@]}" \
  >>"${LOG_DIR}/assemble.log" 2>&1
asm_rc=$?
cat "${LOG_DIR}/assemble.log" >>"${PIPELOG}"
if [[ ${asm_rc} -ne 0 ]]; then
  # Retry permissively so a missing cell still yields a usable CSV to inspect.
  say "assemble reported missing cells; retrying with --allow-partial"
  "${PY}" "${SCRIPT_DIR}/assemble_order_table.py" --log-root "${REPO_ROOT}/log" --allow-partial \
    >>"${LOG_DIR}/assemble.log" 2>&1 || fail assemble "assembler failed even permissively"
  PARTIAL=1
fi

if [[ "${PARTIAL}" == "1" ]]; then
  set_status done PARTIAL "some rungs or cells missing -- see pipeline.log"
else
  set_status done OK "all 11 rungs trained, evaluated, assembled"
fi
say "=== pipeline finished (partial=${PARTIAL}) ==="
