#!/usr/bin/env bash
# End-to-end pipeline for the ladder's downstream sweep.
#
#   resolve -> smoke -> benchmark (reg + pl) -> pair_lp (static LP) -> assemble
#
# Runs entirely ON TUCKER inside tmux so it survives laptop sleep, VPN drops, and the
# controlling session going away. Progress is one parseable line in
# run_logs/pipeline_status.txt, so a watcher only has to read one file.
#
#   tmux new-session -d -s nmld_pipeline \
#     'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
#      bash scripts/experiments/setup/nm_ladder_downstream/run_pipeline_tucker.sh'
#
#   EVAL_GPUS="0,1,2" LP_GPU=0 ./run_pipeline_tucker.sh
#   SKIP_SMOKE=1 ./run_pipeline_tucker.sh      # smoke already green
#   ONLY=pair_lp ./run_pipeline_tucker.sh      # rerun a single phase
#
# No training happens here -- all 21 encoders already exist. The smoke phase is the one
# hard stop: it scores a single encoder on the smallest graph through the pair evaluator
# and checks the validity reads, because a checkpoint/architecture mismatch would
# otherwise produce 336 confidently-wrong rows.
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
PY="${PY:-/home/mhchu/miniconda3/envs/prodigy/bin/python}"

STATE_DIR="${STATE_DIR:-/dataMeR1/phil/gfm/prodigy/state}"
DATA_ROOT="${DATA_ROOT:-/dataMeR1/phil/data}"
# GPU occupancy checked 2026-07-27: rdorn runs a vLLM worker pinned across GPUs 2 AND 3
# (~76 GB each), so only 0 and 1 are ours in practice. This moves -- run
# `nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv` before launching
# and override if the picture has changed.
EVAL_GPUS="${EVAL_GPUS:-0,1}"
LP_GPU="${LP_GPU:-0}"
ONLY="${ONLY:-}"

say() { echo "[$(date +%F_%T)] $*" | tee -a "${PIPELOG}"; }
set_status() {
  echo "PHASE=$1 STATUS=$2 TS=$(date +%F_%T) NOTE=${3:-}" > "${STATUS_FILE}"
  say "PHASE=$1 STATUS=$2 ${3:-}"
}
fail() { set_status "$1" FAILED "${2:-}"; say "PIPELINE FAILED at $1"; exit 1; }
want() { [[ -z "${ONLY}" || "${ONLY}" == "$1" ]]; }

say "=== pipeline start (state='${STATE_DIR}' eval_gpus='${EVAL_GPUS}' lp_gpu='${LP_GPU}') ==="

# ------------------------------------------------------------------ phase 1: resolve
if want resolve; then
  set_status resolve RUNNING
  STATE_DIR="${STATE_DIR}" "${PY}" "${SCRIPT_DIR}/make_model_list.py" \
    >> "${PIPELOG}" 2>&1 \
    || fail resolve "some encoder has no state_dict_40000.ckpt under ${STATE_DIR}"
  n=$(grep -cve '^\s*$' "${SCRIPT_DIR}/model_list.txt")
  [[ "${n}" == "21" ]] || fail resolve "model_list.txt has ${n} lines, expected 21"
  set_status resolve OK "21 encoders resolved"
fi

# -------------------------------------------------------------------- phase 2: smoke
# One encoder, the smallest static-LP graph. Catches an encoder-reconstruction mismatch
# (wrong --emb-dim/--gnn-type/--layers) in minutes instead of after the full sweep.
if want smoke && [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  set_status smoke RUNNING
  head -1 "${SCRIPT_DIR}/model_list.txt" > "${LOG_DIR}/model_list_smoke.txt"
  MODEL_LIST="${LOG_DIR}/model_list_smoke.txt" \
  OUT_DIR="${LOG_DIR}/smoke_pair_lp" \
  DATASETS="cp_hk_twitter" GPU="${LP_GPU}" DATA_ROOT="${DATA_ROOT}" \
    bash "${SCRIPT_DIR}/run_pair_lp_sweep.sh" >> "${PIPELOG}" 2>&1 \
    || fail smoke "pair_link_sweep failed on cp_hk_twitter"
  "${PY}" - "${LOG_DIR}/smoke_pair_lp/cp_hk_twitter__pair_lp.csv" <<'PYCHECK' \
    >> "${PIPELOG}" 2>&1 || fail smoke "validity reads out of range -- see pipeline.log"
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1]))
        if r["model"] != "__floor__" and r["negative_kind"] == "degree_matched"]
assert rows, "no model rows emitted"
for r in rows:
    leak, sens = float(r["leakage_edges"]), float(r["endpoint_sensitivity"])
    perm = float(r["endpoint_permutation_auc"])
    print(f"{r['model']}: auc={r['auc']} leak={leak} sens={sens} perm={perm}")
    assert leak == 0, f"leakage_edges={leak}"
    assert sens > 0.99, f"endpoint_sensitivity={sens}"
    assert abs(perm - 0.5) < 0.05, f"endpoint_permutation_auc={perm}"
print("SMOKE OK")
PYCHECK
  set_status smoke OK
fi

# ---------------------------------------------------------------- phase 3: benchmark
# 252 regression + 84 classification jobs across the eval GPUs.
if want benchmark; then
  set_status benchmark RUNNING
  MODEL_LIST="${SCRIPT_DIR}/model_list.txt" DATA_ROOT="${DATA_ROOT}" \
    bash "${SCRIPT_DIR}/run_eval_sweep.sh" --gpus "${EVAL_GPUS}" >> "${PIPELOG}" 2>&1 \
    || fail benchmark "reg/pl sweep returned nonzero"
  grep -q NM_LADDER_DOWNSTREAM_EVAL_SWEEP_DONE "${PIPELOG}" \
    || fail benchmark "sweep did not reach its done marker"
  set_status benchmark OK
fi

# ------------------------------------------------------------------ phase 4: pair_lp
# 5 graph passes, serial (the adjacency build is the memory peak).
if want pair_lp; then
  set_status pair_lp RUNNING
  MODEL_LIST="${SCRIPT_DIR}/model_list.txt" DATA_ROOT="${DATA_ROOT}" \
  GPU="${LP_GPU}" bash "${SCRIPT_DIR}/run_pair_lp_sweep.sh" >> "${PIPELOG}" 2>&1 \
    || fail pair_lp "pair_link_sweep returned nonzero"
  set_status pair_lp OK
fi

# ----------------------------------------------------------------- phase 5: assemble
if want assemble; then
  set_status assemble RUNNING
  "${PY}" "${SCRIPT_DIR}/assemble_downstream_tables.py" >> "${PIPELOG}" 2>&1
  rc=$?
  if [[ ${rc} -eq 1 ]]; then
    say "assemble reported missing cells; re-running with --allow-partial"
    "${PY}" "${SCRIPT_DIR}/assemble_downstream_tables.py" --allow-partial \
      >> "${PIPELOG}" 2>&1 || fail assemble "assembler failed even with --allow-partial"
    set_status assemble PARTIAL "some cells missing -- see pipeline.log"
  elif [[ ${rc} -ne 0 ]]; then
    fail assemble "assembler exited ${rc}"
  else
    set_status assemble OK
  fi
fi

say "=== pipeline done ==="
grep -q "STATUS=PARTIAL" "${STATUS_FILE}" || set_status pipeline OK
