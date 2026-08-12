#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
GRAPH="${GRAPH:-/dataMeR1/phil/data/merged/graphs/ukr_rus_covid_midterm_all9_facebook_final_core_split_seed0.pt}"
IDENTITY_DB="${IDENTITY_DB:-/dataMeR1/phil/gfm/prodigy-identityaudit/state/identity_overlap_audit/v002/identity_overlap.duckdb}"
TRAINING_STATE_ROOT="${TRAINING_STATE_ROOT:-/dataMeR1/phil/gfm/prodigy-final-core/state/final_core}"
ORIGINAL_RESULTS_ROOT="${ORIGINAL_RESULTS_ROOT:-/dataMeR1/phil/gfm/prodigy-final-core-fixed-test/log/final_core_fixed_test/production/bs32/results}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/state/entity_disjoint_eval/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/entity_disjoint_eval/${RUN_ID}}"
EXCLUSION_ROOT="${EXCLUSION_ROOT:-${REPO_ROOT}/state/entity_disjoint_eval/center_clean_v001/exclusions}"
WORKER_COUNT=2

mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$LOG_ROOT/ready"
cd "$REPO_ROOT"
PIDS=()

cleanup() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
for target in ukr_rus covid midterm; do
  [[ -f "$EXCLUSION_ROOT/${target}.pt" ]] || {
    echo "missing frozen exclusion artifact: $EXCLUSION_ROOT/${target}.pt" >&2
    exit 1
  }
done
[[ -f "$EXCLUSION_ROOT/summary.json" ]] || {
  echo "missing frozen exclusion summary: $EXCLUSION_ROOT/summary.json" >&2
  exit 1
}

conda activate bio-embeddings-v001
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
GRAPH_SHA256="$(python -c 'import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); from protocol import sha256_file; print(sha256_file(Path(sys.argv[2])))' "$SCRIPT_DIR" "$GRAPH")"
IDENTITY_DB_SHA256="$(python -c 'import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); from protocol import sha256_file; print(sha256_file(Path(sys.argv[2])))' "$SCRIPT_DIR" "$IDENTITY_DB")"

conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1
export ENTITY_DISJOINT_CPU_THREADS="${ENTITY_DISJOINT_CPU_THREADS:-24}"

for gpu in 0 1; do
  used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if (( used > 1000 )); then
    echo "GPU $gpu is busy (${used} MiB); refusing to launch" >&2
    exit 1
  fi
done

available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
available_gib=$((available_kib / 1024 / 1024))
if (( available_gib < 380 )); then
  echo "Only ${available_gib} GiB host RAM available; need at least 380 GiB" >&2
  exit 1
fi

for worker in 0 1; do
  CUDA_VISIBLE_DEVICES="$worker" python -u "$SCRIPT_DIR/evaluate_center_disjoint.py" \
    --worker-index "$worker" --worker-count "$WORKER_COUNT" \
    --targets ukr_rus,covid,midterm --batch-size 32 --candidate-multiplier 2 \
    --exclusion-level induced_subgraph \
    --config "$SCRIPT_DIR/../final_core/training.yaml" \
    --training-state-root "$TRAINING_STATE_ROOT" --training-run-stamp 20260807 \
    --evaluation-state-root "$RUN_ROOT/eval_state_worker${worker}" \
    --evaluation-log-root "$LOG_ROOT/internal_worker${worker}" \
    --results-root "$RUN_ROOT/results" \
    --evaluation-run-stamp "$RUN_ID" \
    --exclusion-root "$EXCLUSION_ROOT" \
    --graph-sha256 "$GRAPH_SHA256" --identity-db-sha256 "$IDENTITY_DB_SHA256" \
    --original-results-root "$ORIGINAL_RESULTS_ROOT" \
    --ready-dir "$LOG_ROOT/ready" --expected-workers "$WORKER_COUNT" \
    --min-host-reserve-gib 128 \
    > "$LOG_ROOT/worker${worker}.log" 2>&1 &
  PIDS+=("$!")
done
status=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || status=1
done
PIDS=()
(( status == 0 )) || { echo "one or more evaluation workers failed" >&2; exit 1; }

python "$SCRIPT_DIR/aggregate_center_disjoint.py" \
  --variant induced \
  --results-root "$RUN_ROOT/results" \
  --original-results-root "$ORIGINAL_RESULTS_ROOT" \
  --output-root "$RUN_ROOT/summary" \
  > "$LOG_ROOT/aggregate.log" 2>&1
date -u +%FT%TZ > "$RUN_ROOT/complete_utc.txt"
echo "INDUCED_DISJOINT_COMPLETE run_root=$RUN_ROOT"
