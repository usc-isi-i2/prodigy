#!/usr/bin/env bash
# Evaluate the four seed-0 PinSAGE checkpoints on the frozen final-core NM streams.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
FINAL_CORE_DIR="${REPO_ROOT}/scripts/experiments/setup/final_core"
STATE_ROOT="${STATE_ROOT:-${REPO_ROOT}/state}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPUS_TEXT="${GPUS:-0 1 2 3}"
TARGETS="${TARGETS:-covid_political,election2020,ukr_rus_suspended,twibot20}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/log/pinsage_finalcore_fixed_test/${RUN_ID}}"
REFERENCE_FINGERPRINTS="${REFERENCE_FINGERPRINTS:-${REPO_ROOT}/scripts/experiments/analysis/transfer/matrices/cross_model/final_core/data/prodigy_final_core/fixed_test/summary/episode_fingerprints.tsv}"
EPISODE_PLAN_ROOT="${EPISODE_PLAN_ROOT:-/dataMeR1/phil/gfm/final_core_episode_plans_045ba527}"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1
PYTHON="${CONDA_PREFIX}/bin/python"
mkdir -p "$OUT_ROOT/queue" "$OUT_ROOT/results" "$OUT_ROOT/ready"

arms=(covid election loo_twibot20 twibot20)
sources=(covid election2020 "ukr_rus_suspended,election2020,ukr_rus,facebook_page_reference,cp_hk,covid_political,covid,midterm" twibot20)
manifest="$OUT_ROOT/jobs.json"
"$PYTHON" - "$STATE_ROOT" "$manifest" "${arms[@]}" -- "${sources[@]}" <<'PY'
import glob, json, pathlib, sys
state_root, output = sys.argv[1:3]
sep = sys.argv.index("--")
arms, sources = sys.argv[3:sep], sys.argv[sep + 1:]
rows = []
for arm, source_text in zip(arms, sources):
    pattern = f"{state_root}/nm_{arm}_pinsage_s0_*/checkpoint/state_dict_2500.ckpt"
    matches = sorted(glob.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"expected one checkpoint for {arm}, found {matches}")
    rows.append({"model_id": f"pinsage_{arm}", "seed": 0,
                 "sources": source_text.split(","), "aliases": [f"pinsage:{arm}"],
                 "checkpoint": matches[0]})
pathlib.Path(output).write_text(json.dumps(rows, indent=2) + "\n")
PY

read -r -a gpus <<< "$GPUS_TEXT"
(( ${#gpus[@]} > 0 )) || { echo "GPUS is empty" >&2; exit 2; }
workers=${#gpus[@]}
pids=()
for ((worker=0; worker<workers; worker++)); do
  gpu=${gpus[$worker]}
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  (( used < 12000 )) || { echo "GPU $gpu is too busy (${used} MiB)" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES="$gpu" FINAL_CORE_CPU_THREADS=16 "$PYTHON" -u \
    "$FINAL_CORE_DIR/evaluate_fixed_grid.py" \
    --worker-index "$worker" --worker-count "$workers" --job-manifest "$manifest" \
    --seeds 0 --targets "$TARGETS" --batch-size 32 --episode-count 512 \
    --config "$SCRIPT_DIR/configs/train_covid_pinsage_s0.yaml" \
    --plan-config "$FINAL_CORE_DIR/training.yaml" \
    --plan-member-policy lowest_sorted --plan-member-seed -1 \
    --episode-plan-root "$EPISODE_PLAN_ROOT" \
    --training-state-root "$STATE_ROOT" --evaluation-state-root "$OUT_ROOT/state" \
    --evaluation-log-root "$OUT_ROOT/internal" --results-root "$OUT_ROOT/results" \
    --evaluation-run-stamp "$RUN_ID" --ready-dir "$OUT_ROOT/ready" \
    --expected-workers "$workers" --reference-fingerprints "$REFERENCE_FINGERPRINTS" \
    > "$OUT_ROOT/queue/worker${worker}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || { echo "PinSAGE fixed-test worker failed" >&2; exit 1; }
echo "PINSAGE_FIXED_TEST_COMPLETE results=$OUT_ROOT/results"
